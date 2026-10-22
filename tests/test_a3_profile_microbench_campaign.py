from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "a3_profile_microbench_campaign.py"
MANIFEST = ROOT / "experiments" / "a3-profile-hypotheses.json"


def load_module():
    spec = importlib.util.spec_from_file_location("a3_profile_microbench_campaign", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evidence(selector: str, duration: float, *, handle="remote:bz-a3-1:job:x"):
    return {
        "ok": True,
        "handle": handle,
        "evidence": {
            "capture": {
                "observed_kernel_names": [selector],
                "selected_kernel": selector,
                "rows": [{
                    "block_id": "0", "sub_block_id": "vector0",
                    "aiv_time(us)": str(duration), "aiv_vec_ratio": "0.6",
                    "aiv_mte2_ratio": "0.2", "aiv_scalar_ratio": "0.3",
                }],
                "sources": [{"sha256": "a" * 64, "role": "PipeUtilization"}],
            },
            "device": {"physical_id": 2, "logical_id": 0},
            "dispatch_key": "key",
        },
    }


def test_timeline_capture_is_not_blocked_by_unstable_latency(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["id"] == "repeated-phases/aba/before")
    calls = []
    values = iter([1.0, 2.0, 4.0])

    def acquire(phase, current, attempt, selector):
        calls.append(phase)
        result = evidence(current["kernel_names"][0], 0)
        if phase == "pipe":
            return evidence(selector, next(values))
        if phase == "timeline":
            result["evidence"]["capture"].update(
                availability="unavailable", diagnostics={"classification": "raw_binary_only"},
                timeline_semantics={"event_count": 0, "events": []},
            )
        return result

    result = module.run_cell(cell, tmp_path, acquire, samples=3,
                             max_attempts=3, max_relative_range=0.05)
    assert result["status"] == "inconclusive"
    assert result["measurements"]["stable"] is False
    assert calls == ["basic", "pipe", "pipe", "pipe", "timeline"]
    assert result["timeline"]["status"] == "unavailable"
    assert result["timeline"]["diagnostics"]["classification"] == "raw_binary_only"


def test_zero_exit_timeline_collection_failure_preserves_timing_but_fails_cell(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["id"] == "ordered-phases/balanced/before")

    def acquire(phase, current, attempt, selector):
        result = evidence(current["kernel_names"][0], 3.0)
        if phase == "timeline":
            result["evidence"]["capture"].update(
                availability="unavailable", diagnostics={"classification": "collection_failed"},
                reason="kernel context save failed", timeline_semantics={"events": []},
            )
        return result

    result = module.run_cell(cell, tmp_path, acquire, samples=3,
                             max_attempts=3, max_relative_range=0.05)
    assert result["status"] == "profiling_failure"
    assert result["measurements"]["stable"] is True
    assert result["measurements"]["median"] == 3.0
    assert result["timeline"]["status"] == "profiling_failure"


def test_enumeration_is_complete_and_shards_are_disjoint():
    module = load_module()
    manifest = json.loads(MANIFEST.read_text())
    cells = module.enumerate_cells(manifest)
    assert len(cells) == 52
    assert len({cell["id"] for cell in cells}) == len(cells)
    shards = [module.select_shard(cells, index, 3) for index in range(3)]
    assert set().union(*(set(x["id"] for x in shard) for shard in shards)) == {
        cell["id"] for cell in cells
    }
    assert not (set(x["id"] for x in shards[0]) & set(x["id"] for x in shards[1]))
    for cell in cells[::2]:
        pair = {cell["id"], cell["id"].replace("/before", "/after")}
        assert any(pair <= {item["id"] for item in shard} for shard in shards)


def test_cell_collects_three_stable_exact_pipe_samples(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["id"] == "loop-bound/short-local/before")
    calls = []
    durations = iter((10.0, 10.2, 9.9))

    def acquire(phase, current, attempt, selector):
        calls.append((phase, attempt, selector))
        if phase == "basic":
            result = evidence("a3_vector_probe", 0)
            result["evidence"]["capture"]["observed_kernel_names"] = [
                "framework_setup", "a3_vector_probe"
            ]
            return result
        return evidence(selector, next(durations))

    result = module.run_cell(cell, tmp_path, acquire, samples=3, max_attempts=5,
                             max_relative_range=0.05)
    assert result["status"] == "success"
    assert len(result["measurements"]["samples"]) == 3
    assert result["measurements"]["median"] == 10.0
    assert result["signals"]["aiv_vec_ratio"]["mean"] == 0.6
    assert [call[0] for call in calls] == ["basic", "pipe", "pipe", "pipe"]
    assert len(result["provenance"]["handles"]) == 4


def test_early_outlier_does_not_poison_first_consecutive_stable_window(tmp_path):
    module = load_module()
    cell = module.enumerate_cells(json.loads(MANIFEST.read_text()))[0]
    values = iter((100.0, 10.0, 10.2, 9.9))

    pipe_index = 0

    def acquire(phase, current, attempt, selector):
        nonlocal pipe_index
        if phase == "basic":
            return evidence(current["kernel_names"][0], 0)
        result = evidence(selector, next(values))
        result["evidence"]["capture"]["rows"][0]["aiv_vec_ratio"] = str(
            (9.0, 0.4, 0.5, 0.6)[pipe_index]
        )
        pipe_index += 1
        return result

    result = module.run_cell(cell, tmp_path, acquire, samples=3, max_attempts=7,
                             max_relative_range=0.05)
    assert result["status"] == "success"
    assert result["measurements"]["samples"] == [100.0, 10.0, 10.2, 9.9]
    assert result["measurements"]["selected_window"] == [10.0, 10.2, 9.9]
    assert result["measurements"]["median"] == 10.0
    assert result["signals"]["aiv_vec_ratio"]["values"] == [0.4, 0.5, 0.6]


def test_infrastructure_failure_is_discarded_but_candidate_failure_is_counted(tmp_path):
    module = load_module()
    cell = module.enumerate_cells(json.loads(MANIFEST.read_text()))[0]
    counter = 0

    def acquire(phase, current, attempt, selector):
        nonlocal counter
        counter += 1
        if counter == 1:
            return {"ok": False, "classification": "transport_error", "message": "flaky"}
        if phase == "basic":
            return evidence(current["kernel_names"][0], 0)
        return evidence(selector, 10.0)

    interrupted = module.run_cell(cell, tmp_path, acquire, samples=1, max_attempts=3,
                                  max_relative_range=0.05)
    assert interrupted["status"] == "running"
    assert interrupted["attempts"][0]["identity_consumed"] is False
    result = module.run_cell(cell, tmp_path, acquire, samples=1, max_attempts=3,
                             max_relative_range=0.05)
    assert result["status"] == "success"
    assert result["discarded_infrastructure_attempts"] == 1

    def compile_failure(*_args):
        return {"ok": False, "classification": "workload_failure", "message": "compile"}

    failed = module.run_cell(cell, tmp_path / "candidate", compile_failure,
                             samples=1, max_attempts=3, max_relative_range=0.05)
    assert failed["status"] == "candidate_failure"
    assert failed["failure"]["classification"] == "workload_failure"

    def profiler_failure(*_args):
        return {"ok": False, "classification": "profiler_failure", "message": "metric"}

    profiled = module.run_cell(cell, tmp_path / "profiler", profiler_failure,
                               samples=1, max_attempts=3, max_relative_range=0.05)
    assert profiled["status"] == "profiling_failure"


def test_unstable_samples_retry_to_cap_and_remain_inconclusive(tmp_path):
    module = load_module()
    cell = module.enumerate_cells(json.loads(MANIFEST.read_text()))[0]
    values = iter((10.0, 20.0, 10.0, 20.0))

    def acquire(phase, current, attempt, selector):
        if phase == "basic":
            return evidence(current["kernel_names"][0], 0)
        return evidence(selector, next(values))

    result = module.run_cell(cell, tmp_path, acquire, samples=3, max_attempts=4,
                             max_relative_range=0.05)
    assert result["status"] == "inconclusive"
    assert result["measurements"]["stable"] is False
    assert len(result["measurements"]["samples"]) == 4


def test_timeline_absence_is_unavailable_not_failure(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["id"] == "repeated-phases/aba/after")

    def acquire(phase, current, attempt, selector):
        return evidence(current["kernel_names"][0], 3.0)

    result = module.run_cell(cell, tmp_path, acquire, samples=1, max_attempts=1,
                             max_relative_range=0.05)
    assert result["status"] == "success"
    assert result["timeline"]["status"] == "unavailable"
    assert "attempted acquisition" in result["timeline"]["reason"]
    assert result["timeline"]["handle"] == "remote:bz-a3-1:job:x"


def test_timeline_runs_once_only_after_stable_timing(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["kind"] == "timeline")
    calls = []

    def acquire(phase, current, attempt, selector):
        calls.append((phase, attempt))
        return evidence(current["kernel_names"][0], 3.0)

    result = module.run_cell(cell, tmp_path, acquire, samples=3, max_attempts=7,
                             max_relative_range=0.05)
    assert result["status"] == "success"
    assert calls == [("basic", 0), ("pipe", 0), ("pipe", 1), ("pipe", 2),
                     ("timeline", 3)]


def test_timeline_transport_failure_stays_running_and_resumes_same_attempt(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["kind"] == "timeline")
    calls = []
    timeline_calls = 0

    def acquire(phase, current, attempt, selector):
        nonlocal timeline_calls
        calls.append((phase, attempt))
        if phase == "timeline":
            timeline_calls += 1
            if timeline_calls == 1:
                return {
                    "ok": False, "classification": "transport_error",
                    "phase": "observe", "message": "observer disconnected",
                    "handle": "remote:gz-a3:job:timeline-kept",
                }
        return evidence(current["kernel_names"][0], 3.0,
                        handle="remote:gz-a3:job:timeline-kept")

    interrupted = module.run_cell(
        cell, tmp_path, acquire, samples=1, max_attempts=3,
        max_relative_range=0.05,
    )
    assert interrupted["status"] == "running"
    assert interrupted["timeline"]["status"] == "pending"
    assert interrupted["timeline"]["handle"] == "remote:gz-a3:job:timeline-kept"

    resumed = module.run_cell(
        cell, tmp_path, acquire, samples=1, max_attempts=3,
        max_relative_range=0.05,
    )
    assert resumed["status"] == "success"
    assert resumed["timeline"]["status"] == "unavailable"
    assert [item for item in calls if item[0] == "pipe"] == [("pipe", 0)]
    assert [item for item in calls if item[0] == "timeline"] == [
        ("timeline", 1), ("timeline", 1),
    ]


def test_timeline_terminal_infrastructure_failure_retries_without_kernel_failure(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["kind"] == "timeline")
    timeline_calls = 0

    def acquire(phase, current, attempt, selector):
        nonlocal timeline_calls
        if phase == "timeline":
            timeline_calls += 1
            if timeline_calls == 1:
                return {
                    "ok": False, "classification": "host_environment",
                    "phase": "logs", "message": "terminal logs missing",
                    "handle": "remote:gz-a3:job:timeline-host-failure",
                }
        return evidence(current["kernel_names"][0], 3.0)

    interrupted = module.run_cell(
        cell, tmp_path, acquire, samples=1, max_attempts=3,
        max_relative_range=0.05,
    )
    assert interrupted["status"] == "running"
    assert interrupted["discarded_infrastructure_attempts"] == 1
    assert "remote:gz-a3:job:timeline-host-failure" in interrupted["provenance"]["handles"]

    resumed = module.run_cell(
        cell, tmp_path, acquire, samples=1, max_attempts=3,
        max_relative_range=0.05,
    )
    assert resumed["status"] == "success"
    assert resumed["timeline"]["status"] == "unavailable"


def test_timeline_profiler_failure_is_not_reported_as_unavailable(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["kind"] == "timeline")

    def acquire(phase, current, attempt, selector):
        if phase == "timeline":
            return {
                "ok": False, "classification": "profiler_failure",
                "phase": "profiler", "message": "capture failed",
                "handle": "remote:gz-a3:job:timeline-profiler-failure",
            }
        return evidence(current["kernel_names"][0], 3.0)

    result = module.run_cell(
        cell, tmp_path, acquire, samples=1, max_attempts=2,
        max_relative_range=0.05,
    )
    assert result["status"] == "profiling_failure"
    assert result["timeline"]["status"] == "profiling_failure"


def test_timeline_passes_every_expected_selector_and_retains_semantics(tmp_path):
    module = load_module()
    cell = next(x for x in module.enumerate_cells(json.loads(MANIFEST.read_text()))
                if x["id"] == "fused-vs-split/balanced/after")
    selectors = []

    def acquire(phase, current, attempt, selector):
        if phase == "timeline":
            selectors.append(selector)
            result = evidence(current["kernel_names"][0], 3.0)
            result["evidence"]["capture"].update({
                "availability": "available",
                "reason": "semantic timeline parsed",
                "timeline_semantics": {
                    "selectors_complete": True,
                    "kernel_order": current["kernel_names"],
                    "phase_order": ["load", "compute", "store"],
                    "phase_duration_us": {"load": 3.0, "compute": 7.0, "store": 3.0},
                    "load_compute_overlap_us": 0.0,
                    "kernel_boundaries": [
                        {"name": name, "start_us": index * 3.0, "duration_us": 3.0}
                        for index, name in enumerate(current["kernel_names"])
                    ],
                },
            })
            return result
        result = evidence(current["kernel_names"][0] if phase == "basic" else selector, 3.0)
        if phase == "basic":
            result["evidence"]["capture"]["observed_kernel_names"] = current["kernel_names"]
        return result

    result = module.run_cell(cell, tmp_path, acquire, samples=1, max_attempts=1,
                             max_relative_range=0.05)

    assert selectors == [["a3_phase_copy", "a3_phase_compute", "a3_phase_store"]]
    assert result["timeline"]["semantics"]["selectors_complete"] is True
    assert len(result["timeline"]["semantics"]["kernel_boundaries"]) == 3


def test_pair_runner_interleaves_before_and_after_attempts(tmp_path):
    module = load_module()
    cells = module.enumerate_cells(json.loads(MANIFEST.read_text()))[:2]
    calls = []

    def acquire_for(cell, cell_dir):
        def acquire(phase, current, attempt, selector):
            calls.append((current["variant"], phase, attempt))
            return evidence(current["kernel_names"][0], 10.0)
        return acquire

    results = module.run_pair_interleaved(
        cells, tmp_path, acquire_for, samples=3, max_attempts=7,
        max_relative_range=0.05, identities={cell["id"]: cell["id"] for cell in cells}
    )
    assert [result["status"] for result in results] == ["success", "success"]
    pipe_order = [variant for variant, phase, _ in calls if phase == "pipe"]
    assert pipe_order == ["before", "after"] * 3


def test_qualification_receipt_requires_target_sources_and_stability(tmp_path, monkeypatch):
    module = load_module()
    identity = module.campaign_identity("gz-a3", json.loads(MANIFEST.read_text()))
    assert identity["runtime"] == "py311-torch"
    probes = []
    for probe_id in module.QUALIFICATION_PROBES:
        probes.append({
            "id": probe_id, "samples": [10.0, 10.2, 9.9, 10.1, 10.0],
            "handles": [f"remote:gz-a3:job:{probe_id}-{i}" for i in range(5)],
            "devices": [{"physical_id": i % 2, "logical_id": 0} for i in range(5)],
            "sources": [{"sha256": f"{i:064x}", "role": "PipeUtilization"}
                        for i in range(5)],
        })
    receipt = module.build_qualification_receipt(identity, probes)
    assert receipt["passed"] is True
    path = tmp_path / "qualification.json"
    path.write_bytes(module.canonical(receipt) + b"\n")
    module.validate_qualification(path, identity)
    helper = tmp_path / "profile_device.py"
    helper.write_bytes(module.DEVICE_HELPER.read_bytes() + b"\n# changed selector\n")
    monkeypatch.setattr(module, "DEVICE_HELPER", helper)
    with pytest.raises(ValueError, match="identity"):
        module.validate_qualification(path, module.campaign_identity("gz-a3", json.loads(MANIFEST.read_text())))
    with pytest.raises(ValueError, match="identity"):
        module.validate_qualification(path, {**identity, "target": "bz-a3-1"})
    # A previously passing default-runtime gate must not admit named-runtime work.
    receipt["identity"]["runtime"] = None
    path.write_bytes(module.canonical(receipt) + b"\n")
    with pytest.raises(ValueError, match="identity"):
        module.validate_qualification(path, module.campaign_identity(
            "gz-a3", json.loads(MANIFEST.read_text())
        ))


@pytest.mark.parametrize(
    "probe_ids",
    [
        ("wrong-probe-one", "wrong-probe-two"),
        ("loop-bound/short-local/before", "loop-bound/short-local/before"),
        ("loop-bound/short-local/before",),
    ],
)
def test_qualification_requires_exact_unique_designated_probes(probe_ids):
    module = load_module()
    identity = {"target": "gz-a3", "runtime": None,
                "manifest_sha256": "a" * 64, "benchmark_sha256": "b" * 64}
    probes = [{
        "id": probe_id, "samples": [10.0] * 5,
        "handles": [f"remote:gz-a3:job:{index}" for index in range(5)],
        "devices": [{"physical_id": index, "logical_id": 0} for index in range(5)],
        "sources": [{"sha256": f"{index:064x}", "role": "PipeUtilization"}
                    for index in range(5)],
    } for probe_id in probe_ids]
    assert module.build_qualification_receipt(identity, probes)["passed"] is False


def test_qualification_rejects_all_five_range_above_ten_percent():
    module = load_module()
    identity = {"target": "gz-a3", "runtime": None,
                "manifest_sha256": "a" * 64, "benchmark_sha256": "b" * 64}
    probes = [{"id": module.QUALIFICATION_PROBES[0], "samples": [10, 10, 10, 10, 12],
               "handles": [], "devices": [], "sources": []},
              {"id": module.QUALIFICATION_PROBES[1], "samples": [10, 10, 10, 10, 10],
               "handles": [], "devices": [], "sources": []}]
    receipt = module.build_qualification_receipt(identity, probes)
    assert receipt["passed"] is False
    assert "all-five" in receipt["reasons"][0]


def test_existing_dispatch_receipt_resumes_same_handle(tmp_path, monkeypatch):
    module = load_module()
    evidence_path = tmp_path / "basic.json"
    receipt = Path(str(evidence_path) + ".dispatch.json")
    receipt.write_text(json.dumps({"handle": "remote:bz-a3-1:job:kept"}))
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        evidence_path.write_text(json.dumps({"capture": {"observed_kernel_names": ["k"]}}))
        return module.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module.invoke_acquisition(
        phase="basic", target="bz-a3-1", dispatch_key="key",
        bundle=tmp_path, entrypoint="benchmarks/a3_profile_microbench.py",
        workload_args=["--contract"], evidence_path=evidence_path,
        selector=None, basic_evidence=None, cpl_remote=None,
    )
    assert result["ok"] is True
    assert "--resume-handle" in observed["command"]
    assert "remote:bz-a3-1:job:kept" in observed["command"]


def test_failed_new_dispatch_preserves_handle_written_by_acquirer(tmp_path, monkeypatch):
    module = load_module()
    receipt_path = tmp_path / "basic.json.dispatch.json"

    def fake_run(command, **kwargs):
        receipt_path.write_text(json.dumps({"handle": "remote:gz-a3:job:kept"}))
        return module.subprocess.CompletedProcess(
            command, 2, "",
            "acquisition failed: classification=host_environment phase=logs "
            "handle=remote:gz-a3:job:kept: missing diagnostics\n",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module.invoke_acquisition(
        phase="basic", target="gz-a3", dispatch_key="key", bundle=tmp_path,
        entrypoint="main.py", workload_args=[], evidence_path=tmp_path / "basic.json",
        selector=None, basic_evidence=None, cpl_remote=None,
    )
    assert result["ok"] is False
    assert result["classification"] == "host_environment"
    assert result["handle"] == "remote:gz-a3:job:kept"
    assert result["retired_receipt"].endswith(".retired")


def test_cached_acquisition_is_bound_to_bundle_and_dispatch_identity(tmp_path):
    campaign = load_module()
    acquire_spec = importlib.util.spec_from_file_location(
        "test_campaign_acquire", ROOT / "scripts/acquire_profile.py"
    )
    acquire = importlib.util.module_from_spec(acquire_spec)
    assert acquire_spec.loader
    acquire_spec.loader.exec_module(acquire)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    workload = bundle / "main.py"
    workload.write_text("print('one')\n")
    identity_value = acquire.build_bundle(
        bundle=bundle, entrypoint="main.py", arguments=[]
    ).identity
    request = {
        "schema_version": 1, "mode": "basic", "product": "a3",
        "target": "bz-a3-1", "runtime": "py311-torch", "dispatch_key": "test-key",
        "workload": identity_value, "kernel_names": [], "launch_count": 1,
    }
    output = tmp_path / "evidence.json"
    empty_digest = hashlib.sha256(b"").hexdigest()
    output.write_text(json.dumps({
        "schema_version": 2, "schema": "cpl.profile-acquisition.v2",
        "status": "success", "phase": "basic", "product": "a3",
        "target": "bz-a3-1", "runtime": "py311-torch", "dispatch_key": "test-key",
        "workload": identity_value, "provenance": {"product": "a3", "target": "bz-a3-1"},
        "normal_run": {"exit_code": 0,
                       "stdout": {"bytes": 0, "sha256": empty_digest},
                       "stderr": {"bytes": 0, "sha256": empty_digest}},
        "capture": {"metric": "BasicInfo", "observed_kernel_names": ["kernel"],
                    "rows": [], "sources": []},
        "saturation": {"state": "unknown"},
    }))
    receipt = Path(str(output) + ".dispatch.json")
    receipt.write_text(json.dumps({
        "schema_version": 1, "handle": "remote:bz-a3-1:job:cached",
        "request": request,
        "request_sha256": hashlib.sha256(campaign.canonical(request)).hexdigest(),
    }))
    result = campaign.invoke_acquisition(
        phase="basic", target="bz-a3-1", dispatch_key="test-key", bundle=bundle,
        entrypoint="main.py", workload_args=[], evidence_path=output, selector=None,
        basic_evidence=None, cpl_remote=None, launch_count=1,
    )
    assert result["ok"] is True

    workload.write_text("print('two')\n")
    stale = campaign.invoke_acquisition(
        phase="basic", target="bz-a3-1", dispatch_key="test-key", bundle=bundle,
        entrypoint="main.py", workload_args=[], evidence_path=output, selector=None,
        basic_evidence=None, cpl_remote=None, launch_count=1,
    )
    assert stale["ok"] is False
    assert stale["classification"] == "evidence_failure"


def test_merge_rejects_overlap_and_combines_disjoint_results(tmp_path):
    module = load_module()
    manifest = {
        "hypotheses": [{
            "id": "pair", "expected": {"direction": "no_change", "minimum_relative_change": 0.1},
            "signal_predictions": [], "variants": {"before": "b", "after": "a"},
            "kernel_names": {"before": ["k"], "after": ["k"]},
            "sweeps": [{"id": "one"}], "expected_evidence": "pair", "falsifier": "none",
        }],
        "timeline_probes": [],
    }
    digest = hashlib.sha256(module.canonical(manifest)).hexdigest()
    identity = module.campaign_identity("gz-a3", manifest)
    expected = module.enumerate_cells(manifest)
    one = {"schema_version": 1, "manifest_sha256": digest,
           "campaign_identity": identity, "shard": {"index": 0, "count": 2},
           "raw_cells": [expected[0]]}
    two = {"schema_version": 1, "manifest_sha256": digest,
           "campaign_identity": identity, "shard": {"index": 1, "count": 2},
           "raw_cells": [expected[1]]}
    merged = module.merge_shards([one, two], manifest)
    assert [x["id"] for x in merged["raw_cells"]] == sorted(x["id"] for x in expected)
    two["raw_cells"] = [expected[0]]
    try:
        module.merge_shards([one, two], manifest)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate cell was accepted")


def test_merge_rejects_manifest_digest_mismatch():
    module = load_module()
    shard = {"schema_version": 1, "manifest_sha256": "stale",
             "campaign_identity": {}, "raw_cells": []}
    try:
        module.merge_shards([shard], {"hypotheses": [], "timeline_probes": []})
    except ValueError as error:
        assert "manifest" in str(error)
    else:
        raise AssertionError("stale manifest was accepted")


def test_merge_rejects_cross_target_shards():
    module = load_module()
    manifest = {"hypotheses": [], "timeline_probes": []}
    digest = hashlib.sha256(module.canonical(manifest)).hexdigest()
    one = {"manifest_sha256": digest,
           "campaign_identity": module.campaign_identity("gz-a3", manifest),
           "raw_cells": [{"id": "a"}]}
    two = {"manifest_sha256": digest,
           "campaign_identity": module.campaign_identity("bz-a3-1", manifest),
           "raw_cells": [{"id": "b"}]}
    with pytest.raises(ValueError, match="campaign identity"):
        module.merge_shards([one, two], manifest)


def test_merge_rejects_missing_duplicate_or_incomplete_shards():
    module = load_module()
    manifest = json.loads(MANIFEST.read_text())
    digest = hashlib.sha256(module.canonical(manifest)).hexdigest()
    identity = module.campaign_identity("gz-a3", manifest)
    cells = module.enumerate_cells(manifest)
    shard_zero = {
        "manifest_sha256": digest, "campaign_identity": identity,
        "shard": {"index": 0, "count": 2},
        "raw_cells": module.select_shard(cells, 0, 2),
    }
    shard_one = {
        "manifest_sha256": digest, "campaign_identity": identity,
        "shard": {"index": 1, "count": 2},
        "raw_cells": module.select_shard(cells, 1, 2),
    }
    with pytest.raises(ValueError, match="missing shard"):
        module.merge_shards([shard_zero], manifest)
    with pytest.raises(ValueError, match="duplicate shard"):
        module.merge_shards([shard_zero, shard_zero], manifest)
    shard_one["raw_cells"] = shard_one["raw_cells"][:-1]
    with pytest.raises(ValueError, match="cell coverage"):
        module.merge_shards([shard_zero, shard_one], manifest)


def test_campaign_identity_binds_coordinator_and_acquisition_sources():
    module = load_module()
    manifest = json.loads(MANIFEST.read_text())
    value = module.campaign_identity("gz-a3", manifest)
    assert value["coordinator_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert value["acquisition_sha256"] == hashlib.sha256(
        (ROOT / "scripts/acquire_profile.py").read_bytes()
    ).hexdigest()
    assert value["device_helper_sha256"] == hashlib.sha256(module.DEVICE_HELPER.read_bytes()).hexdigest()


def test_cached_evidence_without_valid_receipt_is_retired(tmp_path):
    module = load_module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    entry = bundle / "work.py"
    entry.write_text("print('ok')\n")
    evidence_path = tmp_path / "basic.json"
    evidence_path.write_text("{}")
    receipt = Path(str(evidence_path) + ".dispatch.json")
    receipt.write_text("not-json")

    result = module.invoke_acquisition(
        phase="basic", target="gz-a3", dispatch_key="attempt-0",
        bundle=bundle, entrypoint="work.py", workload_args=[],
        evidence_path=evidence_path, selector=None, basic_evidence=None,
        cpl_remote=None,
    )

    assert result["ok"] is False
    assert result["classification"] == "evidence_failure"
    assert not evidence_path.exists()
    assert not receipt.exists()
    assert list(tmp_path.glob("basic.json.retired*"))
    assert list(tmp_path.glob("basic.json.dispatch.json.retired*"))
    assert module._fresh_artifact_path(evidence_path).name == "basic-retry-1.json"


def cell(experiment, sweep, variant, median, signals, status="success"):
    return {
        "id": f"{experiment}/{sweep}/{variant}", "experiment": experiment,
        "sweep": sweep, "variant": variant, "kind": "hypothesis", "status": status,
        "measurements": {"median": median, "metric": "proxy", "unit": "us"},
        "signals": {name: {"mean": value} for name, value in signals.items()},
    }


def test_summarize_pairs_into_current_report_hypothesis_contract():
    module = load_module()
    manifest = {
        "hypotheses": [{
            "id": "loop", "question": "Local bound", "falsifier": "does not fall",
            "expected": {"direction": "decrease", "minimum_relative_change": 0.05},
            "signal_predictions": [{
                "name": "aiv_scalar_ratio", "direction": "decrease",
                "minimum_absolute_change": 0.02,
            }],
            "variants": {"before": "b", "after": "a"}, "kernel_names": {},
            "sweeps": [{"id": "short"}],
        }],
        "timeline_probes": [],
    }
    raw = [
        cell("loop", "short", "before", 10.0, {"aiv_scalar_ratio": 0.5}),
        cell("loop", "short", "after", 8.0, {"aiv_scalar_ratio": 0.4}),
    ]
    summarized = module.summarize_hypotheses(raw, manifest)
    assert summarized == [{
        "hypothesis_id": "loop", "title": "Local bound",
        "expected": {"direction": "decrease", "minimum_relative_change": 0.05},
        "falsification": "does not fall",
        "measurements": [{
            "status": "valid", "sweep": "short", "before": 10.0, "after": 8.0,
            "metric": "proxy", "unit": "us",
            "before_cell_id": "loop/short/before", "after_cell_id": "loop/short/after",
        }],
        "signals": [{
            "name": "aiv_scalar_ratio", "relationship": "tracks_outcome",
            "summary": "short: before=0.5, after=0.4; signal and latency track the declared outcome",
        }],
    }]
    # This mirrors build_profile_reverse_report.py's current hypothesis input.
    assert module.validate_report_hypotheses({"schema_version": 1, "results": summarized})


def test_summary_propagates_failure_and_timeline_not_exported():
    module = load_module()
    manifest = {
        "hypotheses": [],
        "timeline_probes": [{
            "id": "timeline", "question": "Can phases be seen?", "falsifier": "absent",
            "expected": {"direction": "no_change", "minimum_relative_change": 0.05},
            "signal_predictions": [{"name": "timeline_phase_order", "availability": "required"}],
            "variants": {"before": "b", "after": "a"}, "kernel_names": {},
            "sweeps": [{"id": "order"}],
        }],
    }
    before = cell("timeline", "order", "before", None, {}, "infrastructure_exhausted")
    after = cell("timeline", "order", "after", 2.0, {})
    before["kind"] = after["kind"] = "timeline"
    result = module.summarize_hypotheses([before, after], manifest)[0]
    assert result["measurements"] == [{
        "status": "infrastructure_failure", "sweep": "order",
        "before_status": "infrastructure_exhausted", "after_status": "success",
    }]
    assert result["signals"] == [{
        "name": "timeline_phase_order", "relationship": "not_exported",
        "summary": ("TimelineDetail acquisition was absent or failed for one or more "
                    "controlled captures; exporter support was not evaluated for those captures"),
    }]


@pytest.mark.parametrize(
    ("experiment", "signal", "before_semantics", "after_semantics"),
    [
        ("reversed-phases", "timeline_phase_order",
         {"phase_order": ["load", "compute"]}, {"phase_order": ["compute", "load"]}),
        ("duration-scaled-phases", "timeline_phase_duration",
         {"phase_duration_us": {"compute": 4.0}},
         {"phase_duration_us": {"compute": 8.0}}),
        ("serialized-vs-overlapped", "timeline_pipe_overlap",
         {"load_compute_overlap_us": 0.0}, {"load_compute_overlap_us": 2.0}),
        ("fused-vs-split", "timeline_kernel_boundaries",
         {"kernel_boundaries": [{"name": "fused"}]},
         {"kernel_boundaries": [{"name": "copy"}, {"name": "compute"}]}),
    ],
)
def test_summary_evaluates_compact_timeline_semantics(
    experiment, signal, before_semantics, after_semantics,
):
    module = load_module()
    manifest = {
        "hypotheses": [],
        "timeline_probes": [{
            "id": experiment, "question": "timeline", "falsifier": "absent",
            "expected": {"direction": "increase", "minimum_relative_change": 0.1},
            "signal_predictions": [{"name": signal, "availability": "required"}],
            "variants": {"before": "b", "after": "a"}, "kernel_names": {},
            "sweeps": [{"id": "control"}],
        }],
    }
    before = cell(experiment, "control", "before", 2.0, {})
    after = cell(experiment, "control", "after", 3.0, {})
    for item, semantics in ((before, before_semantics), (after, after_semantics)):
        item["kind"] = "timeline"
        item["timeline"] = {
            "status": "available",
            "semantics": {"selectors_complete": True, **semantics},
        }

    result = module.summarize_hypotheses([before, after], manifest)[0]

    assert result["signals"][0]["relationship"] == "tracks_outcome"
    assert "context_only" not in result["signals"][0]["summary"]


def test_summary_accepts_same_server_devices_but_rejects_cross_server_pair():
    module = load_module()
    spec = {
        "id": "loop", "question": "Local bound", "falsifier": "no",
        "expected": {"direction": "decrease", "minimum_relative_change": 0.05},
        "signal_predictions": [{"name": "aiv_scalar_ratio", "direction": "decrease",
                                "minimum_absolute_change": 0.02}],
        "sanity": {"signals": ["aiv_scalar_ratio"], "minimum_absolute_change": 0.001},
        "variants": {"before": "b", "after": "a"}, "kernel_names": {},
        "sweeps": [{"id": "short"}],
    }
    before = cell("loop", "short", "before", 10.0, {"aiv_scalar_ratio": 0.5})
    after = cell("loop", "short", "after", 8.0, {"aiv_scalar_ratio": 0.4})
    before["provenance"] = {"placements": [{"target": "bz-a3-1", "physical_id": 0}]}
    after["provenance"] = {"placements": [{"target": "bz-a3-1", "physical_id": 1}]}
    result = module.summarize_hypotheses(
        [before, after], {"hypotheses": [spec], "timeline_probes": []}
    )[0]
    assert result["measurements"][0]["status"] == "valid"

    after["provenance"] = {"placements": [{"target": "bz-a3-2", "physical_id": 0}]}
    result = module.summarize_hypotheses(
        [before, after], {"hypotheses": [spec], "timeline_probes": []}
    )[0]
    assert result["measurements"][0]["status"] == "infrastructure_failure"
    assert result["measurements"][0]["reason"] == "before/after were not measured on one server target"

    after["provenance"] = before["provenance"]
    after["signals"]["aiv_scalar_ratio"]["mean"] = 0.5
    result = module.summarize_hypotheses(
        [before, after], {"hypotheses": [spec], "timeline_probes": []}
    )[0]
    assert result["measurements"][0]["status"] == "profiling_failure"
    assert "sanity" in result["measurements"][0]["reason"]


def test_cached_cell_is_bound_to_exact_run_contract_and_identical_resume(tmp_path):
    module = load_module()
    cell_value = module.enumerate_cells(json.loads(MANIFEST.read_text()))[0]
    calls = []

    def acquire(phase, current, attempt, selector):
        calls.append((phase, attempt, selector))
        return evidence(current["kernel_names"][0] if phase == "basic" else selector, 3.0)

    first = module.run_cell(
        cell_value, tmp_path, acquire, samples=1, max_attempts=3,
        max_relative_range=0.05, campaign_identity_sha256="campaign-a",
    )
    assert first["status"] == "success"
    first_calls = list(calls)
    resumed = module.run_cell(
        cell_value, tmp_path, lambda *_: pytest.fail("identical terminal cell reran"),
        samples=1, max_attempts=3, max_relative_range=0.05,
        campaign_identity_sha256="campaign-a",
    )
    assert resumed == first

    changed = module.run_cell(
        cell_value, tmp_path, acquire, samples=2, max_attempts=4,
        max_relative_range=0.02, campaign_identity_sha256="campaign-a",
    )
    assert changed["status"] == "success"
    assert changed["run_contract"]["samples"] == 2
    assert changed["run_contract"]["max_attempts"] == 4
    assert changed["run_contract"]["max_relative_range"] == 0.02
    assert len(calls) > len(first_calls)
    assert list(tmp_path.glob("result.json.retired*"))


def test_default_runtime_checkpoint_is_retired_before_named_runtime_work(tmp_path):
    module = load_module()
    manifest = json.loads(MANIFEST.read_text())
    cell = module.enumerate_cells(manifest)[0]
    identity = module.campaign_identity("gz-a3", manifest)
    legacy_identity = {**identity, "runtime": None}
    legacy_digest = hashlib.sha256(module.canonical(legacy_identity)).hexdigest()
    current_digest = hashlib.sha256(module.canonical(identity)).hexdigest()
    calls = []

    def acquire(phase, current, attempt, selector):
        calls.append((phase, attempt))
        return evidence(current["kernel_names"][0] if phase == "basic" else selector, 3.0)

    old = module.run_cell(
        cell, tmp_path, acquire, samples=1, max_attempts=3,
        max_relative_range=0.05, campaign_identity_sha256=legacy_digest,
    )
    assert old["status"] == "success"
    calls.clear()
    new = module.run_cell(
        cell, tmp_path, acquire, samples=1, max_attempts=3,
        max_relative_range=0.05, campaign_identity_sha256=current_digest,
    )
    assert calls
    assert new["status"] == "success"
    retired = list(tmp_path.glob("result.json.retired*"))
    assert len(retired) == 1
    assert json.loads(retired[0].read_text()) == old
    assert new["run_contract"]["campaign_identity_sha256"] == current_digest


@pytest.mark.parametrize(
    "mutation",
    ["campaign", "samples", "max_attempts", "max_relative_range", "cell"],
)
def test_each_run_contract_dimension_retires_a_terminal_cache(tmp_path, mutation):
    module = load_module()
    cell_value = module.enumerate_cells(json.loads(MANIFEST.read_text()))[0]
    calls = []

    def acquire(phase, current, attempt, selector):
        calls.append((phase, attempt, selector))
        return evidence(current["kernel_names"][0] if phase == "basic" else selector, 3.0)

    initial = {
        "samples": 1,
        "max_attempts": 3,
        "max_relative_range": 0.05,
        "campaign_identity_sha256": "campaign-a",
    }
    first = module.run_cell(cell_value, tmp_path, acquire, **initial)
    call_count = len(calls)
    changed_cell = json.loads(json.dumps(cell_value))
    changed = dict(initial)
    if mutation == "cell":
        changed_cell["sweep_parameters"]["n"] += 1
    elif mutation == "campaign":
        changed["campaign_identity_sha256"] = "campaign-b"
    elif mutation == "samples":
        changed["samples"] = 2
    elif mutation == "max_attempts":
        changed["max_attempts"] = 4
    else:
        changed["max_relative_range"] = 0.02

    second = module.run_cell(changed_cell, tmp_path, acquire, **changed)

    assert len(calls) > call_count
    assert second["campaign_identity_sha256"] != first["campaign_identity_sha256"]
    assert second["run_contract"]["cell"] == changed_cell
    assert list(tmp_path.glob("result.json.retired*"))


def test_legacy_terminal_cache_without_exact_contract_is_retired(tmp_path):
    module = load_module()
    cell_value = module.enumerate_cells(json.loads(MANIFEST.read_text()))[0]
    state = tmp_path / "result.json"
    state.write_text(json.dumps({
        "status": "success", "campaign_identity_sha256": "campaign-a",
    }))

    result = module.run_cell(
        cell_value, tmp_path,
        lambda phase, current, attempt, selector: evidence(
            current["kernel_names"][0] if phase == "basic" else selector, 3.0
        ),
        samples=1, max_attempts=3, max_relative_range=0.05,
        campaign_identity_sha256="campaign-a",
    )

    assert result["status"] == "success"
    assert result["run_contract"]["cell"]["sweep_parameters"]["n"] > 0
    assert (tmp_path / "result.json.retired").exists()


def test_real_summary_is_accepted_by_current_report_classifier():
    report_path = ROOT.parent / "reverse-profile-report" / "scripts" / "build_profile_reverse_report.py"
    if not report_path.exists():
        pytest.skip("report worktree is not present; equivalent contract validator runs above")
    campaign = load_module()
    spec = {
        "id": "loop", "question": "Local bound", "falsifier": "does not fall",
        "expected": {"direction": "decrease", "minimum_relative_change": 0.05},
        "signal_predictions": [], "variants": {"before": "b", "after": "a"},
        "kernel_names": {}, "sweeps": [{"id": "short"}],
    }
    summary = campaign.summarize_hypotheses([
        cell("loop", "short", "before", 10.0, {}),
        cell("loop", "short", "after", 8.0, {}),
    ], {"hypotheses": [spec], "timeline_probes": []})[0]
    report_spec = importlib.util.spec_from_file_location("current_reverse_report", report_path)
    report = importlib.util.module_from_spec(report_spec)
    assert report_spec.loader
    report_spec.loader.exec_module(report)
    assert report.classify_hypothesis(summary)[0] == "supported"
