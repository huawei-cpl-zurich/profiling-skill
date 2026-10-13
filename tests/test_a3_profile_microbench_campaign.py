from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
    assert [call[0] for call in calls] == ["basic", "pipe"] * 3
    assert len(result["provenance"]["handles"]) == 6


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
    assert result["timeline"] == {
        "status": "unavailable",
        "reason": "A3 msprof-op acquisition emitted no TimelineDetail evidence",
    }


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


def test_merge_rejects_overlap_and_combines_disjoint_results(tmp_path):
    module = load_module()
    one = {"schema_version": 1, "manifest_sha256": "x", "shard": {"index": 0, "count": 2},
           "results": [{"id": "a"}]}
    two = {"schema_version": 1, "manifest_sha256": "x", "shard": {"index": 1, "count": 2},
           "results": [{"id": "b"}]}
    merged = module.merge_shards([one, two])
    assert [x["id"] for x in merged["results"]] == ["a", "b"]
    two["results"] = [{"id": "a"}]
    try:
        module.merge_shards([one, two])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate cell was accepted")
