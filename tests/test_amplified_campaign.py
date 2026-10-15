import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a3_profile_amplified_campaign as campaign


@pytest.fixture
def pair(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "probe.py").write_text("print('example')\n")
    return {"id": "test", "bundle": str(bundle), "target": "bz-a3-1", "cells": [
        {"id": f"test/{variant}", "variant": variant, "kind": "original",
         "entrypoint": "probe.py", "arguments": [variant], "kernel_names": ["probe"]}
        for variant in ("before", "after")]}


def evidence(phase, duration=10, amplified=False):
    result = {"ok": True, "handle": "remote:bz-a3-1:job:example", "evidence": {
        "capture": {"observed_kernel_names": ["probe", "a3_amplified_probe"],
                    "rows": [{"Op Name": "probe", "Task Duration(us)": str(duration)}]
                    if phase == "basic" else [{"aiv_time(us)": "999", "aiv_scalar_ratio": "0.4"}]}}}
    if amplified:
        result["evidence"]["normal_run"] = {"workload_result": {"correctness": True,
            "compiler": [{"stage": "ttir", "loop_evidence": ["scf.for %i = %c0 to %repeat"]}]}}
    return result


def test_manifest_eighteen_unique_seeded_runtime_cells():
    pairs = campaign.manifest()
    cells = [cell for pair in pairs for cell in pair["cells"]]
    assert len(cells) == len({cell["id"] for cell in cells}) == 18
    assert {cell["repetitions"] for cell in cells} == {1, 32, 128}
    assert {(cell["experiment"], cell["sweep"]) for cell in cells} == set(campaign.SWEEPS.items())


def test_interleaved_independent_timing_pipe_resume(pair, tmp_path):
    calls = []
    def acquire(phase, cell, attempt, path, selector, basic):
        calls.append((phase, cell["variant"], str(path)))
        return evidence(phase)
    output = tmp_path / "result"
    result = campaign.run_pair(pair, output, acquire, campaign="fresh")
    assert result["terminal"]
    assert [row["status"] for row in result["cells"]] == ["success", "success"]
    assert [side for _, side, _ in calls] == ["before", "after"] * 7
    assert [row["timing"]["median_us"] for row in result["cells"]] == [10, 10]
    assert all(len(row["pipe"]) == 3 for row in result["cells"])
    assert len(calls) == 14
    assert campaign.run_pair(pair, output, lambda *a: pytest.fail("redispatched"), campaign="fresh") == result


def test_source_or_campaign_change_rejected(pair, tmp_path):
    output = tmp_path / "results"
    campaign.run_pair(pair, output, lambda *a: evidence(a[0]), campaign="fresh")
    with pytest.raises(ValueError, match="contract"):
        campaign.run_pair(pair, output, lambda *a: None, campaign="different")
    (Path(pair["bundle"]) / "probe.py").write_text("changed")
    with pytest.raises(ValueError, match="contract"):
        campaign.run_pair(pair, output, lambda *a: None, campaign="fresh")


def test_pending_reuses_same_path_without_attempt_budget(pair, tmp_path):
    paths = []
    def unavailable(phase, cell, attempt, path, selector, basic):
        paths.append(path)
        return {"ok": False, "classification": "transport_error", "handle": "retained"}
    output = tmp_path / "results"
    first = campaign.run_pair(pair, output, unavailable, campaign="fresh")
    assert not first["terminal"]
    second = campaign.run_pair(pair, output, unavailable, campaign="fresh")
    assert not second["terminal"]
    assert paths[:2] == paths[2:]
    assert all(not row["errors"] for row in second["cells"])
    assert all(not row["attempts"] for row in second["cells"])


@pytest.mark.parametrize("classification,count,status", [
    ("device_unavailable", 3, "infrastructure_exhausted"),
    ("host_environment", 3, "infrastructure_exhausted"),
    ("profiler_failure", 2, "failure"),
    ("evidence_failure", 2, "failure"),
    ("workload_failure", 1, "failure"),
    ("controller_failure", 1, "failure"),
])
def test_failures_bounded_without_blocking_other_side(pair, tmp_path, classification, count, status):
    calls = []
    def acquire(phase, cell, attempt, path, selector, basic):
        calls.append((cell["variant"], str(path)))
        if cell["variant"] == "before":
            return {"ok": False, "classification": classification}
        return evidence(phase)
    result = campaign.run_pair(pair, tmp_path / "r", acquire, campaign="fresh")
    assert result["terminal"]
    assert [row["status"] for row in result["cells"]] == [status, "success"]
    assert len([call for call in calls if call[0] == "before"]) == count
    assert len({path for side, path in calls if side == "before"}) == count


def test_unstable_seven_valid_preserves_all_samples_and_still_profiles(pair, tmp_path):
    counts = {"before": 0, "after": 0}
    def acquire(phase, cell, attempt, path, selector, basic):
        counts[cell["variant"]] += phase == "basic"
        return evidence(phase, duration=10 if counts[cell["variant"]] % 2 else 20)
    result = campaign.run_pair(pair, tmp_path / "r", acquire, campaign="fresh")
    for row in result["cells"]:
        assert row["status"] == "inconclusive"
        assert len(row["timing_samples_us"]) == 7
        assert row["timing"]["accepted_sample_indexes"] == []
        assert len(row["pipe"]) == 3


def test_first_stable_window_not_best_window():
    result = campaign.stable([20, 10, 10.1, 10.2, 10.2, 10.2])
    assert result["accepted_sample_indexes"] == [1, 2, 3]
    assert result["median_us"] == 10.1


def test_multiselectors_sums_sequential_kernel_medians_not_pipeline_times():
    value = {"capture": {"rows": [
        {"Op Name": "first", "Task Duration(us)": "3"},
        {"Op Name": "first", "Task Duration(us)": "5"},
        {"Op Name": "second", "Task Duration(us)": "7"},
        {"Op Name": "other", "Task Duration(us)": "999"}]}}
    assert campaign.basic_duration(value, ["first", "second"]) == 11


@pytest.mark.parametrize("count", [2, 3])
def test_generic_multikernel_pair_has_every_metric_selector(pair, tmp_path, count):
    names = [f"kernel-{index}" for index in range(count)]
    for cell in pair["cells"]:
        cell["kernel_names"] = names
    observed_pipe = []
    def acquire(phase, cell, attempt, path, selector, basic):
        result = evidence(phase)
        result["evidence"]["capture"]["observed_kernel_names"] = names
        if phase == "basic":
            result["evidence"]["capture"]["rows"] = [
                {"Op Name": name, "Task Duration(us)": str(index + 1)}
                for index, name in enumerate(names)]
        else:
            observed_pipe.append((cell["variant"], selector))
        return result
    result = campaign.run_pair(pair, tmp_path / "r", acquire, campaign="fresh")
    assert all(row["status"] == "success" for row in result["cells"])
    assert all(row["timing"]["median_us"] == sum(range(1, count + 1)) for row in result["cells"])
    assert all(len(row["pipe"]) == 3 * count for row in result["cells"])
    assert {name for _, name in observed_pipe} == set(names)


@pytest.mark.parametrize("raw", ["NA", "nan", "inf", "0", "-1"])
def test_invalid_basic_durations_rejected(raw):
    value = evidence("basic")["evidence"]
    value["capture"]["rows"][0]["Task Duration(us)"] = raw
    with pytest.raises(ValueError):
        campaign.basic_duration(value, ["probe"])


def test_amplified_requires_correctness_and_compiler_loop_proof(pair, tmp_path):
    for cell in pair["cells"]:
        cell["kind"] = "amplified"
    failed = campaign.run_pair(pair, tmp_path / "bad", lambda *a: evidence(a[0]), campaign="fresh")
    assert all(row["status"] == "failure" for row in failed["cells"])
    passed = campaign.run_pair(pair, tmp_path / "good", lambda *a: evidence(a[0], amplified=True), campaign="fresh")
    assert all(row["status"] == "success" and row["compiler_loop_proof"] for row in passed["cells"])


def test_timeline_cells_do_not_duplicate_separate_timeline_owner(pair, tmp_path):
    for cell in pair["cells"]:
        cell["kind"] = "timeline"
    def acquire(phase, *args):
        assert phase in {"basic", "pipe"}
        return evidence(phase)
    assert campaign.run_pair(pair, tmp_path / "r", acquire, campaign="fresh")["terminal"]
