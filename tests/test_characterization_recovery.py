from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import recover_a3_profile_campaign as recovery


def evidence(cell, duration=10):
    return {"ok": True, "handle": "remote:gz-a3:job:sample", "evidence": {
        "capture": {"observed_kernel_names": cell["kernel_names"],
                    "rows": [{"aiv_time(us)": duration, "aiv_vec_ratio": 0.6}],
                    "sources": [{"sha256": "a" * 64}]},
        "device": {"physical_id": 0}}}


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "producer"
    for relative in ("scripts/a3_profile_microbench_campaign.py", "scripts/acquire_profile.py",
                     "benchmarks/a3_profile_microbench.py", "experiments/a3-profile-hypotheses.json"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    # A distinct frozen producer identity must remain usable for reporting.
    script = root / "scripts/a3_profile_microbench_campaign.py"
    script.write_bytes(script.read_bytes() + b"\n# frozen producer\n")
    producer = recovery.load_producer(root)
    manifest = json.loads(producer.DEFAULT_MANIFEST.read_bytes())
    identity = producer.campaign_identity("gz-a3", manifest)
    probes = [{"id": probe_id, "samples": [10] * 5,
               "handles": ["remote:gz-a3:job:sample"] * 5,
               "devices": [{"physical_id": 0}] * 5,
               "sources": [{"sha256": "a" * 64}] * 5}
              for probe_id in producer.QUALIFICATION_PROBES]
    qualification = tmp_path / "qualification.json"
    producer._atomic_json(qualification, producer.build_qualification_receipt(identity, probes))
    campaign = tmp_path / "original"
    for index in range(4):
        for cell in producer.select_shard(producer.enumerate_cells(manifest), index, 4):
            contract = {"schema_version": 1, "cell": cell, **identity, "samples": 3,
                        "max_attempts": 7, "max_relative_range": 0.05}
            record = producer.run_cell(
                cell, campaign / f"shard-{index}/cells" / cell["id"].replace("/", "__"),
                lambda phase, current, attempt, selector: evidence(current), samples=3,
                max_attempts=7, max_relative_range=0.05,
                campaign_identity_sha256=hashlib.sha256(producer.canonical(contract)).hexdigest())
            assert record["status"] == "success"
    return SimpleNamespace(producer=producer, manifest=manifest, campaign=campaign,
                           qualification=qualification, output=tmp_path / "recovered")


def record_path(saved):
    return saved.campaign / "shard-0/cells/loop-bound__short-local__before/result.json"


@pytest.mark.parametrize("left,right", [(None, 0.4), (0.5, None), (None, None), (0.5, 0.4)])
def test_signal_summary_handles_independently_missing_sides(left, right):
    module = recovery.reporting
    manifest = json.loads(module.DEFAULT_MANIFEST.read_bytes())
    spec = manifest["hypotheses"][0]
    spec["sanity"] = {}
    spec["sweeps"] = spec["sweeps"][:1]
    spec["signal_predictions"] = spec["signal_predictions"][:1]
    signal = spec["signal_predictions"][0]["name"]
    rows = []
    for variant, value in (("before", left), ("after", right)):
        rows.append({"id": f"{spec['id']}/{spec['sweeps'][0]['id']}/{variant}",
                     "experiment": spec["id"], "sweep": spec["sweeps"][0]["id"],
                     "variant": variant, "status": "success",
                     "measurements": {"median": 10, "metric": "proxy", "unit": "us"},
                     "signals": {} if value is None else {signal: {"mean": value}}})
    result = module.summarize_hypotheses(rows, {"hypotheses": [spec], "timeline_probes": []})[0]
    summary = result["signals"][0]
    assert summary["relationship"] == ("not_exported" if None in (left, right) else "context_only")
    assert f"before={'not_exported' if left is None else str(left)}" in summary["summary"]
    assert f"after={'not_exported' if right is None else str(right)}" in summary["summary"]


def test_offline_recovery_preserves_all_cells_and_original_identity(saved, monkeypatch):
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: pytest.fail("remote call"))
    before = {p: p.read_bytes() for p in saved.campaign.rglob("*") if p.is_file()}
    merged = recovery.recover_report(saved.campaign, saved.output, saved.producer,
                                     saved.qualification, 4)
    assert len(merged["raw_cells"]) == 52
    assert merged["campaign_identity"] != recovery.reporting.campaign_identity("gz-a3", saved.manifest)
    assert merged["reporting_revision"] == recovery.reporting_revision()
    assert len(list(saved.output.glob("shard-*/shard-result.json"))) == 4
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (saved.output / "report.md").read_text().startswith("# Characterization recovery report")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "identity", "qualification"])
def test_recovery_rejects_invalid_original_coverage_and_identity(saved, mutation):
    path = record_path(saved)
    if mutation == "missing":
        path.unlink()
    elif mutation == "duplicate":
        duplicate = path.parent.with_name("duplicate") / "result.json"
        duplicate.parent.mkdir()
        shutil.copy2(path, duplicate)
    elif mutation == "qualification":
        value = json.loads(saved.qualification.read_bytes())
        value["identity"]["coordinator_sha256"] = "wrong"
        saved.producer._atomic_json(saved.qualification, value)
    else:
        value = json.loads(path.read_bytes())
        value["run_contract"]["samples"] = 99
        saved.producer._atomic_json(path, value)
    with pytest.raises(ValueError):
        recovery.recover_report(saved.campaign, saved.output, saved.producer, saved.qualification, 4)
    assert not saved.output.exists()


def make_retryable(saved, classification="host_environment", message="host unavailable"):
    path = record_path(saved)
    value = json.loads(path.read_bytes())
    value["status"] = "infrastructure_exhausted" if classification == "host_environment" else "profiling_failure"
    value["attempts"] = value["attempts"][:2] + [{"classification": classification,
                          "message": message, "handle": "remote:gz-a3:job:failure"}]
    value["measurements"]["samples"] = [10, 10]
    saved.producer._atomic_json(path, value)
    return path, value


@pytest.mark.parametrize("directory_order", ["reverse", "timeline_first"])
def test_selective_retry_preserves_valid_captures_and_other_cells(saved, monkeypatch, directory_order):
    original_glob = Path.glob
    def reordered_glob(directory, pattern):
        matches = list(original_glob(directory, pattern))
        if pattern == "*/result.json":
            if directory_order == "timeline_first":
                matches.sort(key=lambda path: (json.loads(path.read_bytes())["kind"] != "timeline", str(path)))
            else:
                matches.sort(reverse=True)
        return iter(matches)
    monkeypatch.setattr(Path, "glob", reordered_glob)
    path, prior = make_retryable(saved, "remote_failure", "remote job failed without one valid failure receipt")
    before = {p.relative_to(saved.campaign): p.read_bytes() for p in
              saved.campaign.glob("shard-0/cells/*/result.json") if p != path}
    observations, acquisitions = [], []
    def observe(command, **kwargs):
        observations.append(command)
        return SimpleNamespace(returncode=1, stdout=json.dumps({"target": "gz-a3", "handle": command[3], "state": "failed"}))
    monkeypatch.setattr(recovery.subprocess, "run", observe)
    def acquire(**kwargs):
        acquisitions.append(kwargs)
        cell = next(cell for cell in saved.producer.enumerate_cells(saved.manifest)
                    if cell["id"] == prior["id"])
        return evidence(cell)
    monkeypatch.setattr(saved.producer, "invoke_acquisition", acquire)
    assert recovery.retry_shard(saved.campaign, saved.output, saved.producer,
                                saved.qualification, 4, 0, Path("cpl-remote")) == 0
    retried = json.loads((saved.output / path.relative_to(saved.campaign)).read_bytes())
    assert retried["status"] == "success"
    assert retried["attempts"][:len(prior["attempts"])] == prior["attempts"]
    assert len(retried["measurements"]["samples"]) == 3
    assert retried["run_contract"] == prior["run_contract"]
    assert observations[0][1:] == ["--json", "observe", "remote:gz-a3:job:failure"]
    assert len(acquisitions) == 2  # one discovery and one additional capture
    assert all((saved.output / p).read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("state", ["running", "reconnecting", "observation-unavailable", "completed"])
def test_nonfailed_handle_is_observed_without_new_dispatch(saved, monkeypatch, state):
    _, prior = make_retryable(saved)
    commands = []
    def observe(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"target": "gz-a3", "handle": command[3], "state": state}))
    monkeypatch.setattr(recovery.subprocess, "run", observe)
    monkeypatch.setattr(saved.producer, "invoke_acquisition", lambda **k: pytest.fail("new dispatch"))
    with pytest.raises(ValueError, match="terminal failure"):
        recovery.retry_shard(saved.campaign, saved.output, saved.producer,
                             saved.qualification, 4, 0, Path("cpl-remote"))
    assert len(commands) == 1
    assert commands[0][3] == prior["attempts"][-1]["handle"]


def test_real_profiler_failure_is_not_selected(saved, monkeypatch):
    _, prior = make_retryable(saved, "profiler_failure", "msprof rejected metric")
    assert not recovery.reporting.recovery_eligible(prior)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: pytest.fail("remote call"))
    assert recovery.retry_shard(saved.campaign, saved.output, saved.producer,
                                saved.qualification, 4, 0, Path("cpl-remote")) == 0


def test_retry_missing_diagnostic_retries_only_after_terminal_recheck(saved, monkeypatch):
    path, prior = make_retryable(saved)
    observed = []
    def observe(command, **kwargs):
        observed.append(command[3])
        return SimpleNamespace(returncode=1, stdout=json.dumps({"target": "gz-a3", "handle": command[3], "state": "failed"}))
    monkeypatch.setattr(recovery.subprocess, "run", observe)
    attempts = []
    def acquire(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return {"ok": False, "classification": "remote_failure", "phase": "remote-job",
                    "handle": "remote:gz-a3:job:new-failure",
                    "message": "remote job failed without one valid failure receipt"}
        return evidence(prior)
    monkeypatch.setattr(saved.producer, "invoke_acquisition", acquire)
    recovery.retry_shard(saved.campaign, saved.output, saved.producer,
                         saved.qualification, 4, 0, Path("cpl-remote"))
    assert observed == ["remote:gz-a3:job:failure", "remote:gz-a3:job:new-failure"]
    result = json.loads((saved.output / path.relative_to(saved.campaign)).read_bytes())
    assert result["status"] == "success"
    assert result["attempts"][len(prior["attempts"])]["classification"] == "host_environment"
    assert attempts[0]["evidence_path"] != attempts[1]["evidence_path"]


def test_recovery_budget_exhaustion_does_not_create_another_round(saved, monkeypatch):
    path, _ = make_retryable(saved)
    monkeypatch.setattr(recovery.subprocess, "run", lambda command, **k: SimpleNamespace(
        returncode=1, stdout=json.dumps({"target": "gz-a3", "handle": command[3], "state": "failed"})))
    calls = []
    def acquire(**kwargs):
        calls.append(kwargs)
        return {"ok": False, "classification": "host_environment", "phase": "logs",
                "handle": "remote:gz-a3:job:new-failure", "message": "missing logs"}
    monkeypatch.setattr(saved.producer, "invoke_acquisition", acquire)
    recovery.retry_shard(saved.campaign, saved.output, saved.producer,
                         saved.qualification, 4, 0, Path("cpl-remote"))
    assert len(calls) == 7
    second_producer = recovery.load_producer(saved.producer.ROOT)
    monkeypatch.setattr(second_producer, "invoke_acquisition", lambda **k: pytest.fail("new round"))
    recovery.retry_shard(saved.campaign, saved.output, second_producer,
                         saved.qualification, 4, 0, Path("cpl-remote"))
    result = json.loads((saved.output / path.relative_to(saved.campaign)).read_bytes())
    assert len(result["attempts"]) == 10


@pytest.mark.parametrize("receipt", [
    {"target": "bz-a3-1", "handle": "remote:gz-a3:job:failure", "state": "failed"},
    {"target": "gz-a3", "handle": "remote:gz-a3:job:other", "state": "failed"},
    [],
])
def test_terminal_failure_requires_exact_target_and_handle(monkeypatch, receipt):
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=json.dumps(receipt)))
    with pytest.raises(ValueError, match="terminal failure"):
        recovery.verify_terminal(Path("cpl-remote"), {"id": "cell", "attempts": [
            {"handle": "remote:gz-a3:job:failure"}]})


@pytest.mark.parametrize("status", ["not_attempted", "infrastructure_failure", "unavailable"])
def test_timeline_summary_distinguishes_missing_acquisition_from_missing_export(status):
    module = recovery.reporting
    manifest = json.loads(module.DEFAULT_MANIFEST.read_bytes())
    spec = manifest["timeline_probes"][0]
    rows = [{**cell, "status": "infrastructure_exhausted", "timeline": {"status": status}}
            for cell in module.enumerate_cells(manifest) if cell["experiment"] == spec["id"]]
    result = module.summarize_hypotheses(rows, {"hypotheses": [], "timeline_probes": [spec]})[0]
    summary = result["signals"][0]
    assert summary["relationship"] == "not_exported"
    if status == "unavailable":
        assert summary["summary"] == "TimelineDetail was unavailable for every controlled capture"
    else:
        assert "acquisition was absent or failed" in summary["summary"]
        assert "exporter support was not evaluated" in summary["summary"]
