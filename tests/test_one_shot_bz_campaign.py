from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("one_shot_bz_campaign", ROOT / "scripts" / "one_shot_bz_campaign.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(module)


def inputs(tmp_path: Path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("same prompt\n")
    model = {"name": "model", "reasoning_effort": "low"}
    skills = {"cannbot": ["triton-op-coding", "ops-profiling"],
              "project-cannbot": ["triton-op-coding", "ascend-profiling"],
              "project-guarded": ["ascend-profiling", "triton-guarded-kernel"]}
    digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
    manifest = {"prompt": str(prompt), "prompt_sha256": digest, "model": model,
        "model_sha256": hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "treatments": {name: {"skills": values, "skill_sha256": {v: "frozen-" + v for v in values}}
                       for name, values in skills.items()}}
    assets = {}
    for name in ("baseline", "case_spec", "runner"):
        path = tmp_path / name
        path.write_text("frozen\n")
        assets[name] = str(path)
    config = {"prompt": str(prompt), "prompt_sha256": digest, "assets": assets,
              "timeouts": {"agent": 360, "cell": 600, "wave": 600}, "waves": 4}
    placements = {
        "cannbot": [{"profile": "bz-a3-1", "device": 0}, {"profile": "bz-a3-2", "device": 2}],
        "project-cannbot": [{"profile": "bz-a3-1", "device": 1}, {"profile": "bz-a3-2", "device": 3}],
        "project-guarded": [{"profile": "bz-a3-2", "device": 0}, {"profile": "bz-a3-1", "device": 2}],
    }
    return config, manifest, placements


class Agent:
    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()

    def launch(self, request, timeout):
        with self.lock:
            self.requests.append((request, timeout))
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        return {"status": "ok", "controller_usage": {"billed": 1, "calls": [
            {"arguments": ["check", "--scope", "development", "--round", "1"]}]}}


class Client:
    def __init__(self, fail_once=False, candidate_failure=False, retain_handle=True):
        self.requests = []
        self.observations = []
        self.fail_once, self.candidate_failure = fail_once, candidate_failure
        self.retain_handle = retain_handle
        self.failed = set()

    def run(self, request):
        self.requests.append(request)
        key = (request["wave"], request["cell"])
        if self.fail_once and key not in self.failed:
            self.failed.add(key)
            handle = f"{request['profile']}:kept" if self.retain_handle else None
            return {"status": "infrastructure_error", "failure_type": "device_error",
                    "handle": handle, "profile": request["profile"], "device": request["device"]}
        if self.candidate_failure and request["wave"] == "1" and request["cell"] == "cannbot":
            return {"status": "compile_error", "diagnostics": "full compiler traceback"}
        return {"status": "ok", "passed": True, "artifacts": {"candidate_sha256": "remote"}}

    def observe(self, request, handle):
        self.observations.append((request, handle))
        return {"status": "ok", "passed": True, "handle": handle,
                "profile": request["profile"], "device": request["device"]}


def test_end_to_end_maps_controller_submission_and_bz_assignments(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    agent, client = Agent(), Client()
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, client)

    assert ledger["status"] == "complete" and len(agent.requests) == len(client.requests) == 12
    assert all(timeout == 360 for _, timeout in agent.requests)
    assert all(request["controller_contract"] == {"billed_limit": 1,
        "command": ["check", "--scope", "development", "--round", "1"]} for request, _ in agent.requests)
    assert all(request["cases"] == list(range(7)) for request in client.requests)
    assert {(r["profile"], r["device"]) for r in client.requests[:3]} == {
        ("bz-a3-1", 0), ("bz-a3-1", 1), ("bz-a3-2", 0)}
    assert all(Path(r["candidate"]).name == "candidate.py" for r in client.requests)
    wire = json.dumps([request for request, _ in agent.requests])
    assert "profile" not in wire and "msprof" not in wire
    assert json.loads((tmp_path / "campaign" / "ledger.json").read_text()) == ledger


def test_retained_terminal_job_is_observed_without_duplicate_run_or_agent(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    agent, client = Agent(), Client(fail_once=True)
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, client)

    assert len(agent.requests) == 12
    assert len(client.requests) == len(client.observations) == 12
    assert not ledger["reschedule"]
    for wave in ledger["waves"]:
        for cell in wave["cells"]:
            assert cell["retry"]["candidate_sha256"] == cell["candidate_sha256"]
            assert cell["retry"]["terminal_resumed"] is True
            assert cell["retry"]["attempt"] == cell["attempt"] == 1
    first = client.requests[0]
    observed, handle = next(item for item in client.observations
                            if item[0]["wave"] == first["wave"]
                            and item[0]["cell"] == first["cell"])
    assert (observed["profile"], observed["device"]) == (first["profile"], first["device"])
    assert handle == f"{first['profile']}:kept"


def test_pre_dispatch_infrastructure_without_handle_replays_frozen_submission(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    agent, client = Agent(), Client(fail_once=True, retain_handle=False)
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, client)

    assert len(agent.requests) == 12
    assert len(client.requests) == 24
    assert not client.observations and not ledger["reschedule"]
    for wave in ledger["waves"]:
        for cell in wave["cells"]:
            assert cell["retry"]["agent"]["submission_replayed"] is True
            assert cell["retry"]["attempt"] == 2


def test_counted_compile_failure_is_not_retried(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    agent, client = Agent(), Client(candidate_failure=True)
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, client)
    failed = ledger["waves"][0]["cells"][0]
    assert failed["outcome"] == "compile_error" and "retry" not in failed
    assert len(agent.requests) == len(client.requests) == 12
    assert failed["terminal"]["diagnostics"] == "full compiler traceback"


def test_placement_contract_rejects_shared_primary_device(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    placements["project-cannbot"][0] = placements["cannbot"][0]
    try:
        module.run(config, manifest, placements, tmp_path / "campaign", Agent(), Client())
    except module.DiagnosticError as error:
        assert "distinct physical devices" in str(error)
    else:
        raise AssertionError("shared primary device accepted")


def test_adaptive_bz_api_runs_one_wave_and_accepts_external_receipt(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    root, agent, client = tmp_path / "campaign", Agent(), Client()
    paused = module.run_wave(config, manifest, placements, root, agent, client, 1)

    assert paused["status"] == "awaiting_curation"
    assert len(paused["waves"]) == 1
    receipt = {"wave": 1, "accepted": True,
               "stable_ref_citations": ["ref://profiling-skill/triton-ascend/debugging/wave-1"],
               "librarian_query_ids": ["query-1"]}
    ready = module.acknowledge_curation(config, manifest, placements, root, receipt)
    assert ready["status"] == "ready_for_next"

    revised = tmp_path / "revised.md"
    revised.write_text("same experiment, curated clarification\n")
    config = {**config, "prompt": str(revised),
              "prompt_sha256": hashlib.sha256(revised.read_bytes()).hexdigest()}
    manifest = {**manifest, "prompt": config["prompt"],
                "prompt_sha256": config["prompt_sha256"]}
    resumed = module.run_wave(config, manifest, placements, root, agent, client, 2)
    assert len(resumed["waves"]) == 2
    assert len(agent.requests) == 6


def test_adaptive_bz_rejects_placement_drift_before_launch(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    root, agent = tmp_path / "campaign", Agent()
    module.run_wave(config, manifest, placements, root, agent, Client(), 1)
    module.acknowledge_curation(config, manifest, placements, root, {
        "wave": 1, "accepted": True,
        "stable_ref_citations": ["ref://profiling-skill/common/debugging/wave-1"],
        "librarian_query_ids": ["query-1"],
    })
    placements["cannbot"][0]["device"] = 7
    with pytest.raises(module.DiagnosticError, match="drift"):
        module.run_wave(config, manifest, placements, root, agent, Client(), 2)
    assert len(agent.requests) == 3
