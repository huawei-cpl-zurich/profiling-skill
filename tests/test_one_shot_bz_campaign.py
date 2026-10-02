from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path


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
    skills = {name: list(module.diagnostic_campaign.TREATMENT_SKILLS[name])
              for name in module.TREATMENTS}
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
        self.cancelled = False

    def launch(self, request, timeout):
        with self.lock:
            self.requests.append((request, timeout))
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        return {"status": "ok", "controller_usage": {"billed": 1, "calls": [
            {"arguments": ["check", "--scope", "development", "--round", "1"]}]}}

    def cancel(self):
        self.cancelled = True


class Client:
    def __init__(self, fail_once=False, candidate_failure=False):
        self.requests = []
        self.fail_once, self.candidate_failure = fail_once, candidate_failure
        self.failed = set()

    def run(self, request):
        self.requests.append(request)
        key = (request["wave"], request["cell"].rsplit("-attempt-", 1)[0])
        if self.fail_once and key not in self.failed:
            self.failed.add(key)
            return {"status": "infrastructure_error", "failure_type": "device_error", "handle": "kept"}
        if self.candidate_failure and request["wave"] == "1" and request["cell"] == "cannbot-attempt-1":
            return {"status": "compile_error", "diagnostics": "full compiler traceback"}
        return {"status": "ok", "passed": True, "artifacts": {"candidate_sha256": "remote"}}


def test_end_to_end_maps_controller_submission_and_bz_assignments(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    agent, client = Agent(), Client()
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, client)

    assert ledger["status"] == "complete" and len(agent.requests) == len(client.requests) == 12
    assert all(timeout == 360 for _, timeout in agent.requests)
    assert all(request["controller_contract"] == {"billed_limit": 1,
        "command": ["check", "--scope", "development", "--round", "1"]} for request, _ in agent.requests)
    assert all(request["protocol_version"] == 2 and
               module.diagnostic_campaign.request_prompt_bytes(request) == b"same prompt\n"
               for request, _ in agent.requests)
    assert all(request["cases"] == list(range(7)) for request in client.requests)
    assert {(r["profile"], r["device"]) for r in client.requests[:3]} == {
        ("bz-a3-1", 0), ("bz-a3-1", 1), ("bz-a3-2", 0)}
    assert all(Path(r["candidate"]).name == "candidate.py" for r in client.requests)
    wire = json.dumps([request for request, _ in agent.requests])
    assert "profile" not in wire and "msprof" not in wire
    assert json.loads((tmp_path / "campaign" / "ledger.json").read_text()) == ledger


def test_infrastructure_retries_same_frozen_candidate_without_second_agent(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    agent, client = Agent(), Client(fail_once=True)
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, client)

    assert len(agent.requests) == 12
    assert len(client.requests) == 24
    assert not ledger["reschedule"]
    for wave in ledger["waves"]:
        for cell in wave["cells"]:
            assert cell["retry"]["candidate_sha256"] == cell["candidate_sha256"]
            assert cell["retry"]["agent"]["submission_replayed"] is True
    first = client.requests[0]
    treatment = first["cell"].rsplit("-attempt-", 1)[0]
    retry = next(r for r in client.requests if r["wave"] == first["wave"]
                 and r["cell"] == treatment + "-attempt-2")
    assert (first["profile"], first["device"]) != (retry["profile"], retry["device"])
    assert first["cell"] == treatment + "-attempt-1"


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


def test_frozen_launcher_delegates_cancellation():
    agent = Agent()
    module.FrozenAgentLauncher(agent).cancel()
    assert agent.cancelled is True
