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
    def __init__(self, fail_once=False, candidate_failure=False, observer_once=False):
        self.requests = []
        self.fail_once, self.candidate_failure = fail_once, candidate_failure
        self.observer_once = observer_once
        self.failed = set()

    def run(self, request):
        self.requests.append(request)
        key = (request["wave"], request["cell"].rsplit("-attempt-", 1)[0])
        if self.observer_once and key not in self.failed:
            self.failed.add(key)
            return {"status": "infrastructure_error", "failure_type": "observer_error",
                    "handle": "bz-a3-1:retained"}
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
    assert set(ledger["assets"]["sha256"]) == {"baseline", "case_spec", "runner"}
    assert {request["campaign"] for request in client.requests} == {ledger["campaign_id"]}
    assert all(Path(r[name]).parent == Path(ledger["assets"]["root"])
               for r in client.requests for name in ("baseline", "case_spec", "runner"))
    wire = json.dumps([request for request, _ in agent.requests])
    assert "profile" not in wire and "msprof" not in wire
    assert json.loads((tmp_path / "campaign" / "ledger.json").read_text()) == ledger


def test_common_assets_are_snapshotted_before_any_cell_runs(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)

    class MutatingClient(Client):
        def run(self, request):
            ledger = json.loads((tmp_path / "campaign" / "ledger.json").read_text())
            assert set(ledger["assets"]["sha256"]) == {"baseline", "case_spec", "runner"}
            for source in config["assets"].values():
                Path(source).write_text("changed\n")
            assert all(Path(request[name]).read_text() == "frozen\n"
                       for name in ("baseline", "case_spec", "runner"))
            return super().run(request)

    module.run(config, manifest, placements, tmp_path / "campaign", Agent(), MutatingClient())


def test_common_assets_with_same_basename_remain_distinct(tmp_path: Path):
    config, _manifest, _placements = inputs(tmp_path)
    for index, name in enumerate(("baseline", "case_spec", "runner")):
        source = tmp_path / "sources" / name / "main.py"
        source.parent.mkdir(parents=True)
        source.write_text(f"asset-{index}\n")
        config["assets"][name] = str(source)
    frozen, _evidence = module.freeze_assets(config["assets"], tmp_path / "run", "unique")
    assert [Path(frozen[name]).read_text() for name in ("baseline", "case_spec", "runner")] == [
        "asset-0\n", "asset-1\n", "asset-2\n"]


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


def test_launcher_exception_is_replayed_without_second_agent_invocation(tmp_path: Path):
    class RaisingAgent:
        calls = 0

        def launch(self, _request, _timeout):
            self.calls += 1
            raise RuntimeError("controller unavailable")

    launcher = RaisingAgent()
    frozen = module.FrozenAgentLauncher(launcher)
    request = {"cell_id": "wave-1-cannbot", "workspace": str(tmp_path / "one")}
    Path(request["workspace"]).mkdir()
    first = frozen.launch(request, 10)
    request["workspace"] = str(tmp_path / "two")
    Path(request["workspace"]).mkdir()
    second = frozen.launch(request, 5)
    assert launcher.calls == 1
    assert first["failure_type"] == second["failure_type"] == "launcher_error"
    assert second["submission_replayed"] is True


def test_observer_interruption_reuses_original_receipt_and_placement(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    agent, client = Agent(), Client(observer_once=True)
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, client)

    assert len(agent.requests) == 12 and len(client.requests) == 24
    assert not ledger["reschedule"]
    by_cell = {}
    for request in client.requests:
        by_cell.setdefault((request["wave"], request["cell"]), []).append(request)
    assert len(by_cell) == 12
    assert all(len(requests) == 2
               and {**requests[1], "observe_timeout": None}
               == {**requests[0], "observe_timeout": None}
               and requests[1]["observe_timeout"] == requests[0]["timeout"]
               and requests[0]["cell"].endswith("-attempt-1")
               for requests in by_cell.values())


def test_retained_observation_uses_remaining_attempt_budget(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    client = Client(observer_once=True)
    hook = module.BzTerminalHook(client, placements, config["assets"], "unique")
    first = tmp_path / "attempt-1" / "workspace"
    second = tmp_path / "attempt-2" / "workspace"
    for workspace in (first, second):
        workspace.mkdir(parents=True)
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
    hook.check({"cell_id": "wave-1-cannbot", "workspace": str(first)}, 240)
    hook.check({"cell_id": "wave-1-cannbot", "workspace": str(second)}, 7)
    assert client.requests[1]["cell"] == "cannbot-attempt-1"
    assert client.requests[1]["timeout"] == 240
    assert client.requests[1]["observe_timeout"] == 7


def test_failed_retained_job_uses_fallback_in_same_retry(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)

    class RetainedFailure(Client):
        def run(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {"status": "infrastructure_error", "failure_type": "observer_error",
                        "handle": "bz-a3-1:retained"}
            if len(self.requests) == 2:
                return {"status": "infrastructure_error", "failure_type": "device_error",
                        "handle": "bz-a3-1:retained"}
            return {"status": "ok", "passed": True}

    client = RetainedFailure()
    hook = module.BzTerminalHook(client, placements, config["assets"], "unique")
    for attempt in (1, 2):
        workspace = tmp_path / f"attempt-{attempt}" / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        result = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace)}, 240)
    assert result["status"] == "ok"
    assert len(client.requests) == 3
    assert client.requests[1]["cell"] == "cannbot-attempt-1"
    assert (client.requests[2]["profile"], client.requests[2]["device"]) == ("bz-a3-2", 2)
    assert client.requests[2]["cell"] == "cannbot-attempt-2"


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


def test_placement_contract_rejects_primary_as_fallback(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    placements["cannbot"][1] = placements["cannbot"][0]
    try:
        module.run(config, manifest, placements, tmp_path / "campaign", Agent(), Client())
    except module.DiagnosticError as error:
        assert "fallback placement must differ" in str(error)
    else:
        raise AssertionError("primary placement accepted as fallback")


def test_frozen_launcher_delegates_cancellation():
    agent = Agent()
    module.FrozenAgentLauncher(agent).cancel()
    assert agent.cancelled is True


def test_terminal_cancel_waits_for_durable_client_completion(tmp_path: Path):
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()

    class BlockingClient:
        def run(self, _request):
            entered.set()
            release.wait()
            return {"status": "ok", "passed": True, "handle": "bz-a3-1:kept"}

    config, _manifest, placements = inputs(tmp_path)
    workspace = tmp_path / "attempt-1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "candidate.py").write_text("candidate\n")
    (workspace / "candidate.manifest.json").write_text("{}\n")
    hook = module.BzTerminalHook(BlockingClient(), placements, config["assets"], "unique")
    worker = threading.Thread(target=hook.check, args=({"cell_id": "wave-1-cannbot",
        "workspace": str(workspace)}, 30))
    worker.start()
    assert entered.wait(1)
    stopper = threading.Thread(target=lambda: (hook.cancel(), cancelled.set()))
    stopper.start()
    assert not cancelled.wait(0.05)
    release.set()
    worker.join(1), stopper.join(1)
    assert cancelled.is_set()


def test_each_campaign_uses_a_globally_unique_bz_receipt_namespace(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    first, second = Client(), Client()
    module.run(config, manifest, placements, tmp_path / "one" / "same", Agent(), first)
    module.run(config, manifest, placements, tmp_path / "two" / "same", Agent(), second)
    assert {r["campaign"] for r in first.requests}.isdisjoint(
        {r["campaign"] for r in second.requests})
