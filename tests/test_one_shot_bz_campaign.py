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
                    "handle": f"{request['profile']}:retained"}
        if self.fail_once and key not in self.failed:
            self.failed.add(key)
            return {"status": "infrastructure_error", "failure_type": "device_error"}
        if self.candidate_failure and request["wave"] == "1" and request["cell"] == "cannbot-attempt-1":
            return {"status": "compile_error", "diagnostics": "full compiler traceback"}
        return {"status": "ok", "passed": True, "artifacts": {"candidate_sha256": "remote"}}

    def resume(self, requests, handle, observe_timeout):
        matches = [request for request in requests
                   if handle.startswith(request["profile"] + ":")]
        assert len(matches) == 1
        return self.run({**matches[0], "retained_handle": handle,
                         "observe_timeout": observe_timeout})


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


def test_adaptive_bz_reuses_controller_snapshot_across_prompt_revision(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    root, agent, client = tmp_path / "campaign", Agent(), Client()
    first = module.run_wave(config, manifest, placements, root, agent, client, 1)
    receipt = {"wave": 1, "campaign_id": first["campaign_id"],
               "wave_sha256": module.DiagnosticCampaign._wave_sha256(first["waves"][0]),
               "curator_operation_id": "curate-wave-1", "accepted": True,
               "stable_ref_citations": ["ref://profiling-skill/common/debugging/wave-1"],
               "librarian_query_ids": ["query-1"]}
    module.acknowledge_curation(config, manifest, placements, root, receipt)
    revised = tmp_path / "revised.md"
    revised.write_text("curated prompt revision\n")
    config = {**config, "prompt": str(revised),
              "prompt_sha256": hashlib.sha256(revised.read_bytes()).hexdigest()}
    manifest = {**manifest, "prompt": config["prompt"],
                "prompt_sha256": config["prompt_sha256"]}
    second = module.run_wave(config, manifest, placements, root, agent, client, 2)

    assert first["campaign_id"] == second["campaign_id"]
    assert first["assets"] == second["assets"]
    assert len(agent.requests) == len(client.requests) == 6
    assert len({request["campaign"] for request in client.requests}) == 1
    assert len([request for request, _ in agent.requests if request["wave"] == 1]) == 3


def test_adaptive_rejects_nonfresh_root_before_freezing_assets(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "existing").write_text("keep\n")
    with pytest.raises(module.DiagnosticError, match="not fresh"):
        module.run_wave(config, manifest, placements, root, Agent(), Client(), 1)
    assert list(tmp_path.glob(".campaign-inputs-*")) == []


def test_partial_asset_snapshot_failure_is_cleaned_up(tmp_path: Path):
    config, manifest, placements = inputs(tmp_path)
    Path(config["assets"]["case_spec"]).unlink()
    with pytest.raises(module.DiagnosticError, match="asset is missing"):
        module.run_wave(config, manifest, placements, tmp_path / "campaign",
                        Agent(), Client(), 1)
    assert list(tmp_path.glob(".campaign-inputs-*")) == []


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


def test_snapshot_failure_cannot_invoke_agent_twice(tmp_path: Path, monkeypatch):
    agent = Agent()
    frozen = module.FrozenAgentLauncher(agent)
    request = {"cell_id": "wave-1-cannbot", "workspace": str(tmp_path / "one")}
    Path(request["workspace"]).mkdir()
    original = Path.read_bytes

    def unreadable(path):
        if path.name == "candidate.py":
            raise PermissionError("candidate is unreadable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    try:
        frozen.launch(request, 10)
    except PermissionError:
        pass
    else:
        raise AssertionError("unreadable candidate accepted")
    request["workspace"] = str(tmp_path / "two")
    Path(request["workspace"]).mkdir()
    replay = frozen.launch(request, 5)
    assert len(agent.requests) == 1
    assert replay["status"] == "infrastructure_error"
    assert replay["failure_type"] == "snapshot_error" and replay["submission_replayed"] is True


def test_first_terminal_attempt_uses_frozen_submission(tmp_path: Path, monkeypatch):
    agent = Agent()
    frozen = module.FrozenAgentLauncher(agent)
    workspace = tmp_path / "attempt-1" / "workspace"
    workspace.mkdir(parents=True)
    request = {"cell_id": "wave-1-cannbot", "workspace": str(workspace)}
    original = Path.read_bytes

    def mutate_after_read(path):
        content = original(path)
        if path.name == "candidate.py":
            path.write_text("background mutation\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    result = frozen.launch(request, 10)
    assert result["status"] == "ok" and len(agent.requests) == 1
    assert result["candidate_sha256"] == {
        "candidate.py": hashlib.sha256(b"candidate\n").hexdigest(),
        "candidate.manifest.json": hashlib.sha256(b"{}\n").hexdigest(),
    }
    assert (workspace / "candidate.py").read_text() == "candidate\n"
    assert (workspace / "candidate.py").stat().st_mode & 0o777 == 0o444
    snapshot = workspace.parent / "frozen-submission"
    assert (snapshot / "candidate.py").read_text() == "candidate\n"
    assert snapshot.stat().st_mode & 0o777 == 0o555
    monkeypatch.setattr(Path, "read_bytes", original)
    (workspace / "candidate.py").unlink()
    (workspace / "candidate.py").write_text("delayed mutation\n")
    (tmp_path / "inputs").mkdir()
    _config, _manifest, placements = inputs(tmp_path / "inputs")
    client = Client()
    hook = module.BzTerminalHook(client, placements, _config["assets"], "unique")
    hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                "candidate_sha256": result["candidate_sha256"]}, 240)
    assert Path(client.requests[0]["candidate"]).read_text() == "candidate\n"
    assert Path(client.requests[0]["candidate"]).parent == snapshot


def test_snapshot_failure_remains_reschedulable_in_campaign(tmp_path: Path, monkeypatch):
    config, manifest, placements = inputs(tmp_path)
    agent = Agent()
    original = Path.read_bytes

    def unreadable(path):
        if path.name == "candidate.py":
            raise PermissionError("candidate is unreadable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    ledger = module.run(config, manifest, placements, tmp_path / "campaign", agent, Client())
    failed = ledger["waves"][0]["cells"][0]
    assert len(agent.requests) == 12
    assert failed["category"] == "infrastructure"
    assert failed["retry"]["outcome"] == "snapshot_error"
    assert "wave-1-cannbot" in ledger["reschedule"]


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
    for requests in by_cell.values():
        assert len(requests) == 2
        first, resumed = requests
        assert resumed["retained_handle"] == f"{first['profile']}:retained"
        assert resumed["observe_timeout"] == first["timeout"]
        assert {key: value for key, value in resumed.items()
                if key not in {"retained_handle", "observe_timeout"}} == first
        assert first["cell"].endswith("-attempt-1")


def test_retained_observation_uses_remaining_attempt_budget(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    client = Client(observer_once=True)
    hook = module.BzTerminalHook(client, placements, config["assets"], "unique")
    first = tmp_path / "attempt-1" / "workspace"
    second = tmp_path / "attempt-2" / "workspace"
    for workspace in (first, second):
        workspace.mkdir(parents=True)
        snapshot = workspace.parent / "frozen-submission"
        snapshot.mkdir()
        (snapshot / "candidate.py").write_text("candidate\n")
        (snapshot / "candidate.manifest.json").write_text("{}\n")
    hook.check({"cell_id": "wave-1-cannbot", "workspace": str(first)}, 240)
    result = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(second)}, 7)
    assert client.requests[1]["cell"] == "cannbot-attempt-1"
    assert client.requests[1]["timeout"] == 240
    assert client.requests[1]["observe_timeout"] == 7
    assert result["terminal_attempt"] == 1


def test_failed_retained_observation_never_dispatches_fallback(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)

    class RetainedFailure(Client):
        def run(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {"status": "infrastructure_error", "failure_type": "observer_error",
                        "handle": "bz-a3-1:retained"}
            if len(self.requests) == 2:
                return {"status": "infrastructure_error", "failure_type": "transport_error",
                        "handle": "bz-a3-1:retained"}
            raise AssertionError("retained handle incorrectly dispatched a fallback")

    client = RetainedFailure()
    hook = module.BzTerminalHook(client, placements, config["assets"], "unique")
    for attempt in (1, 2):
        workspace = tmp_path / f"attempt-{attempt}" / "workspace"
        workspace.mkdir(parents=True)
        snapshot = workspace.parent / "frozen-submission"
        snapshot.mkdir()
        (snapshot / "candidate.py").write_text("candidate\n")
        (snapshot / "candidate.manifest.json").write_text("{}\n")
        result = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace)}, 240)
    assert result["status"] == "infrastructure_error"
    assert len(client.requests) == 2
    assert client.requests[1]["cell"] == "cannbot-attempt-1"
    assert (client.requests[1]["profile"], client.requests[1]["device"]) == ("bz-a3-1", 0)
    assert client.requests[1]["observe_timeout"] == client.requests[1]["timeout"]


def test_terminal_retained_failure_dispatches_new_fallback_attempt(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)

    class RetainedThenTerminalFailure(Client):
        def run(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {"status": "infrastructure_error", "failure_type": "observer_error",
                        "handle": "bz-a3-1:retained"}
            if len(self.requests) == 2:
                return {"status": "infrastructure_error", "failure_type": "device_error",
                        "handle": "bz-a3-1:retained"}
            return {"status": "ok", "passed": True}

    client = RetainedThenTerminalFailure()
    hook = module.BzTerminalHook(client, placements, config["assets"], "unique")
    workspace = tmp_path / "attempt-1" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name in ("candidate.py", "candidate.manifest.json"):
        (snapshot / name).write_text("{}\n")
    first = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace)}, 240)
    result = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                         "retained_terminal_request": first["retained_terminal_request"]}, 240)
    assert result["status"] == "ok" and result["terminal_attempt"] == 2
    assert [request["cell"] for request in client.requests] == [
        "cannbot-attempt-1", "cannbot-attempt-1", "cannbot-attempt-2"]
    assert (client.requests[2]["profile"], client.requests[2]["device"]) == ("bz-a3-2", 2)
    assert "observe_timeout" not in client.requests[2]


def test_exhausted_retained_failure_preserves_attempt_for_next_resume(
        tmp_path: Path, monkeypatch):
    config, _manifest, placements = inputs(tmp_path)

    class TerminalThenSuccess(Client):
        def run(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {"status": "infrastructure_error", "failure_type": "observer_error",
                        "handle": "bz-a3-2:retained"}
            if len(self.requests) == 2:
                return {"status": "infrastructure_error", "failure_type": "device_error",
                        "handle": "bz-a3-2:retained"}
            return {"status": "ok", "passed": True}

    client = TerminalThenSuccess()
    hook = module.BzTerminalHook(client, placements, config["assets"], "campaign")
    workspace = tmp_path / "attempt-2" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name in ("candidate.py", "candidate.manifest.json"):
        (snapshot / name).write_text("{}\n")
    ticks = iter((0, 0, 31))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    uncertain = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                            "terminal_attempt": 2}, 30)
    assert uncertain["retained_terminal_request"]["cell"] == "cannbot-attempt-2"
    first = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                        "terminal_attempt": 2}, 30)
    assert first["terminal_attempt"] == 2 and not hook._uncertain
    monkeypatch.setattr(module.time, "monotonic", lambda: 0)
    second = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                         "terminal_attempt": first["terminal_attempt"] + 1}, 30)
    assert second["status"] == "ok"
    assert [request["cell"] for request in client.requests] == [
        "cannbot-attempt-2", "cannbot-attempt-2", "cannbot-attempt-3"]


def test_terminal_infrastructure_result_uses_fallback(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)

    class TerminalFailure(Client):
        def run(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {"status": "infrastructure_error",
                        "failure_type": "remote_infrastructure_error",
                        "handle": "bz-a3-1:terminal", "artifacts": {"remote_run_root": "kept"}}
            return {"status": "ok", "passed": True}

    client = TerminalFailure()
    hook = module.BzTerminalHook(client, placements, config["assets"], "unique")
    for attempt in (1, 2):
        workspace = tmp_path / f"attempt-{attempt}" / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        result = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace)}, 240)
    assert result["status"] == "ok" and len(client.requests) == 2
    assert client.requests[0]["cell"] == "cannbot-attempt-1"
    assert client.requests[1]["cell"] == "cannbot-attempt-2"
    assert (client.requests[1]["profile"], client.requests[1]["device"]) == ("bz-a3-2", 2)


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


def test_frozen_launcher_replays_durable_receipt_after_restart(tmp_path: Path):
    cell_root = tmp_path / "cells" / "wave-1-cannbot"
    first_workspace = cell_root / "attempt-1" / "workspace"
    first_workspace.mkdir(parents=True)
    request = {"cell_id": "wave-1-cannbot", "workspace": str(first_workspace)}
    first_agent = Agent()
    first = module.FrozenAgentLauncher(first_agent).launch(request, 10)

    class MustNotLaunch:
        def launch(self, _request, _timeout):
            raise AssertionError("durably completed agent was relaunched")

    for name in ("candidate.py", "candidate.manifest.json"):
        (first_workspace / name).unlink()
    same_attempt = module.FrozenAgentLauncher(MustNotLaunch()).launch(request, 10)
    assert same_attempt["submission_replayed"] is True
    assert (first_workspace / "candidate.py").read_text() == "candidate\n"

    second_workspace = cell_root / "attempt-2" / "workspace"
    second_workspace.mkdir(parents=True)
    request["workspace"] = str(second_workspace)
    replay = module.FrozenAgentLauncher(MustNotLaunch()).launch(request, 10)
    assert first["status"] == replay["status"] == "ok"
    assert replay["submission_replayed"] is True
    assert (second_workspace / "candidate.py").read_text() == "candidate\n"


def test_frozen_launcher_rejects_changed_durable_snapshot(tmp_path: Path):
    cell_root = tmp_path / "cells" / "wave-1-cannbot"
    first_workspace = cell_root / "attempt-1" / "workspace"
    first_workspace.mkdir(parents=True)
    request = {"cell_id": "wave-1-cannbot", "workspace": str(first_workspace)}
    module.FrozenAgentLauncher(Agent()).launch(request, 10)
    (first_workspace.parent / "frozen-submission" / "candidate.py").chmod(0o644)
    (first_workspace.parent / "frozen-submission" / "candidate.py").write_text("changed\n")

    class MustNotLaunch:
        def launch(self, _request, _timeout):
            raise AssertionError("corrupt durable snapshot caused relaunch")

    second_workspace = cell_root / "attempt-2" / "workspace"
    second_workspace.mkdir(parents=True)
    request["workspace"] = str(second_workspace)
    with pytest.raises(module.DiagnosticError, match="snapshot digest mismatch"):
        module.FrozenAgentLauncher(MustNotLaunch()).launch(request, 10)


def test_adaptive_crash_window_reuses_snapshot_and_terminal_receipt(tmp_path: Path):
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    config, manifest, placements = inputs(input_root)
    root = tmp_path / "campaign"
    first = module.run_wave(config, manifest, placements, root, Agent(), Client(), 1)
    ledger = json.loads((root / "ledger.json").read_text())
    ledger["status"] = "running"
    ledger["waves"][0]["cells"] = [
        cell for cell in ledger["waves"][0]["cells"] if cell["treatment"] != "cannbot"]
    (root / "ledger.json").write_text(json.dumps(ledger))

    class MustNotLaunch:
        def launch(self, _request, _timeout):
            raise AssertionError("durable agent submission was relaunched")

    class MustNotRun:
        def run(self, _request):
            raise AssertionError("completed terminal result was redispatched")

    recovered = module.run_wave(
        config, manifest, placements, root, MustNotLaunch(), MustNotRun(), 1)
    assert recovered["status"] == "awaiting_curation"
    cell = next(cell for cell in recovered["waves"][0]["cells"]
                if cell["treatment"] == "cannbot")
    assert cell["category"] == "counted"
    assert cell["agent"]["status"] == "ok"


def test_uncertain_durable_launch_is_never_billed_twice(tmp_path: Path):
    class Interrupted:
        calls = 0

        def launch(self, _request, _timeout):
            self.calls += 1
            raise KeyboardInterrupt

    cell_root = tmp_path / "cells" / "wave-1-cannbot"
    first_workspace = cell_root / "attempt-1" / "workspace"
    first_workspace.mkdir(parents=True)
    request = {"cell_id": "wave-1-cannbot", "workspace": str(first_workspace)}
    agent = Interrupted()
    with pytest.raises(KeyboardInterrupt):
        module.FrozenAgentLauncher(agent).launch(request, 10)
    second_workspace = cell_root / "attempt-2" / "workspace"
    second_workspace.mkdir(parents=True)
    request["workspace"] = str(second_workspace)
    result = module.FrozenAgentLauncher(agent).launch(request, 10)
    assert agent.calls == 1
    assert result["failure_type"] == "uncertain_agent_launch"


def test_terminal_resume_rejects_changed_frozen_submission(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    workspace = tmp_path / "attempt-1" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name, content in (("candidate.py", b"candidate\n"),
                          ("candidate.manifest.json", b"{}\n")):
        (snapshot / name).write_bytes(content)
    expected = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
                for name in ("candidate.py", "candidate.manifest.json")}
    (snapshot / "candidate.py").write_text("changed\n")
    hook = module.BzTerminalHook(Client(), placements, config["assets"], "campaign")
    with pytest.raises(module.DiagnosticError, match="digest mismatch"):
        hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                    "candidate_sha256": expected}, 30)


def test_rescheduled_terminal_uses_new_durable_attempt_identity(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    workspace = tmp_path / "attempt-2" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name in ("candidate.py", "candidate.manifest.json"):
        (snapshot / name).write_text("{}\n")
    client = Client()
    hook = module.BzTerminalHook(client, placements, config["assets"], "campaign")
    result = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                         "terminal_attempt": 3}, 30)
    assert result["status"] == "ok"
    assert client.requests[0]["cell"] == "cannbot-attempt-3"


def test_retained_terminal_request_survives_hook_restart(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    workspace = tmp_path / "attempt-1" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name in ("candidate.py", "candidate.manifest.json"):
        (snapshot / name).write_text("{}\n")

    class ObserverThenSuccess(Client):
        def run(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {"status": "infrastructure_error", "failure_type": "observer_error",
                        "handle": "bz-a3-1:retained"}
            return {"status": "ok", "passed": True, "handle": "bz-a3-1:retained"}

    client = ObserverThenSuccess()
    first = module.BzTerminalHook(client, placements, config["assets"], "campaign").check(
        {"cell_id": "wave-1-cannbot", "workspace": str(workspace),
         "terminal_attempt": 1}, 30)
    assert first["retained_terminal_request"]["cell"] == "cannbot-attempt-1"
    second = module.BzTerminalHook(client, placements, config["assets"], "campaign").check(
        {"cell_id": "wave-1-cannbot", "workspace": str(workspace),
         "terminal_attempt": 1,
         "retained_terminal_request": first["retained_terminal_request"]}, 30)
    assert second["status"] == "ok"
    assert client.requests[1]["cell"] == "cannbot-attempt-1"
    assert "observe_timeout" in client.requests[1]


def test_manual_handle_reconciliation_uses_bz_resume_without_redispatch(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    workspace = tmp_path / "attempt-1" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name in ("candidate.py", "candidate.manifest.json"):
        (snapshot / name).write_text("{}\n")

    class ResumeOnlyClient(Client):
        def resume(self, requests, handle, observe_timeout):
            assert len(requests) == 2
            request = requests[0]
            self.requests.append(request)
            assert handle == "bz-a3-1:reconciled"
            assert observe_timeout == 30
            return {"status": "ok", "passed": True,
                    "handle": handle, "cell": request["cell"]}

    client = ResumeOnlyClient()
    hook = module.BzTerminalHook(client, placements, config["assets"], "campaign")
    result = hook.resume(
        {"cell_id": "wave-1-cannbot", "workspace": str(workspace),
         "terminal_attempt": 1}, "bz-a3-1:reconciled", 30)

    assert result["status"] == "ok"
    assert result["terminal_attempt"] == 1
    assert len(client.requests) == 1


def test_reconcile_terminal_cli_updates_uncertain_receipt(
    tmp_path: Path, monkeypatch, capsys,
):
    config, manifest, placements = inputs(tmp_path)
    root = tmp_path / "campaign"
    cell = "wave-1-cannbot"
    workspace = root / "cells" / cell / "attempt-1" / "workspace"
    workspace.mkdir(parents=True)
    for name in ("candidate.py", "candidate.manifest.json"):
        (workspace / name).write_text("{}\n")

    class LocalTimeout:
        def check(self, request, timeout_seconds):
            return {"status": "transport_or_observer_error",
                    "invocation_timeout": True}

    campaign = module.DiagnosticCampaign(
        manifest, root, module.CommandLauncher(["false"]), LocalTimeout(),
        campaign_id="campaign", campaign_identity={"config_sha256": "fixed"},
    )
    request = {
        "protocol_version": 1, "operation": "terminal_check", "cell_id": cell,
        "workspace": str(workspace), "benchmark": "streaming-matmul-add",
        "cases": list(range(7)), "terminal_attempt": 1,
        "candidate_sha256": {name: module.diagnostic_campaign.sha256_file(workspace / name)
                             for name in ("candidate.py", "candidate.manifest.json")},
    }
    campaign._durable_terminal_check(request, 30)
    (root / "ledger.json").write_text(json.dumps({"campaign_id": "campaign"}))
    paths = {}
    for name, value in (("config", config), ("manifest", manifest),
                        ("placements", placements)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    monkeypatch.setattr(sys, "argv", [
        "one_shot_bz_campaign.py", "--action", "reconcile-terminal",
        "--config", str(paths["config"]), "--manifest", str(paths["manifest"]),
        "--placements", str(paths["placements"]), "--run-root", str(root),
        "--cell-id", cell, "--agent-attempt", "1", "--terminal-attempt", "1",
        "--terminal-handle", "bz-a3-1:recovered",
    ])

    assert module.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "reconciled"
    assert output["receipt"]["state"] == "started"
    assert output["receipt"]["result"]["handle"] == "bz-a3-1:recovered"


def test_retained_terminal_request_rejects_cross_cell_snapshot(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    workspace = tmp_path / "attempt-1" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name in ("candidate.py", "candidate.manifest.json"):
        (snapshot / name).write_text("{}\n")
    other = tmp_path / "other" / "frozen-submission"
    other.mkdir(parents=True)
    for name in ("candidate.py", "candidate.manifest.json"):
        (other / name).write_text("other\n")
    client = Client()
    retained = {
        "campaign": "campaign", "wave": "1", "cell": "cannbot-attempt-1",
        "profile": "bz-a3-1", "device": 0, "timeout": 30,
        "candidate": str(other / "candidate.py"),
        "candidate_manifest": str(other / "candidate.manifest.json"),
        **config["assets"], "cases": list(range(7)),
    }
    hook = module.BzTerminalHook(client, placements, config["assets"], "campaign")
    with pytest.raises(module.DiagnosticError, match="retained terminal request mismatch"):
        hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                    "retained_terminal_request": retained}, 30)
    assert client.requests == []


def test_retained_terminal_request_rejects_mutated_snapshot(tmp_path: Path):
    config, _manifest, placements = inputs(tmp_path)
    workspace = tmp_path / "attempt-1" / "workspace"
    snapshot = workspace.parent / "frozen-submission"
    workspace.mkdir(parents=True)
    snapshot.mkdir()
    for name in ("candidate.py", "candidate.manifest.json"):
        (snapshot / name).write_text("{}\n")
    client = Client(observer_once=True)
    hook = module.BzTerminalHook(client, placements, config["assets"], "campaign")
    first = hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace)}, 30)
    (snapshot / "candidate.py").write_text("mutated\n")

    with pytest.raises(module.DiagnosticError, match="retained terminal request mismatch"):
        hook.check({"cell_id": "wave-1-cannbot", "workspace": str(workspace),
                    "retained_terminal_request": first["retained_terminal_request"]}, 30)
    assert len(client.requests) == 1


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
