import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "simple_experiment_runner",
    Path(__file__).parents[1] / "scripts" / "simple_experiment_runner.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], text=True, capture_output=True, check=True
    ).stdout.strip()


def source_repo(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.invalid")
    (path / "README.md").write_text("fixture\n")
    git(path, "add", "README.md")
    git(path, "commit", "-m", "fixture")
    return path


def config(tmp_path, **updates):
    skill = tmp_path / "skill"
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("test skill\n")
    value = {
        "campaign": "trial",
        "product": "a3",
        "task": "matmul",
        "treatment": "guarded",
        "source_repo": str(source_repo(tmp_path)),
        "base_ref": "main",
        "target": "bz-a3-1",
        "runtime": "cann",
        "remote_workspace": "/tmp/simple-runner",
        "rounds": 2,
        "infrastructure_retries": 2,
        "candidate_repairs": 2,
        "prompt": "Implement the exact requested kernel.\n",
        "skill_paths": [str(skill)],
    }
    value.update(updates)
    return value


def evidence(label="v1"):
    return {
        "candidate": f"KERNEL = {label!r}\n",
        "remote_script": "#!/bin/bash\npython candidate.py\n",
        "script_fragment": "python \"$CANDIDATE_PATH\"\n",
        "commands": ["python candidate.py"],
        "reasoning": f"reasoning {label}",
        "postmortem": f"postmortem {label}",
        "hypothesis": f"hypothesis {label}",
        "session_id": "session-1",
        "retrieved_knowledge": ["ref://example"],
    }


class Agent:
    def __init__(self, values=None):
        self.values = list(values or [evidence()])
        self.requests = []
        self.analyses = []

    def prepare(self, request):
        self.requests.append(request)
        return self.values.pop(0)

    def analyze(self, request):
        self.analyses.append(request)
        return {"reasoning": "final reasoning", "postmortem": "final postmortem"}


class Remote:
    def __init__(self, observations):
        self.observations = list(observations)
        self.submitted = []
        self.observed = []
        self.preflights = []

    def preflight(self, request):
        self.preflights.append(request)
        return {"capabilities": ["run", "observe", "logs"], "status": "completed"}

    def submit(self, request, script):
        handle = f"remote:{request['target']}:job:{len(self.submitted) + 1}"
        self.submitted.append((request, script.read_text()))
        return handle

    def observe(self, handle):
        self.observed.append(handle)
        value = self.observations.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def initialized(tmp_path, remote, agent=None, **updates):
    cell = runner.ExperimentCell(tmp_path / "runs", config(tmp_path, **updates), agent or Agent(), remote)
    cell.init()
    return cell


def success(us=12.5):
    return {"outcome": "success", "stdout": "PASS", "stderr": "", "profile_summary": {"duration_us": us}}


def test_init_creates_isolated_repo_and_branch(tmp_path):
    cell = initialized(tmp_path, Remote([]))
    assert cell.path == tmp_path / "runs" / "trial" / "a3-matmul-guarded"
    assert git(cell.repo, "branch", "--show-current") == "experiment/trial/a3-matmul-guarded"
    state = json.loads((cell.path / "state.json").read_text())
    assert state["round"] == 1 and state["status"] == "ready"
    assert not (cell.repo / "rounds").exists()
    copied = cell.agent_home / ".agents/skills/skill/SKILL.md"
    assert copied.read_text() == "test skill\n" and not copied.is_symlink()


def test_run_and_resume_reuse_exact_retained_handle(tmp_path):
    handle = "remote:bz-a3-1:job:1"
    remote = Remote([runner.ObservationLost("connection dropped"), success()])
    cell = initialized(tmp_path, remote)
    assert cell.run()["status"] == "infrastructure_unknown"
    assert (cell.path / "attempts/1/remote-job-id.txt").read_text().strip() == handle
    assert cell.resume()["status"] == "ready"
    assert remote.observed == [handle, handle]
    assert len(remote.submitted) == 1
    assert json.loads((cell.path / "state.json").read_text())["round"] == 2


@pytest.mark.parametrize("persisted_status", ["observing", "ready"])
def test_observing_state_blocks_run_and_resume_uses_pending_handle(tmp_path, persisted_status):
    remote = Remote([success()])
    agent = Agent()
    cell = initialized(tmp_path, remote, agent)
    attempt = cell.path / "attempts/1"
    attempt.mkdir()
    (attempt / "remote-job-id.txt").write_text("remote:bz-a3-1:job:42\n")
    (attempt / "agent-result.json").write_text(json.dumps(evidence()))
    (attempt / "run.sh").write_text(cell._script(evidence()["candidate"], evidence()["script_fragment"]))
    state = json.loads(cell.state_path.read_text())
    state.update(status=persisted_status, attempts=1, pending_attempt=1)
    cell.state_path.write_text(json.dumps(state))
    with pytest.raises(runner.RunnerError, match="resume"):
        cell.run()
    assert not remote.submitted and not agent.requests
    assert cell.resume()["round"] == 2
    assert remote.observed == ["remote:bz-a3-1:job:42"]


def test_known_infrastructure_failures_retry_without_advancing_round(tmp_path):
    infra = {"outcome": "infrastructure_failure", "diagnostics": "device unavailable"}
    remote = Remote([infra, infra, success()])
    cell = initialized(tmp_path, remote)
    result = cell.run()
    assert result["status"] == "ready" and result["round"] == 2
    assert len(remote.submitted) == 3
    assert [json.loads(p.read_text())["round"] for p in sorted((cell.path / "attempts").glob("*/request.json"))] == [1, 1, 1]


def test_retry_limit_is_clear_infrastructure_failure(tmp_path):
    infra = {"outcome": "infrastructure_failure", "diagnostics": "offline"}
    cell = initialized(tmp_path, Remote([infra, infra, infra]))
    result = cell.run()
    assert result["status"] == "infrastructure_failure"
    assert result["round"] == 1 and result["infrastructure_failures"] == 3


def test_run_does_not_restart_exhausted_infrastructure_budget(tmp_path):
    remote = Remote([{"outcome": "infrastructure_failure", "diagnostics": "offline"}])
    agent = Agent()
    cell = initialized(tmp_path, remote, agent, infrastructure_retries=0)
    exhausted = cell.run()
    assert cell.run() == exhausted
    assert len(agent.requests) == 1
    assert len(remote.submitted) == 1


def test_candidate_failure_repairs_same_round_and_passes_diagnostics(tmp_path):
    failed = {"outcome": "candidate_failure", "diagnostics": "compile: bad tl.load"}
    agent = Agent([evidence("bad"), evidence("fixed")])
    cell = initialized(tmp_path, Remote([failed, success()]), agent)
    result = cell.run()
    assert result["round"] == 2
    assert agent.requests[1]["round"] == 1
    assert agent.requests[1]["candidate_diagnostics"] == "compile: bad tl.load"
    assert agent.requests[1]["prior_candidate"] == evidence("bad")["candidate"]
    assert agent.requests[1]["prior_result"]["outcome"] == "candidate_failure"
    assert agent.requests[1]["prior_attempt"].endswith("attempts/1")
    assert agent.requests[1]["session_id"] == "session-1"
    assert len(list((cell.path / "attempts").iterdir())) == 2


def test_run_does_not_restart_exhausted_candidate_budget(tmp_path):
    remote = Remote([{"outcome": "candidate_failure", "diagnostics": "compile error"}])
    agent = Agent()
    cell = initialized(tmp_path, remote, agent, candidate_repairs=0)
    exhausted = cell.run()
    assert cell.run() == exhausted
    assert len(agent.requests) == 1
    assert len(remote.submitted) == 1


def test_completed_round_is_one_commit_with_required_evidence(tmp_path):
    cell = initialized(tmp_path, Remote([success(9.25)]))
    before = int(git(cell.repo, "rev-list", "--count", "HEAD"))
    cell.run()
    assert int(git(cell.repo, "rev-list", "--count", "HEAD")) == before + 1
    names = set(git(cell.repo, "show", "--pretty=", "--name-only", "HEAD").splitlines())
    expected = {
        "rounds/01/candidate.py", "rounds/01/commands.jsonl",
        "rounds/01/profile-summary.json", "rounds/01/reasoning.md",
        "rounds/01/postmortem.md", "rounds/01/retrieved-knowledge.json",
        "rounds/01/result.json",
        "rounds/01/run.sh", "rounds/01/inputs.json", "rounds/01/hypothesis.md",
    }
    assert expected <= names
    assert json.loads((cell.repo / "rounds/01/profile-summary.json").read_text())["duration_us"] == 9.25


def test_runner_materializes_exact_candidate_and_submitted_script(tmp_path):
    remote, agent = Remote([success()]), Agent([evidence("exact")])
    cell = initialized(tmp_path, remote, agent)
    cell.run()
    submitted = remote.submitted[0][1]
    assert "CANDIDATE_PATH" in submitted and "S0VSTkVM" in submitted
    assert (cell.repo / "rounds/01/candidate.py").read_text() == evidence("exact")["candidate"]
    assert (cell.repo / "rounds/01/run.sh").read_text() == submitted
    assert subprocess.run(["bash", str(cell.path / "attempts/1/run.sh")], check=False).returncode == 0
    inputs = json.loads((cell.repo / "rounds/01/inputs.json").read_text())
    assert inputs["prompt"] == "Implement the exact requested kernel.\n"
    assert inputs["treatment"] == "guarded" and inputs["allowed_skills"] == ["skill"]
    assert agent.analyses[0]["phase"] == "analyze"
    assert agent.analyses[0]["candidate"] == evidence("exact")["candidate"]
    assert (cell.repo / "rounds/01/postmortem.md").read_text() == "final postmortem\n"


def test_candidate_size_is_bounded_before_remote_submission(tmp_path):
    remote = Remote([])
    cell = initialized(tmp_path, remote, Agent([evidence("x" * 1_100_000)]))
    with pytest.raises(runner.RunnerError, match="candidate.*large"):
        cell.run()
    assert not remote.submitted


def test_rejected_candidate_creates_no_attempt_and_retry_can_succeed(tmp_path):
    remote = Remote([success()])
    agent = Agent([evidence("x" * 1_100_000), evidence("fixed")])
    cell = initialized(tmp_path, remote, agent)
    with pytest.raises(runner.RunnerError, match="candidate.*large"):
        cell.run()
    assert not any((cell.path / "attempts").iterdir())
    assert json.loads(cell.state_path.read_text())["attempts"] == 0
    assert cell.run()["round"] == 2
    assert len(remote.submitted) == 1


def test_analysis_interruption_resumes_without_remote_rerun(tmp_path):
    class Interrupted(Agent):
        def analyze(self, request):
            self.analyses.append(request)
            if len(self.analyses) == 1:
                raise runner.AgentInterrupted("controller stopped")
            return {"reasoning": "recovered", "postmortem": "recovered"}

    remote, agent = Remote([success()]), Interrupted()
    cell = initialized(tmp_path, remote, agent)
    assert cell.run()["status"] == "analysis_pending"
    assert cell.resume()["round"] == 2
    assert len(remote.submitted) == len(remote.observed) == 1


def test_finalization_reuses_commit_after_state_save_interruption(tmp_path, monkeypatch):
    remote = Remote([success(), success()])
    cell = initialized(tmp_path, remote)
    original = cell._save
    interrupted = False

    def fail_after_commit(state):
        nonlocal interrupted
        if state.get("round") == 2 and not interrupted:
            interrupted = True
            raise runner.RunnerError("state write interrupted")
        return original(state)

    monkeypatch.setattr(cell, "_save", fail_after_commit)
    with pytest.raises(runner.RunnerError, match="state write"):
        cell.run()
    commit = git(cell.repo, "rev-parse", "HEAD")
    monkeypatch.setattr(cell, "_save", original)
    assert cell.resume()["round"] == 2
    assert git(cell.repo, "rev-parse", "HEAD") == commit
    assert len(remote.submitted) == 1


def test_finalization_reuses_evidence_after_commit_interruption(tmp_path, monkeypatch):
    remote = Remote([success(), success()])
    cell = initialized(tmp_path, remote)
    original = runner._git
    interrupted = False

    def fail_commit(repo, *args):
        nonlocal interrupted
        if args and args[0] == "commit" and not interrupted:
            interrupted = True
            raise runner.RunnerError("commit interrupted")
        return original(repo, *args)

    monkeypatch.setattr(runner, "_git", fail_commit)
    with pytest.raises(runner.RunnerError, match="commit interrupted"):
        cell.run()
    monkeypatch.setattr(runner, "_git", original)
    assert cell.resume()["round"] == 2
    assert git(cell.repo, "log", "--format=%s").splitlines().count("experiment: complete round 1") == 1


def test_subprocess_agent_isolated_home_cwd_prompt_and_skills(tmp_path):
    home, repo = tmp_path / "home", tmp_path / "repo"
    home.mkdir(); repo.mkdir()
    program = tmp_path / "agent.py"
    program.write_text(
        "import json,os,sys\n"
        "r=json.load(sys.stdin); print(json.dumps({'candidate':'x','script_fragment':'true',"
        "'commands':[],'hypothesis':os.getcwd(),'retrieved_knowledge':[],"
        "'session_id':os.environ['HOME']+'|'+os.environ['CODEX_HOME']}))\n"
    )
    agent = runner.SubprocessAgent(["python", str(program)], repo, home)
    value = agent.prepare({"phase": "prepare", "prompt": "exact", "allowed_skills": ["one"]})
    assert value["hypothesis"] == str(repo)
    assert value["session_id"] == f"{home}|{home / '.codex'}"


def test_subprocess_agent_repair_and_analysis_flow(tmp_path):
    program = tmp_path / "agent-flow.py"
    program.write_text(
        "import json,os,sys\n"
        "r=json.load(sys.stdin)\n"
        "with open('agent-requests.jsonl','a') as f: f.write(json.dumps(r)+'\\n')\n"
        "if r['phase']=='analyze':\n"
        " print(json.dumps({'reasoning':'measured result','postmortem':'matched expectation'}))\n"
        "else:\n"
        " repaired=bool(r.get('prior_result'))\n"
        " print(json.dumps({'candidate':'FIXED = '+str(repaired)+'\\n',"
        "'script_fragment':'true','commands':['true'],'hypothesis':'repair',"
        "'retrieved_knowledge':[],'session_id':r.get('session_id') or 'durable-1'}))\n"
    )
    remote = Remote([
        {"outcome": "candidate_failure", "diagnostics": "compile failed"}, success()
    ])
    cfg = config(tmp_path, rounds=1, agent_command=["python", str(program)])
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, None, remote)
    cell.init()
    cell.agent = runner.SubprocessAgent(cfg["agent_command"], cell.repo, cell.agent_home)
    assert cell.run()["status"] == "complete"
    requests = [json.loads(line) for line in (cell.repo / "agent-requests.jsonl").read_text().splitlines()]
    assert [item["phase"] for item in requests] == ["prepare", "prepare", "analyze"]
    assert requests[1]["prior_candidate"] == "FIXED = False\n"
    assert requests[1]["session_id"] == requests[2]["session_id"] == "durable-1"
    assert remote.preflights == [{"target": "bz-a3-1"}]
    assert (cell.path / "preflights/1.json").is_file()
    assert (cell.repo / "rounds/01/reasoning.md").read_text() == "measured result\n"


def test_init_is_transactional_and_retryable(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    original = runner._git
    monkeypatch.setattr(runner, "_git", lambda *a, **k: (_ for _ in ()).throw(runner.RunnerError("checkout")))
    with pytest.raises(runner.RunnerError):
        cell.init()
    assert not cell.path.exists()
    monkeypatch.setattr(runner, "_git", original)
    assert cell.init()["status"] == "ready"


def test_report_summarizes_progress_and_failures(tmp_path):
    cell = initialized(tmp_path, Remote([success()]), rounds=1)
    cell.run()
    report = cell.report()
    assert report == {
        "cell": "a3-matmul-guarded", "status": "complete", "round": 2,
        "rounds_completed": 1, "attempts": 1,
        "outcomes": {"success": 1}, "branch": "experiment/trial/a3-matmul-guarded",
    }


def test_cpl_remote_adapter_uses_file_and_exact_handle(tmp_path, monkeypatch):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        handle = "remote:bz-a3-1:job:77"
        if "run" in command:
            payload = {"handle": handle, "target": "bz-a3-1", "state": "running"}
        elif "observe" in command:
            payload = {"handle": handle, "target": "bz-a3-1", "state": "completed", "exit": 0}
        else:
            content = json.dumps({"schema": runner.RESULT_SCHEMA, "outcome": "success"}) if "stdout" in command else ""
            payload = {"handle": handle, "target": "bz-a3-1", "content": content}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    request = {"target": "bz-a3-1", "runtime": "cann", "remote_workspace": "/work"}
    handle = adapter.submit(request, script)
    assert handle == "remote:bz-a3-1:job:77"
    assert calls[0] == ["/global/cpl-remote", "--json", "run", "bz-a3-1", "--runtime", "cann", "--file", str(script), "--cwd", "/work"]
    assert adapter.observe(handle)["outcome"] == "success"
    assert calls[1] == ["/global/cpl-remote", "--json", "observe", handle, "--wait"]


def test_cpl_remote_returns_remote_compile_diagnostics(tmp_path):
    def invoke(command, **kwargs):
        identity = {"handle": "remote:bz-a3-1:job:9", "target": "bz-a3-1"}
        if "observe" in command:
            value, code = {**identity, "state": "failed", "exit": 1}, 1
        elif "logs" in command and "stderr" in command:
            value, code = {**identity, "content": "triton compiler: invalid tl.load"}, 0
        else:
            value, code = {**identity, "content": ""}, 0
        return subprocess.CompletedProcess(command, code, json.dumps(value), "")

    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    result = adapter.observe("remote:bz-a3-1:job:9")
    assert result["outcome"] == "candidate_failure"
    assert "triton compiler: invalid tl.load" in result["diagnostics"]


@pytest.mark.parametrize(
    "state,exit_code,envelope",
    [
        ("failed", 1, {"schema": "simple-experiment-result/v1", "outcome": "success"}),
        ("cancelled", 0, {"schema": "simple-experiment-result/v1", "outcome": "success"}),
        ("completed", 1, {"schema": "simple-experiment-result/v1", "outcome": "success"}),
        ("completed", 0, None),
        ("missing", None, {"schema": "simple-experiment-result/v1", "outcome": "success"}),
    ],
)
def test_remote_success_envelope_fails_closed(state, exit_code, envelope):
    handle = "remote:bz-a3-1:job:9"

    def invoke(command, **kwargs):
        if "observe" in command:
            value, code = {"handle": handle, "target": "bz-a3-1", "state": state, "exit": exit_code}, int(state == "failed")
        else:
            content = json.dumps(envelope) if envelope and "stdout" in command else ""
            value, code = {"handle": handle, "target": "bz-a3-1", "content": content}, 0
        return subprocess.CompletedProcess(command, code, json.dumps(value), "")

    result = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote")).observe(handle)
    assert result["outcome"] != "success"


@pytest.mark.parametrize("state,exit_code", [("missing", None), ("failed", 125), ("failed", 200)])
def test_retained_or_setup_failure_without_envelope_is_infrastructure(state, exit_code):
    handle = "remote:bz-a3-1:job:infra"

    def invoke(command, **kwargs):
        if "observe" in command:
            value, code = {"handle": handle, "target": "bz-a3-1", "state": state, "exit": exit_code}, 1
        else:
            value, code = {"handle": handle, "target": "bz-a3-1", "content": "remote setup failed"}, 0
        return subprocess.CompletedProcess(command, code, json.dumps(value), "")

    result = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote")).observe(handle)
    assert result["outcome"] == "infrastructure_failure"
    assert "remote setup failed" in result["diagnostics"]


def test_explicit_failure_envelope_is_preserved_over_transport_exit():
    handle = "remote:bz-a3-1:job:explicit"
    envelope = {"schema": runner.RESULT_SCHEMA, "outcome": "candidate_failure", "diagnostics": "compile"}

    def invoke(command, **kwargs):
        if "observe" in command:
            value = {"handle": handle, "target": "bz-a3-1", "state": "failed", "exit": 125}
        else:
            content = json.dumps(envelope) if "stdout" in command else ""
            value = {"handle": handle, "target": "bz-a3-1", "content": content}
        return subprocess.CompletedProcess(command, 1 if "observe" in command else 0, json.dumps(value), "")

    result = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote")).observe(handle)
    assert result["outcome"] == "candidate_failure"
    assert result["diagnostics"] == "compile"


def test_logs_receipt_must_match_handle_and_target():
    handle = "remote:bz-a3-1:job:9"

    def invoke(command, **kwargs):
        if "observe" in command:
            value = {"handle": handle, "target": "bz-a3-1", "state": "completed", "exit": 0}
        else:
            value = {"handle": "remote:bz-a3-2:job:9", "target": "bz-a3-2", "content": ""}
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    with pytest.raises(runner.ObservationLost, match="logs.*identity"):
        runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote")).observe(handle)


def test_preflight_requires_run_observe_logs_and_records_receipts(tmp_path):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        if "capabilities" in command:
            value = {"target": "bz-a3-1", "capabilities": ["run", "observe", "logs"]}
        else:
            value = {"target": "bz-a3-1", "state": "completed"}
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    remote = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    result = remote.preflight({"target": "bz-a3-1"})
    assert result["capabilities"]["capabilities"] == ["run", "observe", "logs"]
    assert calls == [
        ["/global/cpl-remote", "--json", "capabilities", "bz-a3-1"],
        ["/global/cpl-remote", "--json", "preflight", "bz-a3-1"],
    ]


@pytest.mark.parametrize("phase", ["capabilities", "preflight"])
def test_cpl_remote_reports_failed_preflight_phase(phase):
    def invoke(command, **kwargs):
        if "capabilities" in command:
            capabilities = ["run", "observe"] if phase == "capabilities" else ["run", "observe", "logs"]
            value = {"target": "bz-a3-1", "state": "available", "capabilities": capabilities}
        else:
            value = {"target": "bz-a3-1", "state": "failed"}
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    remote = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    with pytest.raises(runner.PreflightFailure) as failure:
        remote.preflight({"target": "bz-a3-1"})
    assert failure.value.phase == phase


@pytest.mark.parametrize("phase", ["capabilities", "preflight"])
def test_preflight_failure_is_retained_infrastructure_without_agent_or_attempt(tmp_path, phase):
    class FailedPreflight(Remote):
        def preflight(self, request):
            raise runner.PreflightFailure(phase, f"{phase} unavailable")

    remote, agent = FailedPreflight([]), Agent()
    cell = initialized(tmp_path, remote, agent)
    state = cell.run()
    assert state["status"] == "infrastructure_failure" and state["round"] == 1
    assert state["attempts"] == 0 and not agent.requests
    evidence = json.loads((cell.path / "preflights/1.json").read_text())
    assert evidence == {
        "diagnostics": f"{phase} unavailable", "outcome": "infrastructure_failure",
        "phase": phase,
    }


def test_preflight_failure_can_retry_without_missing_attempt(tmp_path):
    class FlakyPreflight(Remote):
        def preflight(self, request):
            if not self.preflights:
                self.preflights.append(request)
                raise runner.PreflightFailure("preflight", "temporarily unavailable")
            return super().preflight(request)

    remote, agent = FlakyPreflight([success()]), Agent()
    cell = initialized(tmp_path, remote, agent)
    assert cell.run()["status"] == "infrastructure_failure"
    assert cell.run()["round"] == 2
    assert len(agent.requests) == len(remote.submitted) == 1


def test_cli_init_run_resume_report_end_to_end(tmp_path):
    skill = tmp_path / "cli-skill"
    skill.mkdir(); (skill / "SKILL.md").write_text("cli skill\n")
    fake_home = tmp_path / "fake-home"
    client = fake_home / ".agents/skills/remote-access/scripts/cpl-remote"
    client.parent.mkdir(parents=True)
    transport_state = tmp_path / "transport.json"
    client.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
        f"p=Path({str(transport_state)!r}); s=json.loads(p.read_text()) if p.exists() else {{'observe':0,'run':0}}\n"
        "a=sys.argv[2]; handle='remote:bz-a3-1:job:e2e'\n"
        "if a=='capabilities': out={'target':'bz-a3-1','state':'available','capabilities':{'run':True,'observe':True,'logs':True}}\n"
        "elif a=='preflight': out={'target':'bz-a3-1','state':'completed'}\n"
        "elif a=='run': s['run']+=1; out={'target':'bz-a3-1','state':'running','handle':handle}\n"
        "elif a=='observe':\n"
        " s['observe']+=1; out={'target':'bz-a3-1','handle':handle,'state':'observation-unavailable' if s['observe']==1 else 'completed','exit':None if s['observe']==1 else 0}\n"
        "elif a=='logs':\n"
        " content=json.dumps({'schema':'simple-experiment-result/v1','outcome':'success','profile_summary':{'duration_us':1.0}}) if 'stdout' in sys.argv else ''\n"
        " out={'target':'bz-a3-1','handle':handle,'state':'completed','content':content}\n"
        "p.write_text(json.dumps(s)); print(json.dumps(out))\n"
    )
    client.chmod(0o700)
    agent = tmp_path / "agent.py"
    agent.write_text(
        "import json,sys\nr=json.load(sys.stdin)\n"
        "print(json.dumps({'reasoning':'final','postmortem':'done'} if r['phase']=='analyze' else "
        "{'candidate':'VALUE = 1\\n','script_fragment':'true','commands':['true'],"
        "'hypothesis':'works','retrieved_knowledge':[],'session_id':'cli-session'}))\n"
    )
    cfg = config(tmp_path, rounds=1, skill_paths=[str(skill)],
                 agent_command=["python", str(agent)])
    config_path = tmp_path / "cell.json"
    config_path.write_text(json.dumps(cfg))
    runs = tmp_path / "cli-runs"
    script = Path(__file__).parents[1] / "scripts/simple_experiment_runner.py"
    environment = {**__import__("os").environ, "HOME": str(fake_home), "PYTHONDONTWRITEBYTECODE": "1"}

    def cli(command):
        result = subprocess.run(
            ["python", str(script), command, "--root", str(runs), "--config", str(config_path)],
            text=True, capture_output=True, env=environment, check=True,
        )
        return json.loads(result.stdout)

    assert cli("init")["status"] == "ready"
    assert cli("run")["status"] == "infrastructure_unknown"
    assert cli("resume")["status"] == "complete"
    assert cli("report")["rounds_completed"] == 1
    assert json.loads(transport_state.read_text()) == {"observe": 2, "run": 1}
