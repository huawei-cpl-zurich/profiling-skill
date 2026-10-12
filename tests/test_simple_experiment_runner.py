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
    }
    value.update(updates)
    return value


def evidence(label="v1"):
    return {
        "candidate": f"KERNEL = {label!r}\n",
        "remote_script": "#!/bin/bash\npython candidate.py\n",
        "commands": ["python candidate.py"],
        "reasoning": f"reasoning {label}",
        "postmortem": f"postmortem {label}",
        "retrieved_knowledge": ["ref://example"],
    }


class Agent:
    def __init__(self, values=None):
        self.values = list(values or [evidence()])
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return self.values.pop(0)


class Remote:
    def __init__(self, observations):
        self.observations = list(observations)
        self.submitted = []
        self.observed = []

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
    }
    assert expected <= names
    assert json.loads((cell.repo / "rounds/01/profile-summary.json").read_text())["duration_us"] == 9.25


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
        payload = ({"REMOTE_HANDLE": "remote:bz-a3-1:job:77", "REMOTE_STATE": "running"}
                   if "run" in command else {"REMOTE_HANDLE": "remote:bz-a3-1:job:77", "REMOTE_STATE": "completed", "REMOTE_EXIT": 0})
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
        if "observe" in command:
            value, code = {"handle": "remote:bz-a3-1:job:9", "state": "failed", "exit": 1}, 1
        elif "logs" in command and "stderr" in command:
            value, code = {"content": "triton compiler: invalid tl.load"}, 0
        else:
            value, code = {"content": ""}, 0
        return subprocess.CompletedProcess(command, code, json.dumps(value), "")

    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    result = adapter.observe("remote:bz-a3-1:job:9")
    assert result["outcome"] == "candidate_failure"
    assert result["diagnostics"] == "triton compiler: invalid tl.load"
