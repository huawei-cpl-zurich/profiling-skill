import importlib.util
import hashlib
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
    gate = tmp_path / "terminal-gate.sh"
    gate.write_text('python "$CANDIDATE_PATH"\n')
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
        "terminal_gate_path": str(gate),
        "billed_checks": 10,
        "prompt": "Implement the exact requested kernel.\n",
        "skill_paths": [str(skill)],
    }
    value.update(updates)
    return value


def campaign_gate_config(tmp_path, **updates):
    bundle = tmp_path / "a3-gate.tar"
    bundle.write_bytes(b"host bundle")
    manifest = tmp_path / "candidate.manifest.json"
    manifest.write_text(json.dumps({
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "streaming_matmul_add_kernel_mix_aic",
        "entrypoint": "streaming_matmul_add_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }))
    return config(
        tmp_path, terminal_gate_bundle_path=str(bundle),
        candidate_manifest_path=str(manifest), **updates,
    )


def evidence(label="v1"):
    return {
        "candidate": f"KERNEL = {label!r}\n",
        "commands": ["python candidate.py"],
        "hypothesis": f"hypothesis {label}",
        "session_id": "session-1",
        "retrieved_knowledge": ["ref://example"],
    }


class Agent:
    def __init__(self, values=None):
        self.values = list(values or [evidence()])
        self.requests = []
        self.tunes = []
        self.analyses = []

    def prepare(self, request):
        self.requests.append(request)
        return self.values.pop(0)

    def analyze(self, request):
        self.analyses.append(request)
        return {"reasoning": "final reasoning", "postmortem": "final postmortem"}

    def tune(self, request):
        self.tunes.append(request)
        return {"configurations": [
            {"id": "block-64", "parameters": {"BLOCK": 64},
             "hypothesis": "test a wider block"},
        ]}


class Remote:
    def __init__(self, observations):
        self.observations = list(observations)
        self.submitted = []
        self.observed = []
        self.preflights = []

    def preflight(self, request):
        self.preflights.append(request)
        return {"capabilities": {"run": True, "observe": True, "logs": True}, "state": "completed"}

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


def autotune_candidate():
    return "AUTOTUNE_CONFIG = {'BLOCK': 32, 'WAVES': 1}\nKERNEL = 'candidate'\n"


def autotune_success(plan, winner="block-64", failed=()):
    rows = []
    for index, option in enumerate(plan["options"]):
        if option["id"] in failed:
            rows.append({
                "id": option["id"], "source_sha256": option["source_sha256"],
                "outcome": "candidate_failure", "diagnostics": "compile failed",
            })
        else:
            rows.append({
                "id": option["id"], "source_sha256": option["source_sha256"],
                "outcome": "success", "cheap_device_timing_us": 10.0 + index,
            })
    finalists = [winner]
    selected = next(option for option in plan["options"] if option["id"] == winner)
    return {
        "outcome": "success", "stdout": "PASS", "stderr": "",
        "profile_summary": {
            "timing": {"profiled_ids": finalists},
            "pipe_activity": {"available": True},
            "timeline": {"available": False, "reason": "unsupported"},
        },
        "autotune_summary": {
            "schema": runner.AUTOTUNE_RESULT_SCHEMA,
            "optimization_round": 1,
            "options": rows, "finalists": finalists,
            "profiled_finalists": finalists, "winner_id": winner,
            "winner_source_sha256": selected["source_sha256"],
            "winner_full_gate": True,
            "commands": ["compile all", "time survivors", "msprof finalists"],
            "finalist_profiles": {
                identifier: {"geomean_us": 9.0, "repeats": 3}
                for identifier in finalists
            },
        },
    }


def candidate_contract():
    return {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "streaming_matmul_add_kernel_mix_aic",
        "entrypoint": "streaming_matmul_add_kernel",
        "fusion": {
            "schema_version": 1,
            "mode": "single-logical-launch",
            "complete_operator": True,
        },
    }


def test_init_creates_isolated_repo_and_branch(tmp_path):
    cell = initialized(tmp_path, Remote([]))
    assert cell.path == tmp_path / "runs" / "trial" / "a3-matmul-guarded"
    assert git(cell.repo, "branch", "--show-current") == "experiment/trial/a3-matmul-guarded"
    state = json.loads((cell.path / "state.json").read_text())
    assert state["round"] == 1 and state["status"] == "ready"
    assert not (cell.repo / "rounds").exists()
    copied = cell.agent_home / ".agents/skills/skill/SKILL.md"
    assert copied.read_text() == "test skill\n" and not copied.is_symlink()
    assert (cell.path / "terminal-gate.sh").read_text() == 'python "$CANDIDATE_PATH"\n'
    assert state["billed_checks_used"] == 0
    assert state["task_sha256"] is None and not (cell.repo / "TASK.md").exists()


def test_campaign_gate_init_is_device_neutral_and_inputs_are_snapshotted(tmp_path):
    cfg = campaign_gate_config(tmp_path)
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    state = cell.init()
    source_bundle = Path(cfg["terminal_gate_bundle_path"])
    source_manifest = Path(cfg["candidate_manifest_path"])
    assert "physical_device" not in state
    assert state["gate_bundle_sha256"] == hashlib.sha256(source_bundle.read_bytes()).hexdigest()
    assert state["candidate_manifest_sha256"] == hashlib.sha256(
        source_manifest.read_bytes()
    ).hexdigest()

    gate = (
        'test "$ASCEND_RT_VISIBLE_DEVICES" = 3\n'
        'test "$ASCEND_DEVICE_ID" = 0\n'
        'test "$EXPERIMENT_ROUND" = 4\n'
        'test "$(cat "$TERMINAL_GATE_BUNDLE_PATH")" = "host bundle"\n'
        'python - "$CANDIDATE_MANIFEST_PATH" <<\'PY\'\n'
        'import json,sys\n'
        'assert json.load(open(sys.argv[1]))["entrypoint"] == "streaming_matmul_add_kernel"\n'
        'PY\n'
        'python "$CANDIDATE_PATH"\n'
    )
    script = tmp_path / "materialized.sh"
    script.write_text(runner.ExperimentCell._script(
        'print("materialized")\n', gate, bundle=source_bundle.read_bytes(),
        manifest=source_manifest.read_bytes(), physical_device=3, round_number=4,
    ))
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True, check=True)
    assert result.stdout == "materialized\n"


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"candidate_manifest_path": None}, "require terminal_gate_bundle_path"),
        ({"terminal_gate_bundle_path": None}, "require terminal_gate_bundle_path"),
    ],
)
def test_campaign_gate_inputs_fail_closed(tmp_path, updates, message):
    cfg = campaign_gate_config(tmp_path)
    cfg.update(updates)
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    with pytest.raises(runner.RunnerError, match=message):
        cell.init()


@pytest.mark.parametrize("physical_device", [None, -1, True, "2"])
def test_campaign_run_requires_valid_launch_device_before_remote_work(
        tmp_path, physical_device):
    cfg = campaign_gate_config(tmp_path)
    remote, agent = Remote([]), Agent()
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, agent, remote)
    cell.init()
    if physical_device is not None:
        cell.config["physical_device"] = physical_device
    with pytest.raises(runner.RunnerError, match="physical_device.*run time"):
        cell.run()
    assert not remote.preflights and not remote.submitted and not agent.requests


def test_campaign_manifest_rejects_unrelated_selector(tmp_path):
    cfg = campaign_gate_config(tmp_path)
    path = Path(cfg["candidate_manifest_path"])
    value = json.loads(path.read_text())
    value["kernel_name"] = "unrelated_kernel"
    path.write_text(json.dumps(value))
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    with pytest.raises(runner.RunnerError, match="declared entrypoint"):
        cell.init()


@pytest.mark.parametrize("filename", ["terminal-gate-bundle.tar", "candidate-manifest.json"])
def test_campaign_gate_snapshot_drift_blocks_dispatch(tmp_path, filename):
    cfg = campaign_gate_config(tmp_path)
    remote = Remote([])
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), remote)
    cell.init()
    (cell.path / filename).chmod(0o600)
    (cell.path / filename).write_bytes(b"drift")
    with pytest.raises(runner.RunnerError, match="changed after initialization"):
        cell.run()
    assert not remote.preflights and not remote.submitted


def test_campaign_round_commit_retains_contract_and_bundle_digest(tmp_path):
    cfg = campaign_gate_config(tmp_path, rounds=1)
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([success()]))
    cell.init()
    cell.config["physical_device"] = 3
    assert cell.run()["status"] == "complete"
    evidence_dir = cell.repo / "rounds/01"
    assert json.loads((evidence_dir / "candidate-manifest.json").read_text())["schema"].endswith("/v2")
    assert (evidence_dir / "terminal-gate-bundle.sha256").read_text().strip() == (
        cell._state()["gate_bundle_sha256"]
    )
    assert json.loads((evidence_dir / "inputs.json").read_text())["physical_device"] == 3


def test_init_copies_hash_bound_canonical_skill_alias_and_private_auth(tmp_path):
    source = tmp_path / "frozen-profiler-new"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: ascend-profiling\n---\n")
    digest = runner._tree_sha256(source)
    auth = tmp_path / "auth.json"
    auth.write_text('{"token":"private"}\n')
    auth.chmod(0o600)
    cfg = config(
        tmp_path,
        skill_paths=None,
        skill_bindings=[{"name": "ascend-profiling", "path": str(source),
                         "sha256": digest}],
        agent_auth_path=str(auth),
    )
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    cell.init()

    copied = cell.agent_home / ".agents/skills/ascend-profiling/SKILL.md"
    assert copied.read_text() == (source / "SKILL.md").read_text()
    assert cell.skill_names == ["ascend-profiling"]
    private_auth = cell.agent_home / ".codex/auth.json"
    assert private_auth.read_bytes() == auth.read_bytes()
    assert private_auth.stat().st_mode & 0o077 == 0
    assert not (cell.agent_home / ".codex/config.toml").exists()
    assert not (cell.repo / ".codex").exists()
    for child in source.iterdir():
        child.unlink()
    source.rmdir()
    assert cell.skill_names == ["ascend-profiling"]


def test_init_rejects_skill_alias_that_disagrees_with_declared_name(tmp_path):
    source = tmp_path / "skill-source"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: wrong-skill\n---\n")
    cfg = config(tmp_path, skill_paths=None, skill_bindings=[{
        "name": "ascend-profiling", "path": str(source),
        "sha256": runner._tree_sha256(source),
    }])
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    with pytest.raises(runner.RunnerError, match="name mismatch"):
        cell.init()
    assert not cell.path.exists()


def test_subprocess_agent_does_not_inherit_global_skill_or_auth_environment(tmp_path, monkeypatch):
    program = tmp_path / "agent.py"
    program.write_text(
        "import json,os,sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'candidate':'x','commands':[],'hypothesis':"
        "json.dumps(sorted(k for k in os.environ if 'SKILL' in k or 'TOKEN' in k))}))\n"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    monkeypatch.setenv("GLOBAL_SKILL_PATH", "/global/skills")
    monkeypatch.setenv("SECRET_TOKEN", "secret")
    agent = runner.SubprocessAgent(["python", str(program)], repo, home)
    value = agent.prepare({"phase": "prepare"})
    assert value["hypothesis"] == "[]"


def test_init_snapshots_optional_task_and_records_it_with_round_evidence(tmp_path):
    task = tmp_path / "benchmark-task.md"
    exact_task = b"\xef\xbb\xbfOptimize the supplied benchmark.\r\n"
    task.write_bytes(exact_task)
    remote = Remote([success()])
    cell = initialized(tmp_path, remote, task_path=str(task), rounds=1)
    expected = hashlib.sha256(task.read_bytes()).hexdigest()

    task.write_text("source changed after init\n")
    assert (cell.repo / "TASK.md").read_bytes() == exact_task
    assert cell._state()["task_sha256"] == expected
    assert cell.run()["status"] == "complete"
    inputs = json.loads((cell.repo / "rounds/01/inputs.json").read_text())
    assert inputs["task_sha256"] == expected
    retained = subprocess.run(
        ["git", "-C", str(cell.repo), "show", "HEAD:rounds/01/TASK.md"],
        capture_output=True, check=True,
    ).stdout
    assert retained == exact_task


@pytest.mark.parametrize("operation", ["run", "resume"])
def test_task_snapshot_drift_blocks_run_or_resume(tmp_path, operation):
    task = tmp_path / "benchmark-task.md"
    task.write_text("immutable task\n")
    remote = Remote([runner.ObservationLost("observer disconnected")])
    cell = initialized(tmp_path, remote, task_path=str(task), rounds=1)
    if operation == "resume":
        assert cell.run()["status"] == "infrastructure_unknown"
    (cell.repo / "TASK.md").write_text("agent changed task\n")

    with pytest.raises(runner.RunnerError, match="TASK.md changed"):
        getattr(cell, operation)()

    if operation == "run":
        assert not remote.preflights and not remote.submitted


@pytest.mark.parametrize("phase", ["prepare", "analyze"])
def test_agent_cannot_change_task_during_a_phase(tmp_path, phase):
    task = tmp_path / "benchmark-task.md"
    task.write_text("immutable task\n")

    class MutatingAgent(Agent):
        def prepare(self, request):
            value = super().prepare(request)
            if phase == "prepare":
                (Path(request["repo"]) / "TASK.md").write_text("prepare mutation\n")
            return value

        def analyze(self, request):
            value = super().analyze(request)
            if phase == "analyze":
                (Path(request["repo"]) / "TASK.md").write_text("analyze mutation\n")
            return value

    remote = Remote([success()])
    cell = initialized(
        tmp_path, remote, MutatingAgent(), task_path=str(task), rounds=1
    )
    with pytest.raises(runner.RunnerError, match="TASK.md changed"):
        cell.run()

    if phase == "prepare":
        assert not remote.submitted and cell._state()["billed_checks_used"] == 0
    else:
        assert len(remote.submitted) == 1 and cell._state()["status"] == "analyzing"


@pytest.mark.parametrize("task_path", ["", "missing.md"])
def test_init_rejects_invalid_configured_task_path(tmp_path, task_path):
    cfg = config(tmp_path, task_path=task_path)
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    with pytest.raises(runner.RunnerError, match="task_path"):
        cell.init()
    assert not cell.path.exists()


def test_terminal_gate_is_host_owned_and_snapshotted_at_init(tmp_path):
    cfg = config(tmp_path, rounds=1)
    gate = Path(cfg["terminal_gate_path"])
    gate.write_text("echo HOST_GATE\n")
    remote = Remote([success()])
    prepared = evidence()
    prepared["script_fragment"] = "echo AGENT_FRAGMENT"
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent([prepared]), remote)
    cell.init()
    gate.write_text("echo DRIFTED_GATE\n")
    cell.run()
    submitted = remote.submitted[0][1]
    assert "HOST_GATE" in submitted
    assert "DRIFTED_GATE" not in submitted and "AGENT_FRAGMENT" not in submitted
    assert submitted.endswith("echo HOST_GATE\n")
    assert (cell.repo / "rounds/01/terminal-gate.sh").read_text() == "echo HOST_GATE\n"


@pytest.mark.parametrize("billed_checks", [None, True, 0, -1, 1.5])
def test_init_requires_positive_integer_billed_checks(tmp_path, billed_checks):
    cfg = config(tmp_path, billed_checks=billed_checks)
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    with pytest.raises(runner.RunnerError, match="positive integer"):
        cell.init()
    assert not cell.path.exists()


def test_init_requires_existing_terminal_gate(tmp_path):
    cfg = config(tmp_path, terminal_gate_path=str(tmp_path / "missing-gate.sh"))
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent(), Remote([]))
    with pytest.raises(runner.RunnerError, match="terminal_gate_path"):
        cell.init()
    assert not cell.path.exists()


def test_host_gate_result_is_after_candidate_stdout(tmp_path):
    cfg = config(tmp_path)
    Path(cfg["terminal_gate_path"]).write_text(
        'python "$CANDIDATE_PATH"\n'
        "printf '%s\\n' '{\"schema\":\"simple-experiment-result/v1\",\"outcome\":\"candidate_failure\"}'\n"
    )
    prepared = evidence()
    prepared["candidate"] = (
        "print('{\"schema\":\"simple-experiment-result/v1\",\"outcome\":\"success\"}')\n"
    )
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, Agent([prepared]), Remote([]))
    cell.init()
    wrapper = cell._script(prepared["candidate"], (cell.path / "terminal-gate.sh").read_text())
    script = tmp_path / "wrapper.sh"
    script.write_text(wrapper)
    output = subprocess.run(["bash", str(script)], text=True, capture_output=True, check=True).stdout
    assert json.loads(output.rstrip().splitlines()[-1])["outcome"] == "candidate_failure"


def test_prepare_does_not_require_or_return_execution_fragment(tmp_path):
    program = tmp_path / "agent.py"
    program.write_text(
        "import json\n"
        "print(json.dumps({'candidate':'x','commands':[],'hypothesis':'host gate runs it'}))\n"
    )
    value = runner.SubprocessAgent(["python", str(program)], tmp_path, tmp_path).prepare({})
    assert "script_fragment" not in value


def test_run_and_resume_reuse_exact_retained_handle(tmp_path):
    handle = "remote:bz-a3-1:job:1"
    remote = Remote([runner.ObservationLost("connection dropped"), success()])
    cell = initialized(tmp_path, remote)
    interrupted = cell.run()
    assert interrupted["status"] == "infrastructure_unknown"
    assert interrupted["billed_checks_used"] == 1
    assert (cell.path / "attempts/1/remote-job-id.txt").read_text().strip() == handle
    resumed = cell.resume()
    assert resumed["status"] == "ready" and resumed["billed_checks_used"] == 1
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
    gate = (cell.path / "terminal-gate.sh").read_text()
    (attempt / "run.sh").write_text(cell._script(evidence()["candidate"], gate))
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


def test_one_billed_check_never_submits_candidate_repair(tmp_path):
    remote = Remote([{"outcome": "candidate_failure", "diagnostics": "compile"}])
    agent = Agent([evidence("bad"), evidence("unused")])
    cell = initialized(tmp_path, remote, agent, billed_checks=1)
    state = cell.run()
    assert state["status"] == "billed_checks_exhausted"
    assert state["billed_checks_used"] == 1 and state["last_outcome"] == "candidate_failure"
    assert len(remote.submitted) == len(agent.requests) == 1


def test_billed_check_limit_is_snapshotted_at_init(tmp_path):
    remote = Remote([{"outcome": "candidate_failure", "diagnostics": "compile"}])
    agent = Agent([evidence("bad"), evidence("unused")])
    cell = initialized(tmp_path, remote, agent, billed_checks=1)
    cell.config["billed_checks"] = 2
    state = cell.run()
    assert state["status"] == "billed_checks_exhausted"
    assert state["billed_checks_limit"] == state["billed_checks_used"] == 1
    assert len(remote.submitted) == 1


def test_one_billed_check_never_submits_infrastructure_retry(tmp_path):
    remote = Remote([{"outcome": "infrastructure_failure", "diagnostics": "device"}])
    cell = initialized(tmp_path, remote, billed_checks=1)
    state = cell.run()
    assert state["status"] == "billed_checks_exhausted"
    assert state["billed_checks_used"] == 1 and state["last_outcome"] == "infrastructure_failure"
    assert len(remote.submitted) == 1


def test_uncertain_dispatch_consumes_check_before_transport_returns(tmp_path):
    class LostDispatch(Remote):
        def submit(self, request, script):
            self.submitted.append((request, script.read_text()))
            raise runner.ObservationLost("receipt lost")

    cell = initialized(tmp_path, LostDispatch([]), billed_checks=1)
    state = cell.run()
    assert state["status"] == "infrastructure_unknown"
    assert state["billed_checks_used"] == state["billed_checks_limit"] == 1
    with pytest.raises(runner.RunnerError, match="no handle"):
        cell.resume()


def test_proven_pre_dispatch_failure_preserves_candidate_without_billing(tmp_path):
    class FailedUpload(Remote):
        def submit(self, request, script):
            self.submitted.append((request, script.read_text()))
            if len(self.submitted) == 1:
                raise runner.PreDispatchFailure("upload unavailable")
            return f"remote:{request['target']}:job:2"

    remote, agent = FailedUpload([success()]), Agent()
    cell = initialized(tmp_path, remote, agent, rounds=1)
    state = cell.run()
    assert state["status"] == "complete"
    assert state["attempts"] == state["billed_checks_used"] == 1
    assert len(agent.requests) == 1 and len(remote.submitted) == 2
    archived = cell.path / "pre-dispatch/1"
    assert json.loads((archived / "infrastructure.json").read_text())["dispatch_state"] == "not_dispatched"
    assert json.loads((archived / "agent-result.json").read_text()) == evidence()


def test_initial_and_repair_requests_include_exact_candidate_contract(tmp_path):
    failed = {"outcome": "candidate_failure", "diagnostics": "compile failed"}
    agent = Agent([evidence("initial"), evidence("repair")])
    cell = runner.ExperimentCell(
        tmp_path / "runs", campaign_gate_config(tmp_path, rounds=1),
        agent, Remote([failed, success()]),
    )
    cell.init()
    cell.config["physical_device"] = 0
    contract = candidate_contract()

    completed = cell.run()

    assert completed["status"] == "complete"
    assert len(agent.requests) == 2
    assert [request["candidate_contract"] for request in agent.requests] == [
        contract, contract,
    ]
    assert [request["prompt"] for request in agent.requests] == [
        "Implement the exact requested kernel.\n",
        "Implement the exact requested kernel.\n",
    ]
    assert agent.requests[1]["prior_result"] == failed


@pytest.mark.parametrize("workspace", ["", "bad\npath", None])
def test_invalid_workspace_is_recoverable_without_billing(tmp_path, workspace):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        if command[2] == "capabilities":
            value = {"target": "bz-a3-1", "capabilities": {
                "run": True, "observe": True, "logs": True, "upload": True,
            }}
        elif command[2] == "preflight":
            value = {"target": "bz-a3-1", "state": "completed"}
        else:
            raise AssertionError("invalid workspace must fail before upload or dispatch")
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    cfg = config(tmp_path, remote_workspace=workspace, infrastructure_retries=0)
    agent = Agent()
    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    cell = runner.ExperimentCell(tmp_path / "runs", cfg, agent, adapter)
    cell.init()

    state = cell.run()

    assert state["status"] == "infrastructure_failure"
    assert state["attempts"] == state["billed_checks_used"] == 0
    assert state["infrastructure_failures"] == 1
    assert "pending_attempt" not in state
    assert [command[2] for command in calls] == ["capabilities", "preflight"]
    archived = cell.path / "pre-dispatch/1"
    assert json.loads((archived / "agent-result.json").read_text()) == evidence()
    assert json.loads((archived / "infrastructure.json").read_text()) == {
        "diagnostics": "remote_workspace must be a safe non-empty path",
        "dispatch_state": "not_dispatched",
        "phase": "stage",
    }

    corrected = runner.ExperimentCell(
        tmp_path / "runs", {
            **cfg, "remote_workspace": "/work", "infrastructure_retries": 2,
        }, agent,
        Remote([success()]),
    )
    completed = corrected.run()
    assert completed["status"] == "ready" and completed["round"] == 2
    assert len(agent.requests) == 1
    assert json.loads(
        (corrected.path / "attempts/1/agent-result.json").read_text()
    ) == evidence()


def test_recover_legacy_e2big_checkpoint_without_rebuilding_candidate(tmp_path):
    remote, agent = Remote([success()]), Agent()
    cell = initialized(tmp_path, remote, agent)
    attempt = cell.path / "attempts/1"
    attempt.mkdir()
    _prepared = evidence("retained")
    (attempt / "agent-result.json").write_text(json.dumps(_prepared))
    (attempt / "infrastructure.json").write_text(json.dumps({
        "phase": "dispatch", "diagnostics": "error: [Errno 7] Argument list too long: 'ssh'",
    }))
    state = cell._state()
    state.update(status="infrastructure_unknown", pending_attempt=1, attempts=1,
                 billed_checks_used=1)
    cell._save(state)

    repaired = cell.recover()
    assert repaired["status"] == "infrastructure_failure"
    assert repaired["attempts"] == repaired["billed_checks_used"] == 0
    assert not (cell.path / "attempts/1").exists()
    assert (cell.path / "pre-dispatch/1/agent-result.json").is_file()
    completed = cell.run()
    assert completed["status"] == "ready" and completed["round"] == 2
    assert not agent.requests
    assert json.loads((cell.path / "attempts/1/agent-result.json").read_text()) == _prepared


def test_recover_refuses_unknown_dispatch_without_proof(tmp_path):
    cell = initialized(tmp_path, Remote([]))
    attempt = cell.path / "attempts/1"
    attempt.mkdir()
    (attempt / "infrastructure.json").write_text(json.dumps({
        "phase": "dispatch", "diagnostics": "observer disconnected",
    }))
    state = cell._state()
    state.update(status="infrastructure_unknown", pending_attempt=1, attempts=1,
                 billed_checks_used=1)
    cell._save(state)
    with pytest.raises(runner.RunnerError, match="not proven"):
        cell.recover()


def test_larger_billed_budget_preserves_candidate_repair(tmp_path):
    failed = {"outcome": "candidate_failure", "diagnostics": "compile"}
    cell = initialized(
        tmp_path, Remote([failed, success()]), Agent([evidence("bad"), evidence("fixed")]),
        billed_checks=2, rounds=1,
    )
    state = cell.run()
    assert state["status"] == "complete" and state["billed_checks_used"] == 2


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
        "rounds/01/run.sh", "rounds/01/terminal-gate.sh",
        "rounds/01/inputs.json", "rounds/01/hypothesis.md",
    }
    assert expected <= names
    assert json.loads((cell.repo / "rounds/01/profile-summary.json").read_text())["duration_us"] == 9.25


def test_materialize_autotune_is_deterministic_and_supports_128_agent_options():
    proposed = [
        {"id": f"option-{index:03d}", "parameters": {"BLOCK": index + 1},
         "hypothesis": f"test block {index + 1}"}
        for index in range(128)
    ]
    first = runner.materialize_autotune(autotune_candidate(), proposed)
    second = runner.materialize_autotune(autotune_candidate(), proposed)

    assert first == second
    assert len(first["options"]) == 129
    assert first["options"][0]["id"] == "incumbent"
    assert first["options"][-1]["parameters"] == {"BLOCK": 128, "WAVES": 1}
    assert first["options"][-1]["source_sha256"] == hashlib.sha256(
        first["options"][-1]["source"].encode()
    ).hexdigest()


@pytest.mark.parametrize(
    "candidate,proposed,message",
    [
        ("VALUE = 1\n", [], "exactly one"),
        (autotune_candidate(), [
            {"id": "bad", "parameters": {"UNDECLARED": 1}, "hypothesis": "bad"}
        ], "undeclared"),
        (autotune_candidate(), [
            {"id": "bad", "parameters": {"BLOCK": [1]}, "hypothesis": "bad"}
        ], "JSON scalar"),
    ],
)
def test_materialize_autotune_rejects_invalid_spaces(candidate, proposed, message):
    with pytest.raises(runner.RunnerError, match=message):
        runner.materialize_autotune(candidate, proposed)


def test_autotune_round_commits_selected_source_and_complete_search_evidence(tmp_path):
    prepared = evidence("structural")
    prepared["candidate"] = autotune_candidate()
    class HundredOptionAgent(Agent):
        def tune(self, request):
            self.tunes.append(request)
            return {"configurations": [
                {"id": f"option-{index:03d}",
                 "parameters": {"BLOCK": 64 if index == 0 else index + 65},
                 "hypothesis": "broad deterministic canary"}
                for index in range(100)
            ]}

    agent = HundredOptionAgent([prepared])

    class TuningRemote(Remote):
        def observe(self, handle):
            self.observed.append(handle)
            request_path = Path(self.submitted[0][1])
            del request_path
            attempt = cell.path / "attempts/1/autotune-request.json"
            return autotune_success(json.loads(attempt.read_text()), winner="option-000")

    remote = TuningRemote([])
    cell = initialized(tmp_path, remote, agent, rounds=1, autotune=True)
    completed = cell.run()

    assert completed["status"] == "complete"
    assert len(agent.tunes) == 1
    assert agent.tunes[0]["declared_autotune_config"] == {"BLOCK": 32, "WAVES": 1}
    assert agent.tunes[0]["max_configurations"] == 128
    folder = cell.repo / "rounds/01"
    assert "'BLOCK': 64" not in (folder / "candidate.py").read_text()
    assert '"BLOCK": 64' in (folder / "candidate.py").read_text()
    assert (folder / "structural-candidate.py").read_text() == autotune_candidate()
    retained = json.loads((folder / "autotune-request.json").read_text())
    assert len(retained["options"]) == 101
    assert retained["optimization_round"] == 1
    assert all("source" not in option for option in retained["options"])
    result = json.loads((folder / "autotune-result.json").read_text())
    assert result["winner_id"] == "option-000" and result["optimization_round"] == 1
    assert completed["round"] == 2 and completed["attempts"] == 1
    assert (folder / "autotune-commands.jsonl").read_text().count("command") == 3
    assert "AUTOTUNE_REQUEST_PATH" in remote.submitted[0][1]
    assert agent.analyses[0]["candidate"] == (folder / "candidate.py").read_text()
    assert agent.analyses[0]["structural_candidate"] == autotune_candidate()


def test_autotune_observation_resume_reuses_one_job_and_one_tune_phase(tmp_path):
    prepared = evidence("structural")
    prepared["candidate"] = autotune_candidate()
    agent = Agent([prepared])

    class ResumeRemote(Remote):
        def observe(self, handle):
            self.observed.append(handle)
            if len(self.observed) == 1:
                raise runner.ObservationLost("observer disconnected")
            plan = json.loads((cell.path / "attempts/1/autotune-request.json").read_text())
            return autotune_success(plan)

    remote = ResumeRemote([])
    cell = initialized(tmp_path, remote, agent, rounds=1, autotune=True)
    assert cell.run()["status"] == "infrastructure_unknown"
    assert cell.resume()["status"] == "complete"
    assert len(remote.submitted) == len(agent.tunes) == 1
    assert remote.observed == ["remote:bz-a3-1:job:1"] * 2


def test_autotune_success_rejects_more_than_three_profiled_finalists(tmp_path):
    prepared = evidence("structural")
    prepared["candidate"] = autotune_candidate()

    class FourAgent(Agent):
        def tune(self, request):
            self.tunes.append(request)
            return {"configurations": [
                {"id": f"v{index}", "parameters": {"BLOCK": index + 33},
                 "hypothesis": "variant"} for index in range(4)
            ]}

    class InvalidRemote(Remote):
        def observe(self, handle):
            plan = json.loads((cell.path / "attempts/1/autotune-request.json").read_text())
            value = autotune_success(plan, winner="v0")
            finalists = [option["id"] for option in plan["options"][:4]]
            value["autotune_summary"]["finalists"] = finalists
            value["autotune_summary"]["profiled_finalists"] = finalists
            return value

    cell = initialized(
        tmp_path, InvalidRemote([]), FourAgent([prepared]), rounds=1, autotune=True
    )
    with pytest.raises(runner.RunnerError, match="one to three"):
        cell.run()


def test_autotune_success_rejects_a_result_from_another_round(tmp_path):
    prepared = evidence("structural")
    prepared["candidate"] = autotune_candidate()

    class WrongRoundRemote(Remote):
        def observe(self, handle):
            plan = json.loads((cell.path / "attempts/1/autotune-request.json").read_text())
            value = autotune_success(plan)
            value["autotune_summary"]["optimization_round"] = 2
            return value

    cell = initialized(tmp_path, WrongRoundRemote([]), Agent([prepared]),
                       rounds=1, autotune=True)
    with pytest.raises(runner.RunnerError, match="optimization round"):
        cell.run()


def test_runner_materializes_exact_candidate_and_submitted_script(tmp_path):
    remote, agent = Remote([success()]), Agent([evidence("exact")])
    cell = initialized(tmp_path, remote, agent)
    cell.run()
    submitted = remote.submitted[0][1]
    assert "CANDIDATE_PATH" in submitted
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
    home.mkdir()
    repo.mkdir()
    program = tmp_path / "agent.py"
    program.write_text(
        "import json,os,sys\n"
        "r=json.load(sys.stdin); print(json.dumps({'candidate':'x',"
        "'commands':[],'hypothesis':os.getcwd(),'retrieved_knowledge':[],"
        "'session_id':os.environ['HOME']+'|'+os.environ['CODEX_HOME']}))\n"
    )
    agent = runner.SubprocessAgent(["python", str(program)], repo, home)
    value = agent.prepare({"phase": "prepare", "prompt": "exact", "allowed_skills": ["one"]})
    assert value["hypothesis"] == str(repo)
    assert value["session_id"] == f"{home}|{home / '.codex'}"


def test_prepare_requires_hypothesis_not_reasoning(tmp_path):
    program = tmp_path / "agent.py"
    program.write_text(
        "import json\n"
        "print(json.dumps({'candidate':'x','commands':[],"
        "'reasoning':'not a hypothesis'}))\n"
    )
    agent = runner.SubprocessAgent(["python", str(program)], tmp_path, tmp_path)
    with pytest.raises(runner.RunnerError, match="incomplete"):
        agent.prepare({"phase": "prepare"})


def test_prepare_rejects_empty_hypothesis(tmp_path):
    program = tmp_path / "agent.py"
    program.write_text(
        "import json\n"
        "print(json.dumps({'candidate':'x','commands':[],"
        "'hypothesis':''}))\n"
    )
    agent = runner.SubprocessAgent(["python", str(program)], tmp_path, tmp_path)
    with pytest.raises(runner.RunnerError, match="incomplete"):
        agent.prepare({"phase": "prepare"})


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
        "'commands':['true'],'hypothesis':'repair',"
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
        "billed_checks_limit": 10, "billed_checks_used": 1,
        "outcomes": {"success": 1}, "branch": "experiment/trial/a3-matmul-guarded",
    }


def test_cpl_remote_adapter_uses_file_and_exact_handle(tmp_path, monkeypatch):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        handle = "remote:bz-a3-1:job:77"
        if "upload" in command:
            payload = {"target": "bz-a3-1", "state": "completed", "exit": 0}
        elif "run" in command:
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
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    assert calls[0] == ["/global/cpl-remote", "--json", "upload", "bz-a3-1", str(script),
                        f"/work/.simple-experiment-{digest}.sh"]
    launcher = script.with_name("dispatch.sh")
    assert calls[1] == ["/global/cpl-remote", "--json", "run", "bz-a3-1", "--runtime", "cann",
                        "--file", str(launcher), "--cwd", "/work"]
    assert script.read_text() not in launcher.read_text()
    assert digest in launcher.read_text()
    failure = subprocess.run(["bash", str(launcher)], text=True, capture_output=True)
    assert failure.returncode == 125
    assert json.loads(failure.stdout.strip())["outcome"] == "infrastructure_failure"
    assert adapter.observe(handle)["outcome"] == "success"
    assert calls[2] == ["/global/cpl-remote", "--json", "observe", handle, "--wait"]


def test_uppercase_remote_receipt_is_not_accepted(tmp_path):
    def invoke(command, **kwargs):
        value = ({"target": "bz-a3-1", "state": "completed", "exit": 0}
                 if "upload" in command else
                 {"REMOTE_HANDLE": "remote:bz-a3-1:job:77", "REMOTE_STATE": "running"})
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    script = tmp_path / "run.sh"
    script.write_text("true\n")
    with pytest.raises(runner.ObservationLost, match="durable handle"):
        adapter.submit({"target": "bz-a3-1", "runtime": "cann", "remote_workspace": "/work"}, script)


@pytest.mark.parametrize("receipt_target", [None, "bz-a3-2"])
def test_submit_receipt_requires_explicit_matching_target(tmp_path, receipt_target):
    def invoke(command, **kwargs):
        if "upload" in command:
            value = {"target": "bz-a3-1", "state": "completed", "exit": 0}
        else:
            value = {"handle": "remote:bz-a3-1:job:77", "state": "running"}
            if receipt_target is not None:
                value["target"] = receipt_target
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    script = tmp_path / "run.sh"
    script.write_text("true\n")
    with pytest.raises(runner.ObservationLost, match="matching durable handle"):
        adapter.submit({"target": "bz-a3-1", "runtime": "cann", "remote_workspace": "/work"}, script)


def test_upload_failure_is_proven_pre_dispatch(tmp_path):
    def invoke(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, json.dumps({"target": "bz-a3-1", "state": "failed", "exit": 1}), ""
        )

    script = tmp_path / "run.sh"
    script.write_text("true\n")
    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    with pytest.raises(runner.PreDispatchFailure, match="not confirmed"):
        adapter.submit(
            {"target": "bz-a3-1", "runtime": "cann", "remote_workspace": "/work"}, script
        )


def test_remote_agent_upload_accepts_relative_workspace_and_absent_exit(tmp_path):
    calls = []
    remote_root = tmp_path / "remote"
    (remote_root / "campaign/cell").mkdir(parents=True)

    def invoke(command, **kwargs):
        calls.append(command)
        if "upload" in command:
            (remote_root / command[-1]).write_bytes(Path(command[-2]).read_bytes())
        value = ({"target": "gz-a3", "backend": "remote-agent", "state": "completed"}
                 if "upload" in command else
                 {"target": "gz-a3", "handle": "remote:gz-a3:job:8", "state": "running"})
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    script = tmp_path / "run.sh"
    script.write_text("true\n")
    adapter = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    handle = adapter.submit({
        "target": "gz-a3", "runtime": "py311-torch",
        "remote_workspace": "campaign/cell",
    }, script)
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    assert handle == "remote:gz-a3:job:8"
    assert calls[0][-1] == f"campaign/cell/.simple-experiment-{digest}.sh"
    assert calls[1][-2:] == ["--cwd", "campaign/cell"]
    launcher = script.with_name("dispatch.sh")
    assert subprocess.run(
        ["bash", str(launcher)], cwd=remote_root / "campaign/cell", check=False,
    ).returncode == 0


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


def test_observation_receipt_requires_explicit_target():
    handle = "remote:bz-a3-1:job:9"

    def invoke(command, **kwargs):
        value = {"handle": handle, "state": "completed", "exit": 0}
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    with pytest.raises(runner.ObservationLost, match="observation.*identity"):
        runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote")).observe(handle)


def test_preflight_requires_run_observe_logs_and_records_receipts(tmp_path):
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        if "capabilities" in command:
            value = {"target": "bz-a3-1", "capabilities": {
                "run": True, "observe": True, "logs": True, "upload": True,
            }}
        else:
            value = {"target": "bz-a3-1", "state": "completed"}
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    remote = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    result = remote.preflight({"target": "bz-a3-1"})
    assert result["capabilities"]["capabilities"] == {
        "run": True, "observe": True, "logs": True, "upload": True,
    }
    assert calls == [
        ["/global/cpl-remote", "--json", "capabilities", "bz-a3-1"],
        ["/global/cpl-remote", "--json", "preflight", "bz-a3-1"],
    ]


@pytest.mark.parametrize("phase", ["capabilities", "preflight"])
def test_cpl_remote_reports_failed_preflight_phase(phase):
    def invoke(command, **kwargs):
        if "capabilities" in command:
            capabilities = ({"run": True, "observe": True}
                            if phase == "capabilities"
                            else {"run": True, "observe": True, "logs": True, "upload": True})
            value = {"target": "bz-a3-1", "state": "available", "capabilities": capabilities}
        else:
            value = {"target": "bz-a3-1", "state": "failed"}
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    remote = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    with pytest.raises(runner.PreflightFailure) as failure:
        remote.preflight({"target": "bz-a3-1"})
    assert failure.value.phase == phase


def test_preflight_rejects_alternate_status_vocabulary():
    def invoke(command, **kwargs):
        value = ({"target": "bz-a3-1", "capabilities": {
                    "run": True, "observe": True, "logs": True, "upload": True,
                 }}
                 if "capabilities" in command else {"target": "bz-a3-1", "status": "ok"})
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    remote = runner.CplRemote(invoke=invoke, executable=Path("/global/cpl-remote"))
    with pytest.raises(runner.PreflightFailure) as failure:
        remote.preflight({"target": "bz-a3-1"})
    assert failure.value.phase == "preflight"


@pytest.mark.parametrize("phase", ["capabilities", "preflight"])
def test_preflight_receipts_require_explicit_target(phase):
    def invoke(command, **kwargs):
        if "capabilities" in command:
            value = {"capabilities": {
                "run": True, "observe": True, "logs": True, "upload": True,
            }}
            if phase != "capabilities":
                value["target"] = "bz-a3-1"
        else:
            value = {"state": "completed"}
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
    assert state["billed_checks_used"] == 0
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
    skill.mkdir()
    (skill / "SKILL.md").write_text("cli skill\n")
    fake_home = tmp_path / "fake-home"
    client = fake_home / ".agents/skills/remote-access/scripts/cpl-remote"
    client.parent.mkdir(parents=True)
    transport_state = tmp_path / "transport.json"
    client.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
        f"p=Path({str(transport_state)!r}); s=json.loads(p.read_text()) if p.exists() else {{'observe':0,'run':0,'upload':0}}\n"
        "a=sys.argv[2]; handle='remote:bz-a3-1:job:e2e'\n"
        "if a=='capabilities': out={'target':'bz-a3-1','state':'available','capabilities':{'run':True,'observe':True,'logs':True,'upload':True}}\n"
        "elif a=='preflight': out={'target':'bz-a3-1','state':'completed'}\n"
        "elif a=='upload': s['upload']+=1; out={'target':'bz-a3-1','state':'completed','exit':0}\n"
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
        "{'candidate':'VALUE = 1\\n','commands':['true'],"
        "'hypothesis':'works','retrieved_knowledge':[],'session_id':'cli-session'}))\n"
    )
    task = tmp_path / "cli-task.md"
    task.write_text("CLI task snapshot\n")
    cfg = config(tmp_path, rounds=1, skill_paths=[str(skill)], task_path=str(task),
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
    task.write_text("source task drift does not alter snapshot\n")
    cell_repo = runs / "trial/a3-matmul-guarded/repo"
    assert (cell_repo / "TASK.md").read_text() == "CLI task snapshot\n"
    assert cli("run")["status"] == "infrastructure_unknown"
    assert cli("resume")["status"] == "complete"
    retained_task = subprocess.run(
        ["git", "-C", str(cell_repo), "show", "HEAD:rounds/01/TASK.md"],
        capture_output=True, check=True,
    ).stdout
    assert retained_task == b"CLI task snapshot\n"
    assert cli("report")["rounds_completed"] == 1
    assert json.loads(transport_state.read_text()) == {"observe": 2, "run": 1, "upload": 1}
