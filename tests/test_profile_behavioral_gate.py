from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_behavioral_gate", ROOT / "scripts/profile_behavioral_gate.py"
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(gate)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Interpret the supplied evidence.\n")
    skills = {}
    for arm in ("current", "candidate"):
        skill = tmp_path / f"{arm}-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"# {arm}\n")
        skills[arm] = {"path": str(skill), "sha256": gate.tree_digest(skill)}
    cases = []
    for number, product in enumerate(("a3", "a3", "a5", "a5"), 1):
        evidence = tmp_path / f"case-{number}.json"
        evidence.write_text(json.dumps({"case": number, "activity": 0.9}) + "\n")
        cases.append({
            "id": f"case-{number}", "product": product,
            "evidence": str(evidence), "evidence_sha256": digest(evidence),
            "required_conclusions": [f"fact-{number}"],
            "capacity_supported_resources": ["MTE2"] if number == 4 else [],
        })
    manifest = {
        "schema": gate.MANIFEST_SCHEMA,
        "prompt": str(prompt), "prompt_sha256": digest(prompt),
        "skills": skills, "cases": cases,
        "acquisition": {"products": ["a3", "a5"], "complete_agents": 3,
                        "max_agent_candidates": 6},
        "interpretation": {"agents_per_arm": 3},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, prompt


class FakeExecutor:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def __call__(self, request, workspace, skill_root):
        self.calls.append((request, workspace, skill_root))
        assert {p.name for p in (workspace / ".agents/skills").iterdir()} == {
            "ascend-profiling"
        }
        assert skill_root == workspace / ".agents/skills/ascend-profiling"
        assert (workspace / "prompt.md").read_bytes() == request["prompt_bytes"]
        if self.responses:
            response = self.responses.pop(0)
            if response.get("status") == "success" and request["mode"] == "acquisition":
                evidence = workspace / "compact-evidence.json"
                evidence.write_text("evidence\n")
                response.update(evidence_file=evidence.name,
                                evidence_sha256=digest(evidence), reasoning="ok")
            return response
        if request["mode"] == "acquisition":
            evidence = workspace / "compact-evidence.json"
            evidence.write_text("evidence\n")
            return {"status": "success", "product": request["product"],
                    "evidence_sha256": digest(evidence),
                    "evidence_file": evidence.name,
                    "handle": "remote:test:job:1", "reasoning": "capture complete",
                    "commands": ["cpl-remote observe --wait HANDLE"], "logs": []}
        case_number = int(request["case_id"].split("-")[-1])
        return {"status": "success", "case_id": request["case_id"],
                "evidence_sha256": request["evidence_sha256"],
                "conclusions": [f"fact-{case_number}"],
                "saturation_claims": ([{"resource": "MTE2"}]
                                      if case_number == 4 else []),
                "reasoning": "bounded structured reasoning", "logs": []}


def test_full_battery_is_blinded_isolated_and_scores_12_of_12(tmp_path: Path):
    manifest_path, _ = inputs(tmp_path)
    executor = FakeExecutor()
    report = gate.Battery(manifest_path, tmp_path / "runs", executor).run()

    interpretation = [call[0] for call in executor.calls
                      if call[0]["mode"] == "interpretation"]
    assert len(interpretation) == 24
    for case_id in {item["case_id"] for item in interpretation}:
        prompts = {item["prompt_sha256"] for item in interpretation
                   if item["case_id"] == case_id}
        evidence = {item["evidence_sha256"] for item in interpretation
                    if item["case_id"] == case_id}
        assert len(prompts) == len(evidence) == 1
    assert all("arm" not in item["agent_request"] for item in interpretation)
    assert report["interpretation"]["candidate"] == {"passed": 12, "total": 12}
    assert report["acceptance"]["candidate_12_of_12"] is True
    assert report["acquisition"]["complete_agents"] == 3
    assert report["acceptance"]["passed"] is False  # candidate must beat, not tie


def test_rejects_changed_evidence_and_unsupported_saturation(tmp_path: Path):
    manifest_path, _ = inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    case = Path(manifest["cases"][0]["evidence"])
    case.write_text("changed\n")
    with pytest.raises(gate.GateError, match="evidence hash mismatch"):
        gate.Battery(manifest_path, tmp_path / "runs", FakeExecutor())

    manifest["cases"][0]["evidence_sha256"] = digest(case)
    manifest_path.write_text(json.dumps(manifest))
    executor = FakeExecutor()
    original = executor.__call__

    def unsupported(request, workspace, skill_root):
        result = original(request, workspace, skill_root)
        if request["mode"] == "interpretation" and request["case_id"] == "case-1":
            result["saturation_claims"] = [{"resource": "MTE2"}]
        return result

    report = gate.Battery(manifest_path, tmp_path / "runs", unsupported).run()
    assert report["interpretation"]["candidate"]["passed"] == 9
    assert report["acceptance"]["candidate_12_of_12"] is False
    scored = json.loads((tmp_path / "runs/report.json").read_text())
    failures = [x for x in scored["records"] if x.get("score", {}).get("unsupported")]
    assert len(failures) == 6


def test_infra_retries_resume_and_counted_failures_are_not_discarded(tmp_path: Path):
    manifest_path, _ = inputs(tmp_path)
    responses = [
        {"status": "failure", "failure_class": "infrastructure",
         "failure_type": "device_busy", "logs": ["busy"]},
        {"status": "success", "product": "a3", "evidence_sha256": "e" * 64,
         "handle": "remote:test:job:2", "commands": [], "logs": []},
        {"status": "failure", "failure_class": "counted",
         "failure_type": "compile", "logs": ["compile failed"]},
    ]
    first = FakeExecutor(responses)
    battery = gate.Battery(manifest_path, tmp_path / "runs", first,
                           stop_after_attempts=3)
    partial = battery.run()
    assert partial["status"] == "incomplete"
    assert [r["classification"] for r in partial["records"]] == [
        "discarded_infrastructure", "success", "counted_failure"
    ]

    resumed = FakeExecutor()
    report = gate.Battery(manifest_path, tmp_path / "runs", resumed).run()
    assert report["status"] == "complete"
    assert report["counts"]["discarded_infrastructure"] == 1
    assert report["counts"]["counted_failure"] == 1
    assert report["acquisition"]["agents_attempted"] == 4
    before = len(resumed.calls)
    gate.Battery(manifest_path, tmp_path / "runs", resumed).run()
    assert len(resumed.calls) == before


@pytest.mark.parametrize("failure_type", [
    "compile", "runtime", "profiler_command", "evidence", "interpretation"
])
def test_all_non_infrastructure_failures_count(tmp_path: Path, failure_type: str):
    manifest_path, _ = inputs(tmp_path)
    executor = FakeExecutor([{"status": "failure", "failure_class": "counted",
                              "failure_type": failure_type, "logs": []}])
    report = gate.Battery(manifest_path, tmp_path / "runs", executor,
                          stop_after_attempts=1).run()
    assert report["counts"]["counted_failure"] == 1
    assert report["records"][0]["failure_type"] == failure_type


def test_undocumented_infrastructure_claim_counts_as_evidence_failure(tmp_path: Path):
    manifest_path, _ = inputs(tmp_path)
    executor = FakeExecutor([{"status": "failure", "failure_class": "infrastructure",
                              "failure_type": "device_busy", "logs": []}])
    report = gate.Battery(manifest_path, tmp_path / "runs", executor,
                          stop_after_attempts=1).run()
    assert report["records"][0]["classification"] == "counted_failure"
    assert report["records"][0]["failure_type"] == "evidence"


def test_manifest_rejects_nonidentical_or_unpinned_black_box_inputs(tmp_path: Path):
    manifest_path, _ = inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["skills"]["candidate"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(gate.GateError, match="skill hash mismatch"):
        gate.Battery(manifest_path, tmp_path / "runs", FakeExecutor())


def test_subprocess_executor_retains_structured_answer_and_logs(tmp_path: Path):
    script = tmp_path / "fake.py"
    script.write_text(
        "import json,os\nfrom pathlib import Path\n"
        "r=json.load(open(os.environ['PROFILE_GATE_REQUEST']))\n"
        "assert os.getcwd()==r['workspace']\n"
        "assert os.environ['HOME'].startswith(r['workspace'])\n"
        "assert os.environ['CODEX_HOME'].startswith(r['workspace'])\n"
        "assert os.environ['AGENTS_HOME'].startswith(r['workspace'])\n"
        "assert not (Path.home()/'.agents/skills').exists()\n"
        "json.dump({'status':'success','case_id':r['case_id'],"
        "'evidence_sha256':r['evidence_sha256'],'conclusions':['fact-1'],"
        "'saturation_claims':[],'reasoning':'ok','logs':['ran']},"
        "open(os.environ['PROFILE_GATE_OUTPUT'],'w'))\n"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = {"case_id": "case-1", "evidence_sha256": "a" * 64,
               "workspace": str(workspace)}
    result = gate.SubprocessExecutor(["python", str(script)])(
        request, workspace, workspace
    )
    assert result["reasoning"] == "ok" and result["logs"] == ["ran"]


def test_scoped_remote_registry_is_available_without_home_skill_leakage(tmp_path: Path):
    manifest_path, _ = inputs(tmp_path)
    registry = tmp_path / "remotes.json"
    registry.write_text('{"targets": {"bz-a3-1": {}}}\n')
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_inputs"] = [{
        "name": "remote-registry.json", "path": str(registry),
        "sha256": digest(registry), "environment": "PROFILE_GATE_REMOTE_REGISTRY",
    }]
    manifest_path.write_text(json.dumps(manifest))

    class RuntimeExecutor(FakeExecutor):
        def __call__(self, request, workspace, skill_root):
            remote = Path(request["_executor_environment"]["PROFILE_GATE_REMOTE_REGISTRY"])
            assert remote.read_bytes() == registry.read_bytes()
            assert remote.is_relative_to(workspace)
            assert not (workspace / ".home/.agents/skills").exists()
            return super().__call__(request, workspace, skill_root)

    gate.Battery(manifest_path, tmp_path / "runs", RuntimeExecutor(),
                 stop_after_attempts=1).run()
    assert not list((tmp_path / "runs/workspaces").rglob(".runtime-inputs"))


@pytest.mark.parametrize("kind", ["escape", "symlink", "wrong_hash"])
def test_acquisition_rejects_unretained_evidence(tmp_path: Path, kind: str):
    manifest_path, _ = inputs(tmp_path)

    def invalid(request, workspace, _skill_root):
        outside = tmp_path / "outside.json"
        outside.write_text("evidence\n")
        evidence = workspace / "compact.json"
        if kind == "symlink":
            evidence.symlink_to(outside)
            name = evidence.name
        elif kind == "escape":
            name = str(outside)
        else:
            evidence.write_text("evidence\n")
            name = evidence.name
        return {"status": "success", "product": request["product"],
                "evidence_sha256": ("0" * 64 if kind == "wrong_hash" else digest(outside)),
                "evidence_file": name, "handle": "remote:test:job:1",
                "commands": ["run"], "reasoning": "done", "logs": []}

    report = gate.Battery(manifest_path, tmp_path / "runs", invalid,
                          stop_after_attempts=1).run()
    assert report["records"][0]["classification"] == "counted_failure"
    assert report["records"][0]["failure_type"] == "evidence"


@pytest.mark.parametrize("failure", ["timeout", "oserror"])
def test_executor_launch_failures_are_counted_and_resumable(tmp_path: Path, failure: str):
    manifest_path, _ = inputs(tmp_path)

    def broken(_request, _workspace, _skill_root):
        raise gate.ExecutorFailure("evidence", failure)

    report = gate.Battery(manifest_path, tmp_path / "runs", broken,
                          stop_after_attempts=1).run()
    assert report["status"] == "incomplete"
    assert report["records"][0]["classification"] == "counted_failure"


def test_mutating_frozen_evidence_is_a_counted_failure(tmp_path: Path):
    manifest_path, _ = inputs(tmp_path)

    def mutating(request, workspace, _skill_root):
        if request["mode"] == "acquisition":
            return {"status": "failure", "failure_class": "counted",
                    "failure_type": "compile", "logs": []}
        (workspace / "evidence.json").write_text("tampered\n")
        return {"status": "success", "case_id": request["case_id"],
                "evidence_sha256": request["evidence_sha256"],
                "conclusions": [], "saturation_claims": [],
                "reasoning": "bad", "logs": []}

    # Skip unsuccessful acquisition candidates quickly, then inspect the first
    # interpretation attempt using a resumed state with valid acquisition rows.
    battery = gate.Battery(manifest_path, tmp_path / "runs", mutating,
                           stop_after_attempts=1)
    battery.state["records"] = [
        {"unit": f"acquisition/agent-{agent}/{product}", "attempt": 1,
         "mode": "acquisition", "classification": "success",
         "failure_type": None, "answer": {}}
        for agent in range(1, 4) for product in ("a3", "a5")
    ]
    report = battery.run()
    assert report["records"][-1]["classification"] == "counted_failure"
    assert report["records"][-1]["failure_type"] == "evidence"
