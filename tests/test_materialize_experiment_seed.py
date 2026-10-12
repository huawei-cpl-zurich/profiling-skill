import hashlib
import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/materialize_experiment_seed.py"
RUNNER = Path(__file__).parents[1] / "scripts/simple_experiment_runner.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_file_tree_sha(path: Path) -> str:
    file = path / "SKILL.md"
    digest = hashlib.sha256()
    digest.update(b"SKILL.md\0")
    digest.update(oct(file.stat().st_mode & 0o7777).encode() + b"\0")
    digest.update(file.read_bytes())
    return digest.hexdigest()


def test_materializes_hash_bound_seed_and_cell_config_reproducibly(tmp_path):
    prompt = tmp_path / "prompt.md"
    task = tmp_path / "task.md"
    candidate = tmp_path / "candidate.py"
    cases = tmp_path / "cases.jsonl"
    prompt.write_text("identical common prompt\n")
    task.write_text("task-specific contract\n")
    candidate.write_text("VALUE = 0\n")
    cases.write_text('{"case": 1}\n')
    skill = tmp_path / "frozen-profiler"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: ascend-profiling\n---\n")
    gate = tmp_path / "gate.sh"
    gate.write_text("true\n")
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n")
    auth.chmod(0o600)
    output = tmp_path / "seed"
    spec = {
        "schema": "profiling-skill/simple-seed-spec/v1",
        "output": str(output),
        "prompt": {"path": str(prompt), "sha256": sha(prompt)},
        "task": {"path": str(task), "sha256": sha(task)},
        "files": [
            {"path": str(candidate), "destination": "candidate.py", "sha256": sha(candidate)},
            {"path": str(cases), "destination": "cases.jsonl", "sha256": sha(cases)},
        ],
        "config": {
            "campaign": "campaign", "product": "a3", "task": "matmul",
            "treatment": "guarded-new-profiler", "target": "bz-a3-1",
            "runtime": "py311-torch", "remote_workspace": "/remote/work",
            "rounds": 4, "billed_checks": 8,
            "terminal_gate_path": str(gate), "agent_auth_path": str(auth),
            "agent_command": ["fake-agent"],
            "skill_bindings": [{"name": "ascend-profiling", "path": str(skill),
                                "sha256": one_file_tree_sha(skill)}],
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    first = subprocess.run(["python", str(SCRIPT), str(spec_path)], text=True,
                           capture_output=True)
    assert first.returncode == 0, first.stderr
    receipt = json.loads(first.stdout)
    config = json.loads((output / "cell.json").read_text())
    assert config["prompt"] == prompt.read_text()
    assert Path(config["task_path"]).read_bytes() == task.read_bytes()
    assert Path(config["source_repo"]) == output / "repo"
    assert (output / "repo/candidate.py").read_bytes() == candidate.read_bytes()
    assert subprocess.run(["git", "-C", str(output / "repo"), "status", "--porcelain"],
                          text=True, capture_output=True, check=True).stdout == ""
    initialized = subprocess.run(
        ["python", str(RUNNER), "init", "--root", str(tmp_path / "runs"),
         "--config", str(output / "cell.json")], text=True, capture_output=True,
    )
    assert initialized.returncode == 0, initialized.stderr
    cell = tmp_path / "runs/campaign/a3-matmul-guarded-new-profiler"
    assert (cell / "repo/candidate.py").read_bytes() == candidate.read_bytes()
    assert (cell / "repo/TASK.md").read_bytes() == task.read_bytes()
    assert (cell / "agent-home/.agents/skills/ascend-profiling/SKILL.md").is_file()

    second_output = tmp_path / "seed-2"
    spec["output"] = str(second_output)
    spec_path.write_text(json.dumps(spec))
    second = subprocess.run(["python", str(SCRIPT), str(spec_path)], text=True,
                            capture_output=True)
    assert second.returncode == 0, second.stderr
    second_receipt = json.loads(second.stdout)
    assert receipt["seed_tree_sha256"] == second_receipt["seed_tree_sha256"]
    assert receipt["commit"] == second_receipt["commit"]
    assert json.loads((second_output / "cell.json").read_text())["prompt"] == config["prompt"]


def test_rejects_hash_mismatch_before_publishing(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("VALUE = 0\n")
    output = tmp_path / "seed"
    spec = {"schema": "profiling-skill/simple-seed-spec/v1", "output": str(output),
            "prompt": {"path": str(source), "sha256": "0" * 64},
            "task": {"path": str(source), "sha256": sha(source)}, "files": [],
            "config": {}}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    result = subprocess.run(["python", str(SCRIPT), str(path)], text=True,
                            capture_output=True)
    assert result.returncode != 0
    assert "hash mismatch" in result.stderr
    assert not output.exists()
