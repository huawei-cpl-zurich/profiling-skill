import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/codex_json_adapter.py"


def fake_codex(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-codex"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "capture=pathlib.Path(os.environ['FAKE_CAPTURE'])\n"
        "(pathlib.Path(os.environ['CODEX_HOME'])/'config.toml').write_text('generated = true\\n')\n"
        "prompt=sys.stdin.read()\n"
        "capture.write_text(json.dumps({'argv':sys.argv[1:],'prompt':prompt,'home':os.environ.get('HOME'),'codex_home':os.environ.get('CODEX_HOME')}))\n"
        "out=pathlib.Path(sys.argv[sys.argv.index('-o')+1])\n"
        "request=json.loads(prompt.split('REQUEST_JSON\\n',1)[1])\n"
        "if request['phase']=='prepare':\n"
        " value={'candidate':'VALUE = 1\\n','commands':['python -m py_compile candidate.py'],'hypothesis':'compile first','retrieved_knowledge':[]}\n"
        "elif request['phase']=='tune':\n"
        " value={'configurations':[{'id':'block-64','parameters':{'BLOCK':64},'hypothesis':'wider block'}]}\n"
        "else:\n"
        " value={'reasoning':'profile supports the change','postmortem':'retain it'}\n"
        "out.write_text(json.dumps(value))\n"
        "print(json.dumps({'type':'thread.started','thread_id':'thread-123'}))\n"
    )
    executable.chmod(0o755)
    return executable


def invoke(tmp_path: Path, request: dict):
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text("{}")
    (codex_home / "auth.json").chmod(0o600)
    capture = tmp_path / "capture.json"
    env = {
        **os.environ,
        "HOME": str(home),
        "CODEX_HOME": str(codex_home),
        "FAKE_CAPTURE": str(capture),
    }
    result = subprocess.run(
        ["python", str(SCRIPT), "--codex", str(fake_codex(tmp_path)),
         "--model", "fake-model", "--reasoning-effort", "low"],
        input=json.dumps(request), text=True, capture_output=True, env=env,
    )
    return result, json.loads(capture.read_text())


def test_prepare_invokes_isolated_codex_and_returns_thread(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    request = {"phase": "prepare", "prompt": "same prompt", "repo": str(repo),
               "round": 1, "allowed_skills": ["ascend-profiling"]}
    result, call = invoke(tmp_path, request)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "candidate": "VALUE = 1\n",
        "commands": ["python -m py_compile candidate.py"],
        "hypothesis": "compile first",
        "retrieved_knowledge": [],
        "session_id": "thread-123",
    }
    assert call["argv"][:2] == ["exec", "--json"]
    assert "--ignore-user-config" in call["argv"]
    assert "--ignore-rules" in call["argv"]
    assert call["home"] == str(tmp_path / "home")
    assert call["codex_home"] == str(tmp_path / "home/.codex")
    assert "same prompt" in call["prompt"]
    assert (tmp_path / "home/.codex/auth.json").is_file()
    assert not (tmp_path / "home/.codex/config.toml").exists()


def test_prepare_prompt_directs_exact_candidate_contract_without_treatment_guidance(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    contract = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "streaming_matmul_add_kernel_mix_aic",
        "entrypoint": "streaming_matmul_add_kernel",
        "fusion": {
            "schema_version": 1, "mode": "single-logical-launch",
            "complete_operator": True,
        },
    }
    request = {
        "phase": "prepare", "prompt": "identical common prompt", "repo": str(repo),
        "round": 1, "allowed_skills": [], "treatment": "guarded",
        "candidate_contract": contract,
    }

    result, call = invoke(tmp_path, request)

    assert result.returncode == 0, result.stderr
    assert "define and launch exactly candidate_contract.entrypoint" in call["prompt"]
    assert "Preserve candidate_contract.kernel_name" in call["prompt"]
    assert "initial prepare and every same-round repair" in call["prompt"]
    forwarded = json.loads(call["prompt"].split("REQUEST_JSON\n", 1)[1])
    assert forwarded["candidate_contract"] == contract


def test_analyze_resumes_exact_thread(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    request = {"phase": "analyze", "prompt": "same prompt", "repo": str(repo),
               "round": 1, "allowed_skills": [], "session_id": "thread-old",
               "candidate": "VALUE=1", "terminal_result": {"outcome": "success"}}
    result, call = invoke(tmp_path, request)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "reasoning": "profile supports the change", "postmortem": "retain it"
    }
    assert call["argv"][:4] == ["exec", "resume", "--json", "--ignore-user-config"]
    assert "thread-old" in call["argv"]


def test_tune_resumes_exact_thread_and_returns_bounded_parameter_space(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    request = {
        "phase": "tune", "prompt": "same prompt", "repo": str(repo),
        "round": 1, "allowed_skills": [], "session_id": "thread-old",
        "candidate": "AUTOTUNE_CONFIG = {'BLOCK': 32}\n",
        "declared_autotune_config": {"BLOCK": 32},
    }
    result, call = invoke(tmp_path, request)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"configurations": [{
        "id": "block-64", "parameters": {"BLOCK": 64},
        "hypothesis": "wider block",
    }]}
    assert call["argv"][:4] == ["exec", "resume", "--json", "--ignore-user-config"]
    assert "thread-old" in call["argv"]


def test_generated_config_is_removed_between_prepare_and_analyze(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    prepare = {
        "phase": "prepare", "prompt": "same prompt", "repo": str(repo),
        "round": 1, "allowed_skills": [],
    }
    first, _ = invoke(tmp_path, prepare)
    assert first.returncode == 0, first.stderr
    assert not (tmp_path / "home/.codex/config.toml").exists()

    analyze = {
        "phase": "analyze", "prompt": "same prompt", "repo": str(repo),
        "round": 1, "allowed_skills": [], "session_id": "thread-123",
        "candidate": "VALUE=1", "terminal_result": {"outcome": "success"},
    }
    second, call = invoke(tmp_path, analyze)
    assert second.returncode == 0, second.stderr
    assert call["argv"][:4] == ["exec", "resume", "--json", "--ignore-user-config"]
    assert (tmp_path / "home/.codex/auth.json").is_file()
    assert not (tmp_path / "home/.codex/config.toml").exists()


def test_adapter_rejects_missing_private_auth(tmp_path):
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    subprocess.run(["git", "-C", str(tmp_path), "init"], check=True, capture_output=True)
    result = subprocess.run(
        ["python", str(SCRIPT), "--codex", "/missing/codex", "--model", "fake"],
        input=json.dumps({"phase": "prepare", "prompt": "p", "repo": str(tmp_path)}),
        text=True, capture_output=True,
        env={**os.environ, "HOME": str(home), "CODEX_HOME": str(home / ".codex")},
    )
    assert result.returncode != 0
    assert "private auth" in result.stderr
