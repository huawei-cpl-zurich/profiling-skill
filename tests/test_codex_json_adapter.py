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
        "prompt=sys.stdin.read()\n"
        "capture.write_text(json.dumps({'argv':sys.argv[1:],'prompt':prompt,'home':os.environ.get('HOME'),'codex_home':os.environ.get('CODEX_HOME')}))\n"
        "out=pathlib.Path(sys.argv[sys.argv.index('-o')+1])\n"
        "request=json.loads(prompt.split('REQUEST_JSON\\n',1)[1])\n"
        "if request['phase']=='prepare':\n"
        " value={'candidate':'VALUE = 1\\n','commands':['python -m py_compile candidate.py'],'hypothesis':'compile first','retrieved_knowledge':[]}\n"
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
    codex_home.mkdir(parents=True)
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
