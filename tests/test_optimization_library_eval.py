from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("optimization_eval", Path(__file__).resolve().parents[1] / "scripts/optimization_library_eval.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def wave_fixture(tmp_path):
    files = {"before.py": "before_only", "profile.json": '{"latency_us": 4}',
             "contract.txt": "sum with fp32 accumulation", "skill.md": "guidance_only",
             "pair.json": '{"before_us": 4, "after_us": 3}',
             "gold.json": json.dumps({"basis": "measured_pair", "status": "conclusive",
                                      "rubric": "sealed_answer", "pair_artifact": "pair.json"})}
    for name, value in files.items():
        (tmp_path / name).write_text(value)
    manifest = {"model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
                "allowed_skills": {"optimization": "skill.md"}, "skill_verified": True,
                "families": [{"id": f"family-{i}", "heldouts": [{"id": str(j),
                    "public": {"source": "before.py", "profile": "profile.json", "contract": "contract.txt"},
                    "gold": "gold.json"} for j in range(2)]} for i in range(12)]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def response(schema):
    result = {key: "reasoned explanation" for key in schema["required"]}
    if schema is m.REPORT_SCHEMA:
        result.update(decision="apply", confidence=0.7)
    else:
        result.update(verdict="supported")
    return result


def test_full_wave_is_fresh_paired_and_blinded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = wave_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    (tmp_path / "common.md").write_text("shared_guidance")
    manifest["common_skills"] = {"profiling": "common.md"}
    path.write_text(json.dumps(manifest))
    wave = m.load_wave(path)
    calls = []
    def invoke(session, prompt, schema, model, timeout):
        assert session.is_absolute()
        assert session.is_dir() and not list(session.glob("input"))
        calls.append((session.name, prompt, schema, model))
        return response(schema)
    result = m.run_wave(wave, Path("results"), invoke)
    assert len(calls) == 192
    assert result["review_count"] == 96
    assert not result["failed_reviews"] and not result["failed_grades"]
    assert len({c[0] for c in calls}) == 192
    assert all(c[3] == wave["model"] for c in calls)
    reviews, grades = calls[:96], calls[96:]
    assert all("sealed_answer" not in c[1] and "after_us" not in c[1] and "family-" not in c[1] for c in reviews)
    assert sum("guidance_only" in c[1] for c in reviews) == 48
    assert all("shared_guidance" in c[1] for c in reviews)
    assert all("guidance_only" not in c[1] and "with_skill" not in c[1] for c in grades)
    assert all("sealed_answer" in c[1] and "after_us" in c[1] for c in grades)
    for index in range(24):
        receipts = [r for r in result["reviews"] if r["case_index"] == index]
        assert {(r["role"], r["with_skill"]) for r in receipts} == {
            (r, s) for r in ("actor", "analyzer") for s in (False, True)}
        assert len({r["artifact_digest"] for r in receipts}) == 1
    with pytest.raises(FileExistsError):
        m.run_wave(wave, tmp_path / "results", invoke)


def test_failures_and_initial_grades_remain_visible(tmp_path):
    wave = m.load_wave(wave_fixture(tmp_path))
    count = 0
    def invoke(session, prompt, schema, model, timeout):
        nonlocal count
        count += 1
        if count == 1:
            raise TimeoutError("partial diagnostics retained")
        if schema is m.GRADE_SCHEMA:
            return {"bad": "schema"}
        return response(schema)
    result = m.run_wave(wave, tmp_path / "results", invoke)
    assert result["failed_reviews"] == 1 and result["failed_grades"] == 95
    assert result["reviews"][0]["status"] == "failed"
    assert "TimeoutError" in result["reviews"][0]["error"]
    assert json.loads((tmp_path / "results/results.json").read_text()) == result
    assert json.loads((tmp_path / "results/grade-001/raw-response.json").read_text()) == {"bad": "schema"}


@pytest.mark.parametrize("change", ["count", "extra_public", "unverified", "unmeasured", "symlink"])
def test_invalid_evidence_never_dispatches(tmp_path, change):
    path = wave_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    if change == "count":
        manifest["families"].pop()
    elif change == "extra_public":
        manifest["families"][0]["heldouts"][0]["public"]["after"] = "pair.json"
    elif change == "unverified":
        manifest["skill_verified"] = False
    elif change == "unmeasured":
        gold = json.loads((tmp_path / "gold.json").read_text())
        gold["basis"] = "expected"
        (tmp_path / "gold.json").write_text(json.dumps(gold))
    else:
        (tmp_path / "before.py").unlink()
        (tmp_path / "before.py").symlink_to(tmp_path / "pair.json")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        m.load_wave(path)


def test_inconclusive_gold_is_explicit_and_numeric_schema_is_checked(tmp_path):
    path = wave_fixture(tmp_path)
    gold = json.loads((tmp_path / "gold.json").read_text())
    gold["status"] = "inconclusive"
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    wave = m.load_wave(path)
    prompt = m.grading_prompt(wave["cases"][0], response(m.REPORT_SCHEMA))
    assert '"status": "inconclusive"' in prompt
    invalid = response(m.REPORT_SCHEMA)
    invalid["confidence"] = True
    with pytest.raises(ValueError):
        m.validate_response(invalid, m.REPORT_SCHEMA)


@pytest.mark.parametrize("tool_attempt", [False, True])
@pytest.mark.parametrize("relative_session", [False, True])
def test_real_filesystem_boundary(tmp_path, monkeypatch, tool_attempt, relative_session):
    if not shutil.which("bwrap"):
        pytest.skip("Bubblewrap unavailable")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    executable = runtime / "codex"
    executable.write_text("#!/bin/sh\ntest ! -e '" + str(tmp_path / "gold") + "' || exit 31\n"
        "test ! -e '" + str(Path(__file__).resolve().parents[1]) + "' || exit 32\n"
        "test -r /input/prompt.txt || exit 33\n"
        "if echo changed > /input/prompt.txt 2>/dev/null; then exit 34; fi\n"
        "echo '{}' > /output/response.json\n" +
        ("echo '{\"item\":{\"type\":\"command_execution\"}}'\n" if tool_attempt else ""))
    executable.chmod(0o755)
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    (tmp_path / "gold").write_text("secret")
    launcher = m.ReportLauncher(codex=str(executable), auth_home=auth)
    probe = subprocess.run([launcher.bwrap, "--ro-bind", "/", "/", "true"], capture_output=True)
    if probe.returncode:
        pytest.skip("Bubblewrap namespace unavailable: " + probe.stderr.decode())
    monkeypatch.chdir(tmp_path)
    session = Path("session") if relative_session else tmp_path / "session"
    session.mkdir()
    if tool_attempt:
        with pytest.raises(m.LaunchError, match="attempted a tool"):
            launcher(session, "immutable context", m.REPORT_SCHEMA,
                     {"name": "gpt-5.6-sol", "reasoning_effort": "low"}, 20)
    else:
        assert launcher(session, "immutable context", m.REPORT_SCHEMA,
                        {"name": "gpt-5.6-sol", "reasoning_effort": "low"}, 20) == {}
    assert (session / "input/prompt.txt").read_text() == "immutable context"
    assert (session / "process.json").is_file()


def test_parallel_sessions_overlap_and_all_learners_finish_before_gold(tmp_path):
    from threading import Barrier, Lock
    wave = m.load_wave(wave_fixture(tmp_path))
    barrier, lock = Barrier(2), Lock()
    active = finished = maximum = 0
    def invoke(session, prompt, schema, model, timeout):
        nonlocal active, finished, maximum
        if schema is m.REPORT_SCHEMA:
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait(timeout=10)  # Would fail if sessions accidentally ran serially.
            with lock:
                active -= 1
                finished += 1
        else:
            with lock:
                assert finished == 96 and active == 0
            assert "sealed_answer" in prompt and "allowed_guidance" not in prompt
        return response(schema)
    parallel = m.run_wave(wave, tmp_path / "parallel", invoke, workers=2)
    serial = m.run_wave(wave, tmp_path / "serial", lambda *a: response(a[2]))
    assert maximum == 2
    assert parallel == serial  # Seeded receipt/session identities survive scheduling.
    with pytest.raises(ValueError, match="workers"):
        m.run_wave(wave, tmp_path / "invalid", invoke, workers=0)
    assert not (tmp_path / "invalid").exists()
