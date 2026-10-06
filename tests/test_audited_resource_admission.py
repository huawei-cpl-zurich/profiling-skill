from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/audited_resource_admission.py"


def load():
    spec = importlib.util.spec_from_file_location("audited_resource_admission", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admission(path: Path, slots: list[dict]) -> str:
    path.write_text(json.dumps({
        "schema": "profiling-skill/bz-a3-admission/v1",
        "slots": slots,
    }))
    return sha256(path)


def install_fake_client(home: Path) -> tuple[Path, Path]:
    executable = home / ".agents/skills/remote-access/scripts/cpl-remote"
    executable.parent.mkdir(parents=True)
    calls = home / "calls.jsonl"
    executable.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]
with pathlib.Path(os.environ["FAKE_CPL_CALLS"]).open("a") as stream:
    stream.write(json.dumps(args)+"\\n")
operation=args[1]; target=args[2]
if operation == "capabilities":
    payload={"target":target,"operation":operation,"state":"available",
             "capabilities":{"run":True,"observe":True,"logs":True,"upload":True}}
else:
    payload={"target":target,"operation":operation,
             "state":"completed" if target == "bz-a3-1" else "failed"}
print(json.dumps(payload,sort_keys=True))
''')
    executable.chmod(0o755)
    return executable, calls


def test_pool_intersects_pinned_slots_with_global_capabilities_and_preflight(
        monkeypatch, tmp_path: Path):
    module = load()
    executable, calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("FAKE_CPL_CALLS", str(calls))
    provider = tmp_path / "admission.json"
    provider_sha = admission(provider, [
        {"target": "bz-a3-1", "device": 7, "healthy": True, "idle": True},
        {"target": "bz-a3-1", "device": 8, "healthy": True, "idle": False},
        {"target": "bz-a3-2", "device": 4, "healthy": True, "idle": True},
    ])

    pool = module.CplRemoteResourcePool(
        provider, provider_sha, cpl_remote_sha256=sha256(executable))

    assert pool.admit() == [
        {"target": "bz-a3-1", "device": 7, "healthy": True, "idle": True},
        {"target": "bz-a3-1", "device": 8, "healthy": True, "idle": False},
    ]
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        ["--json", "capabilities", "bz-a3-1"],
        ["--json", "preflight", "bz-a3-1"],
        ["--json", "capabilities", "bz-a3-2"],
        ["--json", "preflight", "bz-a3-2"],
    ]


@pytest.mark.parametrize("document", [
    {},
    {"schema": "wrong", "slots": []},
    {"schema": "profiling-skill/bz-a3-admission/v1", "slots": [], "extra": 1},
    {"schema": "profiling-skill/bz-a3-admission/v1", "slots": [
        {"target": "gz-a3", "device": 0, "healthy": True, "idle": True}]},
    {"schema": "profiling-skill/bz-a3-admission/v1", "slots": [
        {"target": "bz-a3-1", "device": True, "healthy": True, "idle": True}]},
    {"schema": "profiling-skill/bz-a3-admission/v1", "slots": [
        {"target": "bz-a3-1", "device": 0, "healthy": 1, "idle": True}]},
    {"schema": "profiling-skill/bz-a3-admission/v1", "slots": [
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True},
        {"target": "bz-a3-1", "device": 0, "healthy": False, "idle": False}]},
])
def test_admission_schema_rejects_invalid_or_duplicate_slots(
        tmp_path: Path, document: dict):
    module = load()
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(document))
    with pytest.raises(module.AdmissionError):
        module.load_admission(path, sha256(path))


def test_provider_hash_drift_fails_before_global_client(
        monkeypatch, tmp_path: Path):
    module = load()
    executable, calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("FAKE_CPL_CALLS", str(calls))
    provider = tmp_path / "admission.json"
    provider_sha = admission(provider, [])
    provider.write_text("{}")
    pool = module.CplRemoteResourcePool(
        provider, provider_sha, cpl_remote_sha256=sha256(executable))
    with pytest.raises(module.AdmissionError, match="admission.*hash"):
        pool.admit()
    assert not calls.exists()


def test_global_client_is_mandatory_hash_pinned_and_rechecked(
        monkeypatch, tmp_path: Path):
    module = load()
    executable, calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("FAKE_CPL_CALLS", str(calls))
    provider = tmp_path / "admission.json"
    provider_sha = admission(provider, [])
    with pytest.raises(module.AdmissionError, match="hash"):
        module.CplRemoteResourcePool(
            provider, provider_sha, cpl_remote_sha256="0" * 64)
    pool = module.CplRemoteResourcePool(
        provider, provider_sha, cpl_remote_sha256=sha256(executable))
    executable.write_text(executable.read_text() + "\n# drift\n")
    with pytest.raises(module.AdmissionError, match="hash"):
        pool.admit()
    assert not calls.exists()


def test_probe_rejects_wrong_target_or_malformed_success_receipt(
        monkeypatch, tmp_path: Path):
    module = load()
    executable, _calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    provider = tmp_path / "admission.json"
    provider_sha = admission(provider, [
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True}])
    responses = iter([
        subprocess.CompletedProcess([], 0, json.dumps({
            "target": "bz-a3-2", "state": "available", "capabilities": {
                "run": True, "observe": True, "logs": True, "upload": True}}), ""),
    ])
    pool = module.CplRemoteResourcePool(
        provider, provider_sha, cpl_remote_sha256=sha256(executable),
        invoke=lambda *_args, **_kwargs: next(responses),
    )
    with pytest.raises(module.AdmissionError, match="identity"):
        pool.admit()
