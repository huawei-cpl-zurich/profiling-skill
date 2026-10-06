from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
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


def allowlist_sha256(slots: list[dict]) -> str:
    identities = sorted(
        ({"target": slot["target"], "device": slot["device"]} for slot in slots),
        key=lambda item: (item["target"], item["device"]),
    )
    return hashlib.sha256(json.dumps(
        identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def admission(path: Path, slots: list[dict], *, now: datetime | None = None,
              provider_id: str = "operator-snapshot") -> tuple[str, str]:
    now = now or datetime.now(UTC)
    allowlist = allowlist_sha256(slots)
    path.write_text(json.dumps({
        "schema": "profiling-skill/bz-a3-admission/v2",
        "provider": {"id": provider_id, "allowlist_sha256": allowlist},
        "generated_at": (now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
        "slots": slots,
    }))
    return sha256(path), allowlist


def install_fake_client(home: Path) -> tuple[Path, Path, Path]:
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
    implementation = executable.with_name("cpl_remote.py")
    implementation.write_text("# pinned implementation\n")
    return executable, implementation, calls


def test_pool_intersects_pinned_slots_with_global_capabilities_and_preflight(
        monkeypatch, tmp_path: Path):
    module = load()
    executable, _implementation, calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("FAKE_CPL_CALLS", str(calls))
    provider = tmp_path / "admission.json"
    provider_sha, allowlist = admission(provider, [
        {"target": "bz-a3-1", "device": 7, "healthy": True, "idle": True},
        {"target": "bz-a3-1", "device": 8, "healthy": True, "idle": False},
        {"target": "bz-a3-2", "device": 4, "healthy": True, "idle": True},
    ])

    pool = module.CplRemoteResourcePool(
        provider, provider_sha, provider_id="operator-snapshot",
        allowlist_sha256=allowlist,
        cpl_remote_closure_sha256=module.cpl_remote_closure_sha256(executable))

    admitted = pool.admit()
    snapshot = pool.last_receipt
    assert snapshot is not None
    assert snapshot.receipt_sha256 == provider_sha
    assert snapshot.provider_id == "operator-snapshot"
    assert snapshot.allowlist_sha256 == allowlist
    assert admitted == snapshot.as_slots() == [
        {"target": "bz-a3-1", "device": 7, "healthy": True, "idle": True},
        {"target": "bz-a3-1", "device": 8, "healthy": True, "idle": False},
    ]
    assert pool.last_receipt == snapshot
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        ["--json", "capabilities", "bz-a3-1"],
        ["--json", "preflight", "bz-a3-1"],
        ["--json", "capabilities", "bz-a3-2"],
        ["--json", "preflight", "bz-a3-2"],
    ]


@pytest.mark.parametrize("document", [
    {},
    {"schema": "wrong", "slots": []},
    {"schema": "profiling-skill/bz-a3-admission/v2", "slots": [], "extra": 1},
    {"schema": "profiling-skill/bz-a3-admission/v2", "slots": [
        {"target": "gz-a3", "device": 0, "healthy": True, "idle": True}]},
    {"schema": "profiling-skill/bz-a3-admission/v2", "slots": [
        {"target": "bz-a3-1", "device": True, "healthy": True, "idle": True}]},
    {"schema": "profiling-skill/bz-a3-admission/v2", "slots": [
        {"target": "bz-a3-1", "device": 0, "healthy": 1, "idle": True}]},
    {"schema": "profiling-skill/bz-a3-admission/v2", "slots": [
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
    executable, _implementation, calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("FAKE_CPL_CALLS", str(calls))
    provider = tmp_path / "admission.json"
    provider_sha, allowlist = admission(provider, [])
    provider.write_text("{}")
    pool = module.CplRemoteResourcePool(
        provider, provider_sha, provider_id="operator-snapshot",
        allowlist_sha256=allowlist,
        cpl_remote_closure_sha256=module.cpl_remote_closure_sha256(executable))
    with pytest.raises(module.AdmissionError, match="admission.*hash"):
        pool.admit()
    assert not calls.exists()


def test_global_client_is_mandatory_hash_pinned_and_rechecked(
        monkeypatch, tmp_path: Path):
    module = load()
    executable, implementation, calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("FAKE_CPL_CALLS", str(calls))
    provider = tmp_path / "admission.json"
    provider_sha, allowlist = admission(provider, [])
    with pytest.raises(module.AdmissionError, match="hash"):
        module.CplRemoteResourcePool(
            provider, provider_sha, provider_id="operator-snapshot",
            allowlist_sha256=allowlist,
            cpl_remote_closure_sha256="0" * 64)
    pool = module.CplRemoteResourcePool(
        provider, provider_sha, provider_id="operator-snapshot",
        allowlist_sha256=allowlist,
        cpl_remote_closure_sha256=module.cpl_remote_closure_sha256(executable))
    implementation.write_text("# implementation drift\n")
    with pytest.raises(module.AdmissionError, match="hash"):
        pool.admit()
    assert not calls.exists()


def test_probe_rejects_wrong_target_or_malformed_success_receipt(
        monkeypatch, tmp_path: Path):
    module = load()
    executable, _implementation, _calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    provider = tmp_path / "admission.json"
    provider_sha, allowlist = admission(provider, [
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True}])
    responses = iter([
        subprocess.CompletedProcess([], 0, json.dumps({
            "target": "bz-a3-2", "state": "available", "capabilities": {
                "run": True, "observe": True, "logs": True, "upload": True}}), ""),
    ])
    pool = module.CplRemoteResourcePool(
        provider, provider_sha, provider_id="operator-snapshot",
        allowlist_sha256=allowlist,
        cpl_remote_closure_sha256=module.cpl_remote_closure_sha256(executable),
        invoke=lambda *_args, **_kwargs: next(responses),
    )
    with pytest.raises(module.AdmissionError, match="identity"):
        pool.admit()


def test_expired_or_overlong_admission_receipt_fails_closed(tmp_path: Path):
    module = load()
    now = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)
    path = tmp_path / "admission.json"
    slots = [{"target": "bz-a3-1", "device": 0,
              "healthy": True, "idle": True}]
    receipt_sha, _allowlist = admission(path, slots, now=now)
    document = json.loads(path.read_text())
    document["expires_at"] = now.isoformat().replace("+00:00", "Z")
    path.write_text(json.dumps(document))
    with pytest.raises(module.AdmissionError, match="expired"):
        module.load_admission(path, sha256(path), now=now)

    document["generated_at"] = now.isoformat().replace("+00:00", "Z")
    document["expires_at"] = (now + timedelta(minutes=6)).isoformat().replace(
        "+00:00", "Z")
    path.write_text(json.dumps(document))
    with pytest.raises(module.AdmissionError, match="validity"):
        module.load_admission(path, sha256(path), now=now)
    assert receipt_sha != sha256(path)


def test_complete_receipt_rejects_duplicate_target_device(tmp_path: Path):
    module = load()
    path = tmp_path / "admission.json"
    receipt_sha, _allowlist = admission(path, [
        {"target": "bz-a3-1", "device": 2, "healthy": True, "idle": True},
        {"target": "bz-a3-1", "device": 2, "healthy": False, "idle": False},
    ])
    with pytest.raises(module.AdmissionError, match="duplicated"):
        module.load_admission(path, receipt_sha)


def test_pool_pins_static_provider_and_allowlist_identity(monkeypatch, tmp_path: Path):
    module = load()
    executable, _implementation, _calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    path = tmp_path / "admission.json"
    receipt_sha, allowlist = admission(path, [
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True}])
    closure = module.cpl_remote_closure_sha256(executable)
    with pytest.raises(module.AdmissionError, match="provider identity"):
        module.CplRemoteResourcePool(
            path, receipt_sha, provider_id="different-provider",
            allowlist_sha256=allowlist, cpl_remote_closure_sha256=closure).admit()
    with pytest.raises(module.AdmissionError, match="allowlist identity"):
        module.CplRemoteResourcePool(
            path, receipt_sha, provider_id="operator-snapshot",
            allowlist_sha256="0" * 64,
            cpl_remote_closure_sha256=closure).admit()


def test_pool_clears_retained_receipt_when_snapshot_expires(monkeypatch, tmp_path: Path):
    module = load()
    executable, _implementation, calls = install_fake_client(tmp_path)
    monkeypatch.setattr(module, "_user_home", lambda: tmp_path)
    monkeypatch.setenv("FAKE_CPL_CALLS", str(calls))
    current = [datetime(2026, 10, 6, 9, 0, tzinfo=UTC)]
    path = tmp_path / "admission.json"
    receipt_sha, allowlist = admission(path, [
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True}],
        now=current[0])
    pool = module.CplRemoteResourcePool(
        path, receipt_sha, provider_id="operator-snapshot",
        allowlist_sha256=allowlist,
        cpl_remote_closure_sha256=module.cpl_remote_closure_sha256(executable),
        clock=lambda: current[0])
    assert pool.admit() and pool.last_receipt is not None
    current[0] += timedelta(minutes=3)
    with pytest.raises(module.AdmissionError, match="expired"):
        pool.admit()
    assert pool.last_receipt is None


def test_load_admission_hashes_and_parses_one_atomic_byte_read(
        monkeypatch, tmp_path: Path):
    module = load()
    path = tmp_path / "admission.json"
    first_sha, _allowlist = admission(path, [
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True}])
    replacement = tmp_path / "replacement.json"
    admission(replacement, [
        {"target": "bz-a3-2", "device": 3, "healthy": True, "idle": True}])
    original_read = module.Path.read_bytes
    reads = 0

    def replace_after_read(subject):
        nonlocal reads
        data = original_read(subject)
        if subject == path and reads == 0:
            reads += 1
            replacement.replace(path)
        return data

    monkeypatch.setattr(module.Path, "read_bytes", replace_after_read)
    receipt = module.load_admission(path, first_sha)

    assert reads == 1
    assert receipt.receipt_sha256 == first_sha
    assert receipt.as_slots()[0]["target"] == "bz-a3-1"
    with pytest.raises(module.AdmissionError, match="hash"):
        module.load_admission(path, first_sha)
