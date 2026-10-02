from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import subprocess
import tarfile
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "a3_managed_bundle.py"
SPEC = importlib.util.spec_from_file_location("a3_managed_bundle", SCRIPT)
assert SPEC and SPEC.loader
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


def make_sources(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "kernel.py").write_text("print('kernel')\n")
    runner = root / "run.sh"
    runner.write_text("#!/bin/sh\npython pkg/kernel.py\n")
    runner.chmod(0o755)
    return root


def create(root: Path, output: Path, **kwargs):
    return BUNDLE.create_bundle(root, ["pkg", "run.sh"], output, kwargs.get("max_files", 20), kwargs.get("max_bytes", 10_000))


def test_bundle_is_deterministic_and_manifest_records_behavior(tmp_path: Path) -> None:
    first_root = make_sources(tmp_path / "one")
    second_root = make_sources(tmp_path / "two")
    os.utime(second_root / "run.sh", (1_900_000_000, 1_900_000_000))
    first, manifest = create(first_root, tmp_path / "out-one")
    second, _ = create(second_root, tmp_path / "out-two")
    assert first.read_bytes() == second.read_bytes()
    assert first.name == f"{manifest['root_digest']}.tar"
    assert manifest["file_count"] == 2
    assert [item["path"] for item in manifest["files"]] == ["pkg/kernel.py", "run.sh"]
    assert [item["executable"] for item in manifest["files"]] == [False, True]
    assert manifest["archive_sha256"] == BUNDLE.sha256_file(first)


def test_bundle_archives_stable_snapshot_when_source_changes_after_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_sources(tmp_path)
    original = BUNDLE.canonical_json
    mutated = False

    def mutate_after_snapshot(value):
        nonlocal mutated
        encoded = original(value)
        if isinstance(value, list) and not mutated:
            (root / "run.sh").write_text("changed after snapshot\n")
            mutated = True
        return encoded

    monkeypatch.setattr(BUNDLE, "canonical_json", mutate_after_snapshot)
    archive, manifest = create(root, tmp_path / "out")
    destination = tmp_path / "extracted"
    BUNDLE.verify_extract(archive, destination, manifest["root_digest"], manifest["archive_sha256"], 20, 10_000)
    assert mutated
    assert (destination / "run.sh").read_text() == "#!/bin/sh\npython pkg/kernel.py\n"


def test_verify_extract_checks_hash_and_sets_read_only_modes(tmp_path: Path) -> None:
    archive, manifest = create(make_sources(tmp_path), tmp_path / "out")
    destination = tmp_path / "extracted"
    actual = BUNDLE.verify_extract(archive, destination, manifest["root_digest"], manifest["archive_sha256"], 20, 10_000)
    assert actual == manifest
    assert (destination / "pkg/kernel.py").read_text() == "print('kernel')\n"
    assert stat.S_IMODE((destination / "pkg/kernel.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((destination / "run.sh").stat().st_mode) == 0o555


@pytest.mark.parametrize("include", ["/absolute", "../escape", "pkg/../run.sh", "pkg\\kernel.py", ""])
def test_rejects_unsafe_include_paths(tmp_path: Path, include: str) -> None:
    root = make_sources(tmp_path)
    with pytest.raises(BUNDLE.BundleError, match="unsafe relative path"):
        BUNDLE.create_bundle(root, [include], tmp_path / "out", 20, 10_000)


def test_rejects_symlinks_in_explicit_and_recursive_includes(tmp_path: Path) -> None:
    root = make_sources(tmp_path)
    (root / "link").symlink_to("run.sh")
    with pytest.raises(BUNDLE.BundleError, match="symlinks are forbidden"):
        BUNDLE.create_bundle(root, ["link"], tmp_path / "out-a", 20, 10_000)
    (root / "pkg" / "link").symlink_to("../run.sh")
    with pytest.raises(BUNDLE.BundleError, match="symlinks are forbidden"):
        BUNDLE.create_bundle(root, ["pkg"], tmp_path / "out-b", 20, 10_000)


def test_rejects_special_files_and_limits(tmp_path: Path) -> None:
    root = make_sources(tmp_path)
    fifo = root / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(BUNDLE.BundleError, match="special files"):
        BUNDLE.create_bundle(root, ["pipe"], tmp_path / "out-a", 20, 10_000)
    with pytest.raises(BUNDLE.BundleError, match="file limit"):
        create(root, tmp_path / "out-b", max_files=1)
    with pytest.raises(BUNDLE.BundleError, match="byte limit"):
        create(root, tmp_path / "out-c", max_bytes=1)


def rewrite_tar(source: Path, destination: Path, mutate) -> None:
    with tarfile.open(source, "r:") as old, tarfile.open(destination, "w") as new:
        for member in old.getmembers():
            stream = old.extractfile(member)
            data = stream.read() if stream else b""
            name, data, kind = mutate(member.name, data, "file" if member.isfile() else "other")
            header = tarfile.TarInfo(name)
            header.size = len(data)
            if kind == "symlink":
                header.type, header.linkname, header.size = tarfile.SYMTYPE, "../../outside", 0
                new.addfile(header)
            else:
                new.addfile(header, io.BytesIO(data))


def test_verify_rejects_corrupt_content(tmp_path: Path) -> None:
    archive, _ = create(make_sources(tmp_path), tmp_path / "out")
    corrupt = tmp_path / "corrupt.tar"
    rewrite_tar(archive, corrupt, lambda name, data, kind: (name, b"bad" if name.endswith("kernel.py") else data, kind))
    with pytest.raises(BUNDLE.BundleError, match="content verification failed"):
        BUNDLE.verify_extract(corrupt, tmp_path / "dest", None, BUNDLE.sha256_file(corrupt), 20, 10_000)


def test_verify_rejects_undeclared_and_nonregular_members(tmp_path: Path) -> None:
    archive, _ = create(make_sources(tmp_path), tmp_path / "out")
    corrupt = tmp_path / "link.tar"
    rewrite_tar(archive, corrupt, lambda name, data, kind: (name, data, "symlink" if name.endswith("kernel.py") else kind))
    with pytest.raises(BUNDLE.BundleError, match="non-regular archive member"):
        BUNDLE.verify_extract(corrupt, tmp_path / "dest", None, BUNDLE.sha256_file(corrupt), 20, 10_000)


def test_verify_rejects_duplicate_relative_manifest_paths(tmp_path: Path) -> None:
    archive, _ = create(make_sources(tmp_path), tmp_path / "out")
    corrupt = tmp_path / "duplicate.tar"

    def duplicate(name, data, kind):
        if name == "bundle-manifest.json":
            manifest = json.loads(data)
            manifest["files"][1]["path"] = manifest["files"][0]["path"]
            data = BUNDLE.canonical_json(manifest)
        return name, data, kind

    rewrite_tar(archive, corrupt, duplicate)
    with pytest.raises(BUNDLE.BundleError, match="duplicate manifest path"):
        BUNDLE.verify_extract(corrupt, tmp_path / "dest", None, BUNDLE.sha256_file(corrupt), 20, 10_000)


def fake_client(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "client-log.jsonl"
    client = tmp_path / "client.py"
    client.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, shutil, sys
args=sys.argv[1:]
with pathlib.Path(os.environ['FAKE_LOG']).open('a') as f: f.write(json.dumps(args)+'\\n')
if args[:2] == ['transfer','upload'] or args[:2] == ['transfer','download']:
 print(json.dumps({'id':'transfer-123'}))
elif args[:2] == ['transfer','status']:
 receipt=os.environ.get('RECEIPT_TO_CHECK')
 if receipt:
  saved=json.loads(pathlib.Path(receipt).read_text())
  assert saved['state'] == 'submitted' and saved['transfer_handle'] == 'transfer-123'
 status=os.environ.get('FAKE_STATUS','succeeded')
 if status == 'observer-error': print('listener unavailable',file=sys.stderr); raise SystemExit(8)
 print(json.dumps({'id':'transfer-123','status':status}))
elif args[:2] == ['transfer','fetch']:
 pathlib.Path(args[args.index('--dst')+1]).write_bytes(os.environ.get('FAKE_FETCH','result').encode())
 print(json.dumps({'state':'fetched'}))
else: raise SystemExit(9)
"""
    )
    client.chmod(0o755)
    return client, log


def run_cli(arguments: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(SCRIPT), *arguments], text=True, capture_output=True, env=env, check=False)


def test_stage_uploads_content_addressed_bundle_and_writes_receipt(tmp_path: Path) -> None:
    root = make_sources(tmp_path)
    client, log = fake_client(tmp_path)
    receipt = tmp_path / "receipt.json"
    env = {**os.environ, "FAKE_LOG": str(log), "RECEIPT_TO_CHECK": str(receipt)}
    result = run_cli(["stage", "--client", str(client), "--remote", "a3-gz", "--source-root", str(root), "--include", "pkg", "--include", "run.sh", "--receipt", str(receipt), "--poll-interval", "0.01"], env)
    assert result.returncode == 0, result.stderr
    data = json.loads(receipt.read_text())
    assert data["state"] == "succeeded"
    assert data["transfer_handle"] == "transfer-123"
    assert data["remote_path"] == f"incoming/profiling-workloads/{data['root_digest']}/bundle.tar"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][:2] == ["transfer", "upload"]
    assert calls[1] == ["transfer", "status", "--job-id", "transfer-123"]


def test_download_and_fetch_use_managed_client(tmp_path: Path) -> None:
    client, log = fake_client(tmp_path)
    env = {**os.environ, "FAKE_LOG": str(log)}
    receipt = tmp_path / "download.json"
    digest = "a" * 64
    result_sha = BUNDLE.hashlib.sha256(b"result").hexdigest()
    requested = run_cli(["request-download", "--client", str(client), "--remote", "a3-gz", "--remote-path", f"results/profiling-workloads/{digest}/result.tar", "--expected-sha256", result_sha, "--receipt", str(receipt), "--poll-interval", "0.01"], env)
    assert requested.returncode == 0, requested.stderr
    output = tmp_path / "result.tar"
    fetched = run_cli(["fetch", "--client", str(client), "--handle", "transfer-123", "--output", str(output), "--expected-sha256", result_sha, "--poll-interval", "0.01"], env)
    assert fetched.returncode == 0, fetched.stderr
    assert output.read_bytes() == b"result"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][:2] == ["transfer", "download"]
    assert calls[-1][:2] == ["transfer", "fetch"]


def test_download_resume_observes_existing_handle_without_resubmission(tmp_path: Path) -> None:
    client, log = fake_client(tmp_path)
    receipt = tmp_path / "download.json"
    digest = "a" * 64
    result_sha = BUNDLE.hashlib.sha256(b"result").hexdigest()
    command = ["request-download", "--client", str(client), "--remote", "a3-gz", "--remote-path", f"results/profiling-workloads/{digest}/result.tar", "--expected-sha256", result_sha, "--receipt", str(receipt), "--poll-interval", "0.01"]
    first = run_cli(command, {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": "observer-error"})
    second = run_cli(command, {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": "succeeded"})
    assert first.returncode == 2
    assert second.returncode == 0, second.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(call[:2] == ["transfer", "download"] for call in calls) == 1


@pytest.mark.parametrize("remote_path", ["/tmp/result.tar", "results/other/result.tar", "results/profiling-workloads/not-a-digest/result.tar", f"results/profiling-workloads/{'a' * 64}/../result.tar"])
def test_download_rejects_paths_outside_managed_results(tmp_path: Path, remote_path: str) -> None:
    client, log = fake_client(tmp_path)
    result = run_cli(["request-download", "--client", str(client), "--remote", "a3-gz", "--remote-path", remote_path, "--expected-sha256", "a" * 64, "--receipt", str(tmp_path / "receipt.json")], {**os.environ, "FAKE_LOG": str(log)})
    assert result.returncode == 2
    assert not log.exists()


def test_failed_transfer_persists_terminal_receipt(tmp_path: Path) -> None:
    root = make_sources(tmp_path)
    client = tmp_path / "client.py"
    client.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({'id':'x','status':'failed'}))\n")
    client.chmod(0o755)
    receipt = tmp_path / "receipt.json"
    result = run_cli(["stage", "--client", str(client), "--remote", "a3-gz", "--source-root", str(root), "--include", "run.sh", "--receipt", str(receipt), "--poll-interval", "0.01"], os.environ.copy())
    assert result.returncode == 2
    assert "ended with status failed" in result.stderr
    saved = json.loads(receipt.read_text())
    assert saved["state"] == "failed"
    assert saved["phase"] == "upload"
    assert "ended with status failed" in saved["diagnostics"]
    assert "upload_observe" in saved["phase_timings_seconds"]


def test_observation_failure_persists_receipt_and_resume_never_uploads_twice(tmp_path: Path) -> None:
    root = make_sources(tmp_path)
    client, log = fake_client(tmp_path)
    receipt = tmp_path / "receipt.json"
    command = ["stage", "--client", str(client), "--remote", "a3-gz", "--source-root", str(root), "--include", "run.sh", "--receipt", str(receipt), "--poll-interval", "0.01"]
    interrupted = run_cli(command, {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": "observer-error"})
    assert interrupted.returncode == 2
    assert json.loads(receipt.read_text())["state"] == "observation-unavailable"
    resumed = run_cli(command, {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": "succeeded"})
    assert resumed.returncode == 0, resumed.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(call[:2] == ["transfer", "upload"] for call in calls) == 1
    response = json.loads(resumed.stdout)
    saved = json.loads(receipt.read_text())
    assert response["state"] == saved["state"] == "succeeded"
    assert "diagnostics" not in response
    assert "diagnostics" not in saved


def test_timeout_persists_observation_receipt(tmp_path: Path) -> None:
    root = make_sources(tmp_path)
    client, log = fake_client(tmp_path)
    receipt = tmp_path / "receipt.json"
    result = run_cli(["stage", "--client", str(client), "--remote", "a3-gz", "--source-root", str(root), "--include", "run.sh", "--receipt", str(receipt), "--timeout", "1", "--poll-interval", "0.01"], {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": "running"})
    assert result.returncode == 2
    assert "observation timed out" in result.stderr
    assert json.loads(receipt.read_text())["state"] == "observation-unavailable"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(call[:2] == ["transfer", "upload"] for call in calls) == 1


@pytest.mark.parametrize("terminal", ["cancelled", "rejected"])
def test_terminal_states_are_persisted_without_resubmission(tmp_path: Path, terminal: str) -> None:
    root = make_sources(tmp_path)
    client, log = fake_client(tmp_path)
    receipt = tmp_path / "receipt.json"
    command = ["stage", "--client", str(client), "--remote", "a3-gz", "--source-root", str(root), "--include", "run.sh", "--receipt", str(receipt), "--poll-interval", "0.01"]
    first = run_cli(command, {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": terminal})
    second = run_cli(command, {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": "succeeded"})
    assert first.returncode == second.returncode == 2
    assert json.loads(receipt.read_text())["state"] == terminal
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(call[:2] == ["transfer", "upload"] for call in calls) == 1


def test_archive_hash_fails_before_extract_and_leaves_no_destination(tmp_path: Path) -> None:
    archive, manifest = create(make_sources(tmp_path), tmp_path / "out")
    destination = tmp_path / "destination"
    with pytest.raises(BUNDLE.BundleError, match="archive SHA-256 mismatch"):
        BUNDLE.verify_extract(archive, destination, manifest["root_digest"], "0" * 64, 20, 10_000)
    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.extract-*"))


def test_corrupt_member_never_publishes_partial_tree_and_retry_succeeds(tmp_path: Path) -> None:
    archive, manifest = create(make_sources(tmp_path), tmp_path / "out")
    corrupt = tmp_path / "corrupt.tar"
    rewrite_tar(archive, corrupt, lambda name, data, kind: (name, b"bad" if name.endswith("run.sh") else data, kind))
    destination = tmp_path / "destination"
    with pytest.raises(BUNDLE.BundleError, match="content verification failed"):
        BUNDLE.verify_extract(corrupt, destination, manifest["root_digest"], BUNDLE.sha256_file(corrupt), 20, 10_000)
    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.extract-*"))
    BUNDLE.verify_extract(archive, destination, manifest["root_digest"], manifest["archive_sha256"], 20, 10_000)
    assert (destination / "run.sh").exists()


def test_fetch_hash_mismatch_does_not_publish_output(tmp_path: Path) -> None:
    client, log = fake_client(tmp_path)
    output = tmp_path / "result.tar"
    result = run_cli(["fetch", "--client", str(client), "--handle", "transfer-123", "--output", str(output), "--expected-sha256", "0" * 64, "--poll-interval", "0.01"], {**os.environ, "FAKE_LOG": str(log)})
    assert result.returncode == 2
    assert "fetched result SHA-256 mismatch" in result.stderr
    assert not output.exists()


def test_atomic_json_uses_unique_temporary_paths_under_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "receipt.json"
    barrier = threading.Barrier(2)
    original = BUNDLE.os.replace

    def synchronized_replace(source, target):
        barrier.wait(timeout=2)
        original(source, target)

    monkeypatch.setattr(BUNDLE.os, "replace", synchronized_replace)
    failures = []

    def publish(value):
        try:
            BUNDLE.atomic_json(destination, value)
        except Exception as error:  # pragma: no cover - asserted below
            failures.append(error)

    threads = [threading.Thread(target=publish, args=({"writer": writer},)) for writer in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
    assert json.loads(destination.read_text()) in ({"writer": 1}, {"writer": 2})
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_atomic_json_syncs_parent_directory_after_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "receipt.json"
    events: list[str] = []
    directory_descriptors: set[int] = set()
    original_open = BUNDLE.os.open
    original_fsync = BUNDLE.os.fsync
    original_replace = BUNDLE.os.replace

    def observed_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path) == tmp_path:
            directory_descriptors.add(descriptor)
        return descriptor

    def observed_fsync(descriptor):
        events.append("directory-fsync" if descriptor in directory_descriptors else "file-fsync")
        return original_fsync(descriptor)

    def observed_replace(source, target):
        events.append("replace")
        return original_replace(source, target)

    monkeypatch.setattr(BUNDLE.os, "open", observed_open)
    monkeypatch.setattr(BUNDLE.os, "fsync", observed_fsync)
    monkeypatch.setattr(BUNDLE.os, "replace", observed_replace)

    with BUNDLE.receipt_lock(destination):
        BUNDLE.atomic_json(destination, {"state": "succeeded"})
        assert events[-1] == "directory-fsync"

    assert events == ["file-fsync", "replace", "directory-fsync"]
    assert json.loads(destination.read_text()) == {"state": "succeeded"}


def test_fetch_reuses_digest_verified_existing_output_without_remote_observation(tmp_path: Path) -> None:
    client, log = fake_client(tmp_path)
    output = tmp_path / "result.tar"
    output.write_bytes(b"result")
    expected = BUNDLE.sha256_file(output)
    result = run_cli(["fetch", "--client", str(client), "--handle", "transfer-123", "--output", str(output), "--expected-sha256", expected, "--poll-interval", "0.01"], {**os.environ, "FAKE_LOG": str(log), "FAKE_STATUS": "observer-error", "FAKE_FETCH": "different"})
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["reused"] is True
    assert response["phase"] == "download-local-reuse"
    assert "remote observation was not required" in response["diagnostics"]
    assert "local_verify" in response["phase_timings_seconds"]
    assert output.read_bytes() == b"result"
    assert not log.exists()


def test_identical_concurrent_download_requests_submit_once(tmp_path: Path) -> None:
    client, log = fake_client(tmp_path)
    receipt = tmp_path / "download.json"
    digest = "a" * 64
    expected = BUNDLE.hashlib.sha256(b"result").hexdigest()
    command = ["request-download", "--client", str(client), "--remote", "a3-gz", "--remote-path", f"results/profiling-workloads/{digest}/result.tar", "--expected-sha256", expected, "--receipt", str(receipt), "--poll-interval", "0.01"]
    env = {**os.environ, "FAKE_LOG": str(log)}
    barrier = threading.Barrier(3)
    results = []

    def request():
        barrier.wait(timeout=2)
        results.append(run_cli(command, env))

    threads = [threading.Thread(target=request) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=2)
    for thread in threads:
        thread.join()
    assert all(result.returncode == 0 for result in results)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(call[:2] == ["transfer", "download"] for call in calls) == 1


def test_concurrent_observer_error_cannot_replace_terminal_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "download.json"
    receipt = {
        "protocol": BUNDLE.PROTOCOL,
        "kind": "download",
        "remote": "a3-gz",
        "remote_path": f"results/profiling-workloads/{'a' * 64}/result.tar",
        "expected_sha256": "b" * 64,
        "transfer_handle": "transfer-123",
        "state": "submitted",
    }
    BUNDLE.atomic_json(receipt_path, receipt)
    success_persisted = threading.Event()
    original_atomic_json = BUNDLE.atomic_json

    def tracked_atomic_json(path, value):
        original_atomic_json(path, value)
        if value.get("state") == "succeeded":
            success_persisted.set()

    def conflicting_observation(*_args, **_kwargs):
        if threading.current_thread().name == "late-error":
            assert success_persisted.wait(timeout=2)
            raise BUNDLE.ObservationUnavailable("listener unavailable")
        return {"status": "succeeded"}

    monkeypatch.setattr(BUNDLE, "atomic_json", tracked_atomic_json)
    monkeypatch.setattr(BUNDLE, "wait_transfer", conflicting_observation)
    results = []
    failures = []

    def observe():
        try:
            results.append(
                BUNDLE.observe_receipt(
                    dict(receipt), receipt_path, Path("unused"), [], 1, 0, "download-request"
                )
            )
        except Exception as error:  # pragma: no cover - asserted below
            failures.append(error)

    threads = [
        threading.Thread(target=observe, name="success"),
        threading.Thread(target=observe, name="late-error"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert not failures
    assert len(results) == 2
    assert all(result["state"] == "succeeded" for result in results)
    assert json.loads(receipt_path.read_text())["state"] == "succeeded"


@pytest.mark.parametrize("terminal_state", ["succeeded", "failed", "cancelled", "rejected"])
def test_receipt_merge_preserves_every_terminal_state(tmp_path: Path, terminal_state: str) -> None:
    receipt_path = tmp_path / "upload.json"
    terminal = {
        "protocol": BUNDLE.PROTOCOL,
        "kind": "upload",
        "remote": "a3-gz",
        "root_digest": "a" * 64,
        "archive_sha256": "b" * 64,
        "remote_path": f"incoming/profiling-workloads/{'a' * 64}/bundle.tar",
        "transfer_handle": "transfer-123",
        "state": terminal_state,
    }
    BUNDLE.atomic_json(receipt_path, terminal)
    stale_observation = {**terminal, "state": "observation-unavailable"}

    merged = BUNDLE.merge_receipt(receipt_path, stale_observation)

    assert merged["state"] == terminal_state
    assert json.loads(receipt_path.read_text())["state"] == terminal_state
