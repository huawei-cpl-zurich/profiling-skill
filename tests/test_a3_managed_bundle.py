from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import subprocess
import tarfile
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


def test_verify_extract_checks_hash_and_sets_read_only_modes(tmp_path: Path) -> None:
    archive, manifest = create(make_sources(tmp_path), tmp_path / "out")
    destination = tmp_path / "extracted"
    actual = BUNDLE.verify_extract(archive, destination, manifest["root_digest"], 20, 10_000)
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
        BUNDLE.verify_extract(corrupt, tmp_path / "dest", None, 20, 10_000)


def test_verify_rejects_undeclared_and_nonregular_members(tmp_path: Path) -> None:
    archive, _ = create(make_sources(tmp_path), tmp_path / "out")
    corrupt = tmp_path / "link.tar"
    rewrite_tar(archive, corrupt, lambda name, data, kind: (name, data, "symlink" if name.endswith("kernel.py") else kind))
    with pytest.raises(BUNDLE.BundleError, match="non-regular archive member"):
        BUNDLE.verify_extract(corrupt, tmp_path / "dest", None, 20, 10_000)


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
 print(json.dumps({'id':'transfer-123','status':'succeeded'}))
elif args[:2] == ['transfer','fetch']:
 pathlib.Path(args[args.index('--dst')+1]).write_bytes(b'result')
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
    env = {**os.environ, "FAKE_LOG": str(log)}
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
    requested = run_cli(["request-download", "--client", str(client), "--remote", "a3-gz", "--remote-path", f"results/profiling-workloads/{digest}/result.tar", "--receipt", str(receipt), "--poll-interval", "0.01"], env)
    assert requested.returncode == 0, requested.stderr
    output = tmp_path / "result.tar"
    fetched = run_cli(["fetch", "--client", str(client), "--handle", "transfer-123", "--output", str(output), "--poll-interval", "0.01"], env)
    assert fetched.returncode == 0, fetched.stderr
    assert output.read_bytes() == b"result"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][:2] == ["transfer", "download"]
    assert calls[-1][:2] == ["transfer", "fetch"]


@pytest.mark.parametrize("remote_path", ["/tmp/result.tar", "results/other/result.tar", "results/profiling-workloads/not-a-digest/result.tar", f"results/profiling-workloads/{'a' * 64}/../result.tar"])
def test_download_rejects_paths_outside_managed_results(tmp_path: Path, remote_path: str) -> None:
    client, log = fake_client(tmp_path)
    result = run_cli(["request-download", "--client", str(client), "--remote", "a3-gz", "--remote-path", remote_path, "--receipt", str(tmp_path / "receipt.json")], {**os.environ, "FAKE_LOG": str(log)})
    assert result.returncode == 2
    assert not log.exists()


def test_failed_transfer_does_not_write_receipt(tmp_path: Path) -> None:
    root = make_sources(tmp_path)
    client = tmp_path / "client.py"
    client.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({'id':'x','status':'failed'}))\n")
    client.chmod(0o755)
    receipt = tmp_path / "receipt.json"
    result = run_cli(["stage", "--client", str(client), "--remote", "a3-gz", "--source-root", str(root), "--include", "run.sh", "--receipt", str(receipt), "--poll-interval", "0.01"], os.environ.copy())
    assert result.returncode == 2
    assert "ended with status failed" in result.stderr
    assert not receipt.exists()
