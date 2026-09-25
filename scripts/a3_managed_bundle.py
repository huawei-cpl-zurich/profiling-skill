#!/usr/bin/env python3
"""Content-addressed workload bundles for the managed GZ-A3 transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

PROTOCOL = "a3-managed-bundle/v1"
DEFAULT_MAX_FILES = 512
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class BundleError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise BundleError(f"unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BundleError(f"unsafe relative path: {value!r}")
    return path


def safe_result_path(value: str) -> str:
    path = _safe_relative(value)
    parts = path.parts
    if len(parts) < 4 or parts[:2] != ("results", "profiling-workloads") or not re.fullmatch(r"[0-9a-f]{64}", parts[2]):
        raise BundleError("result path must be beneath results/profiling-workloads/<root-digest>/")
    return path.as_posix()


def collect_files(root: Path, includes: Iterable[str], max_files: int, max_bytes: int) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise BundleError("source root must be a directory")
    selected: dict[str, Path] = {}
    for raw in includes:
        relative = _safe_relative(raw)
        candidate = root.joinpath(*relative.parts)
        try:
            info = candidate.lstat()
        except FileNotFoundError as exc:
            raise BundleError(f"included path does not exist: {raw}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise BundleError(f"symlinks are forbidden: {raw}")
        paths = [candidate]
        if stat.S_ISDIR(info.st_mode):
            paths = sorted(candidate.rglob("*"))
        for path in paths:
            item = path.lstat()
            rel = path.relative_to(root).as_posix()
            _safe_relative(rel)
            if stat.S_ISDIR(item.st_mode):
                continue
            if stat.S_ISLNK(item.st_mode):
                raise BundleError(f"symlinks are forbidden: {rel}")
            if not stat.S_ISREG(item.st_mode):
                raise BundleError(f"special files are forbidden: {rel}")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise BundleError(f"path escapes source root: {rel}")
            selected[rel] = path
    if not selected:
        raise BundleError("bundle contains no files")
    if len(selected) > max_files:
        raise BundleError(f"file limit exceeded: {len(selected)} > {max_files}")
    total = sum(path.stat().st_size for path in selected.values())
    if total > max_bytes:
        raise BundleError(f"byte limit exceeded: {total} > {max_bytes}")
    return [
        {
            "path": rel,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "executable": bool(path.stat().st_mode & 0o111),
            "_source": path,
        }
        for rel, path in sorted(selected.items())
    ]


def create_bundle(root: Path, includes: Iterable[str], output_dir: Path, max_files: int, max_bytes: int) -> tuple[Path, dict[str, Any]]:
    entries = collect_files(root, includes, max_files, max_bytes)
    public_entries = [{key: value for key, value in item.items() if key != "_source"} for item in entries]
    root_digest = hashlib.sha256(canonical_json(public_entries)).hexdigest()
    manifest = {
        "protocol": PROTOCOL,
        "root_digest": root_digest,
        "file_count": len(entries),
        "total_bytes": sum(item["size"] for item in entries),
        "files": public_entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{root_digest}.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
        payload = canonical_json(manifest)
        header = tarfile.TarInfo("bundle-manifest.json")
        header.size, header.mode, header.mtime, header.uid, header.gid = len(payload), 0o444, 0, 0, 0
        header.uname = header.gname = ""
        import io
        tar.addfile(header, io.BytesIO(payload))
        for item in entries:
            header = tar.gettarinfo(str(item["_source"]), arcname=f'payload/{item["path"]}')
            header.mode = 0o555 if item["executable"] else 0o444
            header.mtime, header.uid, header.gid = 0, 0, 0
            header.uname = header.gname = ""
            with item["_source"].open("rb") as stream:
                tar.addfile(header, stream)
    return archive, manifest


def _load_manifest(tar: tarfile.TarFile) -> dict[str, Any]:
    members = tar.getmembers()
    if not members or members[0].name != "bundle-manifest.json" or not members[0].isfile():
        raise BundleError("bundle manifest is missing or misplaced")
    stream = tar.extractfile(members[0])
    try:
        manifest = json.load(stream) if stream else None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BundleError("bundle manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL:
        raise BundleError("unsupported bundle protocol")
    return manifest


def verify_extract(archive: Path, destination: Path, expected_digest: str | None, max_files: int, max_bytes: int) -> dict[str, Any]:
    if destination.exists() and any(destination.iterdir()):
        raise BundleError("destination must be absent or empty")
    with tarfile.open(archive, "r:") as tar:
        manifest = _load_manifest(tar)
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) > max_files:
            raise BundleError("invalid manifest file list or file limit exceeded")
        public_entries: list[dict[str, Any]] = []
        expected_members: dict[str, dict[str, Any]] = {}
        total = 0
        for item in files:
            if not isinstance(item, dict):
                raise BundleError("invalid manifest entry")
            rel = _safe_relative(str(item.get("path", ""))).as_posix()
            if rel in expected_members:
                raise BundleError(f"duplicate manifest path: {rel}")
            normalized = {"path": rel, "size": item.get("size"), "sha256": item.get("sha256"), "executable": item.get("executable")}
            if not isinstance(normalized["size"], int) or normalized["size"] < 0:
                raise BundleError(f"invalid size for {rel}")
            if not isinstance(normalized["sha256"], str) or len(normalized["sha256"]) != 64:
                raise BundleError(f"invalid hash for {rel}")
            if not isinstance(normalized["executable"], bool):
                raise BundleError(f"invalid executable bit for {rel}")
            total += normalized["size"]
            expected_members[f"payload/{rel}"] = normalized
            public_entries.append(normalized)
        root_digest = hashlib.sha256(canonical_json(public_entries)).hexdigest()
        if manifest.get("root_digest") != root_digest or (expected_digest and expected_digest != root_digest):
            raise BundleError("bundle root digest mismatch")
        if manifest.get("file_count") != len(files) or manifest.get("total_bytes") != total or total > max_bytes:
            raise BundleError("bundle totals are invalid or byte limit exceeded")
        actual = tar.getmembers()[1:]
        if len(actual) != len(expected_members):
            raise BundleError("archive member set differs from manifest")
        destination.mkdir(parents=True, exist_ok=True)
        for member in actual:
            item = expected_members.pop(member.name, None)
            if item is None or not member.isfile():
                raise BundleError(f"undeclared or non-regular archive member: {member.name}")
            stream = tar.extractfile(member)
            data = stream.read(max_bytes + 1) if stream else b""
            if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise BundleError(f"content verification failed: {item['path']}")
            target = destination.joinpath(*PurePosixPath(item["path"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o555 if item["executable"] else 0o444)
    return manifest


def run_client(client: Path, prefix: list[str], args: list[str]) -> dict[str, Any]:
    result = subprocess.run([str(client), *prefix, *args], check=False, text=True, capture_output=True)
    if result.returncode:
        raise BundleError(f"managed transfer client failed ({result.returncode}): {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BundleError("managed transfer client returned invalid JSON") from exc


def wait_transfer(client: Path, prefix: list[str], job_id: str, timeout: int, poll: float) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        status = run_client(client, prefix, ["transfer", "status", "--job-id", job_id])
        state = status.get("status")
        if state == "succeeded":
            return status
        if state in ("failed", "cancelled"):
            raise BundleError(f"managed transfer {job_id} ended with status {state}")
        if time.monotonic() - started >= timeout:
            raise BundleError(f"managed transfer {job_id} did not finish within {timeout}s")
        time.sleep(poll)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def command_stage(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="a3-bundle-") as temp:
        archive, manifest = create_bundle(Path(args.source_root), args.include, Path(temp), args.max_files, args.max_bytes)
        remote_path = f"incoming/profiling-workloads/{manifest['root_digest']}/bundle.tar"
        response = run_client(Path(args.client), args.client_arg, ["transfer", "upload", "--remote", args.remote, "--src", str(archive), "--dst", remote_path])
        job_id = str(response.get("id", ""))
        if not job_id:
            raise BundleError("managed upload returned no job id")
        wait_transfer(Path(args.client), args.client_arg, job_id, args.timeout, args.poll_interval)
        receipt = {"protocol": PROTOCOL, "root_digest": manifest["root_digest"], "remote_path": remote_path, "transfer_handle": job_id, "state": "succeeded", "manifest": manifest}
        atomic_json(Path(args.receipt), receipt)
        print(canonical_json(receipt).decode(), end="")


def command_request_download(args: argparse.Namespace) -> None:
    remote_path = safe_result_path(args.remote_path)
    response = run_client(Path(args.client), args.client_arg, ["transfer", "download", "--remote", args.remote, "--src", remote_path])
    job_id = str(response.get("id", ""))
    if not job_id:
        raise BundleError("managed download returned no job id")
    wait_transfer(Path(args.client), args.client_arg, job_id, args.timeout, args.poll_interval)
    receipt = {"protocol": PROTOCOL, "remote_path": remote_path, "transfer_handle": job_id, "state": "succeeded"}
    atomic_json(Path(args.receipt), receipt)
    print(canonical_json(receipt).decode(), end="")


def command_fetch(args: argparse.Namespace) -> None:
    wait_transfer(Path(args.client), args.client_arg, args.handle, args.timeout, args.poll_interval)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_client(Path(args.client), args.client_arg, ["transfer", "fetch", "--job-id", args.handle, "--dst", str(output)])
    print(canonical_json({"protocol": PROTOCOL, "transfer_handle": args.handle, "output": str(output), "state": "succeeded"}).decode(), end="")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--client", required=True)
    common.add_argument("--client-arg", action="append", default=[])
    common.add_argument("--timeout", type=int, default=900)
    common.add_argument("--poll-interval", type=float, default=2)
    sub = result.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage", parents=[common])
    stage.add_argument("--source-root", required=True)
    stage.add_argument("--include", action="append", required=True)
    stage.add_argument("--receipt", required=True)
    stage.add_argument("--remote", required=True)
    stage.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    stage.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    stage.set_defaults(function=command_stage)
    verify = sub.add_parser("verify-extract")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--destination", required=True)
    verify.add_argument("--expected-digest")
    verify.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    verify.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    verify.set_defaults(function=lambda a: print(canonical_json(verify_extract(Path(a.archive), Path(a.destination), a.expected_digest, a.max_files, a.max_bytes)).decode(), end=""))
    download = sub.add_parser("request-download", parents=[common])
    download.add_argument("--remote", required=True)
    download.add_argument("--remote-path", required=True)
    download.add_argument("--receipt", required=True)
    download.set_defaults(function=command_request_download)
    fetch = sub.add_parser("fetch", parents=[common])
    fetch.add_argument("--handle", required=True)
    fetch.add_argument("--output", required=True)
    fetch.set_defaults(function=command_fetch)
    return result


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "timeout", 1) <= 0 or getattr(args, "poll_interval", 1) <= 0:
        raise BundleError("timeouts and poll intervals must be positive")
    if getattr(args, "max_files", 1) <= 0 or getattr(args, "max_bytes", 1) <= 0:
        raise BundleError("bundle limits must be positive")
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
