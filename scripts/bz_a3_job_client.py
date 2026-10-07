#!/usr/bin/env python3
"""Production JSON job client for native BZ-A3 experiment workloads."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import shlex
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


TARGETS = {"bz-a3-1", "bz-a3-2"}
GLOBAL_CPL_REMOTE = Path(".agents/skills/remote-access/scripts/cpl-remote")
RESULTS = {"ok", "submission_error", "compile_error", "runtime_error", "correctness_error",
           "infrastructure_error"}


class JobError(RuntimeError):
    def __init__(self, failure_type: str, message: str, handle: str | None = None,
                 *, dispatch_uncertain: bool = False):
        self.failure_type = failure_type
        self.handle = handle
        self.dispatch_uncertain = dispatch_uncertain
        super().__init__(message)


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _invoke(argv: list[str], timeout: int) -> CommandResult:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout,
                                check=False)
    except subprocess.TimeoutExpired as exc:
        output = _text(exc.stdout) + _text(exc.stderr)
        match = re.search(r"\b(remote:bz-a3-[12]:job:[A-Za-z0-9_.-]+)\b", output)
        handle = match.group(1) if match else None
        raise JobError("observer_error" if handle else "transport_error",
                       f"transport timed out after {timeout}s", handle,
                       dispatch_uncertain=handle is None) from exc
    except OSError as exc:
        raise JobError("transport_error", f"transport unavailable: {exc}") from exc
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _bounded(value: str, limit: int = 64 * 1024) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "\n...[diagnostic truncated]"


def _user_home() -> Path:
    """Return the authenticated account home without trusting caller environment."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _cpl_json_receipt(result: CommandResult, handle: str | None) -> dict:
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    diagnostics = _bounded(result.stdout + result.stderr)
    if not lines:
        raise JobError("transport_error",
                       "cpl-remote JSON receipt is missing: " + diagnostics, handle)
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise JobError("transport_error",
                       "cpl-remote trailing JSON receipt is missing: " + diagnostics,
                       handle) from exc
    if not isinstance(payload, dict):
        raise JobError("transport_error",
                       "cpl-remote trailing JSON receipt is not an object: " + diagnostics,
                       handle)
    for line in lines[:-1]:
        try:
            earlier = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(earlier, dict):
            raise JobError("transport_error",
                           "cpl-remote JSON receipt is ambiguous: " + diagnostics,
                           handle)
    return payload


class GlobalCplRemoteTransport:
    """Hash-pinned access to the installed global remote-access client."""

    def __init__(self, expected_sha256: str, remote_root: str,
                 invoke: Callable = _invoke):
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise JobError("request_error",
                           "cpl-remote digest must be 64 lowercase hexadecimal characters")
        self.executable = _user_home() / GLOBAL_CPL_REMOTE
        try:
            actual = _sha(self.executable)
        except OSError as exc:
            raise JobError("transport_error",
                           f"global cpl-remote is unavailable: {exc}") from exc
        if actual != expected_sha256:
            raise JobError("digest_mismatch", "global cpl-remote digest mismatch")
        if not os.access(self.executable, os.X_OK):
            raise JobError("transport_error", "global cpl-remote is not executable")
        self.expected_sha256 = expected_sha256
        self.remote_root = validate_remote_root(remote_root)
        self.invoke = invoke

    def _call(self, arguments: list[str], timeout: int,
              *, handle: str | None = None) -> dict:
        if _sha(self.executable) != self.expected_sha256:
            raise JobError("digest_mismatch", "global cpl-remote changed after validation",
                           handle)
        try:
            result = self.invoke([str(self.executable), "--json", *arguments], timeout)
        except JobError as exc:
            if handle is None:
                raise
            raise JobError("observer_error", str(exc), handle) from exc
        return _cpl_json_receipt(result, handle)

    @staticmethod
    def _validate_handle(target: str, handle: object) -> str:
        prefix = f"remote:{target}:job:"
        if (not isinstance(handle, str) or not handle.startswith(prefix)
                or not re.fullmatch(r"remote:bz-a3-[12]:job:[A-Za-z0-9_.-]+", handle)):
            raise JobError("transport_error", "cpl-remote returned an invalid job handle")
        return handle

    def upload(self, target: str, source: Path, destination: str, timeout: int) -> None:
        payload = self._call(
            ["upload", "--timeout", str(timeout), target, str(source), destination], timeout)
        if (payload.get("target") != target or payload.get("state") != "completed"):
            raise JobError("staging_error", "cpl-remote upload did not complete")

    def dispatch(self, target: str, device: int, runtime: str, operation: str,
                 script: str, timeout: int) -> str:
        del operation
        wrapped = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                   f"export ASCEND_RT_VISIBLE_DEVICES={device}\n"
                   "export ASCEND_DEVICE_ID=0\n" + script)
        path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as stream:
                stream.write(wrapped)
                stream.flush()
                path = Path(stream.name)
            payload = self._call([
                "run", target, "--runtime", runtime, "--file", str(path),
                "--cwd", self.remote_root, "--timeout", str(timeout),
            ], timeout)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
        handle = self._validate_handle(target, payload.get("handle"))
        state = payload.get("state")
        if state not in {"running", "reconnecting", "observation-unavailable",
                         "completed", "failed", "cancelled"}:
            raise JobError("transport_error", "cpl-remote returned an invalid run state",
                           handle)
        return handle

    def observe(self, target: str, handle: str, timeout: int) -> CommandResult:
        self._validate_handle(target, handle)
        payload = self._call(
            ["observe", handle, "--wait", "--timeout", str(timeout)], timeout,
            handle=handle,
        )
        if payload.get("handle") != handle or payload.get("target") != target:
            raise JobError("transport_error", "cpl-remote observation identity mismatch",
                           handle)
        if payload.get("state") in {"running", "reconnecting", "observation-unavailable"}:
            raise JobError("observer_error", "retained job is not terminal", handle)
        if payload.get("state") not in {"completed", "failed", "cancelled"}:
            raise JobError("observer_error", "retained job returned an invalid state", handle)
        return self._collect(target, handle, timeout)

    def _collect(self, target: str, handle: str, timeout: int) -> CommandResult:
        terminal = self._call(["result", handle], timeout, handle=handle)
        if terminal.get("handle") != handle or terminal.get("target") != target:
            raise JobError("transport_error", "cpl-remote result identity mismatch", handle)
        state = terminal.get("state")
        if state not in {"completed", "failed", "cancelled"}:
            raise JobError("observer_error", "retained result is not terminal", handle)
        streams = {}
        for name, tail in (("stdout", 1), ("stderr", 200)):
            payload = self._call(
                ["logs", handle, "--stream", name, "--tail", str(tail)], timeout,
                handle=handle,
            )
            if (payload.get("handle") != handle or payload.get("target") != target
                    or payload.get("state") != "completed"
                    or not isinstance(payload.get("content"), str)):
                raise JobError("transport_error", "cpl-remote logs identity mismatch",
                               handle)
            streams[name] = payload["content"]
        exit_code = terminal.get("exit")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            exit_code = 0 if state == "completed" else 125
        return CommandResult(exit_code, streams["stdout"], streams["stderr"])


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_placements(value: object) -> dict[int, dict]:
    if not isinstance(value, dict) or not value:
        raise JobError("request_error", "placements must be a non-empty JSON object")
    result = {}
    for logical, placement in value.items():
        try:
            logical_id = int(logical)
        except (TypeError, ValueError) as exc:
            raise JobError("request_error", "placement keys must be logical device IDs") from exc
        if (str(logical_id) != str(logical) or logical_id < 0
                or not isinstance(placement, dict)
                or set(placement) != {"target", "device"}
                or placement.get("target") not in TARGETS
                or isinstance(placement.get("device"), bool)
                or not isinstance(placement.get("device"), int)
                or placement["device"] < 0):
            raise JobError("request_error", f"invalid placement for logical device {logical!r}")
        result[logical_id] = dict(placement)
    return result


def validate_remote_root(value: object) -> str:
    if (not isinstance(value, str) or len(value) < 2 or len(value) > 240
            or not re.fullmatch(r"/[A-Za-z0-9._/-]+", value)
            or str(PurePosixPath(value)) != value
            or any(part in {".", ".."} for part in PurePosixPath(value).parts)):
        raise JobError("request_error", "remote root must be a normalized absolute path")
    return value


def _identity(job: dict) -> dict:
    result = {key: job[key] for key in ("benchmark", "action", "device")}
    if job["action"] == "check":
        result.update(cases=job["cases"], scope=job.get("scope"))
    elif job["action"] == "measure":
        result.update(case=job["case"], phase=job["phase"])
    else:
        result.update(cases=job["cases"], repeats=job["repeats"],
                      round=job["round"],
                      kernel_name=job["profiling"]["kernel_name"])
    return result


def _parse_result(stdout: str, expected: dict) -> dict:
    lines = [line.removeprefix("BZ_PRODUCTION_RESULT=") for line in stdout.splitlines()
             if line.startswith("BZ_PRODUCTION_RESULT=")]
    if len(lines) != 1:
        raise JobError("transport_error",
                       "remote job did not return exactly one structured result")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise JobError("transport_error", "remote result is invalid JSON") from exc
    if not isinstance(result, dict) or result.get("status") not in RESULTS:
        raise JobError("transport_error", "remote result has an invalid status")
    mismatch = [key for key, value in expected.items() if result.get(key) != value]
    if mismatch:
        raise JobError("transport_error",
                       "remote result identity mismatch for " + ", ".join(mismatch))
    result["diagnostics"] = _bounded(str(result.get("diagnostics", "")))
    return result


def _positive_number(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _validate_profile_evidence(result: dict, job: dict) -> None:
    evidence = result.get("profile")
    rows = result.get("profile_cases")
    declared = job["profiling"]["kernel_name"]
    repeats = job["repeats"]
    explicit_names = (isinstance(evidence, dict)
                      and ("declared_kernel_name" in evidence
                           or "resolved_kernel_name" in evidence))
    evidence_declared = evidence.get("declared_kernel_name") if explicit_names else declared
    resolved = evidence.get("resolved_kernel_name") if explicit_names else declared
    if (not isinstance(evidence, dict) or evidence.get("status") != "success"
            or evidence.get("profiler") != "msprof-op"
            or evidence.get("kernel_name") != declared
            or evidence_declared != declared
            or not isinstance(resolved, str) or not resolved
            or result.get("declared_kernel_name", declared) != declared
            or result.get("resolved_kernel_name", resolved) != resolved
            or evidence.get("repeats") != repeats
            or not isinstance(rows, list) or evidence.get("cases") != rows
            or len(rows) != len(job["cases"])
            or not all(isinstance(row, dict) for row in rows)
            or [row.get("case") for row in rows] != job["cases"]):
        raise JobError("profile_tool_error", "compact msprof evidence identity mismatch")
    medians = []
    for row in rows:
        samples = row.get("samples_us")
        median = row.get("median_us")
        if (not isinstance(samples, list) or len(samples) != repeats
                or not all(_positive_number(sample) for sample in samples)
                or not _positive_number(median)
                or not math.isclose(median, statistics.median(samples),
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise JobError("profile_tool_error", "compact msprof case samples are invalid")
        medians.append(median)
    captures = evidence.get("captures")
    expected = [(row["case"], iteration, sample) for row in rows
                for iteration, sample in enumerate(row["samples_us"])]
    if (not isinstance(captures, list) or len(captures) != len(expected)
            or any(not isinstance(capture, dict)
                   or (capture.get("case"), capture.get("iteration")) != identity[:2]
                   or capture.get("kernel_name") != resolved
                   or (explicit_names and
                       (capture.get("declared_kernel_name") != declared
                        or capture.get("resolved_kernel_name") != resolved))
                   or not _positive_number(capture.get("duration_us"))
                   or not math.isclose(capture["duration_us"], identity[2],
                                       rel_tol=1e-12, abs_tol=1e-12)
                   for capture, identity in zip(captures, expected))):
        raise JobError("profile_tool_error", "compact msprof captures are invalid")
    geomean = evidence.get("geomean_us")
    expected_geomean = math.exp(sum(math.log(value) for value in medians) / len(medians))
    if (not _positive_number(geomean)
            or not math.isclose(geomean, expected_geomean,
                                rel_tol=1e-12, abs_tol=1e-12)
            or not _positive_number(result.get("geomean_us", geomean))
            or not math.isclose(result.get("geomean_us", geomean), geomean,
                                rel_tol=1e-12, abs_tol=1e-12)):
        raise JobError("profile_tool_error", "compact msprof geomean is invalid")


class BzA3JobClient:
    def __init__(self, transport: GlobalCplRemoteTransport, state_dir: Path,
                 placements: object, *, runner: Path, profiler: Path,
                 batch_profiler: Path, remote_root: str):
        self.transport = transport
        self.state_dir = state_dir
        self.placements = validate_placements(placements)
        self.placements_sha256 = _json_sha(self.placements)
        self.runner, self.profiler = runner, profiler
        self.batch_profiler = batch_profiler
        self.remote_root = validate_remote_root(remote_root)

    def run(self, job: object, timeout: int = 3600) -> dict:
        identity: dict = {}
        placement = None
        handle = None
        request_sha = None
        dispatch = completed = None
        request_lock = None
        try:
            if not isinstance(job, dict):
                raise JobError("request_error", "job must be a JSON object")
            if (isinstance(timeout, bool) or not isinstance(timeout, int)
                    or timeout <= 25):
                raise JobError("request_error",
                               "effective timeout must be an integer greater than 25 seconds")
            self._validate_job(job)
            identity = _identity(job)
            provenance = self._execution_provenance(job["runtime"])
            logical = job["device"]
            placement = self.placements.get(logical)
            if placement is None:
                raise JobError("request_error",
                               f"no frozen placement for logical device {logical}")
            stage_files, remote_job = self._files_and_job(job)
            file_hashes = {name: _sha(path) for name, path in stage_files}
            request_sha = _json_sha({"job": remote_job, "files": file_hashes,
                                     "campaign_device": logical,
                                     "placement": placement,
                                     "placements_sha256": self.placements_sha256,
                                     "remote_root": self.remote_root,
                                     "execution_provenance": provenance,
                                     "timeout_seconds": timeout})
            state = self.state_dir / request_sha
            archive = state / "payload.tar"
            dispatch, completed = state / "dispatch.json", state / "completed.json"
            deadline = time.monotonic() + timeout
            state.mkdir(parents=True, exist_ok=True)
            request_lock = self._acquire_request_lock(state / "request.lock", deadline)
            if completed.is_file():
                record = json.loads(completed.read_text())
                if record.get("request_sha256") != request_sha:
                    raise JobError("request_error", "completed receipt request mismatch")
                if record.get("execution_provenance") != provenance:
                    raise JobError("request_error", "completed receipt provenance mismatch")
                result = record.get("result")
                if not isinstance(result, dict):
                    raise JobError("request_error", "completed receipt is invalid")
                return result
            if dispatch.is_file():
                record = json.loads(dispatch.read_text())
                if record.get("request_sha256") != request_sha:
                    raise JobError("request_error", "dispatch receipt request mismatch")
                if record.get("execution_provenance") != provenance:
                    raise JobError("request_error", "dispatch receipt provenance mismatch")
                handle = record.get("handle")
                expected = f'remote:{placement["target"]}:job:'
                if not isinstance(handle, str) or not handle.startswith(expected):
                    raise JobError("request_error", "dispatch receipt handle mismatch")
                response = self.transport.observe(
                    placement["target"], handle, self._remaining(deadline, handle))
            else:
                self._archive(archive, stage_files, remote_job)
                # cpl-remote's rsync transport does not create destination
                # parents. The configured root is therefore a pre-provisioned
                # writable staging directory, while the content-addressed
                # filename keeps concurrent requests collision-free.
                remote_archive = f"{self.remote_root}/payload-{request_sha}.tar"
                self.transport.upload(placement["target"], archive, remote_archive,
                                      self._remaining(deadline))
                run_root = f"{self.remote_root}/runs/{request_sha}"
                script = self._remote_script(
                    remote_archive, _sha(archive), run_root,
                    self._workload_timeout(self._remaining(deadline)), job,
                )
                try:
                    handle = self.transport.dispatch(
                        placement["target"], placement["device"], job["runtime"],
                        f"profiling-job-{request_sha[:16]}", script,
                        self._remaining(deadline),
                    )
                    _write_json(dispatch, {"protocol_version": 1,
                                "request_sha256": request_sha,
                                "execution_provenance": provenance,
                                "handle": handle})
                    response = self.transport.observe(
                        placement["target"], handle,
                        self._remaining(deadline, handle))
                except JobError as exc:
                    if exc.handle:
                        _write_json(dispatch, {"protocol_version": 1,
                                    "request_sha256": request_sha,
                                    "execution_provenance": provenance,
                                    "handle": exc.handle})
                    raise
            result = self._complete(response, handle, identity, placement,
                                    request_sha, job, self.remote_root,
                                    self.placements_sha256, provenance)
            _write_json(completed, {"protocol_version": 1,
                        "request_sha256": request_sha,
                        "execution_provenance": provenance, "result": result})
            dispatch.unlink(missing_ok=True)
            return result
        except JobError as exc:
            return {"status": "infrastructure_error",
                    "failure_type": exc.failure_type,
                    "diagnostics": _bounded(str(exc)),
                    "handle": exc.handle or handle,
                    "dispatch_uncertain": exc.dispatch_uncertain,
                    **identity,
                    **({"placement": placement} if placement else {})}
        except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
            return {"status": "infrastructure_error", "failure_type": "staging_error",
                    "diagnostics": _bounded(str(exc)), "handle": handle, **identity,
                    **({"placement": placement} if placement else {})}
        finally:
            if request_lock is not None:
                fcntl.flock(request_lock, fcntl.LOCK_UN)
                request_lock.close()

    def _execution_provenance(self, runtime: str) -> dict[str, str]:
        digest = getattr(self.transport, "expected_sha256", None)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise JobError("request_error", "transport cpl-remote digest is invalid")
        return {"cpl_remote_sha256": digest, "runtime": runtime}

    @staticmethod
    def _acquire_request_lock(path: Path, deadline: float):
        stream = path.open("a+")
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if time.monotonic() >= deadline:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                    raise JobError(
                        "transport_error", "BZ request lock deadline exhausted")
                return stream
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    stream.close()
                    raise JobError(
                        "transport_error", "BZ request lock deadline exhausted")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _remaining(deadline: float, handle: str | None = None) -> int:
        value = math.ceil(deadline - time.monotonic())
        if value < 1:
            raise JobError("observer_error" if handle else "transport_error",
                           "BZ job deadline exhausted", handle)
        return value

    @staticmethod
    def _workload_timeout(remaining: int) -> int:
        if remaining <= 25:
            raise JobError("transport_error", "insufficient workload response grace")
        return remaining - 25

    @staticmethod
    def _validate_job(job: dict) -> None:
        if job.get("protocol_version") != 1 or job.get("action") not in {
                "check", "measure", "profile"}:
            raise JobError("request_error", "unsupported job protocol or action")
        if job.get("runtime") != "py311-torch":
            raise JobError("request_error", "runtime must be py311-torch")
        device = job.get("device")
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise JobError("request_error", "device must be a non-negative logical ID")
        logical_device = job.get("logical_device", 0)
        if isinstance(logical_device, bool) or logical_device != 0:
            raise JobError("request_error", "remote logical_device must be 0")
        cases = job.get("cases")
        valid_cases = (isinstance(cases, list) and bool(cases)
                       and all(not isinstance(case, bool)
                               and isinstance(case, int) and case >= 0
                               for case in cases))
        if job["action"] == "check" and not valid_cases:
            raise JobError("request_error", "check requires non-negative integer cases")
        if job["action"] == "measure":
            case = job.get("case")
            if (isinstance(case, bool) or not isinstance(case, int) or case < 0
                    or not isinstance(job.get("phase"), str)):
                raise JobError("request_error", "invalid measure request")
        if job["action"] == "profile":
            profile = job.get("profiling")
            if (not isinstance(profile, dict) or profile.get("tool") != "msprof op"
                    or not isinstance(profile.get("kernel_name"), str)
                    or not profile["kernel_name"].strip() or not valid_cases
                    or isinstance(job.get("repeats"), bool)
                    or not isinstance(job.get("repeats"), int)
                    or job["repeats"] < 1
                    or isinstance(job.get("round"), bool)
                    or not isinstance(job.get("round"), int)):
                raise JobError("request_error", "invalid msprof profile request")

    def _files_and_job(self, job: dict) -> tuple[list[tuple[str, Path]], dict]:
        source_names = {"candidate.py": job.get("candidate"),
                        "baseline.py": job.get("baseline"),
                        "cases.jsonl": job.get("case_spec")}
        if any(not isinstance(value, str) or not Path(value).is_file()
               for value in source_names.values()):
            raise JobError("staging_error", "a required benchmark asset is missing")
        files = [(name, Path(value)) for name, value in source_names.items()]
        files.append(("runner.py", self.runner))
        supplement = Path(job["baseline"]).with_suffix(".json")
        if supplement.is_file() and supplement != Path(job["case_spec"]):
            files.append(("baseline.json", supplement))
        if job["action"] == "profile":
            files.extend((("profile_a3.py", self.profiler),
                          ("batch_profile_a3.py", self.batch_profiler)))
        if any(not path.is_file() for _name, path in files):
            raise JobError("staging_error", "a required harness file is missing")
        remote = json.loads(json.dumps(job))
        remote.update(candidate="candidate.py", baseline="baseline.py",
                      case_spec="cases.jsonl", device=0,
                      logical_device=0)
        if job["action"] == "profile":
            remote["profiling"]["driver"] = "profile_a3.py"
        return files, remote

    @staticmethod
    def _archive(path: Path, files: list[tuple[str, Path]], job: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        job_path = path.parent / ".job.json"
        job_path.write_text(json.dumps(job, sort_keys=True, separators=(",", ":")) + "\n")
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
            for name, source in sorted(files + [("job.json", job_path)]):
                info = archive.gettarinfo(str(source), arcname=name)
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        job_path.unlink()

    @staticmethod
    def _remote_script(archive: str, archive_sha: str, run_root: str,
                       timeout: int, job: dict) -> str:
        kernel = job.get("profiling", {}).get("kernel_name", "")
        q = {key: shlex.quote(str(value)) for key, value in {
            "archive": archive, "archive_sha": archive_sha,
            "run_root": run_root, "timeout": timeout, "kernel": kernel,
        }.items()}
        if job["action"] == "profile":
            command = (f'python "$work/batch_profile_a3.py" --job "$work/job.json" '
                       f'--runner "$work/runner.py" --profiler "$work/profile_a3.py" '
                       f'--output "$work/profile" --response "$work/response.json" '
                       f'--kernel-name {q["kernel"]}')
            merge = '''python - <<'PY'
import json
from pathlib import Path
response=json.loads(Path("response.json").read_text())
evidence=Path("profile/evidence.json")
if evidence.is_file(): response["profile"]=json.loads(evidence.read_text())
print("BZ_PRODUCTION_RESULT="+json.dumps(response,sort_keys=True,separators=(",",":")))
PY'''
        else:
            command = 'python "$work/runner.py" --job "$work/job.json" --output "$work/response.json"'
            merge = '''printf 'BZ_PRODUCTION_RESULT='; tr -d '\\n' <response.json; printf '\\n' '''
        return f'''set -euo pipefail
archive={q["archive"]}; run_root={q["run_root"]}
test "$(sha256sum "$archive" | cut -d' ' -f1)" = {q["archive_sha"]} || {{ echo payload-digest-mismatch >&2; exit 91; }}
work="$run_root/work"; rm -rf "$work"; mkdir -p "$work"
tar -xf "$archive" -C "$work"; cd "$work"
set +e
timeout --signal=TERM --kill-after=10 {q["timeout"]} {command}
rc=$?
set -e
test "$rc" -ne 124 && test "$rc" -ne 137 || exit 124
test -f response.json || exit "$rc"
{merge}
exit 0'''

    @staticmethod
    def _complete(response: CommandResult, handle: str | None, identity: dict,
                  placement: dict, request_sha: str, job: dict,
                  remote_root: str, placements_sha256: str,
                  execution_provenance: dict[str, str]) -> dict:
        output = response.stdout + response.stderr
        if response.returncode in (124, 137):
            return {"status": "runtime_error", "failure_type": "runtime_error",
                    "diagnostics": _bounded(output), "handle": handle,
                    **identity, "placement": placement}
        if response.returncode:
            lowered = output.lower()
            if "digest-mismatch" in lowered:
                failure = "digest_mismatch"
            elif any(x in lowered for x in ("device", "npu", "davinci")):
                failure = "device_error"
            elif job["action"] == "profile":
                failure = "profile_tool_error"
            else:
                failure = "transport_error"
            raise JobError(failure, _bounded(output), handle)
        remote_identity = {**identity, "device": 0}
        result = _parse_result(response.stdout, remote_identity)
        result["device"] = identity["device"]
        result.update(handle=handle, placement=placement,
                      artifacts={"request_digest": request_sha,
                                 "placements_sha256": placements_sha256,
                                 "execution_provenance": execution_provenance,
                                 "remote_run_root": f"{remote_root}/runs/{request_sha}"})
        if job["action"] == "profile":
            result["artifacts"]["remote_profile_evidence"] = (
                result["artifacts"]["remote_run_root"] + "/work/profile/evidence.json")
            if result["status"] == "ok":
                _validate_profile_evidence(result, job)
        if result["status"] == "infrastructure_error":
            result["failure_type"] = ("profile_tool_error" if job["action"] == "profile"
                                      else "remote_infrastructure_error")
        else:
            result["failure_type"] = ("success" if result["status"] == "ok"
                                      else result["status"])
        return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--placements-json", type=Path, required=True)
    parser.add_argument("--cpl-remote-sha256", required=True)
    parser.add_argument(
        "--remote-root", required=True,
        help="existing writable remote staging root; run artifacts use its runs/ child",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        placements = json.loads(args.placements_json.read_text())
        here = Path(__file__).resolve().parent
        client = BzA3JobClient(
            GlobalCplRemoteTransport(args.cpl_remote_sha256, args.remote_root),
            args.state_dir, placements,
            runner=here / "a3_benchmark_runner.py", profiler=here / "profile_a3.py",
            batch_profiler=here / "batch_profile_a3.py",
            remote_root=args.remote_root,
        )
        job = json.load(sys.stdin)
        result = client.run(job, args.timeout)
    except (OSError, ValueError, json.JSONDecodeError, JobError) as exc:
        failure = exc.failure_type if isinstance(exc, JobError) else "request_error"
        result = {"status": "infrastructure_error", "failure_type": failure,
                  "diagnostics": str(exc), "handle": None}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
