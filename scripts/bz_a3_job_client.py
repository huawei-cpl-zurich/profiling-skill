#!/usr/bin/env python3
"""Production JSON job client for native BZ-A3 experiment workloads."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
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


PROFILES = {"bz-a3-1", "bz-a3-2"}
RESULTS = {"ok", "compile_error", "runtime_error", "correctness_error",
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
        match = re.search(
            r"\b(remote:(?:gz-a3|bz-a3-[12]):job:[A-Za-z0-9_.-]+)\b", output)
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


def validate_runtime_activate(value: object) -> str:
    if (not isinstance(value, str) or len(value) < 2 or len(value) > 240
            or not re.fullmatch(r"/[A-Za-z0-9._/-]+", value)
            or str(PurePosixPath(value)) != value
            or any(part in {".", ".."} for part in PurePosixPath(value).parts)):
        raise JobError(
            "request_error", "runtime activation must be a normalized absolute path")
    return value


def _handle(output: str, profile: str) -> str | None:
    match = re.search(
        rf"\b(remote:{re.escape(profile)}:job:[A-Za-z0-9_.-]+)\b", output)
    return match.group(1) if match else None


def _nonterminal(output: str) -> bool:
    states = ("queued", "dispatching", "running", "reconnecting",
              "observation-unavailable")
    return any(
        marker in output
        for state in states
        for marker in (f'"state": "{state}"', f"REMOTE_STATE={state}")
    )


class RemoteTransport:
    """Global cpl-remote retained execution boundary; replaceable in tests."""

    def __init__(self, remote: list[str], runtime_activate: str,
                 invoke: Callable = _invoke):
        self.remote = remote
        self.runtime_activate = validate_runtime_activate(runtime_activate)
        self.invoke = invoke

    def upload(self, profile: str, source: Path, destination: str, timeout: int) -> None:
        result = self.invoke(
            self.remote + ["upload", profile, str(source), destination], timeout)
        if result.returncode:
            raise JobError("staging_error", _bounded(result.stdout + result.stderr))

    @staticmethod
    def _metadata(result: CommandResult, handle: str | None = None) -> dict:
        try:
            value = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise JobError("observer_error" if handle else "transport_error",
                           "cpl-remote returned invalid JSON", handle) from exc
        if not isinstance(value, dict):
            raise JobError("observer_error" if handle else "transport_error",
                           "cpl-remote returned invalid metadata", handle)
        return value

    def _logs(self, handle: str, stream: str, timeout: int) -> str:
        result = self.invoke(
            self.remote + ["--json", "logs", handle, "--stream", stream], timeout)
        metadata = self._metadata(result, handle)
        if result.returncode or not isinstance(metadata.get("content"), str):
            raise JobError("observer_error", "could not retrieve retained job logs", handle)
        return metadata["content"]

    def _completed(self, result: CommandResult, handle: str, deadline: float) -> CommandResult:
        metadata = self._metadata(result, handle)
        state = metadata.get("state")
        if state in {"running", "reconnecting", "observation-unavailable",
                     "queued", "dispatching"}:
            raise JobError("observer_error", "retained job is not terminal", handle)
        if state not in {"completed", "failed", "cancelled"}:
            raise JobError("observer_error", "retained job returned an unknown state", handle)
        exit_code = metadata.get("exit")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            exit_code = 0 if state == "completed" else 1
        return CommandResult(
            exit_code, self._logs(handle, "stdout", self._remaining(deadline, handle)),
            self._logs(handle, "stderr", self._remaining(deadline, handle)))

    @staticmethod
    def _remaining(deadline: float, handle: str) -> int:
        remaining = math.ceil(deadline - time.monotonic())
        if remaining < 1:
            raise JobError("observer_error", "BZ job deadline exhausted", handle)
        return remaining

    def execute(self, profile: str, device: int, remote_cwd: str, script: str,
                timeout: int) -> tuple[CommandResult, str | None]:
        deadline = time.monotonic() + timeout
        payload = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                   f"source {shlex.quote(self.runtime_activate)}\n"
                   f"export ASCEND_RT_VISIBLE_DEVICES={device}\n"
                   "export DEVICE_ID=0\n" + script)
        with tempfile.NamedTemporaryFile("w", suffix=".sh") as command_file:
            command_file.write(payload)
            command_file.flush()
            result = self.invoke(self.remote + [
                "--json", "run", profile, "--file", command_file.name,
                "--cwd", remote_cwd, "--timeout", str(timeout),
            ], timeout)
        handle = _handle(result.stdout + result.stderr, profile)
        if handle is None:
            raise JobError("transport_error", "cpl-remote run returned no durable handle",
                           dispatch_uncertain=True)
        if _nonterminal(result.stdout + result.stderr):
            remaining = math.ceil(deadline - time.monotonic())
            if remaining < 1:
                raise JobError("observer_error", "BZ job deadline exhausted", handle)
            try:
                result = self.observe(profile, handle, remaining)
            except JobError as exc:
                raise JobError("observer_error", str(exc), exc.handle or handle) from exc
        else:
            result = self._completed(result, handle, deadline)
        return result, handle

    def observe(self, profile: str, handle: str, timeout: int) -> CommandResult:
        deadline = time.monotonic() + timeout
        result = self.invoke(
            self.remote + ["--json", "observe", handle, "--wait", "--timeout",
                           str(timeout)],
            timeout,
        )
        if _nonterminal(result.stdout + result.stderr):
            raise JobError("observer_error", "retained job is not terminal", handle)
        return self._completed(result, handle, deadline)


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


def validate_placements(value: object, profiles: set[str] = PROFILES) -> dict[int, dict]:
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
                or set(placement) != {"profile", "device"}
                or placement.get("profile") not in profiles
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
    kernel = job["profiling"]["kernel_name"]
    repeats = job["repeats"]
    if (not isinstance(evidence, dict) or evidence.get("status") != "success"
            or evidence.get("profiler") != "msprof-op"
            or evidence.get("kernel_name") != kernel
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
                   or capture.get("kernel_name") != kernel
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
    def __init__(self, transport: RemoteTransport, state_dir: Path,
                 placements: object, *, runner: Path, profiler: Path,
                 batch_profiler: Path, remote_root: str,
                 allowed_profiles: set[str] = PROFILES):
        self.transport = transport
        self.state_dir = state_dir
        self.placements = validate_placements(placements, allowed_profiles)
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
                                     "runtime_activate": self.transport.runtime_activate,
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
                result = record.get("result")
                if not isinstance(result, dict):
                    raise JobError("request_error", "completed receipt is invalid")
                return result
            if dispatch.is_file():
                record = json.loads(dispatch.read_text())
                if record.get("request_sha256") != request_sha:
                    raise JobError("request_error", "dispatch receipt request mismatch")
                handle = record.get("handle")
                if (not isinstance(handle, str)
                        or not handle.startswith(f"remote:{placement['profile']}:job:")):
                    raise JobError("request_error", "dispatch receipt handle mismatch")
                response = self.transport.observe(
                    placement["profile"], handle, self._remaining(deadline, handle))
            else:
                self._archive(archive, stage_files, remote_job)
                # cpl-remote's rsync transport does not create destination
                # parents. The configured root is therefore a pre-provisioned
                # writable staging directory, while the content-addressed
                # filename keeps concurrent requests collision-free.
                remote_archive = f"{self.remote_root}/payload-{request_sha}.tar"
                self.transport.upload(placement["profile"], archive, remote_archive,
                                      self._remaining(deadline))
                run_root = f"{self.remote_root}/runs/{request_sha}"
                script = self._remote_script(
                    remote_archive, _sha(archive), run_root,
                    self._workload_timeout(self._remaining(deadline)), job,
                )
                try:
                    response, handle = self.transport.execute(
                        placement["profile"], placement["device"],
                        self.remote_root, script,
                        self._remaining(deadline),
                    )
                except JobError as exc:
                    if exc.handle:
                        _write_json(dispatch, {"protocol_version": 1,
                                    "request_sha256": request_sha,
                                    "handle": exc.handle})
                    raise
                if handle:
                    _write_json(dispatch, {"protocol_version": 1,
                                "request_sha256": request_sha, "handle": handle})
            result = self._complete(response, handle, identity, placement,
                                    request_sha, job, self.remote_root,
                                    self.placements_sha256)
            _write_json(completed, {"protocol_version": 1,
                        "request_sha256": request_sha, "result": result})
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
                  remote_root: str, placements_sha256: str) -> dict:
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


def _command(value: str, name: str) -> list[str]:
    parsed = json.loads(value)
    if (not isinstance(parsed, list) or not parsed
            or not all(isinstance(item, str) and item for item in parsed)):
        raise ValueError(f"{name} must be a non-empty JSON string array")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--placements-json", type=Path, required=True)
    parser.add_argument("--remote-json", default='["cpl-remote"]')
    parser.add_argument("--runtime-activate", required=True)
    parser.add_argument(
        "--remote-root", required=True,
        help="existing writable remote staging root; run artifacts use its runs/ child",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        remote = _command(args.remote_json, "--remote-json")
        placements = json.loads(args.placements_json.read_text())
        here = Path(__file__).resolve().parent
        client = BzA3JobClient(
            RemoteTransport(remote, args.runtime_activate), args.state_dir, placements,
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
