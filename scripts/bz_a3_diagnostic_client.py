#!/usr/bin/env python3
"""Host-owned one-shot correctness jobs for the native-only BZ-A3 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROFILES = {"bz-a3-1", "bz-a3-2"}
REMOTE_RESULTS = {
    "ok", "compile_error", "runtime_error", "correctness_error", "infrastructure_error",
}


class DiagnosticError(RuntimeError):
    def __init__(self, failure_type: str, message: str, handle: str | None = None,
                 *, invocation_timeout: bool = False,
                 dispatch_uncertain: bool = False):
        self.failure_type, self.handle = failure_type, handle
        self.invocation_timeout = invocation_timeout
        self.dispatch_uncertain = dispatch_uncertain
        super().__init__(message)


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _run(argv: list[str], timeout: int) -> CommandResult:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        partial = "".join(_text(value) for value in (exc.stdout, exc.stderr))
        match = re.search(r"\b(remote:bz-a3-[12]:job:[A-Za-z0-9_.-]+)\b", partial)
        handle = match.group(1) if match else None
        raise DiagnosticError("observer_error" if handle else "transport_error",
                              f"transport timed out after {timeout}s", handle,
                              invocation_timeout=True) from exc
    except OSError as exc:
        raise DiagnosticError("transport_error", f"transport unavailable: {exc}") from exc
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


class RemoteTransport:
    """Global cpl-remote retained execution boundary; replaceable in tests."""

    def __init__(self, remote: list[str], invoke: Callable = _run):
        self.remote, self.invoke = remote, invoke

    def upload(self, profile: str, source: Path, destination: str, timeout: int) -> None:
        result = self.invoke(self.remote + ["upload", profile, str(source), destination], timeout)
        if result.returncode:
            raise DiagnosticError("staging_error", _bounded(result.stdout + result.stderr))

    @staticmethod
    def _metadata(result: CommandResult, handle: str | None = None) -> dict:
        try:
            value = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise DiagnosticError(
                "observer_error" if handle else "transport_error",
                "cpl-remote returned invalid JSON", handle) from exc
        if not isinstance(value, dict):
            raise DiagnosticError(
                "observer_error" if handle else "transport_error",
                "cpl-remote returned invalid metadata", handle)
        return value

    def _logs(self, handle: str, stream: str, timeout: int) -> str:
        result = self.invoke(
            self.remote + ["--json", "logs", handle, "--stream", stream], timeout)
        metadata = self._metadata(result, handle)
        if result.returncode or not isinstance(metadata.get("content"), str):
            raise DiagnosticError(
                "observer_error", "could not retrieve retained job logs", handle)
        return metadata["content"]

    def _completed(self, result: CommandResult, handle: str, deadline: float) -> CommandResult:
        metadata = self._metadata(result, handle)
        state = metadata.get("state")
        if state in {"running", "reconnecting", "observation-unavailable",
                     "queued", "dispatching"}:
            raise DiagnosticError("observer_error", "retained job is not terminal", handle)
        if state not in {"completed", "failed", "cancelled"}:
            raise DiagnosticError(
                "observer_error", "retained job returned an unknown state", handle)
        exit_code = metadata.get("exit")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            exit_code = 0 if state == "completed" else 1
        return CommandResult(
            exit_code, self._logs(handle, "stdout", _remaining(deadline, handle)),
            self._logs(handle, "stderr", _remaining(deadline, handle)))

    def execute(self, profile: str, device: int, remote_cwd: str, script: str,
                timeout: int) -> tuple[CommandResult, str | None]:
        deadline = time.monotonic() + timeout
        payload = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                   f"export ASCEND_RT_VISIBLE_DEVICES={device}\n"
                   "export DEVICE_ID=0\n" + script)
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".sh") as command_file:
                command_file.write(payload)
                command_file.flush()
                result = self.invoke(self.remote + [
                    "--json", "run", profile, "--file", command_file.name,
                    "--cwd", remote_cwd, "--timeout", str(timeout),
                ], timeout)
        except DiagnosticError as exc:
            if exc.invocation_timeout and not exc.handle:
                raise DiagnosticError(
                    exc.failure_type, str(exc), dispatch_uncertain=True,
                    invocation_timeout=True) from exc
            raise
        handle = _handle(result.stdout + result.stderr, profile)
        if handle is None:
            raise DiagnosticError(
                "transport_error", "cpl-remote run returned no durable handle",
                dispatch_uncertain=True)
        if _nonterminal(result.stdout + result.stderr):
            try:
                result = self.observe(profile, handle, _remaining(deadline, handle))
            except DiagnosticError as exc:
                if exc.handle:
                    raise
                raise DiagnosticError("observer_error", str(exc), handle) from exc
        else:
            result = self._completed(result, handle, deadline)
        return result, handle

    def observe(self, profile: str, handle: str, timeout: int) -> CommandResult:
        deadline = time.monotonic() + timeout
        result = self.invoke(
            self.remote + ["--json", "observe", handle, "--wait", "--timeout",
                           str(timeout)], timeout)
        if _nonterminal(result.stdout + result.stderr):
            raise DiagnosticError("observer_error", "retained job is not terminal", handle)
        return self._completed(result, handle, deadline)


def _bounded(value: str, limit: int = 64 * 1024) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "\n...[diagnostic truncated]"


def _remaining(deadline: float, handle: str | None = None) -> int:
    remaining = math.ceil(deadline - time.monotonic())
    if remaining < 1:
        raise DiagnosticError("observer_error" if handle else "transport_error",
                              "BZ diagnostic deadline exhausted", handle)
    return remaining


def _workload_timeout(outer_timeout: int) -> int:
    """Reserve kill-after and response grace inside the caller's deadline."""
    if outer_timeout <= 25:
        raise DiagnosticError("transport_error",
                              "insufficient remaining budget for workload and response grace")
    return outer_timeout - 25


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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_receipt(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _archive(output: Path, files: list[tuple[Path, str]]) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as tar:
        for source, name in sorted(files, key=lambda item: item[1]):
            info = tar.gettarinfo(str(source), arcname=name)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as stream:
                tar.addfile(info, stream)
    return _sha(output)


def _safe_id(value: object, name: str) -> str:
    if (not isinstance(value, (str, int)) or len(str(value)) > 40
            or str(value) in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(value))):
        raise DiagnosticError("request_error", f"unsafe or missing {name}")
    return str(value)


def _result(stdout: str) -> dict:
    matches = [line.removeprefix("BZ_DIAGNOSTIC_RESULT=") for line in stdout.splitlines()
               if line.startswith("BZ_DIAGNOSTIC_RESULT=")]
    if len(matches) != 1:
        raise DiagnosticError("transport_error", "remote job did not return exactly one structured result")
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise DiagnosticError("transport_error", "remote result is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("status") not in REMOTE_RESULTS:
        raise DiagnosticError("transport_error", "remote result has an invalid status")
    value["diagnostics"] = _bounded(str(value.get("diagnostics", "")))
    return value


class BzA3DiagnosticClient:
    def __init__(self, transport: RemoteTransport, state_dir: Path, remote_root: str = "/home/m00933363/.profiling-skill/diagnostic"):
        self.transport, self.state_dir, self.remote_root = transport, state_dir, remote_root.rstrip("/")

    def resume(self, requests: list[dict], handle: str, observe_timeout: int) -> dict:
        """Observe the one candidate request whose durable receipt owns handle."""
        if len(requests) != 1:
            return {"status": "infrastructure_error", "failure_type": "request_error",
                    "diagnostics": "resume requires one exact terminal request",
                    "handle": handle}
        matches = []
        for request in requests:
            try:
                local = (self.state_dir / _safe_id(request.get("campaign"), "campaign")
                         / _safe_id(request.get("wave"), "wave")
                         / _safe_id(request.get("cell"), "cell"))
                records = []
                dispatch = local / "dispatch.json"
                completed = local / "completed.json"
                if dispatch.is_file():
                    records.append(json.loads(dispatch.read_text()))
                if completed.is_file():
                    records.append(json.loads(completed.read_text()).get("result", {}))
                if any(record.get("handle") == handle for record in records
                       if isinstance(record, dict)):
                    matches.append(request)
            except (OSError, json.JSONDecodeError, DiagnosticError):
                continue
        if not matches:
            # The generic terminal receipt is authoritative for its selected
            # attempt. A supplied handle can reconstruct that one request when
            # the lower-level process died before persisting dispatch state.
            selected = requests[0]
            campaign = _safe_id(selected.get("campaign"), "campaign")
            wave = _safe_id(selected.get("wave"), "wave")
            cell = _safe_id(selected.get("cell"), "cell")
            selected_local = self.state_dir / campaign / wave / cell
            foreign = False
            for pattern in ("*/*/*/dispatch.json", "*/*/*/completed.json"):
                for path in self.state_dir.glob(pattern):
                    if path.parent == selected_local:
                        continue
                    try:
                        record = json.loads(path.read_text())
                        owner = (record.get("result", {}).get("handle")
                                 if path.name == "completed.json"
                                 else record.get("handle"))
                        foreign = owner == handle
                    except (OSError, json.JSONDecodeError, AttributeError):
                        continue
                    if foreign:
                        break
                if foreign:
                    break
            if (not foreign
                    and handle.startswith(
                        f"remote:{selected.get('profile')}:job:")):
                matches.append(selected)
        if len(matches) != 1:
            return {"status": "infrastructure_error", "failure_type": "request_error",
                    "diagnostics": "retained handle has no unique durable BZ receipt",
                    "handle": handle}
        return self.run({**matches[0], "retained_handle": handle,
                         "observe_timeout": observe_timeout})

    def run(self, request: object) -> dict:
        if not isinstance(request, dict):
            return {"status": "infrastructure_error", "failure_type": "request_error",
                    "diagnostics": "request must be a JSON object", "handle": None}
        benchmark = request.get("benchmark", "matmul")
        identity = {key: request.get(key) for key in (
            "campaign", "wave", "cell", "profile", "device")}
        identity["benchmark"] = benchmark
        handle = None
        dispatch_receipt = None
        request_sha = None
        try:
            profile = request.get("profile")
            device = request.get("device")
            if profile not in PROFILES or isinstance(device, bool) or not isinstance(device, int) or device < 0:
                raise DiagnosticError("request_error", "profile must be bz-a3-1/2 and physical device non-negative")
            if benchmark not in {"matmul", "bsa", "gdn"}:
                raise DiagnosticError("request_error", "benchmark must be matmul, bsa, or gdn")
            campaign = _safe_id(request.get("campaign"), "campaign")
            wave = _safe_id(request.get("wave"), "wave")
            cell = _safe_id(request.get("cell"), "cell")
            timeout = request.get("timeout", 180)
            if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 26 or timeout > 540:
                raise DiagnosticError("request_error", "timeout must be 26..540 seconds")
            observe_timeout = request.get("observe_timeout")
            if (observe_timeout is not None
                    and (isinstance(observe_timeout, bool)
                         or not isinstance(observe_timeout, int)
                         or observe_timeout < 1 or observe_timeout > timeout)):
                raise DiagnosticError("request_error",
                                      "observe_timeout must be 1..timeout seconds")
            retained_handle = request.get("retained_handle")
            if (retained_handle is not None
                    and (not isinstance(retained_handle, str)
                         or not retained_handle.startswith(
                             f"remote:{profile}:job:"))):
                raise DiagnosticError("request_error",
                                      "retained_handle must match the selected profile")
            deadline = time.monotonic() + (observe_timeout or timeout)
            path_names = ("candidate", "candidate_manifest", "baseline", "case_spec", "runner")
            if any(not isinstance(request.get(name, ""), str) for name in path_names):
                raise DiagnosticError("request_error", "asset paths must be JSON strings")
            paths = {name: Path(request.get(name, "")) for name in path_names}
            supplementary = request.get("supplementary_assets", {})
            reserved = {"baseline.py", "cases.jsonl", "runner.py", "candidate.py",
                        "candidate.manifest.json", "AGENTS.md"}
            if (not isinstance(supplementary, dict)
                    or any(not isinstance(name, str) or not name or name in {".", ".."}
                           or name in reserved or Path(name).name != name
                           or not isinstance(source, str)
                           for name, source in supplementary.items())):
                raise DiagnosticError("request_error", "invalid supplementary asset map")
            supplementary_paths = {name: Path(source)
                                   for name, source in supplementary.items()}
            missing_submission = [name for name in ("candidate", "candidate_manifest") if not paths[name].is_file()]
            if missing_submission:
                return {"status": "submission_error", "failure_type": "missing_submission",
                        "diagnostics": "missing " + ", ".join(missing_submission), **identity}
            if any(not paths[name].is_file() for name in ("baseline", "case_spec", "runner")):
                raise DiagnosticError("staging_error", "a frozen common asset is missing")
            if any(not path.is_file() for path in supplementary_paths.values()):
                raise DiagnosticError("staging_error", "a frozen supplementary asset is missing")
            cases = request.get("cases", list(range(7)))
            if not isinstance(cases, list) or not cases or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in cases):
                raise DiagnosticError("request_error", "cases must be a non-empty integer list")

            local = self.state_dir / campaign / wave / cell
            common_tar, candidate_tar = local / "common.tar", local / "candidate.tar"
            common_sha = _archive(common_tar, [(paths["baseline"], "baseline.py"),
                                                (paths["case_spec"], "cases.jsonl"),
                                                (paths["runner"], "runner.py"),
                                                *[(supplementary_paths[name], name)
                                                  for name in sorted(supplementary_paths)]])
            candidate_sha = _archive(candidate_tar, [(paths["candidate"], "candidate.py"),
                                                      (paths["candidate_manifest"], "candidate.manifest.json")])
            common_remote = f"/home/m00933363/diagnostic-common-{common_sha}.tar"
            run_root = f"{self.remote_root}/runs/{campaign}/{wave}/{cell}"
            candidate_remote = f"/home/m00933363/diagnostic-candidate-{campaign}-{wave}-{cell}-{candidate_sha}.tar"
            common_receipt = self.state_dir / "common" / profile / f"{common_sha}.verified"
            dispatch_receipt = local / "dispatch.json"
            completed_receipt = local / "completed.json"
            request_sha = _json_sha({"identity": identity, "timeout": timeout, "cases": cases,
                                     "common_sha256": common_sha, "candidate_sha256": candidate_sha,
                                     "tolerances": request.get("tolerances", {"rtol": 2e-2, "atol": 2e-2})})
            if retained_handle is not None and not dispatch_receipt.is_file():
                _write_receipt(dispatch_receipt, {"protocol_version": 1,
                               "request_sha256": request_sha,
                               "handle": retained_handle})
            if completed_receipt.is_file():
                completed_record = json.loads(completed_receipt.read_text())
                if (completed_record.get("request_sha256") != request_sha
                        or not isinstance(completed_record.get("result"), dict)
                        or (retained_handle is not None
                            and completed_record["result"].get("handle")
                            != retained_handle)):
                    raise DiagnosticError("request_error", "completed receipt does not match request")
                return completed_record["result"]
            had_dispatch_receipt = dispatch_receipt.is_file()
            if had_dispatch_receipt:
                try:
                    prior = json.loads(dispatch_receipt.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise DiagnosticError("request_error", f"invalid dispatch receipt: {exc}") from exc
                if (not isinstance(prior, dict) or prior.get("request_sha256") != request_sha
                        or not isinstance(prior.get("handle"), str)
                        or (retained_handle is not None
                            and prior["handle"] != retained_handle)):
                    raise DiagnosticError("request_error", "dispatch receipt does not match request")
                handle = prior["handle"]
                completed = self.transport.observe(profile, handle, _remaining(deadline, handle))
                observed_output = completed.stdout + completed.stderr
                if (completed.returncode
                        and "common-digest-mismatch" in observed_output.lower()):
                    if retained_handle is not None:
                        return {"status": "infrastructure_error",
                                "failure_type": "digest_mismatch",
                                "diagnostics": _bounded(observed_output),
                                "handle": handle, **identity}
                    common_receipt.unlink(missing_ok=True)
                    dispatch_receipt.unlink(missing_ok=True)
                    handle = None
                else:
                    return self._completed(completed, handle, identity, common_sha, candidate_sha,
                                           run_root, common_remote, common_receipt,
                                           dispatch_receipt, completed_receipt, request_sha)
            if observe_timeout is not None and not had_dispatch_receipt:
                raise DiagnosticError("request_error",
                                      "observe_timeout requires a retained dispatch receipt")
            # A structured prior result proves the remote script verified the common digest.
            common_cached = common_receipt.is_file()
            if not common_cached:
                self.transport.upload(profile, common_tar, common_remote, _remaining(deadline))
            self.transport.upload(profile, candidate_tar, candidate_remote, _remaining(deadline))
            job = {"protocol_version": 1, "benchmark": benchmark, "action": "check", "device": 0,
                   "logical_device": 0, "candidate": "candidate.py", "baseline": "baseline.py",
                   "case_spec": "cases.jsonl", "cases": cases, "scope": "diagnostic",
                   "tolerances": request.get("tolerances", {"rtol": 2e-2, "atol": 2e-2})}
            operation_timeout = _remaining(deadline)
            script = _remote_script(common_remote, common_sha, candidate_remote, candidate_sha,
                                    run_root, _workload_timeout(operation_timeout), job)
            completed, handle = self.transport.execute(
                profile, device, self.remote_root, script, operation_timeout,
            )
            if handle:
                _write_receipt(dispatch_receipt, {"protocol_version": 1,
                               "request_sha256": request_sha, "handle": handle})
            output = completed.stdout + completed.stderr
            if (common_cached and completed.returncode
                    and "common-digest-mismatch" in output.lower()):
                common_receipt.unlink(missing_ok=True)
                dispatch_receipt.unlink(missing_ok=True)
                handle = None
                self.transport.upload(profile, common_tar, common_remote, _remaining(deadline))
                operation_timeout = _remaining(deadline)
                retry_script = _remote_script(
                    common_remote, common_sha, candidate_remote, candidate_sha,
                    run_root, _workload_timeout(operation_timeout), job,
                )
                completed, handle = self.transport.execute(
                    profile, device, self.remote_root, retry_script,
                    operation_timeout)
            if handle:
                _write_receipt(dispatch_receipt, {"protocol_version": 1,
                               "request_sha256": request_sha, "handle": handle})
            if completed.returncode and "common-digest-mismatch" in (
                    completed.stdout + completed.stderr).lower():
                dispatch_receipt.unlink(missing_ok=True)
                handle = None
            return self._completed(completed, handle, identity, common_sha, candidate_sha,
                                   run_root, common_remote, common_receipt, dispatch_receipt,
                                   completed_receipt, request_sha)
        except DiagnosticError as exc:
            if dispatch_receipt is not None and request_sha is not None and (exc.handle or handle):
                _write_receipt(dispatch_receipt, {"protocol_version": 1,
                               "request_sha256": request_sha, "handle": exc.handle or handle})
            return {"status": "infrastructure_error", "failure_type": exc.failure_type,
                    "diagnostics": _bounded(str(exc)), "handle": exc.handle or handle,
                    "invocation_timeout": exc.invocation_timeout,
                    "dispatch_uncertain": exc.dispatch_uncertain, **identity}
        except (OSError, ValueError, tarfile.TarError) as exc:
            return {"status": "infrastructure_error", "failure_type": "staging_error",
                    "diagnostics": _bounded(str(exc)), "handle": handle, **identity}

    @staticmethod
    def _completed(completed: CommandResult, handle: str | None, identity: dict,
                   common_sha: str, candidate_sha: str, run_root: str, common_remote: str,
                   common_receipt: Path, dispatch_receipt: Path,
                   completed_receipt: Path, request_sha: str) -> dict:
        output = completed.stdout + completed.stderr
        if completed.returncode in (124, 137):
            result = {"status": "candidate_timeout", "failure_type": "candidate_timeout",
                      "diagnostics": _bounded(output), "handle": handle, **identity}
            _write_receipt(completed_receipt, {"protocol_version": 1,
                           "request_sha256": request_sha, "result": result})
            dispatch_receipt.unlink(missing_ok=True)
            return result
        if completed.returncode:
            lowered = output.lower()
            if "digest-mismatch" in lowered:
                failure = "digest_mismatch"
            elif any(x in lowered for x in ("device", "davinci", "npu")):
                failure = "device_error"
            else:
                failure = "transport_error"
            raise DiagnosticError(failure, _bounded(output), handle)
        result = _result(completed.stdout)
        common_receipt.parent.mkdir(parents=True, exist_ok=True)
        common_receipt.write_text(common_remote + "\n")
        result.update(identity, handle=handle, artifacts={"common_sha256": common_sha,
                      "candidate_sha256": candidate_sha, "remote_run_root": run_root})
        if result["status"] == "infrastructure_error":
            result["failure_type"] = str(result.get("failure_type") or "remote_infrastructure_error")
        else:
            result["failure_type"] = "success" if result["status"] == "ok" else result["status"]
        result.pop("host_elapsed_us", None)
        for evidence in result.get("case_evidence", []):
            evidence.pop("host_elapsed_us", None)
        _write_receipt(completed_receipt, {"protocol_version": 1,
                       "request_sha256": request_sha, "result": result})
        dispatch_receipt.unlink(missing_ok=True)
        return result


def _remote_script(common: str, common_sha: str, candidate: str, candidate_sha: str,
                   run_root: str, timeout: int, job: dict) -> str:
    values = {"common": common, "common_sha": common_sha, "candidate": candidate,
              "candidate_sha": candidate_sha, "run_root": run_root,
              "job": json.dumps(job, sort_keys=True, separators=(",", ":")), "timeout": timeout}
    q = {key: shlex.quote(str(value)) for key, value in values.items()}
    return f'''set -euo pipefail
common={q["common"]}; candidate={q["candidate"]}; run_root={q["run_root"]}
test "$(sha256sum "$common" | cut -d' ' -f1)" = {q["common_sha"]} || {{ echo common-digest-mismatch >&2; exit 91; }}
test "$(sha256sum "$candidate" | cut -d' ' -f1)" = {q["candidate_sha"]} || {{ echo candidate-digest-mismatch >&2; exit 92; }}
work="$run_root/work"; rm -rf "$work"; mkdir -p "$work"
tar -xf "$common" -C "$work"; tar -xf "$candidate" -C "$work"
printf '%s\\n' {q["job"]} >"$work/job.json"
cd "$work"
set +e
timeout --signal=TERM --kill-after=10 {q["timeout"]} python "$work/runner.py" --job "$work/job.json" --output "$work/response.json"
rc=$?
set -e
test "$rc" -ne 124 && test "$rc" -ne 137 || exit 124
test -f "$work/response.json" || exit "$rc"
printf 'BZ_DIAGNOSTIC_RESULT='; tr -d '\\n' <"$work/response.json"; printf '\\n'
exit 0'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--remote-json", default='["cpl-remote"]')
    args = parser.parse_args()
    try:
        remote = json.loads(args.remote_json)
        if (not isinstance(remote, list) or not remote
                or not all(isinstance(x, str) and x for x in remote)):
            raise ValueError("remote command must be a non-empty JSON string array")
        request = json.load(__import__("sys").stdin)
        result = BzA3DiagnosticClient(RemoteTransport(remote), args.state_dir).run(request)
    except (ValueError, json.JSONDecodeError) as exc:
        result = {"status": "infrastructure_error", "failure_type": "request_error", "diagnostics": str(exc)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
