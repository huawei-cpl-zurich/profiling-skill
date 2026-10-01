#!/usr/bin/env python3
"""Host-owned one-shot correctness jobs for the native-only BZ-A3 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROFILES = {"bz-a3-1", "bz-a3-2"}
CANDIDATE_RESULTS = {"ok", "compile_error", "runtime_error", "correctness_error"}


class DiagnosticError(RuntimeError):
    def __init__(self, failure_type: str, message: str, handle: str | None = None):
        self.failure_type, self.handle = failure_type, handle
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
        raise DiagnosticError("transport_error", f"transport timed out after {timeout}s") from exc
    except OSError as exc:
        raise DiagnosticError("transport_error", f"transport unavailable: {exc}") from exc
    return CommandResult(result.returncode, result.stdout, result.stderr)


class AdapterTransport:
    """Approved transfer and neutral-adapter boundary; replaceable in tests."""

    def __init__(self, remote: list[str], adapter: list[str], invoke: Callable = _run):
        self.remote, self.adapter, self.invoke = remote, adapter, invoke

    def upload(self, profile: str, source: Path, destination: str, timeout: int) -> None:
        result = self.invoke(self.remote + ["upload", profile, str(source), destination], timeout)
        if result.returncode:
            raise DiagnosticError("staging_error", _bounded(result.stdout + result.stderr))

    def execute(self, profile: str, device: int, operation: str, script: str,
                timeout: int) -> tuple[CommandResult, str | None]:
        argv = self.adapter + ["--profile", profile, "--operation", operation, "run",
                               "--native", "--runtime", "py311-torch", "--device", str(device),
                               "--timeout", str(timeout), "--", "bash", "-c", script]
        result = self.invoke(argv, timeout + 30)
        handle = _handle(result.stdout + result.stderr, profile)
        if handle and _nonterminal(result.stdout + result.stderr):
            result = self.observe(profile, handle, timeout)
        return result, handle

    def observe(self, profile: str, handle: str, timeout: int) -> CommandResult:
        result = self.invoke(self.adapter + ["--profile", profile, "observe", "--handle", handle], timeout + 30)
        if _nonterminal(result.stdout + result.stderr):
            raise DiagnosticError("observer_error", "retained job is not terminal", handle)
        return result


def _bounded(value: str, limit: int = 64 * 1024) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "\n...[diagnostic truncated]"


def _handle(output: str, profile: str) -> str | None:
    match = re.search(rf"\b({re.escape(profile)}:[A-Za-z0-9_.-]+)\b", output)
    return match.group(1) if match else None


def _nonterminal(output: str) -> bool:
    return any(marker in output for marker in (
        "CATLASS_VALIDATION_STATE=observation-unavailable",
        "CATLASS_VALIDATION_STATE=running",
        "BZ_A3_JOB_STATE=observation-unavailable",
    ))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if not isinstance(value, dict) or value.get("status") not in CANDIDATE_RESULTS:
        raise DiagnosticError("transport_error", "remote result has an invalid candidate status")
    value["diagnostics"] = _bounded(str(value.get("diagnostics", "")))
    return value


class BzA3DiagnosticClient:
    def __init__(self, transport: AdapterTransport, state_dir: Path, remote_root: str = "/home/m00933363/.profiling-skill/diagnostic"):
        self.transport, self.state_dir, self.remote_root = transport, state_dir, remote_root.rstrip("/")

    def run(self, request: dict) -> dict:
        identity = {key: request.get(key) for key in ("campaign", "wave", "cell", "profile", "device")}
        handle = None
        try:
            profile = request.get("profile")
            device = request.get("device")
            if profile not in PROFILES or isinstance(device, bool) or not isinstance(device, int) or device < 0:
                raise DiagnosticError("request_error", "profile must be bz-a3-1/2 and physical device non-negative")
            campaign = _safe_id(request.get("campaign"), "campaign")
            wave = _safe_id(request.get("wave"), "wave")
            cell = _safe_id(request.get("cell"), "cell")
            timeout = request.get("timeout", 180)
            if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1 or timeout > 540:
                raise DiagnosticError("request_error", "timeout must be 1..540 seconds")
            paths = {name: Path(request.get(name, "")) for name in
                     ("candidate", "candidate_manifest", "baseline", "case_spec", "runner")}
            missing_submission = [name for name in ("candidate", "candidate_manifest") if not paths[name].is_file()]
            if missing_submission:
                return {"status": "submission_error", "failure_type": "missing_submission",
                        "diagnostics": "missing " + ", ".join(missing_submission), **identity}
            if any(not paths[name].is_file() for name in ("baseline", "case_spec", "runner")):
                raise DiagnosticError("staging_error", "a frozen common asset is missing")
            cases = request.get("cases", list(range(7)))
            if not isinstance(cases, list) or not cases or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in cases):
                raise DiagnosticError("request_error", "cases must be a non-empty integer list")

            local = self.state_dir / campaign / wave / cell
            common_tar, candidate_tar = local / "common.tar", local / "candidate.tar"
            common_sha = _archive(common_tar, [(paths["baseline"], "baseline.py"),
                                                (paths["case_spec"], "cases.jsonl"),
                                                (paths["runner"], "runner.py")])
            candidate_sha = _archive(candidate_tar, [(paths["candidate"], "candidate.py"),
                                                      (paths["candidate_manifest"], "candidate.manifest.json")])
            common_remote = f"/home/m00933363/diagnostic-common-{common_sha}.tar"
            run_root = f"{self.remote_root}/runs/{campaign}/{wave}/{cell}"
            candidate_remote = f"/home/m00933363/diagnostic-candidate-{campaign}-{wave}-{cell}-{candidate_sha}.tar"
            common_receipt = self.state_dir / "common" / profile / f"{common_sha}.verified"
            # A structured prior result proves the remote script verified the common digest.
            if not common_receipt.is_file():
                self.transport.upload(profile, common_tar, common_remote, timeout)
            self.transport.upload(profile, candidate_tar, candidate_remote, timeout)
            job = {"protocol_version": 1, "benchmark": "matmul", "action": "check", "device": 0,
                   "logical_device": 0, "candidate": "candidate.py", "baseline": "baseline.py",
                   "case_spec": "cases.jsonl", "cases": cases, "scope": "diagnostic",
                   "tolerances": request.get("tolerances", {"rtol": 2e-2, "atol": 2e-2})}
            script = _remote_script(common_remote, common_sha, candidate_remote, candidate_sha,
                                    run_root, timeout, job)
            operation = f"diagnostic-{campaign}-{wave}-{cell}"[:80]
            completed, handle = self.transport.execute(profile, device, operation, script, timeout)
            output = completed.stdout + completed.stderr
            if completed.returncode == 124:
                return {"status": "candidate_timeout", "failure_type": "candidate_timeout",
                        "diagnostics": _bounded(output), "handle": handle, **identity}
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
            result["failure_type"] = "success" if result["status"] == "ok" else result["status"]
            result.pop("host_elapsed_us", None)
            for evidence in result.get("case_evidence", []):
                evidence.pop("host_elapsed_us", None)
            return result
        except DiagnosticError as exc:
            return {"status": "infrastructure_error", "failure_type": exc.failure_type,
                    "diagnostics": _bounded(str(exc)), "handle": exc.handle or handle, **identity}
        except (OSError, ValueError, tarfile.TarError) as exc:
            return {"status": "infrastructure_error", "failure_type": "staging_error",
                    "diagnostics": _bounded(str(exc)), "handle": handle, **identity}


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
test "$rc" -ne 124 || exit 124
test -f "$work/response.json" || exit "$rc"
printf 'BZ_DIAGNOSTIC_RESULT='; tr -d '\\n' <"$work/response.json"; printf '\\n'
exit 0'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--remote-json", default='["cpl-remote"]')
    parser.add_argument("--adapter-json", required=True)
    args = parser.parse_args()
    try:
        remote, adapter = json.loads(args.remote_json), json.loads(args.adapter_json)
        if not all(isinstance(value, list) and value and all(isinstance(x, str) and x for x in value)
                   for value in (remote, adapter)):
            raise ValueError("transport commands must be non-empty JSON string arrays")
        request = json.load(__import__("sys").stdin)
        result = BzA3DiagnosticClient(AdapterTransport(remote, adapter), args.state_dir).run(request)
    except (ValueError, json.JSONDecodeError) as exc:
        result = {"status": "infrastructure_error", "failure_type": "request_error", "diagnostics": str(exc)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
