#!/usr/bin/env python3
"""Acquire exact-kernel profile evidence for an arbitrary workload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


TARGETS = {
    "a3": {"bz-a3-1": "py311-torch", "bz-a3-2": "py311-torch"},
    "a5": {"bz-a5": "cann91"},
}
TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"queued", "dispatching", "running", "reconnecting", "observation-unavailable"}
B64_MARKER = "ACQUIRE_EVIDENCE_B64="
SHA_MARKER = "ACQUIRE_EVIDENCE_SHA256="


class AcquisitionError(RuntimeError):
    def __init__(self, message: str, *, phase: str = "local", handle: str | None = None):
        super().__init__(message)
        self.phase = phase
        self.handle = handle


def workload_identity(path: Path, arguments: list[str]) -> dict:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arguments": arguments,
    }


def validate_evidence(
    value: object,
    mode: str,
    product: str,
    target: str,
    dispatch_key: str,
    identity: dict,
) -> None:
    if not isinstance(value, dict):
        raise AcquisitionError("evidence must be a JSON object", phase="evidence")
    expected = {
        "schema_version": 1,
        "status": "success",
        "phase": mode,
        "product": product,
        "target": target,
        "runtime": TARGETS[product][target],
        "dispatch_key": dispatch_key,
        "workload": identity,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            if key == "workload":
                raise AcquisitionError("workload identity does not match BasicInfo evidence")
            raise AcquisitionError(f"evidence {key} mismatch", phase="evidence")
    capture = value.get("capture")
    if not isinstance(capture, dict) or not isinstance(capture.get("exported_kernel_names"), list):
        raise AcquisitionError("evidence capture is incomplete", phase="evidence")
    if not capture["exported_kernel_names"]:
        raise AcquisitionError("BasicInfo exported no kernel names", phase="evidence")
    saturation = value.get("saturation")
    if not isinstance(saturation, dict) or saturation.get("state") != "unknown":
        raise AcquisitionError(
            "saturation requires a reviewed capacity denominator; this capture is activity-only",
            phase="evidence",
        )


def _read_basic(path: Path, product: str, target: str, identity: dict) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"cannot read BasicInfo evidence: {exc}") from exc
    validate_evidence(
        value,
        "basic",
        product,
        target,
        str(value.get("dispatch_key", "")),
        identity,
    )
    return value


def _remote_program(
    mode: str,
    product: str,
    target: str,
    dispatch_key: str,
    workload: Path,
    arguments: list[str],
    kernel_name: str | None,
) -> str:
    replacements = {
        "__MODE__": repr(mode),
        "__PRODUCT__": repr(product),
        "__TARGET__": repr(target),
        "__RUNTIME__": repr(TARGETS[product][target]),
        "__DISPATCH_KEY_B64__": repr(base64.b64encode(dispatch_key.encode()).decode()),
        "__WORKLOAD_B64__": repr(base64.b64encode(workload.read_bytes()).decode()),
        "__WORKLOAD_NAME__": repr("workload" + workload.suffix),
        "__WORKLOAD_SHA__": repr(hashlib.sha256(workload.read_bytes()).hexdigest()),
        "__WORKLOAD_ARGS_B64__": repr(
            base64.b64encode(json.dumps(arguments).encode()).decode()
        ),
        "__KERNEL_NAME_B64__": repr(
            base64.b64encode(json.dumps(kernel_name).encode()).decode()
        ),
        "__METRIC_ARG__": repr(
            "--aic-metrics=BasicInfo"
            if mode == "basic"
            else "--aic-metrics=PipeUtilization"
        ),
    }
    program = r'''#!/usr/bin/env python3
import base64, csv, hashlib, json, os, pathlib, shlex, subprocess, sys

MODE = __MODE__
PRODUCT = __PRODUCT__
TARGET = __TARGET__
RUNTIME = __RUNTIME__
DISPATCH_KEY = base64.b64decode(__DISPATCH_KEY_B64__).decode()
WORKLOAD_B64 = __WORKLOAD_B64__
WORKLOAD_NAME = __WORKLOAD_NAME__
WORKLOAD_SHA = __WORKLOAD_SHA__
WORKLOAD_ARGS = json.loads(base64.b64decode(__WORKLOAD_ARGS_B64__))
KERNEL_NAME = json.loads(base64.b64decode(__KERNEL_NAME_B64__))
METRIC_ARG = __METRIC_ARG__
SUCCESS = "Profiling running finished. All task success."
NAME_FIELDS = ("Op Name", "OpName", "Kernel Name", "kernel_name", "Name")

def fail(phase, message):
    print("ACQUIRE_FAILURE_JSON=" + json.dumps(
        {"phase": phase, "message": str(message)}, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)

PROBE = r"""
import torch, torch_npu
torch.npu.set_device(0)
x = torch.arange(16, dtype=torch.float32, device='npu')
y = x + 1
torch.npu.synchronize()
assert float(y.cpu()[3]) == 4.0
"""

def pick_device():
    failures = []
    for physical in range(8):
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical)
        env["ASCEND_DEVICE_ID"] = "0"
        try:
            result = subprocess.run(
                [sys.executable, "-c", PROBE], text=True, capture_output=True,
                env=env, timeout=20,
            )
        except subprocess.TimeoutExpired:
            failures.append("%d:timeout" % physical)
            continue
        if result.returncode == 0:
            return physical, env
        failures.append("%d:%s" % (physical, (result.stderr or result.stdout)[-160:]))
    fail("device-probe", "no device passed bounded functional probe: " + "; ".join(failures))

def csv_files(root, metric):
    preferred = "OpBasicInfo" if metric == "BasicInfo" else "PipeUtilization"
    files = sorted(root.rglob(preferred + "*.csv"))
    if not files:
        files = sorted(p for p in root.rglob("*.csv") if preferred.lower() in p.name.lower())
    return files

def collect(root, metric):
    named_rows, sources = [], []
    for path in csv_files(root, metric):
        raw = path.read_bytes()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            name_field = next((field for field in NAME_FIELDS if field in fields), None)
            if name_field is None:
                name_field = next((field for field in fields
                                   if "name" in field.lower().replace("_", " ")
                                   and ("kernel" in field.lower().replace("_", " ")
                                        or "op" in field.lower().replace("_", " "))), None)
            rows = [
                {str(key): str(value) for key, value in row.items() if value not in (None, "")}
                for row in reader
            ]
        if metric == "PipeUtilization":
            named_rows.extend((KERNEL_NAME, row) for row in rows)
        elif name_field is not None:
            named_rows.extend((row[name_field], row) for row in rows if row.get(name_field))
        sources.append({"name": path.name, "bytes": len(raw), "rows": len(rows),
                        "role": metric,
                        "sha256": hashlib.sha256(raw).hexdigest()})
    if metric == "PipeUtilization":
        info_names = []
        for path in csv_files(root, "BasicInfo"):
            raw = path.read_bytes()
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                name_field = next((field for field in NAME_FIELDS if field in fields), None)
                if name_field is None:
                    name_field = next((field for field in fields
                                       if "name" in field.lower().replace("_", " ")
                                       and ("kernel" in field.lower().replace("_", " ")
                                            or "op" in field.lower().replace("_", " "))), None)
                rows = list(reader)
            if name_field:
                info_names.extend((row.get(name_field) or "").strip() for row in rows)
            sources.append({"name": path.name, "bytes": len(raw), "rows": len(rows),
                            "role": "selector_binding",
                            "sha256": hashlib.sha256(raw).hexdigest()})
        observed = sorted({name for name in info_names if name})
        if observed != [KERNEL_NAME]:
            fail("evidence", "selector-binding BasicInfo mismatch: %r" % observed)
    if not named_rows:
        inventory = sorted(
            "%s:%d" % (path.relative_to(root), path.stat().st_size)
            for path in root.rglob("*") if path.is_file()
        )
        fail("evidence", "no usable %s rows; files=%r" % (metric, inventory[:80]))
    return named_rows, sources

def application_argv(path):
    if path.suffix == ".py":
        return [sys.executable, str(path)] + WORKLOAD_ARGS
    return [str(path)] + WORKLOAD_ARGS

physical, env = pick_device()
retention_key = hashlib.sha256((DISPATCH_KEY + "\\0" + MODE).encode()).hexdigest()[:20]
root = pathlib.Path.cwd() / ".cpl-profile-evidence" / retention_key
try:
    root.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    fail("retention", "dispatch key already has retained remote evidence")
try:
    workload = root / WORKLOAD_NAME
    raw_workload = base64.b64decode(WORKLOAD_B64, validate=True)
    if hashlib.sha256(raw_workload).hexdigest() != WORKLOAD_SHA:
        fail("workload", "embedded workload digest mismatch")
    workload.write_bytes(raw_workload)
    workload.chmod(0o700)
    metric = "BasicInfo" if MODE == "basic" else "PipeUtilization"
    report = root / "report"
    application = shlex.join(application_argv(workload))
    command = ["msprof", "op", "--application=" + application,
               "--output=" + str(report), METRIC_ARG,
               "--warm-up=0", "--launch-count=20", "--kill=off"]
    if MODE == "pipe":
        command.append("--kernel-name=" + KERNEL_NAME)
    try:
        result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=300)
    except subprocess.TimeoutExpired as exc:
        fail("msprof-" + metric, "timeout: " + str(exc))
    transcript = result.stdout + result.stderr
    if result.returncode:
        fail("msprof-" + metric, "exit %d: %s" % (result.returncode, transcript[-2000:]))
    if SUCCESS not in transcript:
        fail("msprof-" + metric, "terminal success marker missing: " + transcript[-1000:])
    rows, sources = collect(report, metric)
    names = sorted({name for name, _ in rows})
    if len(names) > 256:
        fail("evidence", "exported kernel name count exceeds compact bound")
    selected_rows = []
    if MODE == "pipe":
        selected_rows = [row for name, row in rows if name == KERNEL_NAME]
        if not selected_rows:
            fail("evidence", "PipeUtilization has no row for exact selector %r" % KERNEL_NAME)
        if len(selected_rows) > 128:
            fail("evidence", "exact-selector row count exceeds compact bound")
    evidence = {
        "schema_version": 1, "status": "success", "phase": MODE,
        "product": PRODUCT, "target": TARGET, "runtime": RUNTIME,
        "dispatch_key": DISPATCH_KEY,
        "device": {"physical_id": physical, "logical_id": 0},
        "workload": {"sha256": WORKLOAD_SHA, "arguments": WORKLOAD_ARGS},
        "capture": {"metric": metric, "exported_kernel_names": names,
                    "selected_kernel": KERNEL_NAME, "rows": selected_rows,
                    "sources": sources,
                    "log_sha256": hashlib.sha256(transcript.encode()).hexdigest()},
        "saturation": {"state": "unknown",
                       "reason": "activity-only evidence has no reviewed capacity denominator"},
    }
    raw = (json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n").encode()
    print("ACQUIRE_EVIDENCE_SHA256=" + hashlib.sha256(raw).hexdigest())
    print("ACQUIRE_EVIDENCE_B64=" + base64.b64encode(raw).decode())
except Exception as exc:
    fail("controller", repr(exc))
'''
    for marker, value in replacements.items():
        program = program.replace(marker, value)
    return program


def render_remote_payload(
    mode: str,
    product: str,
    target: str,
    dispatch_key: str,
    workload: Path,
    arguments: list[str],
    kernel_name: str | None,
) -> str:
    if product not in TARGETS or target not in TARGETS[product]:
        raise AcquisitionError(f"target {target!r} is not valid for product {product!r}")
    if not workload.is_file():
        raise AcquisitionError("workload must be a readable file")
    if mode == "pipe" and not kernel_name:
        raise AcquisitionError("pipe pass requires --kernel-name")
    return _remote_program(mode, product, target, dispatch_key, workload, arguments, kernel_name)


def _receipt(result: subprocess.CompletedProcess[str], phase: str, handle=None) -> dict:
    objects = []
    for line in result.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    # Terminal failed/cancelled jobs are valid retained receipts even though
    # the CLI mirrors their remote exit as its own nonzero status.
    if len(objects) != 1:
        excerpt = (result.stderr or result.stdout)[-1000:]
        raise AcquisitionError(
            f"cpl-remote returned no unambiguous receipt: {excerpt}",
            phase=phase,
            handle=handle,
        )
    return objects[0]


def _invoke(cpl: Path, argv: list[str], phase: str, handle=None, timeout=660) -> dict:
    try:
        result = subprocess.run(
            [str(cpl), "--json", *argv], text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise AcquisitionError(f"cpl-remote timeout: {exc}", phase=phase, handle=handle) from exc
    return _receipt(result, phase, handle)


def _logs(cpl: Path, handle: str, stream: str, tail: int) -> str:
    value = _invoke(
        cpl,
        ["logs", handle, "--stream", stream, "--tail", str(tail)],
        "logs",
        handle,
    )
    if value.get("handle") != handle or not isinstance(value.get("content"), str):
        raise AcquisitionError("logs receipt identity mismatch", phase="logs", handle=handle)
    return value["content"]


def _decode(stdout: str, mode: str, product: str, target: str, key: str, identity: dict) -> bytes:
    encoded = [line.removeprefix(B64_MARKER) for line in stdout.splitlines() if line.startswith(B64_MARKER)]
    expected = [line.removeprefix(SHA_MARKER) for line in stdout.splitlines() if line.startswith(SHA_MARKER)]
    if len(encoded) != 1 or len(expected) != 1:
        raise AcquisitionError("remote evidence markers are missing or ambiguous", phase="evidence")
    try:
        raw = base64.b64decode(encoded[0], validate=True)
        value = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"remote evidence is invalid: {exc}", phase="evidence") from exc
    if hashlib.sha256(raw).hexdigest() != expected[0]:
        raise AcquisitionError("remote evidence SHA-256 mismatch", phase="evidence")
    validate_evidence(value, mode, product, target, key, identity)
    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # An arbitrary workload argument may itself start with ``-``. Normalize
    # the two-token spelling before argparse interprets it as this CLI's flag.
    raw = list(sys.argv[1:] if argv is None else argv)
    normalized: list[str] = []
    index = 0
    while index < len(raw):
        if raw[index] == "--workload-arg" and index + 1 < len(raw):
            normalized.append("--workload-arg=" + raw[index + 1])
            index += 2
        else:
            normalized.append(raw[index])
            index += 1
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("basic", "pipe"))
    parser.add_argument("--product", choices=sorted(TARGETS), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dispatch-key", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--workload-arg", action="append", default=[])
    parser.add_argument("--basic-evidence", type=Path)
    parser.add_argument("--kernel-name")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--cpl-remote", type=Path,
        default=Path.home() / ".agents/skills/remote-access/scripts/cpl-remote",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--observe-attempts", type=int, default=3, help=argparse.SUPPRESS)
    args = parser.parse_args(normalized)
    if args.mode == "basic" and (args.basic_evidence or args.kernel_name):
        parser.error("basic pass does not accept --basic-evidence or --kernel-name")
    if args.mode == "pipe" and (not args.basic_evidence or not args.kernel_name):
        parser.error("pipe pass requires --basic-evidence and --kernel-name")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handle = None
    try:
        if args.target not in TARGETS[args.product]:
            raise AcquisitionError(
                f"target {args.target!r} is not valid for product {args.product!r}"
            )
        if args.evidence.exists():
            raise AcquisitionError("evidence output already exists")
        if not args.cpl_remote.is_file() or not os.access(args.cpl_remote, os.X_OK):
            raise AcquisitionError("global cpl-remote is unavailable")
        identity = workload_identity(args.workload, args.workload_arg)
        if args.mode == "pipe":
            basic = _read_basic(args.basic_evidence, args.product, args.target, identity)
            names = basic["capture"]["exported_kernel_names"]
            if args.kernel_name not in names:
                raise AcquisitionError(
                    "--kernel-name must be one exact exported kernel from BasicInfo evidence"
                )
        payload = render_remote_payload(
            args.mode,
            args.product,
            args.target,
            args.dispatch_key,
            args.workload,
            args.workload_arg,
            args.kernel_name,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as stream:
            stream.write(payload)
            script = Path(stream.name)
        try:
            run = _invoke(
                args.cpl_remote,
                ["run", args.target, "--runtime", TARGETS[args.product][args.target],
                 "--file", str(script), "--timeout", "600"],
                "dispatch",
            )
        finally:
            script.unlink(missing_ok=True)
        handle = run.get("handle")
        if not isinstance(handle, str) or not handle.startswith(f"remote:{args.target}:job:"):
            raise AcquisitionError("dispatch returned an invalid handle", phase="dispatch")
        state = run.get("state")
        if state not in TERMINAL | ACTIVE:
            raise AcquisitionError("dispatch returned an invalid state", phase="dispatch", handle=handle)
        terminal = run if state in TERMINAL else None
        for _ in range(max(args.observe_attempts, 1)):
            if terminal:
                break
            try:
                observed = _invoke(
                    args.cpl_remote,
                    ["observe", handle, "--wait", "--timeout", "660"],
                    "observe",
                    handle,
                    690,
                )
            except AcquisitionError:
                continue
            if observed.get("handle") != handle:
                raise AcquisitionError("observation identity mismatch", phase="observe", handle=handle)
            if observed.get("state") in TERMINAL:
                terminal = observed
        if terminal is None:
            raise AcquisitionError("observation attempts exhausted", phase="observe", handle=handle)
        result = _invoke(args.cpl_remote, ["result", handle], "result", handle)
        if result.get("handle") != handle:
            raise AcquisitionError("result identity mismatch", phase="result", handle=handle)
        if result.get("state") != "completed" or result.get("exit") not in (None, 0):
            excerpt = _logs(args.cpl_remote, handle, "stderr", 80)[-2000:]
            raise AcquisitionError(
                f"remote job {result.get('state')} exit={result.get('exit')}: {excerpt}",
                phase="remote-job",
                handle=handle,
            )
        raw = _decode(
            _logs(args.cpl_remote, handle, "stdout", 20),
            args.mode,
            args.product,
            args.target,
            args.dispatch_key,
            identity,
        )
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_bytes(raw)
        print(json.dumps({"status": "success", "handle": handle,
                          "evidence": str(args.evidence),
                          "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
        return 0
    except (AcquisitionError, OSError) as exc:
        if not isinstance(exc, AcquisitionError):
            exc = AcquisitionError(str(exc), handle=handle)
        details = f"phase={exc.phase}" + (f" handle={exc.handle}" if exc.handle else "")
        print(f"acquisition failed: {details}: {exc}", file=sys.stderr)
        return 2 if exc.phase == "local" else 1


if __name__ == "__main__":
    raise SystemExit(main())
