#!/usr/bin/env python3
"""Acquire exact-kernel profile evidence for a content-addressed workload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple


TARGETS = {
    "a3": {"bz-a3-1": "py311-torch", "bz-a3-2": "py311-torch"},
    "a5": {"bz-a5": "cann91"},
}
TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"queued", "dispatching", "running", "reconnecting", "observation-unavailable"}
B64_MARKER = "ACQUIRE_EVIDENCE_B64="
SHA_MARKER = "ACQUIRE_EVIDENCE_SHA256="
REMOTE_SHA_MARKER = "REMOTE_CONTENT_SHA256="
FAILURE_MARKER = "ACQUIRE_FAILURE_JSON="


class AcquisitionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: str = "local",
        classification: str = "request_error",
        handle: str | None = None,
        excerpt: str = "",
    ):
        super().__init__(message)
        self.phase = phase
        self.classification = classification
        self.handle = handle
        self.excerpt = excerpt


class BundleSpec(NamedTuple):
    identity: dict
    files: list[dict]


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def resolve_cpl_remote(explicit: Path | None, *, home: Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    else:
        path_client = shutil.which("cpl-remote")
        if path_client:
            candidates.append(Path(path_client))
        candidates.append(
            (home or Path.home()) / ".agents/skills/remote-access/scripts/cpl-remote"
        )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise AcquisitionError("approved cpl-remote client is unavailable")


def launch_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 5000:
        raise argparse.ArgumentTypeError(
            "launch count must be in deployed range 1..5000"
        )
    return parsed


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise AcquisitionError(f"unsafe bundle path: {value!r}")
    return path.as_posix()


def build_bundle(
    *,
    arguments: list[str],
    workload: Path | None = None,
    bundle: Path | None = None,
    entrypoint: str | None = None,
) -> BundleSpec:
    if (workload is None) == (bundle is None):
        raise AcquisitionError("select exactly one of --workload or --bundle")
    if workload is not None:
        if entrypoint is not None:
            raise AcquisitionError("--entrypoint is only valid with --bundle")
        if not workload.is_file() or workload.is_symlink():
            raise AcquisitionError("workload must be one regular non-symlink file")
        root = workload.parent
        selected = workload.name
        paths = [workload]
    else:
        if not bundle.is_dir() or bundle.is_symlink():
            raise AcquisitionError("bundle must be one directory")
        if entrypoint is None:
            raise AcquisitionError("--bundle requires --entrypoint")
        root = bundle
        selected = _safe_relative(entrypoint)
        paths = sorted(path for path in bundle.rglob("*") if path.is_file())
        if not paths:
            raise AcquisitionError("bundle contains no files")
        if any(path.is_symlink() for path in bundle.rglob("*")):
            raise AcquisitionError("bundle symlinks are not supported")

    files = []
    manifest_files = []
    for path in paths:
        relative = _safe_relative(path.relative_to(root).as_posix())
        raw = path.read_bytes()
        mode = stat.S_IMODE(path.stat().st_mode)
        item = {
            "path": relative,
            "mode": mode,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest_files.append(item)
        files.append({**item, "data_b64": base64.b64encode(raw).decode()})
    if selected not in {item["path"] for item in manifest_files}:
        raise AcquisitionError("entrypoint is not a regular file in the bundle")
    manifest = {"schema_version": 1, "entrypoint": selected, "files": manifest_files}
    identity = {
        "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "entrypoint": selected,
        "arguments": arguments,
        "file_count": len(files),
    }
    return BundleSpec(identity=identity, files=files)


def validate_evidence(value, mode, product, target, dispatch_key, identity):
    if not isinstance(value, dict):
        raise AcquisitionError("evidence must be a JSON object", phase="evidence")
    expected = {
        "schema_version": 2,
        "schema": "cpl.profile-acquisition.v2",
        "status": "success",
        "phase": mode,
        "product": product,
        "target": target,
        "runtime": TARGETS[product][target],
        "dispatch_key": dispatch_key,
        "workload": identity,
        "provenance": {"product": product, "target": target},
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            message = (
                "workload identity does not match BasicInfo evidence"
                if key == "workload"
                else f"evidence {key} mismatch"
            )
            raise AcquisitionError(
                message, phase="evidence", classification="evidence_failure"
            )
    normal_run = value.get("normal_run")
    if not isinstance(normal_run, dict) or normal_run.get("exit_code") != 0:
        raise AcquisitionError(
            "evidence lacks successful normal workload run",
            phase="evidence",
            classification="evidence_failure",
        )
    for stream in ("stdout", "stderr"):
        digest = normal_run.get(stream)
        if (
            not isinstance(digest, dict)
            or not isinstance(digest.get("bytes"), int)
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest.get("sha256", "")))
        ):
            raise AcquisitionError(
                f"evidence normal-run {stream} digest is invalid",
                phase="evidence",
                classification="evidence_failure",
            )
    capture = value.get("capture")
    if not isinstance(capture, dict) or not isinstance(
        capture.get("observed_kernel_names"), list
    ):
        raise AcquisitionError(
            "evidence capture is incomplete",
            phase="evidence",
            classification="evidence_failure",
        )
    if not capture["observed_kernel_names"]:
        raise AcquisitionError(
            "BasicInfo observed no kernel names",
            phase="evidence",
            classification="evidence_failure",
        )
    saturation = value.get("saturation")
    if not isinstance(saturation, dict) or saturation.get("state") != "unknown":
        raise AcquisitionError(
            "saturation requires a reviewed capacity denominator; this capture is activity-only",
            phase="evidence",
            classification="evidence_failure",
        )


def _read_basic(path: Path, product: str, target: str, identity: dict) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"cannot read BasicInfo evidence: {exc}") from exc
    try:
        validate_evidence(
            value,
            "basic",
            product,
            target,
            str(value.get("dispatch_key", "")),
            identity,
        )
    except AcquisitionError as exc:
        raise AcquisitionError(str(exc)) from exc
    return value


def _remote_program(mode, product, target, dispatch_key, spec, kernel_name, launches):
    encoded = {
        "__MODE__": repr(mode),
        "__PRODUCT__": repr(product),
        "__TARGET__": repr(target),
        "__RUNTIME__": repr(TARGETS[product][target]),
        "__KEY__": repr(base64.b64encode(dispatch_key.encode()).decode()),
        "__IDENTITY__": repr(base64.b64encode(canonical(spec.identity)).decode()),
        "__FILES__": repr(base64.b64encode(canonical(spec.files)).decode()),
        "__KERNEL__": repr(base64.b64encode(canonical(kernel_name)).decode()),
        "__METRIC__": repr(
            "--aic-metrics=BasicInfo"
            if mode == "basic"
            else "--aic-metrics=PipeUtilization"
        ),
        "__LAUNCHES__": str(launches),
    }
    program = r'''#!/usr/bin/env python3
import base64, csv, hashlib, json, os, pathlib, shlex, subprocess, sys

MODE=__MODE__; PRODUCT=__PRODUCT__; TARGET=__TARGET__; RUNTIME=__RUNTIME__
DISPATCH_KEY=base64.b64decode(__KEY__).decode()
IDENTITY=json.loads(base64.b64decode(__IDENTITY__))
FILES=json.loads(base64.b64decode(__FILES__))
KERNEL_NAME=json.loads(base64.b64decode(__KERNEL__))
METRIC_ARG=__METRIC__; LAUNCH_COUNT=__LAUNCHES__
SUCCESS="Profiling running finished. All task success."
NAME_FIELDS=("Op Name","OpName","Kernel Name","kernel_name","Name")

def fail(phase, classification, message):
    print("ACQUIRE_FAILURE_JSON="+json.dumps(
        {"phase":phase,"classification":classification,"message":str(message)},sort_keys=True),
        file=sys.stderr,flush=True)
    raise SystemExit(1)

PROBE="""
import torch, torch_npu
torch.npu.set_device(0)
x=torch.arange(16,dtype=torch.float32,device='npu'); y=x+1
torch.npu.synchronize(); assert float(y.cpu()[3]) == 4.0
"""

def pick_device():
    failures=[]
    for physical in range(8):
        env=os.environ.copy(); env["ASCEND_RT_VISIBLE_DEVICES"]=str(physical); env["ASCEND_DEVICE_ID"]="0"
        try:
            result=subprocess.run([sys.executable,"-c",PROBE],text=True,capture_output=True,env=env,timeout=20)
        except subprocess.TimeoutExpired:
            failures.append("%d:timeout"%physical); continue
        if result.returncode == 0: return physical,env
        failures.append("%d:%s"%(physical,(result.stderr or result.stdout)[-160:]))
    fail("device-probe","device_unavailable","no device passed bounded probe: "+"; ".join(failures))

def name_field(fields):
    direct=next((field for field in NAME_FIELDS if field in fields),None)
    return direct or next((field for field in fields if "name" in field.lower().replace("_"," ")
                           and ("kernel" in field.lower().replace("_"," ") or "op" in field.lower().replace("_"," "))),None)

def csv_files(root, metric):
    preferred="OpBasicInfo" if metric == "BasicInfo" else "PipeUtilization"
    files=sorted(root.rglob(preferred+"*.csv"))
    return files or sorted(p for p in root.rglob("*.csv") if preferred.lower() in p.name.lower())

def read_csv(path):
    raw=path.read_bytes()
    with path.open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames or []
        rows=[{str(k):str(v) for k,v in row.items() if v not in (None,"")} for row in reader]
    return raw,fields,rows

def collect(root,metric):
    named=[]; sources=[]
    for path in csv_files(root,metric):
        raw,fields,rows=read_csv(path); field=name_field(fields)
        named.extend((KERNEL_NAME,row) for row in rows) if metric == "PipeUtilization" else named.extend((row[field],row) for row in rows if field and row.get(field))
        sources.append({"name":path.name,"bytes":len(raw),"rows":len(rows),"role":metric,"sha256":hashlib.sha256(raw).hexdigest()})
    if metric == "PipeUtilization":
        info=[]
        for path in csv_files(root,"BasicInfo"):
            raw,fields,rows=read_csv(path); field=name_field(fields)
            if field: info.extend((row.get(field) or "").strip() for row in rows)
            sources.append({"name":path.name,"bytes":len(raw),"rows":len(rows),"role":"selector_binding","sha256":hashlib.sha256(raw).hexdigest()})
        if sorted({name for name in info if name}) != [KERNEL_NAME]:
            fail("evidence","evidence_failure","selector-binding BasicInfo mismatch: %r"%sorted(set(info)))
    if not named:
        inventory=sorted("%s:%d"%(p.relative_to(root),p.stat().st_size) for p in root.rglob("*") if p.is_file())
        fail("evidence","evidence_failure","no usable %s rows; files=%r"%(metric,inventory[:80]))
    return named,sources

def materialize(root):
    bundle=root/"bundle"; bundle.mkdir()
    manifest_files=[]
    for item in FILES:
        rel=pathlib.PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts: fail("bundle","bundle_failure","unsafe path")
        raw=base64.b64decode(item["data_b64"],validate=True)
        if len(raw)!=item["size"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
            fail("bundle","bundle_failure","file identity mismatch: "+item["path"])
        path=bundle/pathlib.Path(*rel.parts); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw); path.chmod(item["mode"])
        manifest_files.append({k:item[k] for k in ("path","mode","size","sha256")})
    manifest={"schema_version":1,"entrypoint":IDENTITY["entrypoint"],"files":manifest_files}
    if hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()!=IDENTITY["manifest_sha256"]:
        fail("bundle","bundle_failure","manifest identity mismatch")
    return bundle

physical,env=pick_device()
retention=hashlib.sha256((DISPATCH_KEY+"\0"+MODE).encode()).hexdigest()[:20]
root=pathlib.Path.cwd()/".cpl-profile-evidence"/retention
try: root.mkdir(parents=True,exist_ok=False)
except FileExistsError: fail("retention","host_environment","dispatch key already retained")
try:
    bundle=materialize(root); entry=bundle/pathlib.PurePosixPath(IDENTITY["entrypoint"])
    argv=([sys.executable,str(entry)] if entry.suffix==".py" else [str(entry)])+IDENTITY["arguments"]
    try: check=subprocess.run(argv,cwd=bundle,text=True,capture_output=True,env=env,timeout=300)
    except subprocess.TimeoutExpired as exc: fail("workload","workload_failure","timeout: "+str(exc))
    if check.returncode: fail("workload","workload_failure","exit %d: %s"%(check.returncode,(check.stderr or check.stdout)[-2000:]))
    wrapper=root/"workload.sh"
    wrapper.write_text("#!/bin/sh\nset -eu\ncd "+shlex.quote(str(bundle))+"\nexec "+shlex.join(argv)+"\n"); wrapper.chmod(0o700)
    metric="BasicInfo" if MODE=="basic" else "PipeUtilization"; report=root/"report"
    command=["msprof","op","--application="+str(wrapper),"--output="+str(report),METRIC_ARG,
             "--warm-up=0","--launch-count="+str(LAUNCH_COUNT),"--kill=off"]
    if MODE=="pipe": command.append("--kernel-name="+KERNEL_NAME)
    try: result=subprocess.run(command,text=True,capture_output=True,env=env,timeout=600)
    except subprocess.TimeoutExpired as exc: fail("profiler","profiler_failure","timeout: "+str(exc))
    transcript=result.stdout+result.stderr
    if result.returncode: fail("profiler","profiler_failure","exit %d: %s"%(result.returncode,transcript[-2000:]))
    if SUCCESS not in transcript: fail("profiler","profiler_failure","terminal success marker missing: "+transcript[-1000:])
    rows,sources=collect(report,metric); names=sorted({name for name,_ in rows})
    if len(names)>256: fail("evidence","evidence_failure","observed kernel count exceeds compact bound")
    selected=[] if MODE=="basic" else [row for name,row in rows if name==KERNEL_NAME]
    if MODE=="pipe" and not selected: fail("evidence","evidence_failure","no row for exact selector")
    if len(selected)>128: fail("evidence","evidence_failure","exact-selector rows exceed compact bound")
    def stream_digest(value):
        raw=value.encode(); return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
    evidence={"schema_version":2,"schema":"cpl.profile-acquisition.v2","status":"success","phase":MODE,"product":PRODUCT,"target":TARGET,
      "runtime":RUNTIME,"dispatch_key":DISPATCH_KEY,"device":{"physical_id":physical,"logical_id":0},
      "provenance":{"product":PRODUCT,"target":TARGET},"workload":IDENTITY,
      "normal_run":{"exit_code":0,"stdout":stream_digest(check.stdout),"stderr":stream_digest(check.stderr)},
      "capture":{"metric":metric,"observed_kernel_names":names,
      "inventory_scope":{"launch_bound":LAUNCH_COUNT,"complete":False,
      "reason":"msprof exports observed names but no total application launch count"},
      "selected_kernel":KERNEL_NAME,"rows":selected,"sources":sources,
      "log_sha256":hashlib.sha256(transcript.encode()).hexdigest()},
      "saturation":{"state":"unknown","reason":"activity-only evidence has no reviewed capacity denominator"}}
    raw=(json.dumps(evidence,sort_keys=True,separators=(",",":"))+"\n").encode()
    digest=hashlib.sha256(raw).hexdigest()
    print("ACQUIRE_EVIDENCE_SHA256="+digest,flush=True)
    print("REMOTE_CONTENT_SHA256="+digest,flush=True)
    print("ACQUIRE_EVIDENCE_B64="+base64.b64encode(raw).decode(),flush=True)
except SystemExit: raise
except Exception as exc: fail("controller","host_environment",repr(exc))
'''
    for marker, value in encoded.items():
        program = program.replace(marker, value)
    return program


def render_remote_payload(
    mode, product, target, dispatch_key, workload, arguments, kernel_name, launches=5000
):
    if product not in TARGETS or target not in TARGETS[product]:
        raise AcquisitionError(
            f"target {target!r} is not valid for product {product!r}"
        )
    spec = (
        workload
        if isinstance(workload, BundleSpec)
        else build_bundle(workload=Path(workload), arguments=arguments)
    )
    if mode == "pipe" and not kernel_name:
        raise AcquisitionError("pipe pass requires --kernel-name")
    return _remote_program(
        mode, product, target, dispatch_key, spec, kernel_name, launches
    )


def _receipt(result, phase, handle=None):
    objects = []
    for line in result.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    if len(objects) != 1:
        excerpt = (result.stderr or result.stdout)[-1000:]
        raise AcquisitionError(
            "cpl-remote returned no unambiguous receipt",
            phase=phase,
            classification="transport_error",
            handle=handle,
            excerpt=excerpt,
        )
    return objects[0]


def _invoke(cpl, argv, phase, handle=None, timeout=660):
    try:
        result = subprocess.run(
            [str(cpl), "--json", *argv], text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise AcquisitionError(
            f"cpl-remote timeout: {exc}",
            phase=phase,
            classification="transport_error",
            handle=handle,
        ) from exc
    return _receipt(result, phase, handle)


def _retry(cpl, argv, phase, handle, attempts, timeout=660):
    last = None
    for _ in range(max(attempts, 1)):
        try:
            return _invoke(cpl, argv, phase, handle, timeout)
        except AcquisitionError as exc:
            last = exc
            if exc.classification != "transport_error":
                raise
    raise last


def _logs(cpl, handle, stream, tail, attempts):
    value = _retry(
        cpl,
        ["logs", handle, "--stream", stream, "--tail", str(tail)],
        "logs",
        handle,
        attempts,
    )
    if value.get("handle") != handle or not isinstance(value.get("content"), str):
        raise AcquisitionError(
            "logs receipt identity mismatch",
            phase="logs",
            classification="transport_error",
            handle=handle,
        )
    return value["content"]


def _observe_terminal(cpl, handle, attempts):
    last = None
    for _ in range(max(attempts, 1)):
        try:
            value = _invoke(
                cpl,
                ["observe", handle, "--wait", "--timeout", "900"],
                "observe",
                handle,
                930,
            )
        except AcquisitionError as exc:
            last = exc
            if exc.classification != "transport_error":
                raise
            continue
        if value.get("handle") != handle:
            raise AcquisitionError(
                "observation identity mismatch",
                phase="observe",
                classification="transport_error",
                handle=handle,
            )
        if value.get("state") in TERMINAL:
            return value
        last = AcquisitionError(
            "retained job is not terminal",
            phase="observe",
            classification="transport_error",
            handle=handle,
        )
    raise last


def remote_failure(stderr: str, handle: str) -> AcquisitionError:
    records = []
    for line in stderr.splitlines():
        if not line.startswith(FAILURE_MARKER):
            continue
        try:
            value = json.loads(line.removeprefix(FAILURE_MARKER))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    if len(records) == 1:
        value = records[0]
        return AcquisitionError(
            str(value.get("message", "remote acquisition failed")),
            phase=str(value.get("phase", "remote-job")),
            classification=str(value.get("classification", "remote_failure")),
            handle=handle,
            excerpt=stderr[-2000:],
        )
    return AcquisitionError(
        "remote job failed without one valid failure receipt",
        phase="remote-job",
        classification="remote_failure",
        handle=handle,
        excerpt=stderr[-2000:],
    )


def _decode(stdout, mode, product, target, key, identity):
    encoded = [
        line.removeprefix(B64_MARKER)
        for line in stdout.splitlines()
        if line.startswith(B64_MARKER)
    ]
    digests = [
        line.removeprefix(SHA_MARKER)
        for line in stdout.splitlines()
        if line.startswith(SHA_MARKER)
    ]
    remote_digests = [
        line.removeprefix(REMOTE_SHA_MARKER)
        for line in stdout.splitlines()
        if line.startswith(REMOTE_SHA_MARKER)
    ]
    if len(encoded) != 1 or len(digests) != 1 or len(remote_digests) != 1:
        raise AcquisitionError(
            "remote evidence markers missing or ambiguous",
            phase="evidence",
            classification="evidence_failure",
        )
    try:
        raw = base64.b64decode(encoded[0], validate=True)
        value = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise AcquisitionError(
            f"invalid remote evidence: {exc}",
            phase="evidence",
            classification="evidence_failure",
        ) from exc
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != digests[0] or actual_digest != remote_digests[0]:
        raise AcquisitionError(
            "remote evidence SHA-256 mismatch",
            phase="evidence",
            classification="evidence_failure",
        )
    validate_evidence(value, mode, product, target, key, identity)
    return raw


def request_metadata(args, identity):
    return {
        "schema_version": 1,
        "mode": args.mode,
        "product": args.product,
        "target": args.target,
        "runtime": TARGETS[args.product][args.target],
        "dispatch_key": args.dispatch_key,
        "workload": identity,
        "kernel_name": args.kernel_name,
        "launch_count": args.launch_count,
    }


def persist_receipt(path: Path, request: dict, handle: str):
    value = {
        "schema_version": 1,
        "handle": handle,
        "request": request,
        "request_sha256": hashlib.sha256(canonical(request)).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _validate_resume(path, request, handle):
    if not re.fullmatch(
        rf"remote:{re.escape(request['target'])}:job:[A-Za-z0-9_.-]+", handle
    ):
        raise AcquisitionError("resume handle does not match target")
    if path.exists():
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise AcquisitionError(f"invalid dispatch receipt: {exc}") from exc
        expected_digest = hashlib.sha256(canonical(request)).hexdigest()
        if value.get("request_sha256") != expected_digest:
            raise AcquisitionError(
                "dispatch receipt acquisition metadata hash mismatch"
            )
        if value.get("handle") != handle or value.get("request") != request:
            raise AcquisitionError(
                "resume handle or acquisition metadata does not match dispatch receipt"
            )
    else:
        persist_receipt(path, request, handle)


def _normalized_argv(argv):
    raw = list(sys.argv[1:] if argv is None else argv)
    result = []
    index = 0
    while index < len(raw):
        if raw[index] == "--workload-arg" and index + 1 < len(raw):
            result.append("--workload-arg=" + raw[index + 1])
            index += 2
        else:
            result.append(raw[index])
            index += 1
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("basic", "pipe"))
    parser.add_argument("--product", choices=sorted(TARGETS), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dispatch-key", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workload", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--entrypoint")
    parser.add_argument("--workload-arg", action="append", default=[])
    parser.add_argument("--basic-evidence", type=Path)
    parser.add_argument("--kernel-name")
    parser.add_argument("--launch-count", type=launch_count, default=5000)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--dispatch-receipt", type=Path)
    parser.add_argument("--resume-handle")
    parser.add_argument(
        "--cpl-remote",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--transport-attempts", type=int, default=3, help=argparse.SUPPRESS
    )
    args = parser.parse_args(_normalized_argv(argv))
    if args.mode == "basic" and (args.basic_evidence or args.kernel_name):
        parser.error("basic pass does not accept selector options")
    if args.mode == "pipe" and (not args.basic_evidence or not args.kernel_name):
        parser.error("pipe pass requires --basic-evidence and --kernel-name")
    if args.bundle and not args.entrypoint:
        parser.error("--bundle requires --entrypoint")
    if args.workload and args.entrypoint:
        parser.error("--entrypoint is only valid with --bundle")
    args.dispatch_receipt = args.dispatch_receipt or Path(
        str(args.evidence) + ".dispatch.json"
    )
    return args


def main(argv=None):
    args = parse_args(argv)
    handle = None
    try:
        if args.target not in TARGETS[args.product]:
            raise AcquisitionError(
                f"target {args.target!r} is not valid for product {args.product!r}"
            )
        if args.evidence.exists():
            raise AcquisitionError("evidence output already exists")
        args.cpl_remote = resolve_cpl_remote(args.cpl_remote)
        spec = build_bundle(
            workload=args.workload,
            bundle=args.bundle,
            entrypoint=args.entrypoint,
            arguments=args.workload_arg,
        )
        if args.mode == "pipe":
            basic = _read_basic(
                args.basic_evidence, args.product, args.target, spec.identity
            )
            if args.kernel_name not in basic["capture"]["observed_kernel_names"]:
                raise AcquisitionError(
                    "--kernel-name must be one exact observed BasicInfo kernel"
                )
        request = request_metadata(args, spec.identity)
        if args.resume_handle:
            handle = args.resume_handle
            _validate_resume(args.dispatch_receipt, request, handle)
            print(
                json.dumps(
                    {
                        "status": "resuming",
                        "handle": handle,
                        "receipt": str(args.dispatch_receipt),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            terminal = None
        else:
            if args.dispatch_receipt.exists():
                raise AcquisitionError(
                    "dispatch receipt exists; resume its handle instead of redispatching"
                )
            payload = render_remote_payload(
                args.mode,
                args.product,
                args.target,
                args.dispatch_key,
                spec,
                [],
                args.kernel_name,
                args.launch_count,
            )
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as stream:
                stream.write(payload)
                script = Path(stream.name)
            try:
                run = _invoke(
                    args.cpl_remote,
                    [
                        "run",
                        args.target,
                        "--runtime",
                        TARGETS[args.product][args.target],
                        "--file",
                        str(script),
                        "--timeout",
                        "900",
                    ],
                    "dispatch",
                )
            finally:
                script.unlink(missing_ok=True)
            handle = run.get("handle")
            _validate_resume(
                args.dispatch_receipt,
                request,
                handle if isinstance(handle, str) else "",
            )
            print(
                json.dumps(
                    {
                        "status": "dispatched",
                        "handle": handle,
                        "receipt": str(args.dispatch_receipt),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            state = run.get("state")
            terminal = run if state in TERMINAL else None
            if state not in TERMINAL | ACTIVE:
                raise AcquisitionError(
                    "dispatch returned invalid state",
                    phase="dispatch",
                    classification="transport_error",
                    handle=handle,
                )
        if terminal is None:
            _observe_terminal(args.cpl_remote, handle, args.transport_attempts)
        result = _retry(
            args.cpl_remote,
            ["result", handle],
            "result",
            handle,
            args.transport_attempts,
        )
        if result.get("handle") != handle:
            raise AcquisitionError(
                "result identity mismatch",
                phase="result",
                classification="transport_error",
                handle=handle,
            )
        if result.get("state") != "completed" or result.get("exit") not in (None, 0):
            raise remote_failure(
                _logs(args.cpl_remote, handle, "stderr", 80, args.transport_attempts),
                handle,
            )
        raw = _decode(
            _logs(args.cpl_remote, handle, "stdout", 20, args.transport_attempts),
            args.mode,
            args.product,
            args.target,
            args.dispatch_key,
            spec.identity,
        )
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_bytes(raw)
        print(
            json.dumps(
                {
                    "status": "success",
                    "handle": handle,
                    "evidence": str(args.evidence),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return 0
    except (AcquisitionError, OSError) as exc:
        if not isinstance(exc, AcquisitionError):
            exc = AcquisitionError(str(exc), handle=handle)
        details = f"classification={exc.classification} phase={exc.phase}" + (
            f" handle={exc.handle}" if exc.handle else ""
        )
        excerpt = f" excerpt={exc.excerpt!r}" if exc.excerpt else ""
        print(f"acquisition failed: {details}: {exc}{excerpt}", file=sys.stderr)
        return 2 if exc.classification == "request_error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
