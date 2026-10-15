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
import zlib
from pathlib import Path, PurePosixPath
from typing import NamedTuple


TARGETS = {
    "a3": {"bz-a3-1": "py311-torch", "bz-a3-2": "py311-torch", "gz-a3": "py311-torch"},
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
        compressed = zlib.compress(raw, level=9)
        if len(compressed) < len(raw):
            files.append({**item, "encoding": "zlib+base64",
                          "data_b64": base64.b64encode(compressed).decode()})
        else:
            files.append({**item, "encoding": "base64",
                          "data_b64": base64.b64encode(raw).decode()})
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
    if mode == "timeline":
        availability = capture.get("availability")
        if availability not in {"available", "unavailable"}:
            raise AcquisitionError(
                "TimelineDetail availability is missing",
                phase="evidence", classification="evidence_failure",
            )
        if availability == "available" and not capture.get("sources"):
            raise AcquisitionError(
                "available TimelineDetail evidence has no compact sources",
                phase="evidence", classification="evidence_failure",
            )
        semantics = capture.get("timeline_semantics")
        if not isinstance(semantics, dict) or not isinstance(
            semantics.get("events"), list
        ) or len(semantics["events"]) > 256:
            raise AcquisitionError(
                "TimelineDetail compact semantics are missing or unbounded",
                phase="evidence", classification="evidence_failure",
            )
        if availability == "available" and not semantics.get("selectors_complete"):
            raise AcquisitionError(
                "TimelineDetail did not cover every expected selector",
                phase="evidence", classification="evidence_failure",
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


def _remote_program(mode, product, target, dispatch_key, spec, kernel_names, launches):
    encoded = {
        "__MODE__": repr(mode),
        "__PRODUCT__": repr(product),
        "__TARGET__": repr(target),
        "__RUNTIME__": repr(TARGETS[product][target]),
        "__KEY__": repr(base64.b64encode(dispatch_key.encode()).decode()),
        "__IDENTITY__": repr(base64.b64encode(canonical(spec.identity)).decode()),
        "__FILES__": repr(base64.b64encode(canonical(spec.files)).decode()),
        "__KERNELS__": repr(base64.b64encode(canonical(kernel_names)).decode()),
        "__METRIC__": repr(
            "--aic-metrics=BasicInfo"
            if mode == "basic"
            else ("--aic-metrics=PipeUtilization" if mode == "pipe"
                  else "--aic-metrics=TimelineDetail")
        ),
        "__LAUNCHES__": str(launches),
    }
    program = r'''#!/usr/bin/env python3
import base64, csv, fcntl, hashlib, json, os, pathlib, shlex, subprocess, sys, tempfile, zlib

MODE=__MODE__; PRODUCT=__PRODUCT__; TARGET=__TARGET__; RUNTIME=__RUNTIME__
DISPATCH_KEY=base64.b64decode(__KEY__).decode()
IDENTITY=json.loads(base64.b64decode(__IDENTITY__))
FILES=json.loads(base64.b64decode(__FILES__))
KERNEL_NAMES=json.loads(base64.b64decode(__KERNELS__))
KERNEL_NAME=KERNEL_NAMES[0] if len(KERNEL_NAMES)==1 else None
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
    lease_root=pathlib.Path.home()/".cpl-device-leases"; lease_root.mkdir(exist_ok=True)
    for physical in range(8):
        lease=open(lease_root/("device-lease-%d.lock"%physical),"a+")
        try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close(); failures.append("%d:leased"%physical); continue
        env=os.environ.copy(); env["ASCEND_RT_VISIBLE_DEVICES"]=str(physical); env["ASCEND_DEVICE_ID"]="0"
        try:
            result=subprocess.run([sys.executable,"-c",PROBE],text=True,capture_output=True,env=env,timeout=20)
        except subprocess.TimeoutExpired:
            lease.close(); failures.append("%d:timeout"%physical); continue
        if result.returncode == 0: return physical,env,lease
        lease.close()
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

def file_sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def number(value):
    try:
        parsed=float(value)
        return parsed if parsed==parsed and abs(parsed)!=float("inf") else None
    except (TypeError,ValueError): return None

def first(row,names):
    lowered={str(key).lower().replace("_"," "):value for key,value in row.items()}
    for wanted in names:
        if wanted in lowered: return lowered[wanted],wanted
    return None,None

def timeline_phase(label):
    text=label.lower()
    if any(word in text for word in ("store","write","copy out","mte3")): return "store"
    if any(word in text for word in ("load","read","copy","dma","mte2","mte1")): return "load"
    if any(word in text for word in ("compute","vector","cube","dot")): return "compute"
    return None

def normalize_event(item):
    if not isinstance(item,dict): return None
    name_value,_=first(item,("name","op name","kernel name","task name"))
    name=str(name_value or "").strip()
    args=item.get("args") if isinstance(item.get("args"),dict) else {}
    # Instruction details can mention a waited-on pipe (e.g. BAR PIPE:MTE2).
    # Such operands do not identify either the execution lane or a source phase.
    label=" ".join((name,str(item.get("cat","")),str(item.get("type","")),
                    str(args.get("pipe",""))))
    start,start_field=first(item,("ts","start(us)","start time(us)","start time","start(ns)"))
    duration,duration_field=first(item,("dur","duration(us)","task duration(us)","duration","duration(ns)"))
    start=number(start); duration=number(duration)
    if start is None or duration is None or duration < 0: return None
    if start_field and "(ns)" in start_field: start/=1000.0
    if duration_field and "(ns)" in duration_field: duration/=1000.0
    selector=next((value for value in KERNEL_NAMES if value in label),None)
    phase=timeline_phase(label)
    if not name and not selector and not phase: return None
    return {"name":name or selector or phase,"start_us":start,"duration_us":duration,
            **({"selector":selector} if selector else {}),**({"phase":phase} if phase else {})}

def timeline_items(path):
    try:
      if path.suffix.lower()==".json":
        value=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value,dict): value=value.get("traceEvents",value.get("events",[]))
        return value if isinstance(value,list) else []
      if path.suffix.lower()==".csv":
        with path.open(newline="",encoding="utf-8-sig") as stream: return list(csv.DictReader(stream))
    except (OSError,UnicodeError,json.JSONDecodeError,csv.Error): return []
    return []

def compact_timeline(paths):
    events=[]
    simulator_inputs=0
    for path in paths[:128]:
      if "simulator" in path.parts:
        simulator_inputs+=1
        continue  # Simulator instruction traces are not hardware TimelineDetail.
      for item in timeline_items(path):
        event=normalize_event(item)
        if event is not None: events.append(event)
        if len(events)>=256: break
      if len(events)>=256: break
    events.sort(key=lambda item:(item["start_us"],item["duration_us"],item["name"]))
    kernel_events=[item for item in events if item.get("selector")]
    kernel_order=[]
    for item in kernel_events:
      if not kernel_order or kernel_order[-1]!=item["selector"]: kernel_order.append(item["selector"])
    phase_events=[item for item in events if item.get("phase")]
    phase_order=[]
    for item in phase_events:
      if not phase_order or phase_order[-1]!=item["phase"]: phase_order.append(item["phase"])
    phase_duration={}
    for item in phase_events: phase_duration[item["phase"]]=phase_duration.get(item["phase"],0.0)+item["duration_us"]
    overlap=0.0
    loads=[item for item in phase_events if item["phase"]=="load"]
    computes=[item for item in phase_events if item["phase"]=="compute"]
    for left in loads:
      for right in computes:
        overlap+=max(0.0,min(left["start_us"]+left["duration_us"],right["start_us"]+right["duration_us"])-max(left["start_us"],right["start_us"]))
    observed=sorted({item["selector"] for item in kernel_events})
    starts=[item["start_us"] for item in events]; ends=[item["start_us"]+item["duration_us"] for item in events]
    return {"schema_version":1,"event_count":len(events),"events":events,
      "simulator_inputs_excluded":simulator_inputs,
      "truncated":len(events)>=256,"expected_selectors":KERNEL_NAMES,
      "observed_selectors":observed,"selectors_complete":set(KERNEL_NAMES)<=set(observed),
      "kernel_order":kernel_order,"kernel_boundaries":kernel_events,
      "phase_order":phase_order,"phase_duration_us":phase_duration,
      "load_compute_overlap_us":overlap,
      "bounds_us":({"start":min(starts),"end":max(ends),"duration":max(ends)-min(starts)} if events else None)}

def materialize(root):
    bundle=root/"bundle"; bundle.mkdir()
    manifest_files=[]
    for item in FILES:
        rel=pathlib.PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts: fail("bundle","bundle_failure","unsafe path")
        raw=base64.b64decode(item["data_b64"],validate=True)
        if item.get("encoding")=="zlib+base64": raw=zlib.decompress(raw)
        elif item.get("encoding")!="base64": fail("bundle","bundle_failure","unsupported encoding")
        if len(raw)!=item["size"] or hashlib.sha256(raw).hexdigest()!=item["sha256"]:
            fail("bundle","bundle_failure","file identity mismatch: "+item["path"])
        path=bundle/pathlib.Path(*rel.parts); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw); path.chmod(item["mode"])
        manifest_files.append({k:item[k] for k in ("path","mode","size","sha256")})
    manifest={"schema_version":1,"entrypoint":IDENTITY["entrypoint"],"files":manifest_files}
    if hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()!=IDENTITY["manifest_sha256"]:
        fail("bundle","bundle_failure","manifest identity mismatch")
    return bundle

def device_failure(value):
    text=value.lower()
    return any(marker in text for marker in (
      "tsdopen","lazysetdevice","out of memory","device is busy","set device failed"))

physical,env,device_lease=pick_device()
retention=hashlib.sha256((DISPATCH_KEY+"\0"+MODE).encode()).hexdigest()[:20]
retained_parent=pathlib.Path.cwd()/".cpl-profile-evidence"; retained_parent.mkdir(parents=True,exist_ok=True)
root=pathlib.Path(tempfile.mkdtemp(prefix=retention+"-",dir=retained_parent))
try:
    bundle=materialize(root); entry=bundle/pathlib.PurePosixPath(IDENTITY["entrypoint"])
    argv=([sys.executable,str(entry)] if entry.suffix==".py" else [str(entry)])+IDENTITY["arguments"]
    try: check=subprocess.run(argv,cwd=bundle,text=True,capture_output=True,env=env,timeout=300)
    except subprocess.TimeoutExpired as exc: fail("workload","workload_failure","timeout: "+str(exc))
    (root/"workload.stdout.log").write_text(check.stdout)
    (root/"workload.stderr.log").write_text(check.stderr)
    if check.returncode:
        detail=(check.stderr or check.stdout)[-2000:]
        fail("workload","device_unavailable" if device_failure(detail) else "workload_failure","exit %d: %s"%(check.returncode,detail))
    wrapper=root/"workload.sh"
    wrapper.write_text("#!/bin/sh\nset -eu\ncd "+shlex.quote(str(bundle))+"\nexec "+shlex.join(argv)+"\n"); wrapper.chmod(0o700)
    metric={"basic":"BasicInfo","pipe":"PipeUtilization","timeline":"TimelineDetail"}[MODE]; report=root/"report"
    command=["msprof","op","--application="+str(wrapper),"--output="+str(report),METRIC_ARG,
             "--warm-up=0","--launch-count="+str(LAUNCH_COUNT),"--kill=off"]
    if MODE=="pipe": command.append("--kernel-name="+KERNEL_NAME)
    try: result=subprocess.run(command,text=True,capture_output=True,env=env,timeout=600)
    except subprocess.TimeoutExpired as exc: fail("profiler","profiler_failure","timeout: "+str(exc))
    transcript=result.stdout+result.stderr
    (root/"profiler.stdout.log").write_text(result.stdout)
    (root/"profiler.stderr.log").write_text(result.stderr)
    if result.returncode and device_failure(transcript):
        fail("profiler","device_unavailable","exit %d: %s"%(result.returncode,transcript[-2000:]))
    if result.returncode: fail("profiler","device_unavailable" if device_failure(transcript) else "profiler_failure","exit %d: %s"%(result.returncode,transcript[-2000:]))
    if SUCCESS not in transcript: fail("profiler","profiler_failure","terminal success marker missing: "+transcript[-1000:])
    if MODE=="timeline":
        inventory=[{"name":str(p.relative_to(report)),"bytes":p.stat().st_size}
                   for p in sorted(report.rglob("*")) if p.is_file()]
        timeline_files=sorted(p for p in report.rglob("*") if p.is_file() and p.suffix.lower() in {".json",".csv"} and any(word in str(p.relative_to(report)).lower() for word in ("timeline","trace","event")))
        sources=[{"name":p.name,"bytes":p.stat().st_size,"role":"TimelineDetail","sha256":file_sha256(p)} for p in timeline_files[:128]]
        timeline_semantics=compact_timeline(timeline_files)
        availability="available" if timeline_semantics["event_count"] and timeline_semantics["selectors_complete"] and "TimelineDetail data collection failed" not in transcript else "unavailable"
        basic_rows,basic_sources=collect(report,"BasicInfo") if csv_files(report,"BasicInfo") else ([],[])
        sources.extend(basic_sources)
        rows=[]; names=sorted({name for name,_ in basic_rows}) or timeline_semantics["observed_selectors"]
        timeline_diagnostic=("collection_failed" if "TimelineDetail data collection failed" in transcript else
          "semantic_events" if availability=="available" else
          "selector_coverage_incomplete" if timeline_semantics["event_count"] else
          "unsupported_event_schema" if timeline_files else
          "raw_binary_only" if any(item["name"].endswith(".bin") for item in inventory) else
          "no_timeline_exports")
    else:
        rows,sources=collect(report,metric); names=sorted({name for name,_ in rows})
    if len(names)>256: fail("evidence","evidence_failure","observed kernel count exceeds compact bound")
    selected=[] if MODE in ("basic","timeline") else [row for name,row in rows if name==KERNEL_NAME]
    if MODE=="pipe" and not selected: fail("evidence","evidence_failure","no row for exact selector")
    if len(selected)>128: fail("evidence","evidence_failure","exact-selector rows exceed compact bound")
    def normalized(value):
        return value.replace(str(root),"$RETAINED_ROOT")
    def stream_digest(value):
        raw=normalized(value).encode(); return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
    evidence={"schema_version":2,"schema":"cpl.profile-acquisition.v2","status":"success","phase":MODE,"product":PRODUCT,"target":TARGET,
      "runtime":RUNTIME,"dispatch_key":DISPATCH_KEY,"device":{"physical_id":physical,"logical_id":0},
      "provenance":{"product":PRODUCT,"target":TARGET},"workload":IDENTITY,
      "normal_run":{"exit_code":0,"stdout":stream_digest(check.stdout),"stderr":stream_digest(check.stderr)},
      "capture":{"metric":metric,"observed_kernel_names":names,
      "inventory_scope":{"launch_bound":LAUNCH_COUNT,"complete":False,
      "reason":"msprof exports observed names but no total application launch count"},
      "selected_kernel":KERNEL_NAME,"expected_kernel_names":KERNEL_NAMES,"rows":selected,"sources":sources,
      "log_sha256":hashlib.sha256(normalized(transcript).encode()).hexdigest()},
      "saturation":{"state":"unknown","reason":"activity-only evidence has no reviewed capacity denominator"}}
    if MODE=="timeline":
      evidence["capture"]["availability"]=availability
      evidence["capture"]["diagnostics"]={"classification":timeline_diagnostic,
        "inventory":inventory[:128],"inventory_truncated":len(inventory)>128,
        "profiler_transcript_tail":normalized(transcript[-3000:])}
      evidence["capture"]["timeline_semantics"]=timeline_semantics
      evidence["capture"]["reason"]=("TimelineDetail export parsed into bounded semantic evidence" if availability=="available" else "successful TimelineDetail command produced no supported semantic events")
      if timeline_diagnostic=="collection_failed":
        evidence["capture"]["reason"]="TimelineDetail collection failed despite zero exit status; inspect retained profiler transcript"
    raw=(json.dumps(evidence,sort_keys=True,separators=(",",":"))+"\n").encode()
    if str(root).encode() in raw: fail("evidence","evidence_failure","retained report path leaked into compact evidence")
    digest=hashlib.sha256(raw).hexdigest()
    print("ACQUIRE_EVIDENCE_SHA256="+digest,flush=True)
    print("REMOTE_CONTENT_SHA256="+digest,flush=True)
    print("ACQUIRE_EVIDENCE_B64="+base64.b64encode(raw).decode(),flush=True)
except SystemExit: raise
except Exception as exc: fail("controller","host_environment",repr(exc))
'''
    for marker, value in encoded.items():
        program = program.replace(marker, value)
    if target == "gz-a3":
        # The named runtime supplies PATH. Remove inherited listener Python
        # overrides before interpreter startup, including subprocess probes.
        program = ("#!/bin/sh\nset -eu\nunset PYTHONHOME PYTHONPATH\n"
                   "exec python - <<'CPL_ACQUIRE_PROGRAM'\n" + program
                   + "\nCPL_ACQUIRE_PROGRAM\n")
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
    kernel_names = ([kernel_name] if isinstance(kernel_name, str) else list(kernel_name or []))
    if mode in {"pipe", "timeline"} and not kernel_names:
        raise AcquisitionError(f"{mode} pass requires --kernel-name")
    if mode == "pipe" and len(kernel_names) != 1:
        raise AcquisitionError("pipe pass requires exactly one --kernel-name")
    return _remote_program(
        mode, product, target, dispatch_key, spec, kernel_names, launches
    )


def _receipt(result, phase, handle=None):
    content_match = re.search(r"(?m)^REMOTE_CONTENT=", result.stdout)
    text_receipt = re.search(
        r"(?m)^REMOTE_(?:TARGET|OPERATION|STATE|HANDLE)=", result.stdout
    )
    if text_receipt is None and content_match is None:
        objects = []
        for line in result.stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects.append(value)
        if len(objects) == 1:
            return objects[0]
        return _invalid_receipt(result, phase, handle)

    metadata = (
        result.stdout
        if content_match is None
        else result.stdout[: content_match.start()]
    )
    fields = {}
    for stripped in metadata.splitlines():
        match = re.fullmatch(r"REMOTE_([A-Z_]+)=(.*)", stripped)
        if not match:
            continue
        key = match.group(1).lower()
        if key in fields:
            if fields[key] != match.group(2):
                return _invalid_receipt(result, phase, handle)
            continue
        fields[key] = match.group(2)
    if content_match is not None:
        fields["content"] = result.stdout[content_match.end() :]
    if not fields:
        return _invalid_receipt(result, phase, handle)
    if "exit" in fields:
        try:
            fields["exit"] = int(fields["exit"])
        except ValueError:
            return _invalid_receipt(result, phase, handle)
    return fields


def _invalid_receipt(result, phase, handle):
    excerpt = (result.stderr or result.stdout)[-1000:]
    raise AcquisitionError(
        "cpl-remote returned no unambiguous receipt",
        phase=phase,
        classification="transport_error",
        handle=handle,
        excerpt=excerpt,
    )


def _stage_payload(payload: str, evidence: Path) -> Path:
    parent = evidence.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve()
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix=".acquire-payload-",
        suffix=".py",
        dir=parent,
        delete=False,
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def _invoke(cpl, argv, phase, handle=None, timeout=660):
    try:
        result = subprocess.run(
            [str(cpl), *argv], text=True, capture_output=True, timeout=timeout
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
        "kernel_names": args.kernel_name,
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
    parser.add_argument("mode", choices=("basic", "pipe", "timeline"))
    parser.add_argument("--product", choices=sorted(TARGETS), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dispatch-key", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workload", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--entrypoint")
    parser.add_argument("--workload-arg", action="append", default=[])
    parser.add_argument("--basic-evidence", type=Path)
    parser.add_argument("--kernel-name", action="append", default=[])
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
    if args.mode in {"pipe", "timeline"} and (not args.basic_evidence or not args.kernel_name):
        parser.error(f"{args.mode} pass requires --basic-evidence and --kernel-name")
    if args.mode == "pipe" and len(args.kernel_name) != 1:
        parser.error("pipe pass requires exactly one --kernel-name")
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
        if args.mode in {"pipe", "timeline"}:
            basic = _read_basic(
                args.basic_evidence, args.product, args.target, spec.identity
            )
            if not set(args.kernel_name) <= set(basic["capture"]["observed_kernel_names"]):
                raise AcquisitionError(
                    "every --kernel-name must be an exact observed BasicInfo kernel"
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
            script = _stage_payload(payload, args.evidence)
            run_argv = ["run", args.target]
            runtime = TARGETS[args.product][args.target]
            if runtime is not None:
                run_argv.extend(["--runtime", runtime])
            run_argv.extend(["--file", str(script), "--timeout", "900"])
            if os.environ.get("CPL_REMOTE_MODE") == "retained-broker":
                run_argv.extend(["--dispatch-key", args.dispatch_key])
            try:
                run = _invoke(
                    args.cpl_remote,
                    run_argv,
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
            try:
                stderr = _logs(
                    args.cpl_remote, handle, "stderr", 80, args.transport_attempts
                )
            except AcquisitionError as exc:
                raise AcquisitionError(
                    "terminal remote job failed but its diagnostic log is unavailable",
                    phase="logs",
                    classification="host_environment",
                    handle=handle,
                    excerpt=exc.excerpt or str(exc),
                ) from exc
            raise remote_failure(stderr, handle)
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
