#!/usr/bin/env python3
"""Production A3 adapter for ``bsa_candidate_profiler.py``.

The adapter is deliberately transport-only on the controller.  A compact,
self-contained bundle performs correctness, kernel discovery, msprof capture,
and report parsing in one retained cpl-remote job.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REQUEST_SCHEMA = "profiling-skill/bsa-candidate-executor-request/v1"
REMOTE_SCHEMA = "profiling-skill/bsa-candidate-remote-result/v1"
QUALIFICATION_SCHEMA = "profiling-skill/bsa-a3-qualification/v1"
QUALIFICATION_CASES = [47, 46, 49, 44, 43]
CANDIDATE_FAILURES = {"compile_error", "runtime_error", "correctness_error"}
TARGETS = {"bz-a3-1", "bz-a3-2", "gz-a3"}
LOGICAL_RUNTIME = "py311-torch"


class ExecutorError(RuntimeError):
    def __init__(self, status: str, message: str, *, handle: str | None = None,
                 terminal: bool = True):
        self.status, self.handle, self.terminal = status, handle, terminal
        super().__init__(message)


@dataclass(frozen=True)
class ExecutorConfig:
    target: str
    remote_root: str
    state_dir: Path
    baseline: Path
    cases: Path
    runner: Path
    min_free_device_bytes: int = 32 * 1024**3
    campaign_revision: str | None = None
    qualification_receipt: Path | None = None
    qualification_candidate: Path | None = None


def runtime_activation(target: str) -> str:
    return "target-default" if target == "gz-a3" else "named-runtime"


def execution_environment(target: str) -> dict[str, str]:
    return {
        "target": target,
        "runtime": LOGICAL_RUNTIME,
        "runtime_activation": runtime_activation(target),
    }


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest_json(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_request(value: object, target: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != REQUEST_SCHEMA:
        raise ExecutorError("infrastructure_error", "unsupported executor request schema")
    if value.get("action") not in {"discover", "capture"}:
        raise ExecutorError("infrastructure_error", "action must be discover or capture")
    candidate = value.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("benchmark") != "bsa":
        raise ExecutorError("infrastructure_error", "request requires one BSA candidate")
    source, declared = candidate.get("source"), candidate.get("source_sha256")
    if not isinstance(source, str) or hashlib.sha256(source.encode()).hexdigest() != declared:
        raise ExecutorError("infrastructure_error", "candidate source SHA-256 mismatch")
    environment = candidate.get("environment")
    if (not isinstance(environment, dict) or environment.get("product") != "a3"
            or environment.get("runtime") != LOGICAL_RUNTIME):
        raise ExecutorError(
            "infrastructure_error", "candidate lacks the required A3 logical runtime")
    if value.get("execution_environment") != execution_environment(target):
        raise ExecutorError(
            "infrastructure_error",
            "request execution environment does not match the configured A3 target",
        )
    if not isinstance(candidate.get("declared_kernel_name"), str):
        raise ExecutorError("infrastructure_error", "candidate kernel declaration is missing")
    if value["action"] == "discover":
        cases = value.get("correctness_cases")
        if (value.get("metric") != "BasicInfo" or not isinstance(cases, list) or not cases
                or any(isinstance(case, bool) or not isinstance(case, int) or case < 0
                       for case in cases)):
            raise ExecutorError("infrastructure_error", "invalid BasicInfo discovery request")
    else:
        if (value.get("metric") != "PipeUtilization"
                or isinstance(value.get("case"), bool)
                or not isinstance(value.get("case"), int) or value["case"] < 0
                or not isinstance(value.get("kernel_name"), str)
                or not value["kernel_name"]
                or isinstance(value.get("sample_index"), bool)
                or not isinstance(value.get("sample_index"), int)
                or value["sample_index"] < 0):
            raise ExecutorError("infrastructure_error", "invalid PipeUtilization capture request")
        component = value.get("component")
        if (not isinstance(component, dict)
                or component.get("selector") != value["kernel_name"]):
            raise ExecutorError("infrastructure_error", "capture component selector mismatch")
    retry = value.get("retry")
    if retry is not None:
        if not isinstance(retry, dict) or retry.get("mode") not in {"observe", "new_attempt"}:
            raise ExecutorError("infrastructure_error", "invalid retry request")
        field = "handle" if retry["mode"] == "observe" else "prior_handle"
        if not isinstance(retry.get(field), str):
            raise ExecutorError("infrastructure_error", "retry handle is missing")
    return value


class CplRemoteTransport:
    """The installed global cpl-remote client; no alternate transport exists."""

    def __init__(self, executable: Path | None = None):
        self.executable = executable or (
            Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
        )
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ExecutorError("infrastructure_error", "global cpl-remote is unavailable")

    def _call(self, arguments: list[str], timeout: int = 930) -> dict:
        try:
            run = subprocess.run([str(self.executable), "--json", *arguments], text=True,
                                 capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutorError("infrastructure_error", f"cpl-remote failed: {exc}") from exc
        objects = []
        for line in run.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                objects.append(item)
        if len(objects) != 1:
            excerpt = (run.stderr or run.stdout)[-2000:]
            raise ExecutorError("infrastructure_error",
                                "cpl-remote returned no unambiguous receipt: " + excerpt)
        return objects[0]

    def capabilities(self, target: str) -> None:
        receipt = self._call(["capabilities", target], 60)
        if (receipt.get("target") not in {None, target}
                or receipt.get("state") not in {None, "available", "completed"}):
            raise ExecutorError("infrastructure_error", "capabilities target is unavailable")

    def preflight(self, target: str) -> None:
        receipt = self._call(["preflight", target], 120)
        if receipt.get("target") not in {None, target} or receipt.get("state") == "failed":
            raise ExecutorError("infrastructure_error", "remote preflight failed")

    def upload(self, target: str, source: Path, destination: str) -> None:
        receipt = self._call(["upload", target, str(source), destination], 900)
        if receipt.get("state") not in {"completed", "failed", "cancelled"}:
            handle = receipt.get("handle")
            if (not isinstance(handle, str)
                    or not handle.startswith(f"remote:{target}:transfer:")):
                raise ExecutorError("infrastructure_error",
                                    "bundle upload returned no retained transfer handle")
            receipt = self._call(
                ["observe", handle, "--wait", "--timeout", "900"], 930)
        if receipt.get("target") != target or receipt.get("state") != "completed":
            handle = receipt.get("handle")
            raise ExecutorError(
                "infrastructure_error", "bundle upload did not complete",
                handle=handle if isinstance(handle, str) else None,
                terminal=receipt.get("state") in {"completed", "failed", "cancelled"},
            )

    def run(self, target: str, runtime: str, script: Path, cwd: str) -> str:
        arguments = ["run", target]
        if runtime_activation(target) == "named-runtime":
            arguments.extend(["--runtime", runtime])
        arguments.extend(["--file", str(script), "--cwd", cwd, "--timeout", "900"])
        self.last_command = [str(self.executable), "--json", *arguments]
        receipt = self._call(arguments, 930)
        handle = receipt.get("handle")
        if not isinstance(handle, str) or not handle.startswith(f"remote:{target}:job:"):
            raise ExecutorError("infrastructure_error", "run returned no valid durable handle")
        return handle

    def observe_result(self, target: str, handle: str) -> dict:
        try:
            observed = self._call(["observe", handle, "--wait", "--timeout", "900"], 930)
        except ExecutorError as exc:
            raise ExecutorError("infrastructure_error", str(exc), handle=handle,
                                terminal=False) from exc
        if observed.get("handle") != handle:
            raise ExecutorError("infrastructure_error", "observation identity mismatch",
                                handle=handle)
        if observed.get("state") not in {"completed", "failed", "cancelled"}:
            raise ExecutorError("infrastructure_error", "retained job is not terminal",
                                handle=handle, terminal=False)
        stdout = self._call(["logs", handle, "--stream", "stdout", "--tail", "8"], 120)
        stderr = self._call(["logs", handle, "--stream", "stderr", "--tail", "100"], 120)
        marker = "BSA_CANDIDATE_RESULT="
        rows = [line[len(marker):] for line in str(stdout.get("content", "")).splitlines()
                if line.startswith(marker)]
        if len(rows) != 1:
            raise ExecutorError("infrastructure_error",
                                "remote job returned no compact result: "
                                + str(stderr.get("content", ""))[-2000:], handle=handle)
        try:
            result = json.loads(rows[0])
        except json.JSONDecodeError as exc:
            raise ExecutorError("infrastructure_error", "compact result is invalid JSON",
                                handle=handle) from exc
        return result


def remote_driver_source() -> str:
    """Return the dependency-free program executed inside the staged bundle."""
    return r'''#!/usr/bin/env python3
import csv, fcntl, hashlib, importlib.util, json, math, os, pathlib, re, shlex, statistics, subprocess, sys, time, traceback

SCHEMA="profiling-skill/bsa-candidate-remote-result/v1"
SUCCESS="Profiling running finished. All task success."
NAMES=("Op Name","OpName","Kernel Name","kernel_name","Name")
DURATIONS=("Task Duration(us)","Task Duration (us)","task_duration_us")

def emit(value):
    value.setdefault("schema",SCHEMA)
    print("BSA_CANDIDATE_RESULT="+json.dumps(value,sort_keys=True,separators=(",",":")),flush=True)

def pick_device():
    probe="import json,torch,torch_npu;torch.npu.set_device(0);x=torch.ones(1,device='npu');torch.npu.synchronize();assert float(x.cpu()[0])==1;free,total=torch.npu.mem_get_info();print(json.dumps({'free':free,'total':total}))"
    failures={}; root=pathlib.Path(os.environ["CPL_DEVICE_LEASE_ROOT"]
                                  ); root.mkdir(parents=True,exist_ok=True)
    required=int(os.environ["BSA_MIN_FREE_DEVICE_BYTES"])
    for _attempt in range(1):
      for physical in range(8):
        lease=(root/("device-lease-%d.lock"%physical)).open("a+")
        try: fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: lease.close(); continue
        env=os.environ.copy(); env["ASCEND_RT_VISIBLE_DEVICES"]=str(physical); env["ASCEND_DEVICE_ID"]="0"
        try: run=subprocess.run([sys.executable,"-c",probe],env=env,text=True,capture_output=True,timeout=20)
        except subprocess.TimeoutExpired: failures[physical]="timeout"; lease.close(); continue
        if run.returncode==0:
          try: free=int(json.loads(run.stdout.splitlines()[-1])["free"])
          except (IndexError,KeyError,TypeError,ValueError,json.JSONDecodeError):
            failures[physical]="memory probe malformed"; lease.close(); continue
          if free < required:
            failures[physical]="free=%d required=%d"%(free,required); lease.close(); continue
          return physical,env,lease
        failures[physical]=(run.stderr or run.stdout)[-120:]; lease.close()
      time.sleep(1)
    raise RuntimeError("no device passed bounded probe: "+"; ".join(
      "%d:%s"%(physical,failures[physical]) for physical in sorted(failures)))

def job_for(req):
    cases=req["correctness_cases"] if req["action"]=="discover" else [req["case"]]
    return {"protocol_version":1,"runtime":req["candidate"]["environment"]["runtime"],
      "benchmark":"bsa","action":"check","device":0,"logical_device":0,
      "candidate":"candidate.py","baseline":"baseline.py","case_spec":"cases.jsonl",
      "cases":cases,"scope":"candidate-profile","tolerances":{"rtol":0.02,"atol":0.02},
      "fusion_contract":{"entrypoint":req["candidate"]["declared_kernel_name"],
                         "kernel_name":req["candidate"]["declared_kernel_name"]}}

def is_device_infra(text):
    lowered=text.lower()
    phrases=("tsdopen","lazysetdevice","out of memory","outofmemoryerror",
             "device unavailable","device is unavailable","npu device")
    return any(phrase in lowered for phrase in phrases) or bool(
        re.search(r"(?<![a-z0-9])oom(?![a-z0-9])",lowered))

def run_workload(req,env,output):
    job=job_for(req); jobpath=output.with_suffix(".job.json"); jobpath.write_text(json.dumps(job))
    run=subprocess.run([sys.executable,"runner.py","--job",str(jobpath),"--output",str(output)],
                       text=True,capture_output=True,env=env,timeout=300)
    try: result=json.loads(output.read_text())
    except Exception as exc: return {"status":"infrastructure_error","diagnostics":"runner response missing: "+repr(exc)}
    if run.returncode and result.get("status")=="ok":
        return {"status":"infrastructure_error","diagnostics":"runner exited %d after success"%run.returncode}
    diagnostic=str(result.get("diagnostics","" )).lower()
    if is_device_infra(diagnostic): result["status"]="infrastructure_error"
    return result

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module

def profile_workload():
    req=json.loads(pathlib.Path("request.json").read_text())
    harness=load(pathlib.Path("runner.py"),"profile_runner")
    baseline=load(pathlib.Path("baseline.py"),"profile_baseline")
    candidate=load(pathlib.Path("candidate.py"),"profile_candidate")
    cases=req["correctness_cases"] if req["action"]=="discover" else [req["case"]]
    import torch
    model=candidate.Model()
    try:
      with torch.no_grad():
        for _ in range(1):
            for case in cases:
                inputs=harness.selected_inputs(baseline,pathlib.Path("cases.jsonl"),case)
                model(*harness.clone(inputs)); torch.npu.synchronize()
    except BaseException as exc:
      text=(type(exc).__name__+": "+str(exc)).lower()
      compile_markers=("compile","compiler","triton","lowering","codegen","semantic")
      status=("infrastructure_error" if is_device_infra(text) else
              "compile_error" if any(x in text for x in compile_markers) else "runtime_error")
      pathlib.Path("profile-failure.json").write_text(json.dumps(
        {"status":status,"diagnostics":"".join(traceback.format_exception(type(exc),exc,exc.__traceback__))}))
      raise

def valid_fusion(result,cases):
    evidence=result.get("fusion_evidence")
    if not isinstance(evidence,list) or len(evidence)!=len(cases): return False
    for item in evidence:
      ops=item.get("operators",[]); triton=[op for op in ops if op.get("origin")=="triton"]
      if (len(triton)!=1 or any(op.get("origin")!="triton" for op in ops)
          or item.get("output_launch_id")!=triton[0].get("launch_id")): return False
    return True

def csvs(root,needle):
    found=sorted(root.rglob(needle+"*.csv"))
    return found or sorted(p for p in root.rglob("*.csv") if needle.lower() in p.name.lower())

def read_rows(paths):
    rows=[]
    for path in paths:
        with path.open(newline="",encoding="utf-8-sig") as stream:
            rows.extend({str(k):str(v) for k,v in row.items() if v not in (None,"")}
                        for row in csv.DictReader(stream))
    return rows

def name(row):
    direct=next((row.get(field) for field in NAMES if row.get(field)),None)
    if direct: return direct.strip()
    for field,value in row.items():
        label=field.lower().replace("_"," ")
        if "name" in label and ("op" in label or "kernel" in label): return value.strip()
    return ""

def duration(row):
    field=next((field for field in DURATIONS if row.get(field)),None)
    if not field: return None
    try: value=float(row[field])
    except ValueError: return None
    return value if math.isfinite(value) and value>0 else None

def compact_pipe(rows):
    groups={}
    for row in rows:
        identity=(row.get("block_id",""),row.get("sub_block_id",""))
        if identity in groups:
            raise ValueError("duplicate PipeUtilization identity block_id=%r sub_block_id=%r"%identity)
        groups[identity]=[row]
    result=[]
    for _key,items in sorted(groups.items()):
        merged={}
        for field in items[0]:
            values=[item.get(field,"") for item in items]
            if field in {"block_id","sub_block_id"}:
                merged[field]=values[0]; continue
            try:
                numeric=[float(value) for value in values]
                if not all(math.isfinite(value) for value in numeric): raise ValueError
                merged[field]="%.6f"%statistics.median(numeric)
            except ValueError:
                merged[field]=values[0] if len(set(values))==1 else "varies"
        result.append(merged)
    return result

def run_profile(req,env,report,physical,correctness):
    failure=pathlib.Path("profile-failure.json"); failure.unlink(missing_ok=True)
    application=shlex.join([sys.executable,"remote_driver.py","--profile-workload"])
    command=["msprof","op","--application="+application,"--output="+str(report),
             "--aic-metrics="+req["metric"],"--warm-up=0",
             "--launch-count="+("64" if req["action"]=="discover" else "1"),"--kill=off"]
    if req["action"]=="capture": command.append("--kernel-name="+req["kernel_name"])
    try: prof=subprocess.run(command,text=True,capture_output=True,env=env,timeout=600)
    except subprocess.TimeoutExpired as exc:
        return None,{"status":"infrastructure_error","diagnostics":"msprof timeout: "+str(exc),"physical_device":physical}
    transcript=prof.stdout+prof.stderr
    if prof.returncode or SUCCESS not in transcript:
        if failure.is_file():
            item=json.loads(failure.read_text()); item.update(physical_device=physical,
              correctness=correctness.get("case_evidence",[])); return None,item
        return None,{"status":"infrastructure_error","diagnostics":"msprof failed: "+transcript[-2000:],"physical_device":physical}
    return read_rows(csvs(report,"OpBasicInfo")),None

def main():
    if sys.argv[1:]==["--profile-workload"]: profile_workload(); return
    req=json.loads(pathlib.Path("request.json").read_text())
    try: physical,env,lease=pick_device()
    except Exception as exc: emit({"status":"infrastructure_error","diagnostics":str(exc)}); return
    correctness=run_workload(req,env,pathlib.Path("correctness.json"))
    if correctness.get("status")!="ok":
        status=correctness.get("status","infrastructure_error")
        if status not in {"compile_error","runtime_error","correctness_error"}: status="infrastructure_error"
        emit({"status":status,"diagnostics":str(correctness.get("diagnostics","")),
              "physical_device":physical,"correctness":correctness.get("case_evidence",[])})
        return
    expected_cases=req["correctness_cases"] if req["action"]=="discover" else [req["case"]]
    if not valid_fusion(correctness,expected_cases):
        emit({"status":"correctness_error","diagnostics":"candidate is not one fully fused logical Triton launch",
              "physical_device":physical,"correctness":correctness.get("case_evidence",[])})
        return
    declared=req["candidate"]["declared_kernel_name"]
    if req["action"]=="discover":
        report=pathlib.Path("report-discover")
        basic,failure=run_profile(req,env,report,physical,correctness)
        if failure: emit(failure); return
        names=sorted({name(row) for row in basic if name(row)})
        selected=[item for item in names if item==declared or item.startswith(declared+"_")]
        rows=[row for row in basic if name(row) in selected]
        if not selected:
            emit({"status":"infrastructure_error","diagnostics":"BasicInfo did not observe declared kernel; observed="+repr(names),"physical_device":physical}); return
        components=[]
        for selector in selected:
            kind=("aic" if selector.endswith("_mix_aic") else
                  "aiv" if selector.endswith("_mix_aiv") else None)
            components.append({"selector":selector,**({"kind":kind} if kind else {})})
        emit({"status":"ok","physical_device":physical,"correctness":correctness["case_evidence"],
              "components":components,"basic_info":{"metric":"BasicInfo","rows":rows}}); return
    selector=req["kernel_name"]
    batch_start=(req["sample_index"]//3)*3; captures=[]
    for sample_index in range(batch_start,batch_start+3):
      report=pathlib.Path("report-%d"%sample_index)
      basic,failure=run_profile(req,env,report,physical,correctness)
      if failure: emit(failure); return
      names=sorted({name(row) for row in basic if name(row)})
      if names != [selector]:
          emit({"status":"infrastructure_error","diagnostics":"selector-binding BasicInfo mismatch: "+repr(names),"physical_device":physical}); return
      try: pipe=compact_pipe(read_rows(csvs(report,"PipeUtilization")))
      except ValueError as exc:
          emit({"status":"infrastructure_error","diagnostics":str(exc),"physical_device":physical}); return
      values=[duration(row) for row in basic if name(row)==selector]
      values=[value for value in values if value is not None]
      if not pipe or len(values)!=1:
          emit({"status":"infrastructure_error","diagnostics":"capture lacks one exact PipeUtilization or duration row","physical_device":physical}); return
      captures.append({"sample_index":sample_index,"duration_us":values[0],
                       "pipe_utilization":{"metric":"PipeUtilization","selector":selector,"rows":pipe}})
    emit({"status":"ok","physical_device":physical,"correctness":correctness["case_evidence"],
          "captures":captures})

if __name__=="__main__": main()
'''


def qualification_identity(config: ExecutorConfig, logical_runtime: str,
                           campaign_revision: str) -> dict[str, Any]:
    """Return the immutable environment identity proven by qualification."""
    known_good = getattr(config, "qualification_candidate", None) or (
        Path(__file__).resolve().parents[1] / "benchmarks/bsa/qualification_candidate.py")
    return {
        "target": config.target,
        "runtime_activation": runtime_activation(config.target),
        "logical_runtime": logical_runtime,
        "campaign_revision": campaign_revision,
        "executor_sha256": file_digest(Path(__file__).resolve()),
        "baseline_sha256": file_digest(config.baseline),
        "baseline_data_sha256": file_digest(config.baseline.with_suffix(".json")),
        "cases_sha256": file_digest(config.cases),
        "runner_sha256": file_digest(config.runner),
        "remote_driver_sha256": hashlib.sha256(remote_driver_source().encode()).hexdigest(),
        "known_good_candidate_sha256": file_digest(known_good),
    }


def _stable_window(samples: list[float], width: int = 3,
                   threshold: float = 0.05) -> tuple[list[float], float] | None:
    for start in range(len(samples) - width + 1):
        window = samples[start:start + width]
        median = sorted(window)[width // 2]
        relative_range = (max(window) - min(window)) / median
        if relative_range <= threshold:
            return window, relative_range
    return None


def _candidate(source: str, kernel_name: str) -> dict[str, Any]:
    return {
        "execution_id": "qualification-" + hashlib.sha256(source.encode()).hexdigest()[:20],
        "benchmark": "bsa",
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "parameters": {},
        "environment": {"product": "a3", "runtime": "py311-torch"},
        "declared_kernel_name": kernel_name,
    }


def _intentional_compile_failure_source() -> str:
    return '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def intentional_compile_failure(output):
    tl.static_assert(False, "intentional qualification compile failure")

class Model(nn.Module):
    def forward(self, q, *_args):
        output = torch.empty_like(q)
        intentional_compile_failure[(1,)](output)
        return output
'''


def qualify(executor, config: ExecutorConfig, candidate_path: Path,
            kernel_name: str, campaign_revision: str) -> dict[str, Any]:
    """Exercise the complete GZ BSA path and return a compact gate receipt."""
    source = candidate_path.read_text()
    candidate = _candidate(source, kernel_name)
    identity = qualification_identity(config, LOGICAL_RUNTIME, campaign_revision)
    environment = execution_environment(config.target)
    handles: list[str] = []

    def record_handle(result: dict) -> None:
        handle = result.get("provenance", {}).get("handle")
        if isinstance(handle, str) and handle not in handles:
            handles.append(handle)

    def failed(reason: str, **evidence: Any) -> dict[str, Any]:
        return {"schema": QUALIFICATION_SCHEMA, "passed": False, **identity,
                "cases": QUALIFICATION_CASES, "durable_handles": handles,
                "failure": reason, **evidence}

    if candidate["source_sha256"] != identity["known_good_candidate_sha256"]:
        return failed("known-good candidate does not match the frozen qualification source")

    discovery_request = {
        "schema": REQUEST_SCHEMA, "action": "discover", "candidate": candidate,
        "correctness_cases": QUALIFICATION_CASES, "metric": "BasicInfo",
        "execution_environment": environment,
    }
    discovery = executor.execute(discovery_request)
    record_handle(discovery)
    if discovery.get("status") != "ok":
        return failed("known-good discovery failed", discovery=discovery)
    correctness = discovery.get("correctness")
    if (not isinstance(correctness, list)
            or [item.get("case") for item in correctness] != QUALIFICATION_CASES
            or not all(item.get("passed") is True for item in correctness)):
        return failed("known-good correctness did not cover all qualification cases",
                      discovery=discovery)
    components = discovery.get("components")
    if not isinstance(components, list) or not components:
        return failed("known-good discovery returned no components", discovery=discovery)

    measurements: dict[str, Any] = {}
    for case in QUALIFICATION_CASES:
        logical_samples: list[float] = []
        component_samples = {item["selector"]: [] for item in components}
        stable = None
        for sample_index in range(7):
            sample_values = []
            for component in components:
                capture_request = {
                    "schema": REQUEST_SCHEMA, "action": "capture", "candidate": candidate,
                    "case": case, "sample_index": sample_index,
                    "kernel_name": component["selector"], "component": component,
                    "metric": "PipeUtilization",
                    "execution_environment": environment,
                }
                capture = executor.execute(capture_request)
                record_handle(capture)
                if capture.get("status") != "ok":
                    return failed(f"known-good capture failed for case {case}",
                                  discovery=discovery, measurements=measurements,
                                  failed_capture=capture)
                duration = capture.get("duration_us")
                if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                        or not math.isfinite(duration) or duration <= 0):
                    return failed(f"known-good capture has invalid duration for case {case}")
                value = float(duration)
                component_samples[component["selector"]].append(value)
                sample_values.append(value)
            logical_samples.append(max(sample_values))
            stable = _stable_window(logical_samples)
            if stable is not None:
                break
        if stable is None:
            return failed(f"known-good timing is unstable for case {case}",
                          discovery=discovery, measurements=measurements,
                          unstable_samples_us=logical_samples)
        window, relative_range = stable
        measurements[str(case)] = {
            "samples_us": logical_samples,
            "stable_window_us": window,
            "relative_range": relative_range,
            "component_samples_us": component_samples,
        }

    bad_source = _intentional_compile_failure_source()
    compile_request = {
        "schema": REQUEST_SCHEMA, "action": "discover",
        "candidate": _candidate(bad_source, "intentional_compile_failure"),
        "correctness_cases": [QUALIFICATION_CASES[0]], "metric": "BasicInfo",
        "execution_environment": environment,
    }
    compile_failure = executor.execute(compile_request)
    record_handle(compile_failure)
    if (compile_failure.get("status") != "compile_error"
            or not str(compile_failure.get("diagnostics", "")).strip()):
        return failed("intentional compiler failure was not returned as compile_error",
                      discovery=discovery, measurements=measurements,
                      compile_failure=compile_failure)
    return {
        "schema": QUALIFICATION_SCHEMA, "passed": True, **identity,
        "cases": QUALIFICATION_CASES, "components": components,
        "measurements": measurements,
        "compile_failure": compile_failure, "durable_handles": handles,
    }


class Executor:
    def __init__(self, config: ExecutorConfig, transport, *, enforce_qualification: bool = True):
        if config.target not in TARGETS:
            raise ExecutorError("infrastructure_error", "unsupported A3 target")
        root = PurePosixPath(config.remote_root)
        if (".." in root.parts or str(root) in {"", "."}
                or (config.target == "gz-a3") == root.is_absolute()):
            expectation = "workspace-relative" if config.target == "gz-a3" else "absolute"
            raise ExecutorError(
                "infrastructure_error", f"remote root must be {expectation} and normalized")
        if (isinstance(config.min_free_device_bytes, bool)
                or not isinstance(config.min_free_device_bytes, int)
                or config.min_free_device_bytes <= 0):
            raise ExecutorError("infrastructure_error", "minimum free device memory must be positive")
        self.config, self.transport = config, transport
        self.enforce_qualification = enforce_qualification

    def _qualification(self, logical_runtime: str) -> tuple[str, str]:
        activation = runtime_activation(self.config.target)
        if self.config.target != "gz-a3" or not self.enforce_qualification:
            return activation, ""
        if not self.config.campaign_revision or not self.config.qualification_receipt:
            raise ExecutorError("infrastructure_error",
                                "GZ dispatch requires campaign revision and qualification receipt")
        path = self.config.qualification_receipt
        if not path.is_file():
            raise ExecutorError("infrastructure_error", "GZ qualification receipt is missing")
        receipt = json.loads(path.read_text())
        if receipt.get("schema") != QUALIFICATION_SCHEMA or receipt.get("passed") is not True:
            raise ExecutorError("infrastructure_error", "GZ qualification did not pass")
        expected = qualification_identity(
            self.config, logical_runtime, self.config.campaign_revision)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ExecutorError("infrastructure_error", "GZ qualification identity drifted")
        if receipt.get("cases") != QUALIFICATION_CASES:
            raise ExecutorError("infrastructure_error", "GZ qualification case coverage drifted")
        compile_failure = receipt.get("compile_failure")
        if (not isinstance(compile_failure, dict)
                or compile_failure.get("status") != "compile_error"
                or not str(compile_failure.get("diagnostics", "")).strip()):
            raise ExecutorError("infrastructure_error",
                                "GZ qualification lacks compiler diagnostics")
        measurements = receipt.get("measurements")
        if not isinstance(measurements, dict):
            raise ExecutorError("infrastructure_error", "GZ qualification lacks measurements")
        for case in QUALIFICATION_CASES:
            item = measurements.get(str(case))
            window = item.get("stable_window_us") if isinstance(item, dict) else None
            if (not isinstance(window, list) or len(window) != 3
                    or _stable_window([float(value) for value in window]) is None):
                raise ExecutorError("infrastructure_error",
                                    "GZ qualification has unstable case evidence")
        return activation, file_digest(path)

    def _bundle(self, request: dict, state: Path) -> Path:
        state.mkdir(parents=True, exist_ok=True)
        sources = {
            "candidate.py": request["candidate"]["source"].encode(),
            "baseline.py": self.config.baseline.read_bytes(),
            "baseline.json": self.config.baseline.with_suffix(".json").read_bytes(),
            "cases.jsonl": self.config.cases.read_bytes(),
            "runner.py": self.config.runner.read_bytes(),
            "request.json": canonical(request) + b"\n",
            "remote_driver.py": remote_driver_source().encode(),
        }
        archive = state / "payload.tar"
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as stream:
            for name, raw in sorted(sources.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime, info.uid, info.gid = len(raw), 0o444, 0, 0, 0
                import io
                stream.addfile(info, io.BytesIO(raw))
        return archive

    def execute(self, raw_request: object) -> dict:
        handle = None
        command: list[str] = []
        runtime = ""
        try:
            request = validate_request(raw_request, self.config.target)
            retry = request.get("retry")
            base_request = {key: value for key, value in request.items() if key != "retry"}
            if base_request["action"] == "capture":
                base_request["sample_index"] = (base_request["sample_index"] // 3) * 3
            request_sha = digest_json(base_request)
            environment = request["execution_environment"]
            runtime = environment["runtime"]
            activation = environment["runtime_activation"]
            qualification_activation, qualification_sha = self._qualification(runtime)
            if activation != qualification_activation:
                raise ExecutorError(
                    "infrastructure_error",
                    "request execution environment changed during qualification",
                )
            execution_sha = digest_json({
                "request_sha256": request_sha,
                "product": "a3",
                "target": self.config.target,
                "runtime": runtime,
                "runtime_activation": activation,
                "campaign_revision": self.config.campaign_revision,
                "qualification_receipt_sha256": qualification_sha,
                "min_free_device_bytes": self.config.min_free_device_bytes,
                "baseline_sha256": file_digest(self.config.baseline),
                "baseline_data_sha256": file_digest(self.config.baseline.with_suffix(".json")),
                "cases_sha256": file_digest(self.config.cases),
                "runner_sha256": file_digest(self.config.runner),
                "remote_driver_sha256": hashlib.sha256(remote_driver_source().encode()).hexdigest(),
                "executor_sha256": file_digest(Path(__file__).resolve()),
            })
            state = self.config.state_dir / execution_sha
            attempt, handle, command = self._prepare_attempt(
                base_request, request_sha, execution_sha, state, runtime, activation,
                qualification_sha, retry)
            remote = self.transport.observe_result(self.config.target, handle)
            atomic_json(attempt / "terminal.json", {
                "handle": handle, "status": remote.get("status", "malformed"),
                "terminal": True, "evidence_sha256": digest_json(remote),
            })
            response = self._response(
                request, remote, runtime, activation, command, handle)
            atomic_json(attempt / "terminal.json", {
                "handle": handle, "status": response["status"], "terminal": True,
                "evidence_sha256": response.get("evidence_sha256"),
            })
            return response
        except ExecutorError as exc:
            provenance = None
            if exc.handle or handle:
                provenance = {"product": "a3", "target": self.config.target,
                              "runtime": runtime,
                              "runtime_activation": runtime_activation(self.config.target),
                              "physical_device": -1,
                              "command": command, "handle": exc.handle or handle}
            return {"status": "infrastructure_error", "diagnostics": str(exc),
                    "terminal": exc.terminal,
                    **({"provenance": provenance} if provenance else {})}
        except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
            return {"status": "infrastructure_error", "diagnostics": str(exc),
                    "terminal": True}

    def _prepare_attempt(self, base_request, request_sha, execution_sha, state,
                         runtime, activation, qualification_sha, retry):
        state.mkdir(parents=True, exist_ok=True)
        with (state / "dispatch.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            attempts = sorted(path for path in state.glob("attempt-*") if path.is_dir())
            if not attempts:
                if retry is not None:
                    raise ExecutorError("infrastructure_error", "retry has no retained attempt")
                attempt = state / "attempt-0001"
                dispatch = True
            else:
                attempt = attempts[-1]
                receipt_path = attempt / "dispatch.json"
                if not receipt_path.is_file():
                    intent_path = attempt / "intent.json"
                    if intent_path.is_file():
                        intent = json.loads(intent_path.read_text())
                        if (intent.get("request_sha256") != request_sha
                                or intent.get("target") != self.config.target
                                or intent.get("runtime") != runtime
                                or intent.get("runtime_activation") != activation
                                or intent.get("campaign_revision")
                                != self.config.campaign_revision
                                or intent.get("qualification_receipt_sha256")
                                != qualification_sha):
                            raise ExecutorError("infrastructure_error",
                                                "incomplete attempt intent mismatch")
                    if retry is not None:
                        raise ExecutorError(
                            "infrastructure_error",
                            "incomplete attempt has no handle for explicit retry",
                            terminal=False,
                        )
                    # The remote script is idempotent for this attempt root. Reissuing a
                    # lost dispatch therefore observes or joins the same device work.
                    dispatch = True
                else:
                    receipt = self._receipt(
                        attempt, request_sha, runtime, activation, qualification_sha)
                    handle, command = receipt["handle"], receipt["command"]
                    if retry is None:
                        dispatch = False
                    elif retry["mode"] == "observe":
                        if retry["handle"] != handle:
                            raise ExecutorError("infrastructure_error", "retry handle does not match retained handle")
                        dispatch = False
                    else:
                        if retry["prior_handle"] != handle:
                            raise ExecutorError("infrastructure_error", "prior handle does not match retained handle")
                        terminal_path = attempt / "terminal.json"
                        if not terminal_path.is_file():
                            raise ExecutorError("infrastructure_error", "cannot replace a nonterminal retained handle",
                                                handle=handle, terminal=False)
                        terminal = json.loads(terminal_path.read_text())
                        if terminal.get("handle") != handle:
                            raise ExecutorError("infrastructure_error", "terminal receipt identity mismatch",
                                                handle=handle)
                        attempt = state / f"attempt-{len(attempts) + 1:04d}"
                        dispatch = True

            receipt_path = attempt / "dispatch.json"
            if dispatch:
                atomic_json(attempt / "intent.json", {
                    "schema_version": 1, "request_sha256": request_sha,
                    "target": self.config.target, "runtime": runtime,
                    "runtime_activation": activation,
                    "campaign_revision": self.config.campaign_revision,
                    "qualification_receipt_sha256": qualification_sha,
                })
                self.transport.capabilities(self.config.target)
                self.transport.preflight(self.config.target)
                archive = self._bundle(base_request, attempt)
                if activation == "target-default":
                    remote_archive = (
                        f"{self.config.remote_root}/payload-{file_digest(archive)}.tar")
                    remote_cwd = self.config.remote_root
                else:
                    remote_archive = (
                        f"{self.config.remote_root}/payload-{file_digest(archive)}.tar")
                    remote_cwd = self.config.remote_root
                self.transport.upload(self.config.target, archive, remote_archive)
                if activation == "target-default":
                    script_archive = PurePosixPath(remote_archive).name
                    run_root = f"runs/{execution_sha}/{attempt.name}"
                else:
                    script_archive = remote_archive
                    run_root = f"{self.config.remote_root}/runs/{execution_sha}/{attempt.name}"
                quoted = {"archive": shlex.quote(str(script_archive)),
                          "digest": shlex.quote(file_digest(archive)),
                          "run_root": shlex.quote(run_root)}
                script = f'''#!/usr/bin/env bash
set -euo pipefail
archive={quoted["archive"]}; run_root={quoted["run_root"]}
test "$(sha256sum "$archive" | cut -d' ' -f1)" = {quoted["digest"]}
export CPL_DEVICE_LEASE_ROOT="$HOME/.cpl-device-leases"
export BSA_MIN_FREE_DEVICE_BYTES={self.config.min_free_device_bytes}
mkdir -p "$run_root"; exec 9>"$run_root/execution.lock"; flock 9
if test -s "$run_root/compact-result.log"; then cat "$run_root/compact-result.log"; exit 0; fi
work="$run_root/work"; mkdir -p "$work"; tar -xf "$archive" -C "$work"
cd "$work"; python remote_driver.py >"$run_root/result.tmp"
grep -q '^BSA_CANDIDATE_RESULT=' "$run_root/result.tmp"
mv "$run_root/result.tmp" "$run_root/compact-result.log"
cat "$run_root/compact-result.log"
'''
                script_path = attempt / "run.sh"
                script_path.write_text(script)
                handle = self.transport.run(self.config.target, runtime, script_path,
                                            remote_cwd)
                command = getattr(self.transport, "last_command", [
                    "cpl-remote", "run", self.config.target,
                    *(["--runtime", runtime] if activation == "named-runtime" else []),
                    "--file", str(script_path), "--cwd", remote_cwd,
                ])
                atomic_json(receipt_path, {"schema_version": 1,
                            "request_sha256": request_sha, "handle": handle,
                            "target": self.config.target, "runtime": runtime,
                            "runtime_activation": activation,
                            "campaign_revision": self.config.campaign_revision,
                            "qualification_receipt_sha256": qualification_sha,
                            "command": command})
            return attempt, handle, command

    def _receipt(self, attempt: Path, request_sha: str, runtime: str,
                 activation: str, qualification_sha: str) -> dict:
        receipt_path = attempt / "dispatch.json"
        if not receipt_path.is_file():
            raise ExecutorError("infrastructure_error", "retained attempt lacks dispatch receipt")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("request_sha256") != request_sha
                or receipt.get("target") != self.config.target
                or receipt.get("runtime") != runtime
                or receipt.get("runtime_activation") != activation
                or receipt.get("campaign_revision") != self.config.campaign_revision
                or receipt.get("qualification_receipt_sha256") != qualification_sha):
            raise ExecutorError("infrastructure_error", "dispatch receipt identity mismatch")
        handle, command = receipt.get("handle"), receipt.get("command")
        if not isinstance(handle, str) or not handle.startswith(
                f"remote:{self.config.target}:job:"):
            raise ExecutorError("infrastructure_error", "dispatch handle target mismatch")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ExecutorError("infrastructure_error", "dispatch command is missing")
        return receipt

    def _response(self, request, remote, runtime, activation, command, handle):
        if not isinstance(remote, dict) or remote.get("schema") != REMOTE_SCHEMA:
            raise ExecutorError("infrastructure_error", "remote result schema mismatch",
                                handle=handle)
        status = remote.get("status")
        physical = remote.get("physical_device")
        if isinstance(physical, bool) or not isinstance(physical, int) or physical < 0:
            if status == "infrastructure_error":
                physical = -1
            else:
                raise ExecutorError("infrastructure_error", "remote physical device is missing",
                                    handle=handle)
        provenance = {"product": "a3", "target": self.config.target, "runtime": runtime,
                      "runtime_activation": activation,
                      "physical_device": physical, "command": command, "handle": handle}
        if status in CANDIDATE_FAILURES:
            return {"status": status, "diagnostics": str(remote.get("diagnostics", "")),
                    "terminal": True, "evidence_sha256": digest_json(remote),
                    "provenance": provenance}
        if status != "ok":
            return {"status": "infrastructure_error",
                    "diagnostics": str(remote.get("diagnostics", "remote execution failed")),
                    "terminal": True, "evidence_sha256": digest_json(remote),
                    "provenance": provenance}
        result = {"status": "ok", "diagnostics": "", "evidence_sha256": digest_json(remote),
                  "terminal": True, "provenance": provenance}
        if isinstance(remote.get("correctness"), list):
            result["correctness"] = remote["correctness"]
        if request["action"] == "discover":
            components, evidence = remote.get("components"), remote.get("basic_info")
            if (not isinstance(components, list) or not components
                    or not all(isinstance(item, dict)
                               and isinstance(item.get("selector"), str)
                               and item["selector"]
                               and item.get("kind") in {None, "aic", "aiv"}
                               for item in components)
                    or len({item["selector"] for item in components}) != len(components)
                    or not isinstance(evidence, dict)):
                raise ExecutorError("infrastructure_error", "invalid BasicInfo evidence",
                                    handle=handle)
            result.update(components=components, basic_info=evidence)
        else:
            capture = remote
            if isinstance(remote.get("captures"), list):
                matches = [item for item in remote["captures"]
                           if isinstance(item, dict)
                           and item.get("sample_index") == request["sample_index"]]
                if len(matches) != 1:
                    raise ExecutorError("infrastructure_error",
                                        "retained batch lacks requested independent capture",
                                        handle=handle)
                capture = matches[0]
            evidence, duration = capture.get("pipe_utilization"), capture.get("duration_us")
            if (not isinstance(evidence, dict)
                    or evidence.get("selector") != request["kernel_name"]
                    or isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ExecutorError("infrastructure_error",
                                    "invalid exact-selector PipeUtilization evidence",
                                    handle=handle)
            result.update(duration_us=float(duration), pipe_utilization=evidence)
            if capture is not remote:
                result["evidence_sha256"] = digest_json({
                    "schema": REMOTE_SCHEMA, "status": "ok", "physical_device": physical,
                    "correctness": remote.get("correctness", []), "capture": capture,
                })
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    here = Path(__file__).resolve().parents[1]
    try:
        request = json.loads(args.request.read_text())
        target = os.environ.get("BSA_A3_TARGET", "")
        remote_root = os.environ.get("BSA_A3_REMOTE_ROOT", "")
        if not target or not remote_root:
            raise ExecutorError(
                "infrastructure_error",
                "BSA_A3_TARGET and BSA_A3_REMOTE_ROOT local configuration are required",
            )
        state = Path(os.environ.get(
            "BSA_A3_STATE_DIR", str(args.response.parent / ".bsa-a3-executor-state")))
        minimum = int(os.environ.get("BSA_A3_MIN_FREE_DEVICE_BYTES", str(32 * 1024**3)))
        if minimum <= 0:
            raise ExecutorError("infrastructure_error",
                                "BSA_A3_MIN_FREE_DEVICE_BYTES must be positive")
        receipt_value = os.environ.get("BSA_A3_QUALIFICATION_RECEIPT")
        config = ExecutorConfig(
            target, remote_root, state,
            here / "benchmarks/bsa/baseline.py",
            here / "benchmarks/bsa/cases.jsonl",
            here / "scripts/a3_benchmark_runner.py", minimum,
            os.environ.get("BSA_A3_CAMPAIGN_REVISION"),
            Path(receipt_value) if receipt_value else None,
        )
        result = Executor(config, CplRemoteTransport()).execute(request)
    except (OSError, ValueError, json.JSONDecodeError, ExecutorError) as exc:
        result = {"status": "infrastructure_error", "diagnostics": str(exc)}
    atomic_json(args.response, result)
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
