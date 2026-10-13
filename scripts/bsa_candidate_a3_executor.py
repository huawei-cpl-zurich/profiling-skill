#!/usr/bin/env python3
"""Production A3 adapter for ``bsa_candidate_profiler.py``.

The adapter is deliberately transport-only on the controller.  A compact,
self-contained bundle performs correctness, kernel discovery, msprof capture,
and report parsing in one retained cpl-remote job.
"""

from __future__ import annotations

import argparse
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
CANDIDATE_FAILURES = {"compile_error", "runtime_error", "correctness_error"}
TARGETS = {"bz-a3-1", "bz-a3-2"}


class ExecutorError(RuntimeError):
    def __init__(self, status: str, message: str, *, handle: str | None = None):
        self.status, self.handle = status, handle
        super().__init__(message)


@dataclass(frozen=True)
class ExecutorConfig:
    target: str
    remote_root: str
    state_dir: Path
    baseline: Path
    cases: Path
    runner: Path


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


def validate_request(value: object) -> dict[str, Any]:
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
            or not isinstance(environment.get("runtime"), str)
            or not environment["runtime"]):
        raise ExecutorError("infrastructure_error", "candidate lacks an A3 named runtime")
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
        if receipt.get("target") != target or receipt.get("state") != "completed":
            raise ExecutorError("infrastructure_error", "bundle upload did not complete")

    def run(self, target: str, runtime: str, script: Path, cwd: str) -> str:
        arguments = ["run", target, "--runtime", runtime, "--file", str(script),
                     "--cwd", cwd, "--timeout", "900"]
        self.last_command = [str(self.executable), "--json", *arguments]
        receipt = self._call(arguments, 930)
        handle = receipt.get("handle")
        if not isinstance(handle, str) or not handle.startswith(f"remote:{target}:job:"):
            raise ExecutorError("infrastructure_error", "run returned no valid durable handle")
        return handle

    def observe_result(self, target: str, handle: str) -> dict:
        observed = self._call(["observe", handle, "--wait", "--timeout", "900"], 930)
        if observed.get("handle") != handle:
            raise ExecutorError("infrastructure_error", "observation identity mismatch",
                                handle=handle)
        if observed.get("state") not in {"completed", "failed", "cancelled"}:
            raise ExecutorError("infrastructure_error", "retained job is not terminal",
                                handle=handle)
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
import csv, hashlib, json, math, os, pathlib, shlex, statistics, subprocess, sys, tempfile
import importlib.util

SCHEMA="profiling-skill/bsa-candidate-remote-result/v1"
SUCCESS="Profiling running finished. All task success."
NAMES=("Op Name","OpName","Kernel Name","kernel_name","Name")
DURATIONS=("Task Duration(us)","Task Duration (us)","task_duration_us")

def emit(value):
    value.setdefault("schema",SCHEMA)
    print("BSA_CANDIDATE_RESULT="+json.dumps(value,sort_keys=True,separators=(",",":")),flush=True)

def pick_device():
    probe="import torch,torch_npu;torch.npu.set_device(0);x=torch.ones(1,device='npu');torch.npu.synchronize();assert float(x.cpu()[0])==1"
    failures=[]
    for physical in range(8):
        env=os.environ.copy(); env["ASCEND_RT_VISIBLE_DEVICES"]=str(physical); env["ASCEND_DEVICE_ID"]="0"
        try: run=subprocess.run([sys.executable,"-c",probe],env=env,text=True,capture_output=True,timeout=20)
        except subprocess.TimeoutExpired: failures.append("%d:timeout"%physical); continue
        if run.returncode==0: return physical,env
        failures.append("%d:%s"%(physical,(run.stderr or run.stdout)[-120:]))
    raise RuntimeError("no device passed bounded probe: "+"; ".join(failures))

def job_for(req):
    cases=req["correctness_cases"] if req["action"]=="discover" else [req["case"]]
    return {"protocol_version":1,"runtime":req["candidate"]["environment"]["runtime"],
      "benchmark":"bsa","action":"check","device":0,"logical_device":0,
      "candidate":"candidate.py","baseline":"baseline.py","case_spec":"cases.jsonl",
      "cases":cases,"scope":"candidate-profile","tolerances":{"rtol":0.02,"atol":0.02}}

def run_workload(req,env,output):
    job=job_for(req); jobpath=output.with_suffix(".job.json"); jobpath.write_text(json.dumps(job))
    run=subprocess.run([sys.executable,"runner.py","--job",str(jobpath),"--output",str(output)],
                       text=True,capture_output=True,env=env,timeout=300)
    try: result=json.loads(output.read_text())
    except Exception as exc: return {"status":"infrastructure_error","diagnostics":"runner response missing: "+repr(exc)}
    if run.returncode and result.get("status")=="ok":
        return {"status":"infrastructure_error","diagnostics":"runner exited %d after success"%run.returncode}
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
    with torch.no_grad():
        repeats=1 if req["action"]=="discover" else 7
        for _ in range(repeats):
            for case in cases:
                inputs=harness.selected_inputs(baseline,pathlib.Path("cases.jsonl"),case)
                model(*harness.clone(inputs)); torch.npu.synchronize()

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
        groups.setdefault((row.get("block_id",""),row.get("sub_block_id","")),[]).append(row)
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

def main():
    if sys.argv[1:]==["--profile-workload"]: profile_workload(); return
    req=json.loads(pathlib.Path("request.json").read_text())
    try: physical,env=pick_device()
    except Exception as exc: emit({"status":"infrastructure_error","diagnostics":str(exc)}); return
    correctness=run_workload(req,env,pathlib.Path("correctness.json"))
    if correctness.get("status")!="ok":
        status=correctness.get("status","infrastructure_error")
        if status not in {"compile_error","runtime_error","correctness_error"}: status="infrastructure_error"
        emit({"status":status,"diagnostics":str(correctness.get("diagnostics","")),
              "physical_device":physical,"correctness":correctness.get("case_evidence",[])})
        return
    report=pathlib.Path("report"); application=shlex.join([
        sys.executable,"remote_driver.py","--profile-workload"])
    metric=req["metric"]
    command=["msprof","op","--application="+application,"--output="+str(report),
             "--aic-metrics="+metric,"--warm-up=0",
             "--launch-count="+("64" if req["action"]=="discover" else "7"),"--kill=off"]
    if req["action"]=="capture": command.append("--kernel-name="+req["kernel_name"])
    try: prof=subprocess.run(command,text=True,capture_output=True,env=env,timeout=600)
    except subprocess.TimeoutExpired as exc:
        emit({"status":"infrastructure_error","diagnostics":"msprof timeout: "+str(exc),"physical_device":physical}); return
    transcript=prof.stdout+prof.stderr
    if prof.returncode or SUCCESS not in transcript:
        emit({"status":"infrastructure_error","diagnostics":"msprof failed: "+transcript[-2000:],"physical_device":physical}); return
    basic=read_rows(csvs(report,"OpBasicInfo")); names=sorted({name(row) for row in basic if name(row)})
    declared=req["candidate"]["declared_kernel_name"]
    if req["action"]=="discover":
        selected=[item for item in names if item==declared or item.startswith(declared+"_")]
        rows=[row for row in basic if name(row) in selected]
        if not selected:
            emit({"status":"infrastructure_error","diagnostics":"BasicInfo did not observe declared kernel; observed="+repr(names),"physical_device":physical}); return
        emit({"status":"ok","physical_device":physical,"correctness":correctness["case_evidence"],
              "kernel_names":selected,"basic_info":{"metric":"BasicInfo","rows":rows}}); return
    selector=req["kernel_name"]
    if sorted({item for item in names if item}) != [selector]:
        emit({"status":"infrastructure_error","diagnostics":"selector-binding BasicInfo mismatch: "+repr(names),"physical_device":physical}); return
    pipe=compact_pipe(read_rows(csvs(report,"PipeUtilization"))); values=[duration(row) for row in basic if name(row)==selector]
    values=[value for value in values if value is not None]
    if not pipe or not values:
        emit({"status":"infrastructure_error","diagnostics":"capture lacks exact PipeUtilization or duration rows","physical_device":physical}); return
    emit({"status":"ok","physical_device":physical,"correctness":correctness["case_evidence"],
          "duration_us":values[0],"duration_samples_us":values,"pipe_utilization":{
          "metric":"PipeUtilization","selector":selector,"rows":pipe}})

if __name__=="__main__": main()
'''


class Executor:
    def __init__(self, config: ExecutorConfig, transport):
        if config.target not in TARGETS:
            raise ExecutorError("infrastructure_error", "unsupported A3 target")
        if (not config.remote_root.startswith("/") or ".." in PurePosixPath(config.remote_root).parts):
            raise ExecutorError("infrastructure_error", "remote root must be absolute and normalized")
        self.config, self.transport = config, transport

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
        try:
            request = validate_request(raw_request)
            cached = self._cached_capture(request)
            if cached is not None:
                return cached
            request_sha = digest_json(request)
            execution_sha = digest_json({
                "request_sha256": request_sha,
                "baseline_sha256": file_digest(self.config.baseline),
                "baseline_data_sha256": file_digest(self.config.baseline.with_suffix(".json")),
                "cases_sha256": file_digest(self.config.cases),
                "runner_sha256": file_digest(self.config.runner),
                "remote_driver_sha256": hashlib.sha256(remote_driver_source().encode()).hexdigest(),
                "executor_sha256": file_digest(Path(__file__).resolve()),
            })
            state = self.config.state_dir / execution_sha
            receipt_path = state / "dispatch.json"
            runtime = request["candidate"]["environment"]["runtime"]
            command: list[str] = []
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("request_sha256") != request_sha:
                    raise ExecutorError("infrastructure_error", "dispatch receipt mismatch")
                handle = receipt.get("handle")
                command = receipt.get("command")
                if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
                    raise ExecutorError("infrastructure_error", "dispatch command is missing")
            else:
                self.transport.capabilities(self.config.target)
                self.transport.preflight(self.config.target)
                archive = self._bundle(request, state)
                remote_archive = f"{self.config.remote_root}/payload-{file_digest(archive)}.tar"
                self.transport.upload(self.config.target, archive, remote_archive)
                run_root = f"{self.config.remote_root}/runs/{execution_sha}"
                quoted = {"archive": shlex.quote(remote_archive),
                          "digest": shlex.quote(file_digest(archive)),
                          "run_root": shlex.quote(run_root)}
                script = f'''#!/usr/bin/env bash
set -euo pipefail
archive={quoted["archive"]}; run_root={quoted["run_root"]}
test "$(sha256sum "$archive" | cut -d' ' -f1)" = {quoted["digest"]}
rm -rf "$run_root"; mkdir -p "$run_root"; tar -xf "$archive" -C "$run_root"
cd "$run_root"; exec python remote_driver.py
'''
                script_path = state / "run.sh"
                script_path.write_text(script)
                handle = self.transport.run(self.config.target, runtime, script_path,
                                            self.config.remote_root)
                command = getattr(self.transport, "last_command", [
                    "cpl-remote", "run", self.config.target, "--runtime", runtime,
                    "--file", str(script_path), "--cwd", self.config.remote_root,
                ])
                atomic_json(receipt_path, {"schema_version": 1,
                            "request_sha256": request_sha, "handle": handle,
                            "target": self.config.target, "runtime": runtime,
                            "command": command})
            remote = self.transport.observe_result(self.config.target, handle)
            response = self._response(request, remote, runtime, command, handle)
            self._cache_captures(request, remote, response)
            return response
        except ExecutorError as exc:
            provenance = None
            if exc.handle or handle:
                provenance = {"target": self.config.target, "runtime": (
                    raw_request.get("candidate", {}).get("environment", {}).get("runtime", "")
                    if isinstance(raw_request, dict) else ""),
                    "physical_device": -1, "command": [], "handle": exc.handle or handle}
            return {"status": "infrastructure_error", "diagnostics": str(exc),
                    **({"provenance": provenance} if provenance else {})}
        except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
            return {"status": "infrastructure_error", "diagnostics": str(exc)}

    def _cache_path(self, request: dict) -> Path:
        candidate = request["candidate"]
        identity = {"source_sha256": candidate["source_sha256"],
                    "parameters": candidate.get("parameters", {}),
                    "environment": candidate["environment"], "case": request.get("case"),
                    "kernel_name": request.get("kernel_name"),
                    "baseline_sha256": file_digest(self.config.baseline),
                    "cases_sha256": file_digest(self.config.cases),
                    "runner_sha256": file_digest(self.config.runner),
                    "executor_sha256": file_digest(Path(__file__).resolve())}
        return self.config.state_dir / "capture-cache" / (digest_json(identity) + ".json")

    def _cached_capture(self, request: dict) -> dict | None:
        if request["action"] != "capture":
            return None
        path = self._cache_path(request)
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        index = request.get("sample_index")
        samples = value.get("samples")
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                or not isinstance(samples, list) or index >= len(samples)):
            return None
        item = {"duration_us": samples[index],
                "pipe_utilization": value["pipe_utilization"],
                "sample_index": index, "remote_evidence_sha256": value["evidence_sha256"]}
        return {"status": "ok", "diagnostics": "", "duration_us": samples[index],
                "pipe_utilization": value["pipe_utilization"],
                "evidence_sha256": digest_json(item), "provenance": value["provenance"]}

    def _cache_captures(self, request: dict, remote: dict, response: dict) -> None:
        if request["action"] != "capture" or response.get("status") != "ok":
            return
        samples = remote.get("duration_samples_us", [response["duration_us"]])
        if (not isinstance(samples, list) or not samples or len(samples) > 7
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in samples)):
            return
        atomic_json(self._cache_path(request), {
            "schema": "profiling-skill/bsa-candidate-capture-cache/v1",
            "samples": [float(value) for value in samples],
            "pipe_utilization": response["pipe_utilization"],
            "evidence_sha256": response["evidence_sha256"],
            "provenance": response["provenance"],
        })

    def _response(self, request, remote, runtime, command, handle):
        if not isinstance(remote, dict) or remote.get("schema") != REMOTE_SCHEMA:
            raise ExecutorError("infrastructure_error", "remote result schema mismatch",
                                handle=handle)
        physical = remote.get("physical_device")
        if isinstance(physical, bool) or not isinstance(physical, int) or physical < 0:
            raise ExecutorError("infrastructure_error", "remote physical device is missing",
                                handle=handle)
        provenance = {"target": self.config.target, "runtime": runtime,
                      "physical_device": physical, "command": command, "handle": handle}
        status = remote.get("status")
        if status in CANDIDATE_FAILURES:
            return {"status": status, "diagnostics": str(remote.get("diagnostics", "")),
                    "evidence_sha256": digest_json(remote), "provenance": provenance}
        if status != "ok":
            return {"status": "infrastructure_error",
                    "diagnostics": str(remote.get("diagnostics", "remote execution failed")),
                    "evidence_sha256": digest_json(remote), "provenance": provenance}
        result = {"status": "ok", "diagnostics": "", "evidence_sha256": digest_json(remote),
                  "provenance": provenance}
        if request["action"] == "discover":
            names, evidence = remote.get("kernel_names"), remote.get("basic_info")
            if (not isinstance(names, list) or not names
                    or not all(isinstance(name, str) and name for name in names)
                    or not isinstance(evidence, dict)):
                raise ExecutorError("infrastructure_error", "invalid BasicInfo evidence",
                                    handle=handle)
            result.update(kernel_names=names, basic_info=evidence)
        else:
            evidence, duration = remote.get("pipe_utilization"), remote.get("duration_us")
            samples = remote.get("duration_samples_us")
            sample_index = request.get("sample_index", 0)
            if (isinstance(samples, list) and not isinstance(sample_index, bool)
                    and isinstance(sample_index, int) and 0 <= sample_index < len(samples)):
                duration = samples[sample_index]
            if (not isinstance(evidence, dict)
                    or evidence.get("selector") != request["kernel_name"]
                    or isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ExecutorError("infrastructure_error",
                                    "invalid exact-selector PipeUtilization evidence",
                                    handle=handle)
            result.update(duration_us=float(duration), pipe_utilization=evidence)
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
        config = ExecutorConfig(target, remote_root, state,
                                here / "benchmarks/bsa/baseline.py",
                                here / "benchmarks/bsa/cases.jsonl",
                                here / "scripts/a3_benchmark_runner.py")
        result = Executor(config, CplRemoteTransport()).execute(request)
    except (OSError, ValueError, json.JSONDecodeError, ExecutorError) as exc:
        result = {"status": "infrastructure_error", "diagnostics": str(exc)}
    atomic_json(args.response, result)
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
