from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "scripts/gz_a3_job_client.py"
RUNNER = ROOT / "scripts/a3_benchmark_runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("a3_benchmark_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_client():
    spec = importlib.util.spec_from_file_location("gz_a3_job_client", CLIENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_runner_classifies_source_and_triton_compilation_failures():
    module = load_runner()
    assert module.classify(NameError("name 'tl' is not defined")) == "compile_error"
    assert module.classify(RuntimeError("Triton compilation failed at candidate.py:17")) == "compile_error"
    assert module.classify(RuntimeError("device kernel launch failed")) == "runtime_error"


class Marker:
    def __init__(self, value):
        self.value = value
        self.npu_calls = 0

    def clone(self):
        return Marker(self.value)

    def npu(self):
        self.npu_calls += 1
        return self


def frozen_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_actual_gdn_loader_preserves_original_case_seed():
    runner = load_runner()
    baseline = frozen_module(ROOT / "benchmarks/gdn/baseline.py", "gdn_selection")
    baseline._normalized_tensor = lambda _spec, seed: Marker(seed)
    baseline._random_tensor = lambda _spec, seed, _scale=0.15: Marker(seed)
    baseline._gate_tensor = lambda _spec, seed, _raw: Marker(seed)
    baseline._beta_tensor = lambda _spec, seed, _raw: Marker(seed)
    values = runner.selected_inputs(baseline, ROOT / "benchmarks/gdn/cases.jsonl", 40)
    assert [item.value for item in values[:8]] == list(range(362, 370))
    assert all(item.npu_calls == 1 for item in values[:8])


def test_actual_bsa_loader_uses_index_seed_and_moves_tensors_to_npu():
    runner = load_runner()
    baseline = frozen_module(ROOT / "benchmarks/bsa/baseline.py", "bsa_selection")
    seeds = []

    class Tensor(Marker):
        def uniform_(self, *_args): return self
        def item(self): return 0.25
        def cumsum(self, _dim): return self
        def __iter__(self): return iter(self.value if isinstance(self.value, list) else [0])
        def __getitem__(self, _key): return self
        def __setitem__(self, _key, _value): pass

    fake = SimpleNamespace(
        float16="f16", bfloat16="bf16", int32="i32", bool="bool",
        manual_seed=lambda seed: seeds.append(seed), rand=lambda *_a: Tensor(0.25),
        empty=lambda shape, **_kw: Tensor(shape), normal=lambda _m, _s, shape, **_kw: Tensor(shape),
        tensor=lambda value, **_kw: Tensor(value), zeros=lambda *shape, **_kw: Tensor(shape),
        randperm=lambda size: Tensor(list(range(size))),
    )
    baseline.torch = fake
    values = runner.selected_inputs(baseline, ROOT / "benchmarks/bsa/cases.jsonl", 47)
    assert seeds == [3454]
    assert all(item.npu_calls == 1 for item in values[:8])


def test_reference_setup_failure_is_infrastructure_and_clones_before_timer(monkeypatch, tmp_path: Path):
    runner = load_runner()
    baseline = SimpleNamespace(Model=lambda: lambda *_args: 3)
    candidate = SimpleNamespace(Model=lambda: lambda *_args: 3)
    monkeypatch.setattr(runner, "load", lambda _path, name: baseline if name == "frozen_baseline" else candidate)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext, npu=SimpleNamespace(synchronize=lambda: None), Tensor=()))
    job = {"benchmark": "gdn", "action": "measure", "device": 0, "case": 0,
           "phase": "sample", "baseline": "baseline.py", "candidate": "candidate.py",
           "case_spec": str(tmp_path / "cases.jsonl")}
    (tmp_path / "cases.jsonl").write_text("{}\n")
    monkeypatch.setattr(runner, "selected_inputs", lambda *_args: (_ for _ in ()).throw(ValueError("bad baseline")))
    assert runner.execute(job)["status"] == "infrastructure_error"
    events = []
    monkeypatch.setattr(runner, "selected_inputs", lambda *_args: [Marker(1)])
    original_clone = runner.clone
    monkeypatch.setattr(runner, "clone", lambda value: events.append("clone") or original_clone(value))
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: events.append("timer") or len(events))
    assert runner.execute(job)["status"] == "ok"
    assert events.index("clone") < events.index("timer")


def fake_adapter(tmp_path: Path) -> Path:
    path = tmp_path / "adapter.py"
    path.write_text(r'''#!/usr/bin/env python3
import hashlib, json, os, shutil, sys, tarfile
from pathlib import Path
args=sys.argv[1:]
action=next(x for x in ("stage","run-bundle","fetch-bundle-result") if x in args)
def value(name): return args[args.index(name)+1]
if action == "stage":
 root=Path(value("--source-root")); receipt=Path(value("--receipt"))
 digest=hashlib.sha256(b"payload").hexdigest()
 receipt.write_text(json.dumps({"protocol":"a3-managed-bundle/v1","kind":"upload","root_digest":digest,"archive_sha256":"a"*64,"remote":"a3-gz","remote_path":f"incoming/profiling-workloads/{digest}/bundle.tar","state":"succeeded","transfer_handle":"upload-1"}))
 Path(os.environ["FAKE_ROOT"]).write_text(str(root))
elif action == "run-bundle":
 receipt=Path(value("--run-receipt")); root=Path(os.environ["FAKE_ROOT"]).read_text(); root=Path(root)
 job=json.loads((root/"job.json").read_text()); mode=os.environ.get("FAKE_MODE","ok")
 identity={k:job[k] for k in ("benchmark","action","device")}
 if job["action"]=="check": identity.update(cases=job["cases"],scope=job["scope"])
 else: identity["case"]=job["case"]
 if job["action"]=="measure": identity["phase"]=job["phase"]
 if job["action"]=="profile": identity.update(round=job["round"],kernel_name=job["profiling"]["kernel_name"])
 out=Path(os.environ["FAKE_OUT"]); out.mkdir(exist_ok=True)
 result={"status":mode,"diagnostics":"Triton compilation NameError at candidate.py:17" if mode=="compile_error" else "",**identity,"passed":mode=="ok"}
 if job["action"]=="measure": result["latency_us"]=10.0
 (out/"response.json").write_text(json.dumps(result))
 if job["action"]=="profile":
  (out/"profile").mkdir(exist_ok=True)
  (out/"profile/evidence.json").write_text(json.dumps({"status":"success","kernels":[{"name":identity["kernel_name"],"duration_us":{"median":7.5}}]}))
  (out/"profile/msprof.log").write_text("Profiling finished\n")
 archive=Path(os.environ["FAKE_TAR"])
 with tarfile.open(archive,"w") as stream:
  stream.add(out/"response.json",arcname="response.json")
  if job["action"]=="profile":
   stream.add(out/"profile/evidence.json",arcname="profile/evidence.json")
   stream.add(out/"profile/msprof.log",arcname="profile/msprof.log")
 sha=hashlib.sha256(archive.read_bytes()).hexdigest()
 receipt.write_text(json.dumps({"protocol":"a3-managed-bundle/v1","kind":"run","state":"succeeded","command_handle":"command-1","result_sha256":sha,"result_path":"results/x/result.tar"}))
elif action == "fetch-bundle-result":
 shutil.copyfile(os.environ["FAKE_TAR"],value("--output"))
''')
    path.chmod(0o755)
    return path


def inputs(tmp_path: Path) -> tuple[dict, dict]:
    for name in ("candidate.py", "baseline.py", "cases.jsonl"):
        (tmp_path / name).write_text("{}\n" if name.endswith("jsonl") else "# source\n")
    job = {
        "protocol_version": 1, "profile": "gz-a3", "runtime": "py311-torch",
        "benchmark": "gdn", "action": "profile", "device": 0, "logical_device": 0,
        "candidate": str(tmp_path / "candidate.py"), "baseline": str(tmp_path / "baseline.py"),
        "case_spec": str(tmp_path / "cases.jsonl"), "case": 40, "round": 1,
        "profiling": {"kernel_name": "candidate_kernel"},
    }
    adapter = fake_adapter(tmp_path)
    env = {**os.environ, "FAKE_ROOT": str(tmp_path / "root.txt"),
           "FAKE_OUT": str(tmp_path / "remote"), "FAKE_TAR": str(tmp_path / "result.tar")}
    command = [sys.executable, str(CLIENT), "--adapter-json", json.dumps([str(adapter)]),
               "--state-dir", str(tmp_path / "state")]
    return job, {"env": env, "command": command}


def run(tmp_path: Path, mode: str = "ok"):
    job, config = inputs(tmp_path)
    config["env"]["FAKE_MODE"] = mode
    process = subprocess.run(config["command"], input=json.dumps(job), text=True,
                             capture_output=True, env=config["env"], check=False)
    return process, json.loads(process.stdout)


def test_profile_returns_bound_msprof_evidence_and_handle(tmp_path: Path):
    process, result = run(tmp_path)
    assert process.returncode == 0
    assert result["status"] == "ok"
    assert result["handle"] == "gz-a3:command-1"
    assert result["kernel_name"] == "candidate_kernel"
    assert result["latency_us"] == 7.5
    assert result["profile"]["status"] == "success"
    assert Path(result["artifacts"]["msprof_log"]).read_text() == "Profiling finished\n"


def test_compilation_diagnostic_is_counted_and_retrieved(tmp_path: Path):
    process, result = run(tmp_path, "compile_error")
    assert process.returncode == 2
    assert result["status"] == "compile_error"
    assert "NameError" in result["diagnostics"]
    assert result["handle"] == "gz-a3:command-1"


def test_adapter_failure_is_infrastructure_and_keeps_identity(tmp_path: Path):
    job, config = inputs(tmp_path)
    config["command"][3] = json.dumps(["/missing/adapter"])
    process = subprocess.run(config["command"], input=json.dumps(job), text=True,
                             capture_output=True, env=config["env"], check=False)
    result = json.loads(process.stdout)
    assert result["status"] == "infrastructure_error"
    assert (result["benchmark"], result["case"], result["device"]) == ("gdn", 40, 0)
    assert "managed adapter unavailable" in result["diagnostics"]


def test_same_content_reuses_state_and_receipts(tmp_path: Path):
    first, one = run(tmp_path)
    second, two = run(tmp_path)
    assert first.returncode == second.returncode == 0
    assert one["artifacts"]["request_digest"] == two["artifacts"]["request_digest"]
    assert len(list((tmp_path / "state").iterdir())) == 1


def test_digest_ignores_caller_checkout_in_profile_driver(tmp_path: Path):
    client = load_client()
    job, _config = inputs(tmp_path)
    job["profiling"]["driver"] = "/first/checkout/scripts/profile_a3.py"
    first = client.prepare(job, tmp_path / "state-a", RUNNER, ROOT / "scripts/profile_a3.py")[1]
    job["profiling"]["driver"] = "/other/checkout/scripts/profile_a3.py"
    second = client.prepare(job, tmp_path / "state-b", RUNNER, ROOT / "scripts/profile_a3.py")[1]
    assert first == second


def test_partial_retained_fetch_is_quarantined_and_refetched(tmp_path: Path):
    first, result = run(tmp_path)
    assert first.returncode == 0
    state = tmp_path / "state" / result["artifacts"]["request_digest"]
    shutil.rmtree(state / "result")
    (state / "result.tar").write_bytes(b"partial")
    second, recovered = run(tmp_path)
    assert second.returncode == 0
    assert recovered["status"] == "ok"
    assert list(state.glob("result.invalid-*.tar"))
