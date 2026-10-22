from __future__ import annotations

import json
import importlib.util
import functools
import os
import shutil
import subprocess
import sys
import tarfile
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "scripts/gz_a3_job_client.py"
RUNNER = ROOT / "scripts/a3_benchmark_runner.py"
FUSION_CONTRACT = ROOT / "scripts/fully_fused_contract.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("a3_benchmark_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_fusion_contract():
    spec = importlib.util.spec_from_file_location("fully_fused_contract_test", FUSION_CONTRACT)
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

    class CompilationError(Exception):
        pass

    assert module.classify(NameError("name 'tl' is not defined")) == "compile_error"
    assert module.classify(CompilationError("at 17: invalid operands")) == "compile_error"
    assert module.classify(RuntimeError("Triton compilation failed at candidate.py:17")) == "compile_error"
    assert module.classify(RuntimeError("device kernel launch failed")) == "runtime_error"
    assert module.classify(RuntimeError("Triton device kernel launch failed")) == "runtime_error"


def test_runner_classifies_lazy_jit_failure_during_candidate_execution(monkeypatch, tmp_path: Path):
    module = load_runner()

    class CompilationError(Exception):
        pass

    class BaselineModel:
        def __call__(self, *_inputs):
            return "expected"

    class CandidateModel:
        def __call__(self, *_inputs):
            raise CompilationError("at 23: unsupported tensor index")

    baseline = SimpleNamespace(Model=BaselineModel)
    candidate = SimpleNamespace(Model=CandidateModel)
    fake_torch = SimpleNamespace(
        no_grad=nullcontext,
        npu=SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        module, "load",
        lambda path, _name: baseline if path.name == "baseline.py" else candidate,
    )
    monkeypatch.setattr(module, "selected_inputs", lambda *_args: (object(),))
    monkeypatch.setattr(module, "clone", lambda value: value)

    result = module.execute({
        "benchmark": "bsa",
        "action": "check",
        "device": 0,
        "cases": [47],
        "scope": "development",
        "baseline": str(tmp_path / "baseline.py"),
        "candidate": str(tmp_path / "candidate.py"),
        "case_spec": str(tmp_path / "cases.jsonl"),
    })

    assert result["status"] == "compile_error"
    assert result["case_evidence"] == []
    assert "CompilationError: at 23: unsupported tensor index" in result["diagnostics"]


def test_runner_applies_the_job_specific_tolerance_contract():
    import torch

    runner = load_runner()
    expected = torch.tensor([1.0])
    actual = torch.tensor([1.015])
    assert runner.compare(actual, expected, rtol=1e-2, atol=0)[0] is False
    assert runner.compare(actual, expected, rtol=2e-2, atol=0)[0] is True


def test_runner_acquires_trusted_launch_output_and_framework_evidence():
    runner = load_runner()
    declaration = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "complete_kernel_mix_aiv", "entrypoint": "complete_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }

    class Storage:
        def __init__(self, pointer): self.pointer = pointer
        def data_ptr(self): return self.pointer

    class Tensor:
        device = "npu:0"
        def __init__(self, pointer): self.pointer = pointer
        def untyped_storage(self): return Storage(self.pointer)

    class Kernel:
        def __getitem__(self, _grid): return lambda *_args, **_kwargs: None

    Kernel.__module__ = "triton.runtime.jit"
    candidate = types.ModuleType("candidate")
    candidate.complete_kernel = Kernel()
    source, output = Tensor(1), Tensor(2)
    audit = runner.FusionRuntimeAudit(declaration)
    audit.instrument(candidate)
    candidate.complete_kernel[(1,)](source, output)
    evidence = audit.finish(7, output)
    assert evidence["output_launch_id"] == "launch-0"
    assert evidence["operators"] == [{
        "name": "complete_kernel_mix_aiv", "origin": "triton",
        "entrypoint": "complete_kernel", "launch_id": "launch-0",
        "component": "aiv",
    }]

    audit.framework_ops.append(("aten.matmul.default", "torch"))
    assert audit.finish(7, output)["operators"][-1]["origin"] == "torch"


def test_runner_captures_auxiliary_triton_global_and_gate_rejects_two_launches():
    runner = load_runner()
    contract = load_fusion_contract()
    declaration = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "main_kernel_mix_aiv", "entrypoint": "main_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }

    class Storage:
        def __init__(self, pointer): self.pointer = pointer
        def data_ptr(self): return self.pointer

    class Tensor:
        device = "npu:0"
        def __init__(self, pointer): self.pointer = pointer
        def untyped_storage(self): return Storage(self.pointer)

    class Kernel:
        def __getitem__(self, _grid): return lambda *_args, **_kwargs: None

    Kernel.__module__ = "triton.runtime.jit"
    candidate = types.ModuleType("candidate")
    candidate.main_kernel = Kernel()
    candidate.helper_kernel = Kernel()
    source, temporary, output = Tensor(1), Tensor(2), Tensor(3)
    audit = runner.FusionRuntimeAudit(declaration)
    audit.instrument(candidate)
    candidate.main_kernel[(1,)](source, temporary)
    candidate.helper_kernel[(1,)](temporary, output)
    evidence = audit.finish(4, output)
    assert [(op["entrypoint"], op["launch_id"]) for op in evidence["operators"]] == [
        ("main_kernel", "launch-0"), ("helper_kernel", "launch-1"),
    ]
    with pytest.raises(contract.FusionContractError, match="exactly one logical"):
        contract.validate_fusion_evidence(declaration, [evidence], expected_cases=[4])


def test_runner_instruments_entrypoint_retained_on_model_class():
    runner = load_runner()
    declaration = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "complete_kernel_mix_aiv", "entrypoint": "complete_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }

    class Kernel:
        def __getitem__(self, _grid): return lambda *_args, **_kwargs: None

    Kernel.__module__ = "triton.runtime.jit"
    candidate = types.ModuleType("candidate")
    candidate.complete_kernel = Kernel()

    class Model:
        kernel = candidate.complete_kernel

    candidate.Model = Model
    audit = runner.FusionRuntimeAudit(declaration)
    audit.instrument(candidate)
    model = candidate.Model()
    model.kernel[(1,)]("input", "output")
    evidence = audit.finish(0, "output")
    assert evidence["operators"] == [{
        "name": "complete_kernel_mix_aiv", "origin": "triton",
        "entrypoint": "complete_kernel", "launch_id": "launch-0",
        "component": "aiv",
    }]


def test_runner_observes_direct_jit_run_launch_without_changing_arguments():
    runner = load_runner()
    declaration = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "complete_kernel_mix_aiv", "entrypoint": "complete_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }
    calls = []

    class Kernel:
        def run(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "launched"

        def __getitem__(self, _grid): return lambda *_args, **_kwargs: None

    Kernel.__module__ = "triton.runtime.jit"
    candidate = types.ModuleType("candidate")
    candidate.complete_kernel = Kernel()
    audit = runner.FusionRuntimeAudit(declaration)
    audit.instrument(candidate)

    assert candidate.complete_kernel.run("input", "output", grid=(1,)) == "launched"
    assert calls == [(('input', 'output'), {"grid": (1,)})]
    evidence = audit.finish(0, "output")
    assert evidence["operators"] == [{
        "name": "complete_kernel_mix_aiv", "origin": "triton",
        "entrypoint": "complete_kernel", "launch_id": "launch-0",
        "component": "aiv",
    }]


@pytest.mark.parametrize("retention", ["closure", "default", "partial"])
def test_runner_observes_entrypoint_captured_before_audit(retention):
    runner = load_runner()
    declaration = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "complete_kernel_mix_aiv", "entrypoint": "complete_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }

    class Kernel:
        def run(self, *args, **kwargs):
            return args[-1]

        def __getitem__(self, grid):
            return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    Kernel.__module__ = "triton.runtime.jit"
    candidate = types.ModuleType("candidate")
    candidate.complete_kernel = Kernel()
    captured = candidate.complete_kernel
    if retention == "closure":
        def invoke(source, output):
            return captured[(1,)](source, output)
    elif retention == "default":
        def invoke(source, output, kernel=captured):
            return kernel[(1,)](source, output)
    else:
        def launch(kernel, source, output):
            return kernel[(1,)](source, output)
        invoke = functools.partial(launch, captured)

    audit = runner.FusionRuntimeAudit(declaration)
    audit.instrument(candidate)
    assert invoke("input", "output") == "output"
    evidence = audit.finish(0, "output")
    assert evidence["operators"] == [{
        "name": "complete_kernel_mix_aiv", "origin": "triton",
        "entrypoint": "complete_kernel", "launch_id": "launch-0",
        "component": "aiv",
    }]


@pytest.mark.parametrize(
    "name",
    [
        "aten.slice.Tensor", "aten.select.int", "aten.expand.default",
        "aten.permute.default", "aten.transpose.int", "aten.view.default",
    ],
)
def test_fusion_dispatch_allows_exact_metadata_operator_overloads(name):
    runner = load_runner()
    declaration = {
        "kernel_name": "kernel_mix_aiv", "entrypoint": "kernel",
    }
    audit = runner.FusionRuntimeAudit(declaration)

    class Operator:
        def __str__(self): return name
        def __call__(self, *_args, **_kwargs): return "result"

    assert audit.dispatch_mode().__torch_dispatch__(Operator(), (), (), {}) == "result"
    assert audit.framework_ops == []


@pytest.mark.parametrize(
    "name",
    [
        "aten.slice_scatter.default", "aten.select_scatter.default",
        "aten.expand_copy.default", "aten.permute_copy.default",
        "aten.transpose_copy.int",
    ],
)
def test_fusion_dispatch_rejects_compute_ops_with_metadata_prefixes(name):
    runner = load_runner()
    audit = runner.FusionRuntimeAudit({
        "kernel_name": "kernel_mix_aiv", "entrypoint": "kernel",
    })

    class Operator:
        def __str__(self): return name
        def __call__(self, *_args, **_kwargs): return "result"

    assert audit.dispatch_mode().__torch_dispatch__(Operator(), (), (), {}) == "result"
    assert audit.framework_ops == [(name, "torch")]


def test_fusion_dispatch_allows_aliasing_reshape_but_rejects_copying_reshape():
    import torch

    runner = load_runner()

    class Reshape:
        _schema = SimpleNamespace(name="aten::reshape")

        def __init__(self, result): self.result = result
        def __str__(self): return "aten.reshape.default"
        def __call__(self, *_args, **_kwargs): return self.result

    source = torch.arange(12).reshape(3, 4)
    alias = source.reshape(2, 6)
    copied = source.t().reshape(12)
    assert alias.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert copied.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()

    alias_audit = runner.FusionRuntimeAudit({
        "kernel_name": "kernel_mix_aiv", "entrypoint": "kernel",
    })
    assert alias_audit.dispatch_mode().__torch_dispatch__(
        Reshape(alias), (), (source, [2, 6]), {},
    ) is alias
    assert alias_audit.framework_ops == []

    copy_audit = runner.FusionRuntimeAudit({
        "kernel_name": "kernel_mix_aiv", "entrypoint": "kernel",
    })
    assert copy_audit.dispatch_mode().__torch_dispatch__(
        Reshape(copied), (), (source.t(), [12]), {},
    ) is copied
    assert copy_audit.framework_ops == [("aten.reshape.default", "torch")]


def test_runner_check_emits_fusion_evidence_from_the_real_candidate_forward(
    monkeypatch, tmp_path: Path,
):
    import torch

    runner = load_runner()

    class Kernel:
        def __getitem__(self, _grid):
            def launch(source, output):
                with torch._C._DisableTorchDispatch():
                    output.copy_(source)
            return launch

    Kernel.__module__ = "triton.runtime.jit"
    baseline = types.ModuleType("baseline")
    baseline.Model = lambda: lambda value: value.clone()
    candidate = types.ModuleType("candidate")
    candidate.complete_kernel = Kernel()
    candidate.torch = torch
    exec(
        "class Model:\n"
        " def __init__(self):\n"
        "  self.kernel = complete_kernel\n"
        " def __call__(self, value):\n"
        "  output = torch.empty_like(value)\n"
        "  self.kernel[(1,)](value, output)\n"
        "  return output\n",
        candidate.__dict__,
    )
    monkeypatch.setattr(
        runner, "load",
        lambda _path, name: baseline if name == "frozen_baseline" else candidate,
    )
    monkeypatch.setattr(runner, "selected_inputs", lambda *_args: (torch.tensor([2.0]),))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n")
    declaration = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "complete_kernel_mix_aiv", "entrypoint": "complete_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }
    result = runner.execute({
        "benchmark": "matmul", "action": "check", "device": 0,
        "cases": [0], "scope": "full", "baseline": "baseline.py",
        "candidate": "candidate.py", "case_spec": str(cases),
        "fusion_contract": declaration,
    })
    assert result["status"] == "ok", result.get("diagnostics")
    assert result["fusion_evidence"][0]["output_launch_id"] == "launch-0"
    assert result["fusion_evidence"][0]["operators"][0]["origin"] == "triton"


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


def test_protocol_v1_default_tolerance_and_reference_setup_failure(monkeypatch, tmp_path: Path):
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
    result = runner.execute(job)
    assert result["status"] == "ok"
    assert events.index("clone") < events.index("timer")


def test_attach_profile_evidence_accepts_resolved_and_legacy_exact_selectors(tmp_path: Path):
    module = load_client()
    rows = [{"case": 40, "samples_us": [7.0], "median_us": 7.0}]
    legacy = {
        "status": "success", "kernel_name": "gdn_kernel", "cases": rows,
        "captures": [{"case": 40, "iteration": 0, "duration_us": 7.0,
                      "kernel_name": "gdn_kernel"}],
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(legacy))
    result = {"kernel_name": "gdn_kernel", "profile_cases": rows}
    module.attach_profile_evidence(result, path)

    resolved = json.loads(json.dumps(legacy))
    resolved.update(kernel_name="gdn_kernel_mix_aiv",
                    declared_kernel_name="gdn_kernel_mix_aiv",
                    resolved_kernel_name="gdn_kernel")
    resolved["captures"][0].update(declared_kernel_name="gdn_kernel_mix_aiv",
                                    resolved_kernel_name="gdn_kernel")
    path.write_text(json.dumps(resolved))
    result = {"kernel_name": "gdn_kernel_mix_aiv",
              "declared_kernel_name": "gdn_kernel_mix_aiv",
              "resolved_kernel_name": "gdn_kernel", "profile_cases": rows}
    module.attach_profile_evidence(result, path)


def test_attach_profile_evidence_rejects_selector_identity_drift(tmp_path: Path):
    module = load_client()
    rows = [{"case": 40, "samples_us": [7.0], "median_us": 7.0}]
    evidence = {
        "status": "success", "kernel_name": "gdn_kernel_mix_aiv",
        "declared_kernel_name": "gdn_kernel_mix_aiv", "resolved_kernel_name": "gdn_kernel",
        "cases": rows, "captures": [{"case": 40, "iteration": 0,
                                      "duration_us": 7.0, "kernel_name": "other",
                                      "declared_kernel_name": "gdn_kernel_mix_aiv",
                                      "resolved_kernel_name": "other"}],
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(module.ClientError, match="selector identity"):
        module.attach_profile_evidence(
            {"kernel_name": "gdn_kernel_mix_aiv", "profile_cases": rows}, path)


def test_attach_profile_evidence_rejects_consistent_but_illegal_selector_mapping(tmp_path: Path):
    module = load_client()
    rows = [{"case": 40, "samples_us": [7.0], "median_us": 7.0}]
    evidence = {
        "status": "success", "kernel_name": "gdn_kernel_mix_aiv",
        "declared_kernel_name": "gdn_kernel_mix_aiv",
        "resolved_kernel_name": "unrelated_kernel", "cases": rows,
        "captures": [{"case": 40, "iteration": 0, "duration_us": 7.0,
                      "kernel_name": "unrelated_kernel",
                      "declared_kernel_name": "gdn_kernel_mix_aiv",
                      "resolved_kernel_name": "unrelated_kernel"}],
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    result = {"kernel_name": "gdn_kernel_mix_aiv",
              "declared_kernel_name": "gdn_kernel_mix_aiv",
              "resolved_kernel_name": "unrelated_kernel", "profile_cases": rows}

    with pytest.raises(module.ClientError, match="selector identity"):
        module.attach_profile_evidence(result, path)



def transport_fixture(tmp_path, monkeypatch, status="ok", fail=None):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "shared_bz_tests", ROOT / "tests/test_bz_a3_job_client.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    shared = helper.load()
    transport = helper.FakeTransport(shared, status=status, fail=fail)
    monkeypatch.setattr(shared, "GlobalCplRemoteTransport",
                        lambda *_args, **_kwargs: transport)
    job = helper.profile_job(tmp_path)
    args = SimpleNamespace(
        state_dir=tmp_path / "state", cpl_remote_sha256="f" * 64,
        remote_root="profiling-skill-production", timeout=3600)
    return load_client(), job, args, transport


@pytest.mark.parametrize("status", ["ok", "compile_error", "runtime_error",
                                   "correctness_error", "submission_error"])
def test_gz_native_shared_transport_preserves_results(
        tmp_path, monkeypatch, status):
    client, job, args, transport = transport_fixture(tmp_path, monkeypatch, status)
    result = client.execute(job, args)
    assert result["status"] == status
    assert result["handle"] == "remote:gz-a3:job:job-1"
    assert transport.executions[0][:3] == ("gz-a3", 0, "py311-torch")
    if status == "ok":
        assert result["profile"]["captures"]
        assert result["profile_cases"][0]["median_us"] == 9.0
    if status == "compile_error":
        assert "NameError" in result["diagnostics"]


def test_gz_observer_interruption_retains_and_reuses_handle(tmp_path, monkeypatch):
    client, job, args, transport = transport_fixture(
        tmp_path, monkeypatch, fail="observer")
    interrupted = client.execute(job, args)
    assert interrupted["handle"] == "remote:gz-a3:job:retained"
    assert interrupted["status"] == "infrastructure_error"
    transport.fail = None
    recovered = client.execute(job, args)
    assert recovered["status"] == "ok"
    assert recovered["handle"] == interrupted["handle"]
    assert len(transport.executions) == 1


def test_gz_cached_completion_and_calibration_identity(tmp_path, monkeypatch):
    client, job, args, transport = transport_fixture(tmp_path, monkeypatch)
    first = client.execute(job, args)
    second = client.execute(job, args)
    assert first["artifacts"]["request_digest"] == second["artifacts"]["request_digest"]
    assert len(transport.executions) == 1
    changed = client.execute({**job, "calibration_phase": "after"}, args)
    assert changed["artifacts"]["request_digest"] != first["artifacts"]["request_digest"]
    assert len(transport.executions) == 2


def test_gz_worker_resolves_namespace_paths_before_changing_directory(tmp_path, monkeypatch):
    client, job, args, transport = transport_fixture(tmp_path, monkeypatch)
    job.update(action="check", scope="development")
    client.execute(job, args)
    script = transport.executions[0][4]
    archive = transport.uploads[0][1]
    namespace_destination = transport.uploads[0][2]
    cwd = tmp_path / args.remote_root
    cwd.mkdir()
    remote_archive = tmp_path / namespace_destination
    # Replace the actual device worker with a deterministic host stand-in.
    # Keep the real generated archive/extraction/cwd/response framing protocol.
    with tarfile.open(archive) as stream:
        stream.extractall(tmp_path / "payload", filter="data")
    worker = tmp_path / "payload/runner.py"
    worker.write_text('import json, pathlib\n'
                      'pathlib.Path("response.json").write_text(json.dumps({"status":"ok"}))\n')
    old_digest = __import__("hashlib").sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(remote_archive, "w") as stream:
        for path in (tmp_path / "payload").iterdir():
            stream.add(path, arcname=path.name)
    new_digest = __import__("hashlib").sha256(remote_archive.read_bytes()).hexdigest()
    process = subprocess.run(["bash", "-c", script.replace(old_digest, new_digest)],
                             cwd=cwd, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert 'BZ_PRODUCTION_RESULT={"status": "ok"}' in process.stdout


@pytest.mark.parametrize("root", [".", "/absolute/staging", "../escape", "a//b", "a/./b"])
def test_gz_rejects_non_namespace_roots(tmp_path, monkeypatch, root):
    client, job, args, transport = transport_fixture(tmp_path, monkeypatch)
    args.remote_root = root
    with pytest.raises(transport.module.JobError, match="workspace-relative"):
        client.execute(job, args)
    assert not transport.executions


@pytest.mark.parametrize("job", [[], None, "bad", {"device": "bad"},
                                 {"action": "profile", "profiling": []}])
def test_gz_invalid_request_returns_structured_failure(tmp_path, job):
    command = [
        sys.executable, str(CLIENT), "--cpl-remote-sha256", "invalid",
        "--remote-root", "/srv/profiling-skill-production",
        "--state-dir", str(tmp_path / "state")]
    process = subprocess.run(command, input=json.dumps(job), text=True,
                             capture_output=True, check=False)
    result = json.loads(process.stdout)
    assert process.returncode == 2
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "request_error"
    assert not process.stderr
