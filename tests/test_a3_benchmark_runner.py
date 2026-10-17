from __future__ import annotations

import contextlib
import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "a3_benchmark_runner_for_classification", ROOT / "scripts/a3_benchmark_runner.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Payload:
    def __init__(self, retained):
        retained.append(weakref.ref(self))
        self.retained = retained

    def clone(self):
        return Payload(self.retained)


def fake_torch(empty_cache=lambda: None):
    return types.SimpleNamespace(
        no_grad=contextlib.nullcontext,
        npu=types.SimpleNamespace(synchronize=lambda: None, empty_cache=empty_cache),
    )


def check_job(**overrides):
    job = {
        "benchmark": "bsa", "action": "check", "device": 0,
        "cases": [0, 1], "scope": "full", "baseline": "baseline.py",
        "candidate": "candidate.py", "case_spec": "cases.jsonl",
    }
    job.update(overrides)
    return job


def test_missing_runtime_owned_triton_is_infrastructure(monkeypatch):
    runner = load()

    def fake_load(_path, name):
        if name == "frozen_baseline":
            return object()
        error = ModuleNotFoundError("No module named 'triton'")
        error.name = "triton"
        raise error

    monkeypatch.setattr(runner, "load", fake_load)
    result = runner.execute({
        "benchmark": "matmul", "action": "check", "device": 0,
        "cases": [0], "scope": "full", "baseline": "baseline.py",
        "candidate": "candidate.py", "case_spec": "cases.jsonl",
    })

    assert result["status"] == "infrastructure_error"
    assert "triton" in result["diagnostics"]


def test_missing_candidate_dependency_remains_compile_error(monkeypatch):
    runner = load()

    def fake_load(_path, name):
        if name == "frozen_baseline":
            return object()
        error = ModuleNotFoundError("No module named 'candidate_helper'")
        error.name = "candidate_helper"
        raise error

    monkeypatch.setattr(runner, "load", fake_load)
    result = runner.execute({
        "benchmark": "matmul", "action": "check", "device": 0,
        "cases": [0], "scope": "full", "baseline": "baseline.py",
        "candidate": "candidate.py", "case_spec": "cases.jsonl",
    })

    assert result["status"] == "compile_error"


def test_releases_case_objects_before_building_next_case(monkeypatch):
    runner = load()
    retained = []
    cleanup_calls = []

    class BaselineModel:
        def __call__(self, *_inputs):
            return Payload(retained)

    class CandidateModel:
        def __call__(self, *_inputs):
            return Payload(retained)

    modules = {
        "frozen_baseline": types.SimpleNamespace(Model=BaselineModel),
        "mutable_candidate": types.SimpleNamespace(Model=CandidateModel),
    }

    def selected(_baseline, _case_spec, case):
        gc.collect()
        if case:
            assert retained
            assert all(item() is None for item in retained)
            retained.clear()
        return [Payload(retained)]

    def empty_cache():
        gc.collect()
        assert all(item() is None for item in retained)
        cleanup_calls.append(True)

    monkeypatch.setitem(sys.modules, "torch", fake_torch(empty_cache))
    monkeypatch.setattr(runner, "load", lambda _path, name: modules[name])
    monkeypatch.setattr(runner, "selected_inputs", selected)
    monkeypatch.setattr(runner, "compare", lambda *_args, **_kwargs: (True, 0.0))

    result = runner.execute(check_job())

    assert result["status"] == "ok"
    assert [item["case"] for item in result["case_evidence"]] == [0, 1]
    assert len(cleanup_calls) == 2
    gc.collect()
    assert all(item() is None for item in retained)


def test_failure_releases_objects_and_preserves_prior_evidence(monkeypatch):
    runner = load()
    retained = []
    calls = 0

    class BaselineModel:
        def __call__(self, *_inputs):
            return Payload(retained)

    class CandidateModel:
        def __call__(self, *_inputs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("candidate launch failed")
            return Payload(retained)

    modules = {
        "frozen_baseline": types.SimpleNamespace(Model=BaselineModel),
        "mutable_candidate": types.SimpleNamespace(Model=CandidateModel),
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr(runner, "load", lambda _path, name: modules[name])
    monkeypatch.setattr(
        runner, "selected_inputs",
        lambda _baseline, _case_spec, _case: [Payload(retained)],
    )
    monkeypatch.setattr(runner, "compare", lambda *_args, **_kwargs: (True, 0.0))

    result = runner.execute(check_job())

    assert result["status"] == "runtime_error"
    assert "candidate launch failed" in result["diagnostics"]
    assert [item["case"] for item in result["case_evidence"]] == [0]
    gc.collect()
    assert all(item() is None for item in retained)


def test_cleanup_failure_does_not_replace_candidate_failure(monkeypatch):
    runner = load()
    candidate_calls = cleanup_calls = 0

    class Model:
        def __call__(self, *_inputs):
            return Payload([])

    class CandidateModel:
        def __call__(self, *_inputs):
            nonlocal candidate_calls
            candidate_calls += 1
            if candidate_calls == 2:
                raise RuntimeError("candidate launch failed")
            return Payload([])

    modules = {
        "frozen_baseline": types.SimpleNamespace(Model=Model),
        "mutable_candidate": types.SimpleNamespace(Model=CandidateModel),
    }

    def empty_cache():
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 2:
            raise RuntimeError("allocator cleanup failed")

    monkeypatch.setitem(sys.modules, "torch", fake_torch(empty_cache))
    monkeypatch.setattr(runner, "load", lambda _path, name: modules[name])
    monkeypatch.setattr(
        runner, "selected_inputs", lambda _baseline, _case_spec, _case: [Payload([])],
    )
    monkeypatch.setattr(runner, "compare", lambda *_args, **_kwargs: (True, 0.0))

    result = runner.execute(check_job())

    assert result["status"] == "runtime_error"
    assert "candidate launch failed" in result["diagnostics"]
    assert "allocator cleanup failed" not in result["diagnostics"]
    assert [item["case"] for item in result["case_evidence"]] == [0]


def test_cleanup_only_failure_is_bound_infrastructure_result(monkeypatch):
    runner = load()

    class Model:
        def __call__(self, *_inputs):
            return Payload([])

    modules = {
        "frozen_baseline": types.SimpleNamespace(Model=Model),
        "mutable_candidate": types.SimpleNamespace(Model=Model),
    }

    def empty_cache():
        raise RuntimeError("allocator cleanup failed")

    monkeypatch.setitem(sys.modules, "torch", fake_torch(empty_cache))
    monkeypatch.setattr(runner, "load", lambda _path, name: modules[name])
    monkeypatch.setattr(
        runner, "selected_inputs", lambda _baseline, _case_spec, _case: [Payload([])],
    )
    monkeypatch.setattr(runner, "compare", lambda *_args, **_kwargs: (True, 0.0))

    result = runner.execute(check_job(cases=[0]))

    assert result["status"] == "infrastructure_error"
    assert "allocator cleanup failed" in result["diagnostics"]
    assert result["benchmark"] == "bsa"
    assert result["action"] == "check"
    assert result["device"] == 0
    assert [item["case"] for item in result["case_evidence"]] == [0]


def test_missing_allocator_cache_hook_does_not_mask_case_result(monkeypatch):
    runner = load()

    class Model:
        def __call__(self, *_inputs):
            return Payload([])

    modules = {
        "frozen_baseline": types.SimpleNamespace(Model=Model),
        "mutable_candidate": types.SimpleNamespace(Model=Model),
    }
    torch = fake_torch()
    del torch.npu.empty_cache
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runner, "load", lambda _path, name: modules[name])
    monkeypatch.setattr(
        runner, "selected_inputs", lambda _baseline, _case_spec, _case: [Payload([])],
    )
    monkeypatch.setattr(runner, "compare", lambda *_args, **_kwargs: (True, 0.0))

    result = runner.execute(check_job(cases=[0]))

    assert result["status"] == "ok"
    assert result["passed"] is True


def test_cleanup_runs_after_fusion_evidence_is_retained(monkeypatch):
    runner = load()
    retained = []
    audits = []

    class Model:
        def __call__(self, *_inputs):
            return Payload(retained)

    class Audit:
        def __init__(self, _contract):
            self.closed = False
            audits.append(self)

        def instrument(self, _candidate):
            pass

        def dispatch_mode(self):
            return contextlib.nullcontext()

        def finish(self, case, _actual):
            assert not self.closed
            return {"case": case, "retained": True}

        def _restore_run_hooks(self):
            self.closed = True

    modules = {
        "frozen_baseline": types.SimpleNamespace(Model=Model),
        "mutable_candidate": types.SimpleNamespace(Model=Model),
    }
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr(runner, "load", lambda _path, name: modules[name])
    monkeypatch.setattr(
        runner, "selected_inputs",
        lambda _baseline, _case_spec, _case: [Payload(retained)],
    )
    monkeypatch.setattr(runner, "compare", lambda *_args, **_kwargs: (True, 0.0))
    monkeypatch.setattr(runner, "FusionRuntimeAudit", Audit)

    result = runner.execute(check_job(fusion_contract={"entrypoint": "kernel"}))

    assert result["status"] == "ok"
    assert result["fusion_evidence"] == [
        {"case": 0, "retained": True}, {"case": 1, "retained": True},
    ]
    assert all(audit.closed for audit in audits)
