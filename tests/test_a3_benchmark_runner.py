from __future__ import annotations

import importlib.util
import sys
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
