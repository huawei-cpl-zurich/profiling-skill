#!/usr/bin/env python3
"""Execute one frozen benchmark request inside a managed A3 bundle."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diagnostic(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def classify(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    compile_markers = ("compile", "compiler", "triton", "lowering", "codegen", "semantic")
    compile_types = (ImportError, NameError, SyntaxError)
    return "compile_error" if isinstance(exc, compile_types) or any(item in text for item in compile_markers) else "runtime_error"


def clone(value):
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    return value


def to_npu(value):
    if hasattr(value, "npu"):
        return value.npu()
    if isinstance(value, list):
        return [to_npu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_npu(item) for item in value)
    if isinstance(value, dict):
        return {key: to_npu(item) for key, item in value.items()}
    return value


def selected_inputs(baseline, case_spec: Path, case_index: int):
    cases = [json.loads(line) for line in case_spec.read_text(encoding="utf-8-sig").splitlines() if line]
    selected = cases[case_index]
    sentinel = object()
    old_enumerate = getattr(baseline, "enumerate", sentinel)
    old_open = getattr(baseline, "open", sentinel)
    old_loader = getattr(baseline, "_load_cases", sentinel)
    baseline.enumerate = lambda values: ((case_index, item) for item in values)
    try:
        if old_loader is not sentinel:
            baseline._load_cases = lambda: [selected]
        elif hasattr(baseline, "_json_path"):
            baseline.open = lambda *_args, **_kwargs: io.StringIO(json.dumps(selected) + "\n")
        else:
            raise RuntimeError("frozen baseline has no supported case loader")
        groups = baseline.get_input_groups()
        if len(groups) != 1:
            raise RuntimeError("selected baseline loader did not return exactly one input group")
        return to_npu(groups[0])
    finally:
        for name, value in (("enumerate", old_enumerate), ("open", old_open), ("_load_cases", old_loader)):
            if value is sentinel:
                baseline.__dict__.pop(name, None)
            else:
                setattr(baseline, name, value)


def compare(actual, expected) -> tuple[bool, float]:
    import torch
    if actual is None or expected is None:
        return actual is expected, 0.0
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            return False, float("inf")
        results = [compare(a, e) for a, e in zip(actual, expected)]
        return all(item[0] for item in results), max((item[1] for item in results), default=0.0)
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        return actual == expected, 0.0
    if actual.shape != expected.shape:
        return False, float("inf")
    delta = (actual.float() - expected.float()).abs()
    error = float(delta.max().cpu()) if delta.numel() else 0.0
    return bool(torch.allclose(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)), error


def identity(job: dict) -> dict:
    result = {key: job[key] for key in ("benchmark", "action", "device")}
    if job["action"] == "check":
        result.update(cases=job["cases"], scope=job["scope"])
    else:
        result["case"] = job["case"]
    if job["action"] == "measure":
        result["phase"] = job["phase"]
    if job["action"] == "profile":
        result.update(round=job["round"], kernel_name=job["profiling"]["kernel_name"])
    return result


def execute(job: dict) -> dict:
    bound = identity(job)
    try:
        import torch
    except BaseException as exc:
        return {"status": "infrastructure_error", "diagnostics": diagnostic(exc), **bound}
    try:
        baseline = load(Path(job["baseline"]), "frozen_baseline")
    except BaseException as exc:
        return {"status": "infrastructure_error", "diagnostics": diagnostic(exc), **bound}
    try:
        candidate = load(Path(job["candidate"]), "mutable_candidate")
    except BaseException as exc:
        return {"status": "compile_error", "diagnostics": diagnostic(exc), **bound}
    cases = job["cases"] if job["action"] == "check" else [job["case"]]
    evidence = []
    for case in cases:
        try:
            inputs = selected_inputs(baseline, Path(job["case_spec"]), case)
            reference_inputs = clone(inputs)
            with torch.no_grad():
                expected = baseline.Model()(*reference_inputs)
                torch.npu.synchronize()
        except BaseException as exc:
            return {"status": "infrastructure_error", "diagnostics": diagnostic(exc), **bound,
                    "case_evidence": evidence}
        try:
            candidate_inputs = clone(inputs)
            candidate_model = candidate.Model()
            with torch.no_grad():
                torch.npu.synchronize()
                started = time.perf_counter_ns()
                actual = candidate_model(*candidate_inputs)
                torch.npu.synchronize()
                elapsed_us = (time.perf_counter_ns() - started) / 1000.0
        except BaseException as exc:
            return {"status": classify(exc), "diagnostics": diagnostic(exc), **bound,
                    "case_evidence": evidence}
        try:
            passed, max_abs = compare(actual, expected)
        except BaseException as exc:
            return {"status": "infrastructure_error", "diagnostics": diagnostic(exc), **bound,
                    "case_evidence": evidence}
        evidence.append({"case": case, "passed": passed, "max_abs_error": max_abs,
                         "host_elapsed_us": elapsed_us})
        if not passed:
            return {"status": "correctness_error", "diagnostics": f"case {case} mismatch",
                    **bound, "passed": False, "case_evidence": evidence}
    result = {"status": "ok", "diagnostics": "", **bound, "passed": True,
              "case_evidence": evidence}
    if job["action"] == "measure":
        result["latency_us"] = evidence[0]["host_elapsed_us"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        job = json.loads(args.job.read_text())
        result = execute(job)
    except BaseException as exc:
        result = {"status": "infrastructure_error", "diagnostics": diagnostic(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
