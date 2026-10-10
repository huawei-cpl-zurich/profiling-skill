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
from types import ModuleType


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


_METADATA_OPS = (
    "aten.alias", "aten.as_strided", "aten.detach", "aten.empty",
    "aten.empty_like", "aten.expand", "aten.lift_fresh", "aten.new_empty",
    "aten.permute", "aten.reshape", "aten.select", "aten.slice",
    "aten.squeeze", "aten.t.", "aten.transpose", "aten.unsqueeze",
    "aten.view", "aten._unsafe_view",
)


def _tensor_keys(value) -> set[tuple]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    result = set()
    for item in values:
        if isinstance(item, dict):
            result.update(_tensor_keys(tuple(item.values())))
        elif isinstance(item, (list, tuple)):
            result.update(_tensor_keys(item))
        elif hasattr(item, "untyped_storage"):
            try:
                result.add((str(item.device), item.untyped_storage().data_ptr()))
            except (AttributeError, RuntimeError):
                result.add(("object", id(item)))
    return result


class _KernelProxy:
    def __init__(self, wrapped, audit):
        self._wrapped, self._audit = wrapped, audit

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __getitem__(self, grid):
        launch = self._wrapped[grid]

        def observed(*args, **kwargs):
            self._audit.launches.append(_tensor_keys((args, kwargs)))
            return launch(*args, **kwargs)

        return observed


class FusionRuntimeAudit:
    """Trusted one-forward audit for framework compute and Triton launches."""

    def __init__(self, declaration: dict):
        self.declaration = declaration
        self.launches: list[set[tuple]] = []
        self.framework_ops: list[tuple[str, str]] = []

    def instrument(self, candidate: ModuleType) -> None:
        entrypoint = self.declaration["entrypoint"]
        kernel = getattr(candidate, entrypoint, None)
        if isinstance(kernel, _KernelProxy):
            kernel = kernel._wrapped
        module = type(kernel).__module__ if kernel is not None else ""
        if kernel is None or not module.startswith("triton.") or not hasattr(kernel, "__getitem__"):
            return
        setattr(candidate, entrypoint, _KernelProxy(kernel, self))

    def dispatch_mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        audit = self

        class AuditMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = str(func)
                if not name.startswith(_METADATA_OPS):
                    origin = "acl" if name.startswith(("npu::", "aclnn")) else "torch"
                    audit.framework_ops.append((name, origin))
                return func(*args, **(kwargs or {}))

        return AuditMode()

    def finish(self, case: int, output) -> dict:
        operators = []
        component = "aic" if self.declaration["kernel_name"].endswith("_mix_aic") else "aiv"
        for index, _arguments in enumerate(self.launches):
            operators.append({
                "name": self.declaration["kernel_name"], "origin": "triton",
                "entrypoint": self.declaration["entrypoint"],
                "launch_id": f"launch-{index}", "component": component,
            })
        operators.extend({"name": name, "origin": origin,
                          "launch_id": f"framework-{index}"}
                         for index, (name, origin) in enumerate(self.framework_ops))
        output_keys = _tensor_keys(output)
        producers = [index for index, arguments in enumerate(self.launches)
                     if output_keys and output_keys.issubset(arguments)]
        return {
            "schema": "profiling-skill/fusion-evidence/v1", "case": case,
            "output_launch_id": f"launch-{producers[-1]}" if producers else None,
            "operators": operators,
        }


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


def compare(actual, expected, *, rtol: float, atol: float) -> tuple[bool, float]:
    import torch
    if actual is None or expected is None:
        return actual is expected, 0.0
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            return False, float("inf")
        results = [compare(a, e, rtol=rtol, atol=atol) for a, e in zip(actual, expected)]
        return all(item[0] for item in results), max((item[1] for item in results), default=0.0)
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        return actual == expected, 0.0
    if actual.shape != expected.shape:
        return False, float("inf")
    delta = (actual.float() - expected.float()).abs()
    error = float(delta.max().cpu()) if delta.numel() else 0.0
    return bool(torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)), error


def identity(job: dict) -> dict:
    result = {key: job[key] for key in ("benchmark", "action", "device")}
    if job["action"] == "check":
        result.update(cases=job["cases"], scope=job["scope"])
    elif job["action"] == "profile" and "cases" in job:
        result.update(cases=job["cases"], repeats=job["repeats"])
    else:
        result["case"] = job["case"]
    if job["action"] == "measure":
        result["phase"] = job["phase"]
    if job["action"] == "profile":
        result.update(round=job["round"], kernel_name=job["profiling"]["kernel_name"])
    return result


def execute(job: dict) -> dict:
    bound = identity(job)
    # Protocol v1 jobs predating benchmark-specific tolerances used 1e-2.
    tolerances = job.get("tolerances", {"rtol": 1e-2, "atol": 1e-2})
    if (not isinstance(tolerances, dict)
            or any(isinstance(tolerances.get(name), bool)
                   or not isinstance(tolerances.get(name), (int, float))
                   or tolerances[name] < 0 for name in ("rtol", "atol"))):
        return {"status": "infrastructure_error",
                "diagnostics": "job requires non-negative numeric rtol and atol", **bound}
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
    fusion_evidence = []
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
            audit = None
            if "fusion_contract" in job:
                audit = FusionRuntimeAudit(job["fusion_contract"])
                audit.instrument(candidate)
            with torch.no_grad():
                torch.npu.synchronize()
                started = time.perf_counter_ns()
                if audit is None:
                    actual = candidate_model(*candidate_inputs)
                else:
                    with audit.dispatch_mode():
                        actual = candidate_model(*candidate_inputs)
                torch.npu.synchronize()
                elapsed_us = (time.perf_counter_ns() - started) / 1000.0
            if audit is not None:
                fusion_evidence.append(audit.finish(case, actual))
        except BaseException as exc:
            return {"status": classify(exc), "diagnostics": diagnostic(exc), **bound,
                    "case_evidence": evidence}
        try:
            passed, max_abs = compare(actual, expected, **tolerances)
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
    if "fusion_contract" in job:
        result["fusion_evidence"] = fusion_evidence
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
