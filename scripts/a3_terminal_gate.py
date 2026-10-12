#!/usr/bin/env python3
"""Run one host-owned A3 correctness, fusion, and msprof terminal gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


RESULT_SCHEMA = "simple-experiment-result/v1"
SPECS = {
    "gdn": {
        "cases": list(range(50)), "ranked": [40, 49, 47, 46, 45],
        "rtol": 1e-2, "atol": 1e-2,
        "entrypoint": "chunk_gated_delta_rule_kernel",
        "kernel_name": "chunk_gated_delta_rule_kernel",
    },
    "bsa": {
        "cases": list(range(50)), "ranked": [47, 46, 49, 44, 43],
        "rtol": 1e-2, "atol": 1e-2,
        "entrypoint": "block_sparse_attention_kernel",
        "kernel_name": "block_sparse_attention_kernel",
    },
    "matmul": {
        "cases": list(range(10)), "ranked": [7, 8, 9],
        "rtol": 2e-2, "atol": 2e-2,
        "entrypoint": "streaming_matmul_add_kernel",
        "kernel_name": "streaming_matmul_add_kernel_mix_aic",
    },
}
INFRASTRUCTURE_MARKERS = (
    "507033", "failed to start the device", "tsdopen failed",
    "failed to send the opp", "failed to load the backend extension",
    "device unavailable", "no module named 'torch'", "no module named 'torch_npu'",
    "no module named 'triton'", "msprof could not start", "msprof exited with status 124",
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load host gate module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(outcome: str, diagnostics: str = "", **values) -> int:
    print(json.dumps({
        "schema": RESULT_SCHEMA, "outcome": outcome,
        "diagnostics": diagnostics[-16_000:], **values,
    }, sort_keys=True, separators=(",", ":")))
    return 0 if outcome == "success" else 1


def failure_outcome(status: str, diagnostics: str) -> str:
    lowered = diagnostics.lower()
    if status == "infrastructure_error" or any(marker in lowered for marker in INFRASTRUCTURE_MARKERS):
        return "infrastructure_failure"
    return "candidate_failure"


def validate_environment() -> str | None:
    physical = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    logical = os.environ.get("ASCEND_DEVICE_ID", "")
    if not physical.isdigit() or logical != "0":
        return "host placement must expose one physical device as logical device 0"
    return None


def validate_pipe_activity(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("managed A3 profile is missing pipe activity evidence")
    if value.get("product") != "a3" or value.get("evidence_kind") != "per-block-pipe-activity":
        raise ValueError("managed A3 pipe activity evidence has the wrong product or kind")
    observations = value.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("managed A3 pipe activity evidence has no observations")
    for observation in observations:
        saturation = observation.get("saturation") if isinstance(observation, dict) else None
        if not isinstance(saturation, dict) or saturation.get("state") != "unknown":
            raise ValueError("managed A3 pipe observations must keep saturation unknown")
    return value


def validate_profile_identity(evidence: dict, pipe_activity: dict) -> tuple[str, str]:
    declared = evidence.get("declared_kernel_name")
    resolved = evidence.get("resolved_kernel_name")
    if not isinstance(declared, str) or not declared:
        raise ValueError("managed A3 timing evidence is missing its declared kernel selector")
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("managed A3 timing evidence is missing its resolved kernel selector")
    if evidence.get("kernel_name") != resolved:
        raise ValueError("managed A3 timing kernel identity is not the resolved selector")
    provenance = pipe_activity.get("provenance")
    pipe_selector = provenance.get("kernel_selector") if isinstance(provenance, dict) else None
    if pipe_selector != resolved:
        raise ValueError(
            "managed A3 timing and pipe activity use different exact kernel selectors"
        )
    return declared, resolved


def run_gate(root: Path, candidate: Path, manifest_path: Path, benchmark: str,
             output: Path, round_number: int) -> tuple[str, str, dict]:
    spec = SPECS[benchmark]
    runner = load(root / "scripts/a3_benchmark_runner.py", "a3_gate_runner")
    fusion = load(root / "scripts/fully_fused_contract.py", "a3_gate_fusion")
    batch = load(root / "scripts/batch_profile_a3.py", "a3_gate_batch")
    try:
        manifest = json.loads(manifest_path.read_text())
        declaration = fusion.validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, fusion.FusionContractError) as exc:
        return "infrastructure_failure", f"invalid host candidate contract: {exc}", {}
    if (declaration["entrypoint"] != spec["entrypoint"]
            or declaration["kernel_name"] != spec["kernel_name"]):
        return (
            "infrastructure_failure",
            "host candidate manifest does not match the benchmark contract",
            {},
        )

    common = {
        "protocol_version": 1, "profile": "bz-a3", "product": "a3",
        "runtime": "py311-torch", "benchmark": benchmark, "device": 0,
        "logical_device": 0, "candidate": str(candidate),
        "baseline": str(root / f"benchmarks/{benchmark}/baseline.py"),
        "case_spec": str(root / f"benchmarks/{benchmark}/cases.jsonl"),
        "tolerances": {"rtol": spec["rtol"], "atol": spec["atol"]},
        "fusion_contract": declaration,
    }
    check = runner.execute({
        **common, "action": "check", "scope": "full", "cases": spec["cases"],
    })
    if check.get("status") != "ok" or not check.get("passed"):
        diagnostics = str(check.get("diagnostics", "full correctness failed"))
        return failure_outcome(str(check.get("status", "runtime_error")), diagnostics), diagnostics, {}
    try:
        fusion_summary = fusion.validate_fusion_evidence(
            manifest, check.get("fusion_evidence"), expected_cases=spec["cases"],
        )
    except fusion.FusionContractError as exc:
        return "candidate_failure", f"fully-fused audit failed: {exc}", {}

    profile_job = {
        **common, "action": "profile", "cases": spec["ranked"], "repeats": 3,
        "round": round_number,
        "profiling": {
            "tool": "msprof op", "kernel_name": spec["kernel_name"],
            "pipe_activity": True,
        },
    }
    response_path = output / "profile-response.json"
    profile_output = output / "profile"
    rc = batch.execute(
        profile_job, runner=root / "scripts/a3_benchmark_runner.py",
        profiler=root / "scripts/profile_a3.py", output=profile_output,
        response_path=response_path, kernel_name=spec["kernel_name"],
        normalizer=root / "scripts/normalize_a3_pipe_evidence.py",
        activity_model=root / "references/a3-pipe-activity-model.json",
    )
    try:
        response = json.loads(response_path.read_text())
        evidence = json.loads((profile_output / "evidence.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return "infrastructure_failure", f"profile evidence is unavailable: {exc}", {}
    if rc or response.get("status") != "ok" or evidence.get("status") != "success":
        diagnostics = str(response.get("diagnostics") or evidence.get("failure") or "profiling failed")
        return failure_outcome(str(response.get("status", "infrastructure_error")), diagnostics), diagnostics, {}
    try:
        pipe_activity = validate_pipe_activity(evidence.get("pipe_activity"))
        declared_kernel_name, resolved_kernel_name = validate_profile_identity(
            evidence, pipe_activity,
        )
    except ValueError as exc:
        return "infrastructure_failure", str(exc), {}
    timing = {
        "metric": "BasicInfo",
        "profiler": evidence.get("profiler"),
        "timing_scope": evidence.get("timing_scope"),
        "kernel_name": resolved_kernel_name,
        "declared_kernel_name": declared_kernel_name,
        "resolved_kernel_name": resolved_kernel_name,
        "cases": evidence.get("cases"),
        "repeats": evidence.get("repeats"), "geomean_us": evidence.get("geomean_us"),
        "captures": evidence.get("captures"),
    }
    summary = {
        "schema": "profiling-skill/a3-round-profile-summary/v1",
        "benchmark": benchmark, "correctness_cases": len(spec["cases"]),
        "fusion": fusion_summary, "timing": timing,
        "pipe_activity": pipe_activity,
        "timeline": {
            "available": False,
            "kind": "a3-temporal-pipe-phases",
            "reason": (
                "A3 PipeUtilization rows are non-temporal activity observations; "
                "TimelineDetail flow relationships do not provide A5-style temporal pipe phases"
            ),
        },
    }
    return "success", "", summary


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    environment_error = validate_environment()
    if environment_error:
        return emit("infrastructure_failure", environment_error)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        outcome, diagnostics, summary = run_gate(
            args.root.resolve(), args.candidate.resolve(), args.manifest.resolve(),
            args.benchmark, args.output.resolve(), args.round,
        )
    except BaseException as exc:
        return emit("infrastructure_failure", f"host gate failed unexpectedly: {type(exc).__name__}: {exc}")
    return emit(outcome, diagnostics, **({"profile_summary": summary} if summary else {}))


if __name__ == "__main__":
    raise SystemExit(main())
