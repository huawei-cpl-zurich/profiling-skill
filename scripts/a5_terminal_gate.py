#!/usr/bin/env python3
"""Run correctness/fusion checks, then the launches captured by outer A5 msprof."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    "no module named 'triton'",
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load host gate module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failure_outcome(status: str, diagnostics: str) -> str:
    lowered = diagnostics.lower()
    if status == "infrastructure_error" or any(item in lowered for item in INFRASTRUCTURE_MARKERS):
        return "infrastructure_failure"
    return "candidate_failure"


def run_gate(root: Path, candidate: Path, manifest_path: Path,
             benchmark: str, repeats: int) -> tuple[str, str, dict]:
    spec = SPECS[benchmark]
    runner = load(root / "scripts/a3_benchmark_runner.py", "a5_gate_runner")
    fusion = load(root / "scripts/fully_fused_contract.py", "a5_gate_fusion")
    try:
        manifest = json.loads(manifest_path.read_text())
        declaration = fusion.validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, fusion.FusionContractError) as exc:
        return "candidate_failure", f"invalid host candidate contract: {exc}", {}
    if (declaration["entrypoint"] != spec["entrypoint"]
            or declaration["kernel_name"] != spec["kernel_name"]):
        return "candidate_failure", "candidate manifest does not match the benchmark contract", {}

    common = {
        "protocol_version": 1, "product": "a5", "runtime": "cann91",
        "benchmark": benchmark, "device": 0, "logical_device": 0,
        "candidate": str(candidate),
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

    # The approved wrapper owns the single outer msprof invocation. These are
    # deliberately ordinary runner launches: nesting a profiler here corrupts
    # both timing and failure attribution.
    for _repeat in range(repeats):
        for case in spec["ranked"]:
            result = runner.execute({
                **common, "action": "check", "scope": "ranked-profile", "cases": [case],
            })
            if result.get("status") != "ok" or not result.get("passed"):
                diagnostics = str(result.get("diagnostics", f"ranked case {case} failed"))
                return failure_outcome(str(result.get("status", "runtime_error")), diagnostics), diagnostics, {}

    return "success", "", {
        "benchmark": benchmark, "correctness_cases": len(spec["cases"]),
        "fusion": fusion_summary, "profiler": "msprof-op-outer",
        "kernel_name": spec["kernel_name"], "ranked_cases": spec["ranked"],
        "repeats": repeats, "expected_sample_count": len(spec["ranked"]) * repeats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(SPECS), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        outcome, diagnostics, summary = run_gate(
            args.root.resolve(), args.candidate.resolve(), args.manifest.resolve(),
            args.benchmark, args.repeats,
        )
    except BaseException as exc:
        outcome, diagnostics, summary = (
            "infrastructure_failure",
            f"host gate failed unexpectedly: {type(exc).__name__}: {exc}", {},
        )
    print(json.dumps({
        "schema": RESULT_SCHEMA, "outcome": outcome,
        "diagnostics": diagnostics[-16_000:],
        **({"profile_summary": summary} if summary else {}),
    }, sort_keys=True, separators=(",", ":")))
    return 0 if outcome == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
