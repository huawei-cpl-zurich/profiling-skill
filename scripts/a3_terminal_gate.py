#!/usr/bin/env python3
"""Run one host-owned A3 correctness, fusion, and msprof terminal gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
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
AUTOTUNE_REQUEST_SCHEMA = "simple-experiment-autotune-request/v1"
AUTOTUNE_RESULT_SCHEMA = "simple-experiment-autotune-result/v1"
AUTOTUNE_CHECKPOINT_SCHEMA = "profiling-skill/a3-autotune-checkpoint/v1"
MAX_AUTOTUNE_OPTIONS = 129
MAX_AUTOTUNE_FINALISTS = 3
MIN_AUTOTUNE_PROFILE_REPEATS = 3
AUTOTUNE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


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
        activity = observation.get("activity") if isinstance(observation, dict) else None
        if (not isinstance(activity, dict) or not activity
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) or not 0 <= item <= 1
                       for item in activity.values())):
            raise ValueError(
                "every managed A3 pipe observation requires a finite modeled activity fraction"
            )
    return value


def validate_profile_identity(evidence: dict, pipe_activity: dict) -> tuple[str, str]:
    declared = evidence.get("declared_kernel_name")
    resolved = evidence.get("resolved_kernel_name")
    if not isinstance(declared, str) or not declared:
        raise ValueError("managed A3 timing evidence is missing its declared kernel selector")
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("managed A3 timing evidence is missing its resolved kernel selector")
    if evidence.get("kernel_name") != declared:
        raise ValueError("managed A3 client-facing kernel identity is not the declared selector")
    provenance = pipe_activity.get("provenance")
    pipe_selector = provenance.get("kernel_selector") if isinstance(provenance, dict) else None
    if pipe_selector != resolved:
        raise ValueError(
            "managed A3 timing and pipe activity use different exact kernel selectors"
        )
    return declared, resolved


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_autotune_request(path: Path, round_number: int) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    options = value.get("options") if isinstance(value, dict) else None
    if (not isinstance(value, dict)
            or value.get("schema") != AUTOTUNE_REQUEST_SCHEMA
            or value.get("optimization_round") != round_number
            or not isinstance(options, list)
            or not 1 <= len(options) <= MAX_AUTOTUNE_OPTIONS):
        raise ValueError("invalid autotune request")
    identifiers = set()
    for option in options:
        if (not isinstance(option, dict) or not isinstance(option.get("id"), str)
                or not AUTOTUNE_ID.fullmatch(option["id"]) or option["id"] in identifiers
                or not isinstance(option.get("source"), str)
                or hashlib.sha256(option["source"].encode()).hexdigest()
                != option.get("source_sha256")):
            raise ValueError("invalid autotune option identity")
        identifiers.add(option["id"])
    if value.get("incumbent_id") not in identifiers:
        raise ValueError("autotune incumbent is missing")
    return value, hashlib.sha256(raw).hexdigest()


def profile_candidate(root: Path, common: dict, spec: dict, candidate: Path,
                      output: Path, round_number: int, *, pipe_activity: bool) -> tuple[str, str, dict]:
    batch = load(root / "scripts/batch_profile_a3.py", f"a3_tune_batch_{output.name}")
    job = {
        **common, "candidate": str(candidate), "action": "profile",
        "cases": spec["ranked"], "repeats": MIN_AUTOTUNE_PROFILE_REPEATS,
        "round": round_number,
        "profiling": {
            "tool": "msprof op", "kernel_name": spec["kernel_name"],
            "pipe_activity": pipe_activity,
        },
    }
    response_path = output / "profile-response.json"
    profile_output = output / "profile"
    rc = batch.execute(
        job, runner=root / "scripts/a3_benchmark_runner.py",
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
    declared = evidence.get("declared_kernel_name")
    resolved = evidence.get("resolved_kernel_name")
    geomean = evidence.get("geomean_us")
    repeats = evidence.get("repeats")
    if (declared != spec["kernel_name"] or evidence.get("kernel_name") != declared
            or not isinstance(resolved, str) or not resolved):
        return "infrastructure_failure", "finalist timing is missing selector identity", {}
    if (isinstance(geomean, bool) or not isinstance(geomean, (int, float))
            or not math.isfinite(geomean) or geomean <= 0):
        return "infrastructure_failure", "finalist timing is not positive and finite", {}
    if (isinstance(repeats, bool) or not isinstance(repeats, int)
            or repeats < MIN_AUTOTUNE_PROFILE_REPEATS):
        return "infrastructure_failure", "finalist timing has too few msprof repeats", {}
    summary = {
        "metric": "BasicInfo", "profiler": evidence.get("profiler"),
        "timing_scope": evidence.get("timing_scope"),
        "kernel_name": resolved, "declared_kernel_name": declared,
        "resolved_kernel_name": resolved,
        "cases": evidence.get("cases"),
        "repeats": repeats, "geomean_us": geomean,
        "captures": evidence.get("captures"),
    }
    if pipe_activity:
        try:
            summary["pipe_activity"] = validate_pipe_activity(evidence.get("pipe_activity"))
            validate_profile_identity(evidence, summary["pipe_activity"])
        except ValueError as exc:
            return "infrastructure_failure", str(exc), {}
    return "success", "", summary


def screen_candidate(runner, fusion, manifest: dict, common: dict, spec: dict,
                     candidate: Path) -> tuple[str, str, dict]:
    commands = []
    for retry in range(3):
        retry_infrastructure = False
        for scope in ("autotune-warmup", "autotune-ranked"):
            commands.append(
                f"a3_benchmark_runner check candidate={candidate.name} "
                f"scope={scope} cases={','.join(map(str, spec['ranked']))}"
            )
            check = runner.execute({
                **common, "candidate": str(candidate), "action": "check",
                "scope": scope, "cases": spec["ranked"],
            })
            if check.get("status") != "ok" or not check.get("passed"):
                diagnostics = str(check.get("diagnostics", "autotune screen failed"))
                outcome = failure_outcome(
                    str(check.get("status", "runtime_error")), diagnostics,
                )
                if outcome == "infrastructure_failure" and retry < 2:
                    retry_infrastructure = True
                    break
                return outcome, diagnostics, {"commands": commands}
        if retry_infrastructure:
            continue
        try:
            fusion.validate_fusion_evidence(
                manifest, check.get("fusion_evidence"), expected_cases=spec["ranked"],
            )
            timings = [float(item["host_elapsed_us"]) for item in check["case_evidence"]]
            if not timings or any(not math.isfinite(item) or item <= 0 for item in timings):
                raise ValueError("cheap timings must be positive and finite")
            cheap = math.exp(sum(math.log(item) for item in timings) / len(timings))
            return "success", "", {
                "cheap_device_timing_us": cheap, "case_timings_us": timings,
                "commands": commands,
            }
        except (fusion.FusionContractError, KeyError, TypeError, ValueError) as exc:
            return "candidate_failure", str(exc), {"commands": commands}
    raise AssertionError("unreachable")


def run_autotune_gate(root: Path, request_path: Path, manifest_path: Path,
                      benchmark: str, output: Path, round_number: int) -> tuple[str, str, dict, dict]:
    try:
        request, request_sha = load_autotune_request(request_path, round_number)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return "infrastructure_failure", f"invalid host autotune request: {exc}", {}, {}
    spec = SPECS[benchmark]
    if (request.get("benchmark") != benchmark
            or request.get("required_correctness_cases") != len(spec["cases"])
            or request.get("expected_entrypoint") != spec["entrypoint"]
            or request.get("expected_declared_kernel_name") != spec["kernel_name"]
            or request.get("expected_profile_cases") != spec["ranked"]):
        return "infrastructure_failure", "host autotune request does not match the benchmark contract", {}, {}
    runner = load(root / "scripts/a3_benchmark_runner.py", "a3_tune_runner")
    fusion = load(root / "scripts/fully_fused_contract.py", "a3_tune_fusion")
    try:
        declaration = fusion.validate_manifest(json.loads(manifest_path.read_text()))
    except (OSError, json.JSONDecodeError, fusion.FusionContractError) as exc:
        return "infrastructure_failure", f"invalid host candidate contract: {exc}", {}, {}
    if (declaration["entrypoint"] != spec["entrypoint"]
            or declaration["kernel_name"] != spec["kernel_name"]):
        return "infrastructure_failure", "host candidate manifest does not match the benchmark contract", {}, {}
    common = {
        "protocol_version": 1, "profile": "bz-a3", "product": "a3",
        "runtime": "py311-torch", "benchmark": benchmark, "device": 0,
        "logical_device": 0,
        "baseline": str(root / f"benchmarks/{benchmark}/baseline.py"),
        "case_spec": str(root / f"benchmarks/{benchmark}/cases.jsonl"),
        "tolerances": {"rtol": spec["rtol"], "atol": spec["atol"]},
        "fusion_contract": declaration,
    }
    checkpoint_path = output / "autotune-checkpoint.json"
    if checkpoint_path.is_file():
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return "infrastructure_failure", f"invalid autotune checkpoint: {exc}", {}, {}
        if (not isinstance(checkpoint, dict)
                or checkpoint.get("schema") != AUTOTUNE_CHECKPOINT_SCHEMA
                or checkpoint.get("request_sha256") != request_sha
                or checkpoint.get("optimization_round") != round_number
                or not isinstance(checkpoint.get("screens"), dict)
                or not isinstance(checkpoint.get("profiles"), dict)
                or not isinstance(checkpoint.get("profile_failures", {}), dict)):
            return "infrastructure_failure", "autotune checkpoint request mismatch", {}, {}
    else:
        checkpoint = {
            "schema": AUTOTUNE_CHECKPOINT_SCHEMA,
            "request_sha256": request_sha, "optimization_round": round_number,
            "screens": {}, "profiles": {}, "profile_failures": {},
        }
        write_json(checkpoint_path, checkpoint)
    completed = checkpoint.get("completed_result")
    if isinstance(completed, dict):
        return (
            "success", "", completed["profile_summary"],
            completed["autotune_summary"],
        )
    candidates = output / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    by_id = {}
    manifest = json.loads(manifest_path.read_text())
    for index, option in enumerate(request["options"]):
        identifier = option["id"]
        candidate = candidates / f"{index:03d}.py"
        candidate.write_text(option["source"])
        by_id[identifier] = (option, candidate)
        if identifier in checkpoint["screens"]:
            continue
        outcome, diagnostics, row = screen_candidate(
            runner, fusion, manifest, common, spec, candidate,
        )
        if outcome == "infrastructure_failure":
            return outcome, diagnostics, {}, {}
        row["outcome"] = outcome
        if outcome == "candidate_failure":
            row["diagnostics"] = diagnostics
        checkpoint["screens"][identifier] = row
        write_json(checkpoint_path, checkpoint)
    request_order = {
        option["id"]: index for index, option in enumerate(request["options"])
    }
    survivors = sorted(
        (identifier for identifier, row in checkpoint["screens"].items()
         if row["outcome"] == "success"),
        key=lambda identifier: (
            checkpoint["screens"][identifier]["cheap_device_timing_us"],
            request_order[identifier],
        ),
    )
    if not survivors:
        return "candidate_failure", "all options failed broad autotune screening", {}, {}
    finalists = survivors[:MAX_AUTOTUNE_FINALISTS]
    profile_failures = checkpoint.setdefault("profile_failures", {})
    for identifier in finalists:
        if identifier in checkpoint["profiles"] or identifier in profile_failures:
            continue
        option, candidate = by_id[identifier]
        for retry in range(3):
            outcome, diagnostics, timing = profile_candidate(
                root, common, spec, candidate, output / f"finalist-{identifier}-try-{retry}",
                round_number, pipe_activity=False,
            )
            if outcome != "infrastructure_failure":
                break
        command = f"batch_profile_a3 finalist={identifier} repeats=3 metric=BasicInfo"
        if outcome == "infrastructure_failure":
            return outcome, diagnostics, {}, {}
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if candidate_sha != option["source_sha256"]:
            return "infrastructure_failure", "finalist artifact does not match its declared source", {}, {}
        if outcome == "candidate_failure":
            profile_failures[identifier] = {
                "option_id": identifier,
                "source_sha256": option["source_sha256"],
                "candidate_artifact_sha256": candidate_sha,
                "outcome": "candidate_failure", "diagnostics": diagnostics,
                "commands": [command] * (retry + 1),
            }
            write_json(checkpoint_path, checkpoint)
            continue
        checkpoint["profiles"][identifier] = {
            "option_id": identifier,
            "source_sha256": option["source_sha256"],
            "candidate_artifact_sha256": candidate_sha,
            "timing": timing,
            "commands": [command] * (retry + 1),
        }
        write_json(checkpoint_path, checkpoint)
    profiled_finalists = [identifier for identifier in finalists
                          if identifier in checkpoint["profiles"]]
    official_order = sorted(
        profiled_finalists,
        key=lambda identifier: (
            checkpoint["profiles"][identifier]["timing"]["geomean_us"],
            finalists.index(identifier),
        ),
    )
    winner_failures = checkpoint.setdefault("winner_gate_failures", {})
    winner = None
    profile_summary = {}
    for identifier in official_order:
        if identifier in winner_failures:
            continue
        option, candidate = by_id[identifier]
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if candidate_sha != option["source_sha256"]:
            return "infrastructure_failure", "winner artifact does not match its declared source", {}, {}
        for retry in range(3):
            outcome, diagnostics, profile_summary = run_gate(
                root, candidate, manifest_path, benchmark,
                output / f"winner-{identifier}-try-{retry}", round_number,
            )
            if outcome != "infrastructure_failure":
                break
        commands = [f"a3_terminal_gate full winner={identifier}"] * (retry + 1)
        if outcome == "infrastructure_failure":
            return outcome, diagnostics, {}, {}
        if outcome == "candidate_failure":
            winner_failures[identifier] = {
                "option_id": identifier,
                "source_sha256": option["source_sha256"],
                "candidate_artifact_sha256": candidate_sha,
                "outcome": "candidate_failure",
                "diagnostics": diagnostics,
                "commands": commands,
            }
            write_json(checkpoint_path, checkpoint)
            continue
        winner = identifier
        break
    rows = []
    for option in request["options"]:
        screen = checkpoint["screens"][option["id"]]
        row = {"id": option["id"], "source_sha256": option["source_sha256"],
               "outcome": screen["outcome"]}
        if screen["outcome"] == "success":
            row["cheap_device_timing_us"] = screen["cheap_device_timing_us"]
        else:
            row["diagnostics"] = screen["diagnostics"]
        rows.append(row)
    retained_commands = [command for row in checkpoint["screens"].values()
                         for command in row.get("commands", [])]
    retained_commands += [command for row in checkpoint["profiles"].values()
                          for command in row.get("commands", [])]
    retained_commands += [command for row in profile_failures.values()
                          for command in row.get("commands", [])]
    retained_commands += [command for row in winner_failures.values()
                          for command in row.get("commands", [])]
    if winner is not None:
        retained_commands += [f"a3_terminal_gate full winner={winner}"] * (retry + 1)
    failure_rows = [winner_failures[identifier] for identifier in official_order
                    if identifier in winner_failures]
    profile_failure_rows = [profile_failures[identifier] for identifier in finalists
                            if identifier in profile_failures]
    autotune_summary = {
        "schema": AUTOTUNE_RESULT_SCHEMA, "options": rows,
        "optimization_round": round_number,
        "finalists": finalists, "profiled_finalists": profiled_finalists,
        "profile_failures": profile_failure_rows,
        "winner_gate_failures": failure_rows,
        "commands": retained_commands,
        "finalist_profiles": {
            identifier: {key: value for key, value in checkpoint["profiles"][identifier].items()
                         if key != "commands"}
            for identifier in profiled_finalists
        },
    }
    if winner is None:
        diagnostics = (
            "all selected finalists failed official profiling"
            if not profiled_finalists else
            "all officially profiled finalists failed the complete winner gate"
        )
        return "candidate_failure", diagnostics, {}, autotune_summary
    winner_option, winner_path = by_id[winner]
    candidate_artifact_sha = hashlib.sha256(winner_path.read_bytes()).hexdigest()
    autotune_summary.update({
        "winner_id": winner, "winner_source_sha256": winner_option["source_sha256"],
        "winner_full_gate": {
            "winner_id": winner,
            "source_sha256": winner_option["source_sha256"],
            "candidate_artifact_sha256": candidate_artifact_sha,
            "profile_summary_sha256": canonical_json_sha256(profile_summary),
        },
    })
    checkpoint["completed_result"] = {
        "profile_summary": profile_summary,
        "autotune_summary": autotune_summary,
    }
    write_json(checkpoint_path, checkpoint)
    return "success", "", profile_summary, autotune_summary


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
    parser.add_argument("--autotune-request", type=Path)
    args = parser.parse_args()
    environment_error = validate_environment()
    if environment_error:
        return emit("infrastructure_failure", environment_error)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.autotune_request is not None:
            outcome, diagnostics, summary, autotune_summary = run_autotune_gate(
                args.root.resolve(), args.autotune_request.resolve(), args.manifest.resolve(),
                args.benchmark, args.output.resolve(), args.round,
            )
        else:
            outcome, diagnostics, summary = run_gate(
                args.root.resolve(), args.candidate.resolve(), args.manifest.resolve(),
                args.benchmark, args.output.resolve(), args.round,
            )
            autotune_summary = {}
    except BaseException as exc:
        outcome = "infrastructure_failure"
        diagnostics = f"host gate failed unexpectedly: {type(exc).__name__}: {exc}"
        summary, autotune_summary = {}, {}
    values = {"profile_summary": summary} if summary else {}
    if autotune_summary:
        values["autotune_summary"] = autotune_summary
    checkpoint_path = args.output / "autotune-checkpoint.json"
    if (args.autotune_request is not None and outcome == "infrastructure_failure"
            and checkpoint_path.is_file()):
        try:
            values["autotune_checkpoint"] = json.loads(checkpoint_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return emit(outcome, diagnostics, **values)


if __name__ == "__main__":
    raise SystemExit(main())
