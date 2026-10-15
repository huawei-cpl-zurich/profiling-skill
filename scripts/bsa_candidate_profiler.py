#!/usr/bin/env python3
"""Inventory and resumably profile retained BSA candidate variants.

The profiler delegates device-specific execution to a small JSON executor
contract.  That keeps this coordinator independently testable while requiring
the production executor to return compact BasicInfo and PipeUtilization
evidence rather than vendor report trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any


TREATMENTS = (
    "a3-bsa-cannbot-all",
    "a3-bsa-cannbot-new-profiler",
    "a3-bsa-guarded-new-profiler",
    "a3-bsa-guarded-old-profiler",
)
CANDIDATE_FAILURES = {"compile_error", "runtime_error", "correctness_error"}
SCHEMA = "profiling-skill/bsa-candidate-manifest/v1"
LEGACY_RESULT_SCHEMA = "profiling-skill/bsa-candidate-results/v1"
RESULT_SCHEMA = "profiling-skill/bsa-candidate-results/v2"
MERGED_RESULT_SCHEMA = "profiling-skill/bsa-candidate-merged-results/v1"
SHARD_ALGORITHM = "sha256-mod-v1"
SAMPLE_SELECTION = "first-stable-consecutive-window-v1"
DIRECT_API_CAMPAIGN_REVISION = "direct-api-v1"
TARGET_ACTIVATION = {
    "gz-a3": "named-runtime",
    "bz-a3-1": "named-runtime",
    "bz-a3-2": "named-runtime",
}


class InvalidArtifact(ValueError):
    pass


def extract_features(source: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Extract conservative, reviewable BSA implementation dimensions."""
    if re.search(r"(?:tl\.)?range\s*\(\s*0\s*,\s*(?:sk\b|ke\s*-\s*ks)", source):
        loop_bound = "per_sequence_k"
    elif re.search(r"(?:tl\.)?range\s*\(\s*0\s*,\s*TOTAL_K\b", source):
        loop_bound = "total_packed_k"
    else:
        loop_bound = "unknown"

    linear_rank = re.search(
        r"for\s+\w+\s+in\s+(?:tl\.)?(?:static_)?range\s*\([^\n]*\bH\b", source
    ) and re.search(r"\brank\s*\+?=", source)
    if linear_rank:
        head_rank = "linear_scan"
    elif re.search(r"dense_heads|block_rank\s*=|rank\s*=\s*tl\.maximum\s*\(\s*h", source):
        head_rank = "direct"
    else:
        head_rank = "unknown"

    if re.search(r"mask\s*=\s*(?:keep|load_mask)\b", source):
        mask_style = "final_keep"
    elif re.search(r"mask\s*=\s*(?:valid|kv_mask)\b", source):
        mask_style = "valid_region"
    else:
        mask_style = "unknown"

    if "tl.dot(" in source:
        reduction = "matrix_tl_dot"
    elif re.search(r"tl\.sum\s*\([^\n]*(?:qv|kv|kval|\*\s*q)", source):
        reduction = "row_tl_sum"
    else:
        reduction = "unknown"

    block_m = parameters.get("BLOCK_M")
    if isinstance(block_m, int) and block_m > 1:
        query_tiling = "block_m"
    elif re.search(r"\b(?:qi|qt)\s*=\s*tl\.program_id", source):
        query_tiling = "single_query"
    else:
        query_tiling = "unknown"
    if re.search(r"triton\.cdiv\s*\([^\n]*BLOCK_M[^\n]*\)\s*,\s*[hH]", source):
        grid = "query_tiles_by_head"
    elif re.search(r"\(\s*(?:tq|total_q)\s*\*\s*[hH]\s*,?\s*\)", source):
        grid = "flattened_query_head"
    elif "tl.program_id(1)" in source:
        grid = "query_by_head"
    else:
        grid = "unknown"

    return {
        "loop_bound": loop_bound,
        "head_rank_strategy": head_rank,
        "block_n": parameters.get("BLOCK_N"),
        "whole_tile_skip": bool(re.search(r"\bif\s+has\s*:", source)),
        "scalar_block_decision": bool(re.search(
            r"\bcb\s*=\s*(?:start|start_n)\s*//", source
        )),
        "kv_mask_style": mask_style,
        "num_warps": parameters.get("num_warps"),
        "num_stages": parameters.get("num_stages"),
        "reduction_strategy": reduction,
        "query_tiling": query_tiling,
        "block_m": block_m,
        "grid_strategy": grid,
    }


def _feature_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise InvalidArtifact(f"cannot hash executor {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidArtifact(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidArtifact(f"{path} is not a JSON object")
    return value


def _attempt_number(path: Path) -> tuple[int, str]:
    name = path.parent.name
    return (int(name), name) if name.isdigit() else (2**31, name)


def inventory_campaign(
    root: Path, *, environment: dict[str, Any], starter: dict[str, Any]
) -> dict[str, Any]:
    """Recover every request occurrence and group exact executable duplicates."""
    root = root.resolve()
    occurrences: list[dict[str, Any]] = []
    executions: dict[str, dict[str, Any]] = {}
    treatment_summary: dict[str, dict[str, int]] = {}

    # Sources occasionally disappear from sanitized round records.  Build a
    # campaign-wide hash index before reading individual occurrences.
    sources: dict[str, str] = {}
    for path in root.glob("*/attempts/*/autotune-request.json"):
        try:
            request = _read_json(path)
        except InvalidArtifact:
            continue
        for option in request.get("options", []):
            if not isinstance(option, dict) or not isinstance(option.get("source"), str):
                continue
            digest = hashlib.sha256(option["source"].encode()).hexdigest()
            declared = option.get("source_sha256", digest)
            if declared == digest:
                sources[digest] = option["source"]

    for treatment in TREATMENTS:
        directory = root / treatment
        if not directory.is_dir():
            raise InvalidArtifact(f"missing treatment directory: {treatment}")
        count = recoverable = 0
        treatment_executions: set[str] = set()
        paths = sorted(directory.glob("attempts/*/autotune-request.json"), key=_attempt_number)
        for request_path in paths:
            request = _read_json(request_path)
            if request.get("schema") != "simple-experiment-autotune-request/v1":
                raise InvalidArtifact(f"unsupported request schema in {request_path}")
            attempt = request_path.parent.name
            round_number = request.get("optimization_round")
            result_path = request_path.with_name("result.json")
            result = _read_json(result_path) if result_path.is_file() else {}
            options = request.get("options")
            if not isinstance(options, list):
                raise InvalidArtifact(f"options are not an array in {request_path}")
            request_occurrences = []
            for position, option in enumerate(options):
                if not isinstance(option, dict):
                    raise InvalidArtifact(f"option {position} is not an object in {request_path}")
                count += 1
                digest = option.get("source_sha256")
                source = option.get("source")
                if source is None and isinstance(digest, str):
                    source = sources.get(digest)
                computed = (
                    hashlib.sha256(source.encode()).hexdigest()
                    if isinstance(source, str) else None
                )
                occurrence_id = (
                    f"{treatment}/round-{round_number}/attempt-{attempt}/"
                    f"option-{position}-{option.get('id', 'unnamed')}"
                )
                occurrence = {
                    "occurrence_id": occurrence_id,
                    "treatment": treatment,
                    "round": round_number,
                    "attempt": attempt,
                    "position": position,
                    "option_id": option.get("id"),
                    "parameters": option.get("parameters", {}),
                    "hypothesis": option.get("hypothesis"),
                    "source_sha256": digest,
                    "artifact_outcome": result.get("outcome"),
                    "artifact_diagnostics": result.get("diagnostics"),
                    "request_path": str(request_path.relative_to(root)),
                }
                if not isinstance(source, str):
                    occurrence["recoverability"] = "source_unavailable"
                elif not isinstance(digest, str) or digest != computed:
                    occurrence["recoverability"] = "source_hash_mismatch"
                    occurrence["computed_source_sha256"] = computed
                else:
                    recoverable += 1
                    occurrence["recoverability"] = "recoverable"
                    identity = {
                        "benchmark": "bsa",
                        "source_sha256": digest,
                        "parameters": option.get("parameters", {}),
                        "environment": environment,
                    }
                    execution_id = "candidate-" + _sha(identity)[:20]
                    occurrence["execution_id"] = execution_id
                    treatment_executions.add(execution_id)
                    if execution_id not in executions:
                        executions[execution_id] = {
                            "execution_id": execution_id,
                            **identity,
                            "source": source,
                            "features": extract_features(
                                source, option.get("parameters", {})
                            ),
                            "declared_kernel_name": request.get(
                                "expected_declared_kernel_name",
                                "block_sparse_attention_kernel",
                            ),
                            "occurrence_ids": [],
                        }
                    executions[execution_id]["occurrence_ids"].append(occurrence_id)
                occurrences.append(occurrence)
                request_occurrences.append(occurrence)
            incumbent_id = request.get("incumbent_id")
            incumbents = [
                item for item in request_occurrences
                if item["option_id"] == incumbent_id
            ]
            if isinstance(incumbent_id, str) and len(incumbents) == 1:
                parent = incumbents[0]
                for item in request_occurrences:
                    item["is_incumbent"] = item is parent
                    if item is not parent:
                        item["parent_occurrence_id"] = parent["occurrence_id"]
                        item["lineage_note"] = "request-declared incumbent"
        treatment_summary[treatment] = {
            "occurrences": count,
            "recoverable_occurrences": recoverable,
            "unrecoverable_occurrences": count - recoverable,
            "unique_executions": len(treatment_executions),
        }

    # A next-round incumbent inherits the prior winner only when the round
    # artifact names its source hash and exactly one successful occurrence
    # realizes that winner. Ambiguous retries and partial CANNBot histories
    # deliberately remain unlinked.
    for treatment in TREATMENTS:
        directory = root / treatment
        winners: dict[int, str] = {}
        for result_path in directory.glob("repo/rounds/*/autotune-result.json"):
            try:
                round_number = int(result_path.parent.name)
                winner = _read_json(result_path).get("winner_source_sha256")
            except (InvalidArtifact, ValueError):
                continue
            if isinstance(winner, str):
                winners[round_number] = winner
        for item in occurrences:
            if (
                item["treatment"] != treatment
                or not item.get("is_incumbent")
                or "parent_occurrence_id" in item
                or not isinstance(item.get("round"), int)
                or item["round"] <= 1
            ):
                continue
            winner = winners.get(item["round"] - 1)
            if winner is None or item.get("source_sha256") != winner:
                continue
            parents = [
                parent for parent in occurrences
                if parent["treatment"] == treatment
                and parent.get("round") == item["round"] - 1
                and parent.get("source_sha256") == winner
                and parent.get("artifact_outcome") == "success"
            ]
            if len(parents) == 1:
                item["parent_occurrence_id"] = parents[0]["occurrence_id"]
                item["lineage_note"] = "prior round winner hash"

    by_occurrence = {item["occurrence_id"]: item for item in occurrences}
    for item in occurrences:
        parent_id = item.get("parent_occurrence_id")
        if not isinstance(parent_id, str):
            continue
        parent = by_occurrence[parent_id]
        current_execution = executions.get(item.get("execution_id"))
        parent_execution = executions.get(parent.get("execution_id"))
        if current_execution is not None and parent_execution is not None:
            item["parent_feature_deltas"] = _feature_deltas(
                parent_execution["features"], current_execution["features"]
            )

    starter_source = starter.get("source")
    starter_parameters = starter.get("parameters", {})
    starter_kernel_name = starter.get("declared_kernel_name")
    if (
        not isinstance(starter_source, str)
        or not isinstance(starter_parameters, dict)
        or not isinstance(starter_kernel_name, str)
        or not starter_kernel_name
    ):
        raise InvalidArtifact("starter requires source, parameters, and declared kernel name")
    starter_sha = hashlib.sha256(starter_source.encode()).hexdigest()
    starter_identity = {
        "benchmark": "bsa", "source_sha256": starter_sha,
        "parameters": starter_parameters, "environment": environment,
    }
    starter_execution_id = "candidate-" + _sha(starter_identity)[:20]
    matching_occurrences = sorted(
        (
            item for item in occurrences
            if item.get("execution_id") == starter_execution_id
        ),
        key=lambda item: item["occurrence_id"],
    )
    if matching_occurrences:
        starter_occurrence = matching_occurrences[0]
        executions[starter_execution_id]["declared_kernel_name"] = starter_kernel_name
    else:
        starter_occurrence = {
            "occurrence_id": "frozen-bsa-starter",
            "treatment": None,
            "round": 0,
            "attempt": None,
            "position": 0,
            "option_id": "frozen-bsa-starter",
            "parameters": starter_parameters,
            "hypothesis": None,
            "source_sha256": starter_sha,
            "artifact_outcome": None,
            "artifact_diagnostics": None,
            "request_path": None,
            "recoverability": "recoverable",
            "execution_id": starter_execution_id,
        }
        occurrences.append(starter_occurrence)
        executions[starter_execution_id] = {
            "execution_id": starter_execution_id,
            **starter_identity,
            "source": starter_source,
            "features": extract_features(starter_source, starter_parameters),
            "declared_kernel_name": starter_kernel_name,
            "occurrence_ids": [starter_occurrence["occurrence_id"]],
        }
    starter_occurrence["lineage_role"] = "starter"

    recoverable_count = sum(
        item["recoverability"] == "recoverable" for item in occurrences
    )
    return {
        "schema": SCHEMA,
        "campaign_root": str(root),
        "environment": environment,
        "starter_occurrence_id": starter_occurrence["occurrence_id"],
        "treatments": treatment_summary,
        "occurrences": occurrences,
        "executions": sorted(executions.values(), key=lambda item: item["execution_id"]),
        "summary": {
            "treatments": len(TREATMENTS),
            "occurrences": len(occurrences),
            "recoverable_occurrences": recoverable_count,
            "unrecoverable_occurrences": len(occurrences) - recoverable_count,
            "unique_executions": len(executions),
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def execution_shard(execution_id: str, shard_count: int) -> int:
    return int.from_bytes(hashlib.sha256(execution_id.encode()).digest()[:8], "big") % shard_count


def _validate_shard(shard_count: int, shard_index: int) -> None:
    if type(shard_count) is not int or shard_count < 1:
        raise InvalidArtifact("shard count must be positive")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise InvalidArtifact("shard index must be in [0, shard count)")


def _execution_environment(environment: object) -> dict[str, str]:
    if not isinstance(environment, dict):
        raise InvalidArtifact("execution environment must be an object")
    target = environment.get("target")
    runtime = environment.get("runtime")
    activation = environment.get("runtime_activation")
    expected_activation = TARGET_ACTIVATION.get(target)
    if (
        expected_activation is None
        or not isinstance(runtime, str)
        or not runtime
        or activation != expected_activation
    ):
        raise InvalidArtifact(
            "execution environment requires an allowed A3 target, a logical "
            "runtime, and its matching runtime_activation "
            "(gz-a3/bz-a3-1/bz-a3-2=named-runtime)"
        )
    return {
        "target": target,
        "runtime": runtime,
        "runtime_activation": activation,
    }


def _run_contract(
    cases: list[int], stable_samples: int, max_samples: int,
    variability_threshold: float, require_provenance: bool,
    environment: dict[str, Any], executor_identity: dict[str, str],
) -> dict[str, Any]:
    return {
        "cases": cases,
        "stable_samples": stable_samples,
        "max_samples": max_samples,
        "variability_threshold": variability_threshold,
        "require_provenance": require_provenance,
        "sample_selection": SAMPLE_SELECTION,
        "execution_environment": _execution_environment(environment),
        "executor_identity": executor_identity,
    }


def _executor_identity(executable: Path, campaign_revision: str) -> dict[str, str]:
    if not isinstance(campaign_revision, str) or not campaign_revision.strip():
        raise InvalidArtifact("campaign revision must be a nonempty string")
    if not executable.is_file():
        raise InvalidArtifact(f"executor is not a regular file: {executable}")
    return {
        "campaign_revision": campaign_revision,
        "executable_sha256": _file_sha256(executable),
    }


def _initial_results(
    manifest: dict[str, Any], contract: dict[str, Any],
    shard_count: int, shard_index: int,
) -> dict[str, Any]:
    selected = [
        item for item in manifest["executions"]
        if execution_shard(item["execution_id"], shard_count) == shard_index
    ]
    return {
        "schema": RESULT_SCHEMA,
        "manifest_sha256": _sha(manifest),
        "cases": contract["cases"],
        "run_contract": contract,
        "shard": {
            "algorithm": SHARD_ALGORITHM,
            "count": shard_count,
            "index": shard_index,
        },
        "executions": {
            item["execution_id"]: {
                "status": "pending",
                "kernel_name": None,
                "selectors": None,
                "components": None,
                "basic_info": None,
                "cases": {
                    str(case): {"status": "pending", "samples": []}
                    for case in contract["cases"]
                },
                "infrastructure_failures": [],
            }
            for item in selected
        },
    }


def _load_results(
    path: Path, manifest: dict[str, Any], contract: dict[str, Any],
    shard_count: int, shard_index: int,
) -> dict[str, Any]:
    if not path.exists():
        return _initial_results(manifest, contract, shard_count, shard_index)
    value = _read_json(path)
    if value.get("schema") == LEGACY_RESULT_SCHEMA:
        raise InvalidArtifact(
            "legacy v1 checkpoint cannot represent consecutive-window or "
            "multi-component captures; regenerate it with the v2 runner"
        )
    if value.get("schema") != RESULT_SCHEMA:
        raise InvalidArtifact("result schema mismatch")
    if value.get("manifest_sha256") != _sha(manifest):
        raise InvalidArtifact("result manifest identity mismatch")
    checkpoint_contract = value.get("run_contract")
    if (
        not isinstance(checkpoint_contract, dict)
        or "execution_environment" not in checkpoint_contract
        or "executor_identity" not in checkpoint_contract
    ):
        raise InvalidArtifact(
            "checkpoint predates target/runtime activation or executor identity; "
            "regenerate it"
        )
    if checkpoint_contract != contract:
        raise InvalidArtifact("result run contract mismatch")
    expected_shard = {
        "algorithm": SHARD_ALGORITHM, "count": shard_count, "index": shard_index,
    }
    if value.get("shard") != expected_shard:
        raise InvalidArtifact("result shard identity mismatch")
    return value


def _invoke_executor(
    executable: Path, protocol_dir: Path, request: dict[str, Any]
) -> dict[str, Any]:
    protocol_dir.mkdir(parents=True, exist_ok=True)
    request_path = protocol_dir / "request.json"
    response_path = protocol_dir / "response.json"
    _atomic_json(request_path, request)
    response_path.unlink(missing_ok=True)
    run = subprocess.run(
        [str(executable), "--request", str(request_path), "--response", str(response_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if not response_path.is_file():
        return {
            "status": "infrastructure_error",
            "diagnostics": (
                f"executor exit {run.returncode} produced no response; "
                f"stdout={run.stdout[-2000:]!r}; stderr={run.stderr[-2000:]!r}"
            ),
        }
    try:
        response = _read_json(response_path)
    except InvalidArtifact as exc:
        return {"status": "infrastructure_error", "diagnostics": str(exc)}
    if run.returncode and response.get("status") == "ok":
        return {
            "status": "infrastructure_error",
            "diagnostics": f"executor exited {run.returncode} despite success response",
        }
    return response


def _stable_window(samples: list[dict[str, Any]], wanted: int, threshold: float):
    for start in range(len(samples) - wanted + 1):
        indexes = tuple(range(start, start + wanted))
        values = [float(samples[index]["duration_us"]) for index in indexes]
        median = statistics.median(values)
        relative_range = (max(values) - min(values)) / median
        if relative_range <= threshold:
            return relative_range, indexes, median
    return None


def _complete_stable_case(
    state: dict[str, Any], wanted: int, threshold: float
) -> bool:
    if len(state["samples"]) < wanted:
        return False
    stable = _stable_window(state["samples"], wanted, threshold)
    if stable is None:
        return False
    spread, indexes, median = stable
    state.update({
        "status": "completed",
        "accepted_sample_indexes": list(indexes),
        "median_us": median,
        "relative_range": spread,
    })
    return True


def _failure(response: dict[str, Any]) -> dict[str, Any]:
    kind = response.get("status", "infrastructure_error")
    failure = {
        "kind": kind,
        "category": "candidate" if kind in CANDIDATE_FAILURES else "infrastructure",
        "diagnostics": str(response.get("diagnostics", "")),
        "evidence_sha256": response.get("evidence_sha256"),
        "provenance": response.get("provenance"),
    }
    if "terminal" in response:
        failure["terminal"] = response["terminal"]
    return failure


def _valid_provenance(value: object, environment: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    expected = _execution_environment(environment)
    target = value.get("target")
    runtime = value.get("runtime")
    return (
        target == expected["target"]
        and runtime == expected["runtime"]
        and value.get("runtime_activation") == expected["runtime_activation"]
        and isinstance(value.get("physical_device"), int)
        and isinstance(value.get("command"), (str, list))
        and isinstance(value.get("handle"), str)
        and value["handle"].startswith(f"remote:{target}:")
    )


def _valid_evidence_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise InvalidArtifact("manifest schema mismatch")
    starter_id = manifest.get("starter_occurrence_id")
    occurrences = manifest.get("occurrences")
    starters = [
        item for item in occurrences or []
        if isinstance(item, dict) and item.get("lineage_role") == "starter"
    ]
    if (
        not isinstance(starter_id, str)
        or len(starters) != 1
        or starters[0].get("occurrence_id") != starter_id
    ):
        raise InvalidArtifact(
            "manifest must identify exactly one starter occurrence via "
            "starter_occurrence_id and lineage_role"
        )


def _retry_from_failure(response: dict[str, Any]) -> dict[str, Any] | None:
    provenance = response.get("provenance")
    handle = provenance.get("handle") if isinstance(provenance, dict) else None
    if not isinstance(handle, str) or not handle.startswith("remote:"):
        return None
    if response.get("terminal") is True:
        return {"mode": "new_attempt", "prior_handle": handle}
    retry = {"mode": "observe", "handle": handle}
    if "terminal" not in response:
        retry["terminal_unknown"] = True
    return retry


def _valid_failure_provenance(value: object, environment: dict[str, Any]) -> bool:
    """Allow sparse failure provenance, but never cross execution identities."""
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    expected = _execution_environment(environment)
    for field in ("target", "runtime", "runtime_activation"):
        if field in value and value[field] != expected[field]:
            return False
    handle = value.get("handle")
    return handle is None or (
        isinstance(handle, str)
        and handle.startswith(f"remote:{expected['target']}:")
    )


def _record_infrastructure_failure(
    state: dict[str, Any], response: dict[str, Any], environment: dict[str, Any]
) -> None:
    state.pop("retry", None)
    if not _valid_failure_provenance(response.get("provenance"), environment):
        state["infrastructure_failures"].append({
            "kind": "invalid_provenance",
            "category": "infrastructure",
            "diagnostics": (
                "infrastructure failure provenance does not match the requested "
                "target, runtime activation, or handle namespace"
            ),
            "rejected_provenance": response.get("provenance"),
        })
        return
    state["infrastructure_failures"].append(_failure(response))
    retry = _retry_from_failure(response)
    if retry is not None:
        state["retry"] = retry


def _record_malformed_success(
    state: dict[str, Any], response: dict[str, Any], *, kind: str, diagnostics: str
) -> None:
    """Retain malformed terminal evidence and force a distinct remote attempt."""
    state.pop("retry", None)
    failure = {
        "kind": kind,
        "category": "infrastructure",
        "diagnostics": diagnostics,
        "evidence_sha256": response.get("evidence_sha256"),
        "provenance": response.get("provenance"),
    }
    if "terminal" in response:
        failure["terminal"] = response["terminal"]
    state["infrastructure_failures"].append(failure)
    retry = _retry_from_failure(response)
    if retry is not None:
        state["retry"] = retry


def _reject_inexact_response_provenance(
    state: dict[str, Any], response: dict[str, Any], environment: dict[str, Any],
    evidence_name: str,
) -> bool:
    """Reject success/candidate evidence that came from another environment."""
    if _valid_provenance(response.get("provenance"), environment):
        return False
    state.pop("retry", None)
    state["status"] = "retryable_infrastructure"
    state["infrastructure_failures"].append({
        "kind": "invalid_provenance",
        "category": "infrastructure",
        "diagnostics": (
            f"{evidence_name} provenance must match the requested A3 target, "
            "runtime activation, device, command, and handle"
        ),
        "rejected_provenance": response.get("provenance"),
    })
    return True


def _with_retry(request: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if isinstance(state.get("retry"), dict):
        request["retry"] = state["retry"]
    return request


def _acquisition_exhausted(state: dict[str, Any]) -> bool:
    """Bound new submissions, never observations of an unresolved remote job."""
    if state.get("retry", {}).get("mode") == "observe":
        return False
    failures = state.get("infrastructure_failures", [])
    terminal = [item for item in failures if item.get("terminal") is not False]
    # A successful discovery/capture resets the consecutive failure window.
    terminal = terminal[state.get("acquisition_failure_offset", 0):]
    diagnostics = [item.get("diagnostics", "") for item in terminal]
    deterministic_prefixes = (
        "missing_pipe_rows:", "invalid_duration:", "duration_row_count:",
        "capture lacks one exact PipeUtilization or duration row",
    )
    repeated = (len(diagnostics) >= 2 and diagnostics[-1] == diagnostics[-2]
                and diagnostics[-1].startswith(deterministic_prefixes))
    if len(terminal) < 3 and not repeated:
        return False
    state["status"] = "measurement_unstable"
    state["failure"] = {
        "kind": "acquisition_exhausted", "category": "infrastructure",
        "diagnostics": (
            "Acquisition stopped after repeated deterministic evidence failure"
            if repeated else "Acquisition exhausted initial attempt plus two retries"
        ),
        "last_diagnostics": diagnostics[-1],
    }
    state.pop("retry", None)
    return True


def _acquisition_succeeded(state: dict[str, Any]) -> None:
    state["acquisition_failure_offset"] = sum(
        item.get("terminal") is not False
        for item in state.get("infrastructure_failures", [])
    )


def run_campaign(
    manifest: dict[str, Any],
    output: Path,
    executor: Path,
    *,
    cases: list[int],
    stable_samples: int,
    max_samples: int,
    max_actions: int | None,
    variability_threshold: float = 0.05,
    require_provenance: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
    campaign_revision: str = DIRECT_API_CAMPAIGN_REVISION,
) -> dict[str, Any]:
    _validate_manifest(manifest)
    execution_environment = _execution_environment(manifest.get("environment"))
    if stable_samples < 1 or max_samples < stable_samples:
        raise InvalidArtifact("sample limits are invalid")
    _validate_shard(shard_count, shard_index)
    contract = _run_contract(
        cases, stable_samples, max_samples, variability_threshold,
        require_provenance, manifest["environment"],
        _executor_identity(executor, campaign_revision),
    )
    results = _load_results(
        output, manifest, contract, shard_count, shard_index
    )
    protocol_dir = output.parent / (output.stem + ".protocol")
    actions = 0

    for candidate in manifest["executions"]:
        if execution_shard(candidate["execution_id"], shard_count) != shard_index:
            continue
        state = results["executions"][candidate["execution_id"]]
        if state["status"] in {"completed", "candidate_failure", "measurement_unstable"}:
            continue
        if _acquisition_exhausted(state):
            active_case = state.get("active_case")
            if active_case is None or state.get("selectors") is None:
                _atomic_json(output, results)
                continue
            failed_case = state["cases"][str(active_case)]
            failed_case["status"] = "acquisition_exhausted"
            failed_case["failure"] = state.pop("failure")
            state["status"] = "running"
            _acquisition_succeeded(state)
        if state.get("selectors") is None:
            if max_actions is not None and actions >= max_actions:
                _atomic_json(output, results)
                return results
            state["status"] = "running"
            response = _invoke_executor(executor, protocol_dir, _with_retry({
                "schema": "profiling-skill/bsa-candidate-executor-request/v1",
                "action": "discover",
                "candidate": candidate,
                "correctness_cases": cases,
                "metric": "BasicInfo",
                "execution_environment": execution_environment,
            }, state))
            actions += 1
            status = response.get("status")
            if (
                require_provenance
                and status in CANDIDATE_FAILURES | {"ok"}
                and _reject_inexact_response_provenance(
                    state, response, manifest["environment"], "BasicInfo"
                )
            ):
                _atomic_json(output, results)
                continue
            if status in CANDIDATE_FAILURES:
                state["status"] = "candidate_failure"
                state["failure"] = _failure(response)
                _atomic_json(output, results)
                continue
            if status != "ok":
                state["status"] = "retryable_infrastructure"
                _record_infrastructure_failure(
                    state, response, manifest["environment"]
                )
                _atomic_json(output, results)
                continue
            state.pop("retry", None)
            if not _valid_evidence_digest(response.get("evidence_sha256")):
                state["status"] = "retryable_infrastructure"
                _record_malformed_success(
                    state, response,
                    kind="invalid_evidence_digest",
                    diagnostics=(
                        "BasicInfo response lacks a valid SHA-256 evidence digest"
                    ),
                )
                _atomic_json(output, results)
                continue
            if not isinstance(response.get("basic_info"), (dict, list)):
                state["status"] = "retryable_infrastructure"
                _record_malformed_success(
                    state, response,
                    kind="invalid_basic_info",
                    diagnostics="BasicInfo response lacks compact dict/list evidence",
                )
                _atomic_json(output, results)
                continue
            raw_components = response.get("components")
            components = None
            if (
                isinstance(raw_components, list) and raw_components
                and all(
                    isinstance(item, dict)
                    and isinstance(item.get("selector"), str)
                    and item["selector"]
                    for item in raw_components
                )
            ):
                selectors = [item["selector"] for item in raw_components]
                if len(selectors) == len(set(selectors)):
                    components = raw_components
            names = response.get("kernel_names")
            if components is None and isinstance(names, list) and names and all(
                isinstance(name, str) and name for name in names
            ):
                declared = candidate["declared_kernel_name"]
                selected = declared if declared in names else (names[0] if len(names) == 1 else None)
                if selected is not None:
                    components = [{"selector": selected}]
            if components is None:
                state["status"] = "retryable_infrastructure"
                _record_malformed_success(
                    state, response,
                    kind="invalid_basic_info",
                    diagnostics=(
                        "BasicInfo returned neither distinct explicit components nor "
                        "one unambiguous legacy kernel selector"
                    ),
                )
                _atomic_json(output, results)
                continue
            state["components"] = components
            state["selectors"] = [item["selector"] for item in components]
            # Retained for report consumers which predate split AIC/AIV kernels.
            state["kernel_name"] = state["selectors"][0]
            state["basic_info"] = {
                "kernel_names": (
                    names if isinstance(names, list) else state["selectors"]
                ),
                "components": components,
                "compact": response.get("basic_info"),
                "evidence_sha256": response.get("evidence_sha256"),
                "provenance": response.get("provenance"),
            }
            _acquisition_succeeded(state)
            _atomic_json(output, results)

        interrupted = False
        for case in cases:
            case_state = state["cases"][str(case)]
            if case_state["status"] in {"completed", "acquisition_exhausted"}:
                continue
            while len(case_state["samples"]) < max_samples:
                if _complete_stable_case(
                    case_state, stable_samples, variability_threshold
                ):
                    break
                if max_actions is not None and actions >= max_actions:
                    interrupted = True
                    break
                state["status"] = "running"
                partial = case_state.setdefault("in_progress_sample", {
                    "sample_index": len(case_state["samples"]), "components": {},
                })
                for component in state["components"]:
                    selector = component["selector"]
                    if selector in partial["components"]:
                        continue
                    if max_actions is not None and actions >= max_actions:
                        interrupted = True
                        break
                    state["active_case"] = case
                    response = _invoke_executor(executor, protocol_dir, _with_retry({
                        "schema": "profiling-skill/bsa-candidate-executor-request/v1",
                        "action": "capture",
                        "candidate": candidate,
                        "case": case,
                        "sample_index": len(case_state["samples"]),
                        "kernel_name": selector,
                        "component": component,
                        "metric": "PipeUtilization",
                        "execution_environment": execution_environment,
                    }, state))
                    actions += 1
                    status = response.get("status")
                    if (
                        require_provenance
                        and status in CANDIDATE_FAILURES | {"ok"}
                        and _reject_inexact_response_provenance(
                            state, response, manifest["environment"],
                            "PipeUtilization",
                        )
                    ):
                        interrupted = True
                        break
                    if status in CANDIDATE_FAILURES:
                        state["status"] = "candidate_failure"
                        state["failure"] = _failure(response)
                        interrupted = True
                        break
                    if status != "ok":
                        state["status"] = "retryable_infrastructure"
                        _record_infrastructure_failure(
                            state, response, manifest["environment"]
                        )
                        interrupted = True
                        break
                    state.pop("retry", None)
                    if not _valid_evidence_digest(response.get("evidence_sha256")):
                        state["status"] = "retryable_infrastructure"
                        _record_malformed_success(
                            state, response,
                            kind="invalid_evidence_digest",
                            diagnostics=(
                                "PipeUtilization response lacks a valid SHA-256 "
                                "evidence digest"
                            ),
                        )
                        interrupted = True
                        break
                    duration = response.get("duration_us")
                    if (
                        not isinstance(duration, (int, float))
                        or not math.isfinite(duration)
                        or duration <= 0
                        or not isinstance(response.get("pipe_utilization"), (dict, list))
                    ):
                        state["status"] = "retryable_infrastructure"
                        _record_malformed_success(
                            state, response,
                            kind="invalid_pipe_evidence",
                            diagnostics=(
                                "capture lacks positive duration or compact "
                                "PipeUtilization"
                            ),
                        )
                        interrupted = True
                        break
                    partial["components"][selector] = {
                        "duration_us": float(duration),
                        "pipe_utilization": response["pipe_utilization"],
                        "evidence_sha256": response.get("evidence_sha256"),
                        "provenance": response.get("provenance"),
                    }
                    _acquisition_succeeded(state)
                    _atomic_json(output, results)
                if interrupted:
                    break
                case_state["samples"].append({
                    "duration_us": max(
                        item["duration_us"] for item in partial["components"].values()
                    ),
                    "duration_aggregation": "max_component_duration",
                    "components": partial["components"],
                })
                del case_state["in_progress_sample"]
                _atomic_json(output, results)
            if interrupted:
                break
            _complete_stable_case(
                case_state, stable_samples, variability_threshold
            )
            if case_state["status"] != "completed":
                state["status"] = "measurement_unstable"
                state["failure"] = {
                    "kind": "measurement_instability",
                    "category": "infrastructure",
                    "diagnostics": (
                        f"no {stable_samples}-sample consecutive window within "
                        f"{variability_threshold:.1%} after {max_samples} valid samples"
                    ),
                }
                break
        if not interrupted and any(
            item["status"] == "acquisition_exhausted" for item in state["cases"].values()
        ):
            state["status"] = "measurement_unstable"
            state["failure"] = {
                "kind": "acquisition_exhausted", "category": "infrastructure",
                "diagnostics": "Some cases lack profiling evidence; available independent cases retained",
                "cases": [case for case, item in state["cases"].items()
                          if item["status"] == "acquisition_exhausted"],
            }
        elif not interrupted and all(
            item["status"] == "completed" for item in state["cases"].values()
        ):
            state["status"] = "completed"
            state["geomean_us"] = math.exp(
                sum(math.log(item["median_us"]) for item in state["cases"].values())
                / len(state["cases"])
            )
        _atomic_json(output, results)
    return results


def merge_results(
    manifest: dict[str, Any], inputs: list[Path],
    *, expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_manifest(manifest)
    if not inputs:
        raise InvalidArtifact("merge requires at least one shard checkpoint")
    manifest_sha = _sha(manifest)
    executions: dict[str, Any] = {}
    shard_indexes: set[int] = set()
    shard_count = None
    contract = expected_contract
    sources = []
    for path in inputs:
        value = _read_json(path)
        if value.get("schema") != RESULT_SCHEMA:
            raise InvalidArtifact(f"result schema mismatch in {path}")
        if value.get("manifest_sha256") != manifest_sha:
            raise InvalidArtifact(f"result manifest mismatch in {path}")
        current_contract = value.get("run_contract")
        if not isinstance(current_contract, dict):
            raise InvalidArtifact(f"invalid run contract in {path}")
        if (
            "execution_environment" not in current_contract
            or "executor_identity" not in current_contract
        ):
            raise InvalidArtifact(
                "run contract predates target/runtime activation or executor "
                f"identity in {path}"
            )
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise InvalidArtifact(f"result run contract mismatch in {path}")
        shard = value.get("shard")
        if (
            not isinstance(shard, dict)
            or shard.get("algorithm") != SHARD_ALGORITHM
            or type(shard.get("count")) is not int
            or type(shard.get("index")) is not int
        ):
            raise InvalidArtifact(f"invalid shard identity in {path}")
        _validate_shard(shard["count"], shard["index"])
        if shard_count is None:
            shard_count = shard["count"]
        elif shard["count"] != shard_count:
            raise InvalidArtifact(f"shard count mismatch in {path}")
        if shard["index"] in shard_indexes:
            raise InvalidArtifact(f"overlapping shard index {shard['index']}")
        shard_indexes.add(shard["index"])
        shard_executions = value.get("executions")
        if not isinstance(shard_executions, dict):
            raise InvalidArtifact(f"invalid executions object in {path}")
        for execution_id, state in shard_executions.items():
            if execution_shard(execution_id, shard_count) != shard["index"]:
                raise InvalidArtifact(
                    f"execution {execution_id} is assigned to the wrong shard"
                )
            if execution_id in executions:
                raise InvalidArtifact(f"overlapping execution {execution_id}")
            executions[execution_id] = state
        sources.append({"path": str(path), "shard_index": shard["index"]})
    assert shard_count is not None and contract is not None
    known_ids = {item["execution_id"] for item in manifest["executions"]}
    unknown = set(executions) - known_ids
    if unknown:
        raise InvalidArtifact(f"results contain unknown executions: {sorted(unknown)}")
    return {
        "schema": MERGED_RESULT_SCHEMA,
        "manifest_sha256": manifest_sha,
        "run_contract": contract,
        "shard_count": shard_count,
        "present_shards": sorted(shard_indexes),
        "missing_shards": sorted(set(range(shard_count)) - shard_indexes),
        "missing_execution_ids": sorted(known_ids - set(executions)),
        "sources": sorted(sources, key=lambda item: item["shard_index"]),
        "executions": executions,
    }


def _config(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != "profiling-skill/bsa-profile-candidates-config/v1":
        raise InvalidArtifact("configuration schema mismatch")
    revision = value.get("campaign_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise InvalidArtifact("configuration requires a nonempty campaign_revision")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--campaign", type=Path, required=True)
    inventory.add_argument("--config", type=Path, required=True)
    inventory.add_argument("--starter-source", type=Path, required=True)
    inventory.add_argument("--starter-manifest", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--executor", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-actions", type=int)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    merge = subparsers.add_parser("merge-results")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--executor", type=Path, required=True)
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "inventory":
            config = _config(args.config)
            starter_metadata = _read_json(args.starter_manifest)
            if starter_metadata.get("schema") != "profiling-skill/candidate-kernel/v1":
                raise InvalidArtifact("starter manifest schema mismatch")
            manifest = inventory_campaign(
                args.campaign,
                environment=config["environment"],
                starter={
                    "source": args.starter_source.read_text(),
                    "parameters": starter_metadata.get("parameters", {}),
                    "declared_kernel_name": starter_metadata.get("kernel_name"),
                },
            )
            _atomic_json(args.output, manifest)
            print(json.dumps(manifest["summary"], sort_keys=True))
        elif args.command == "run":
            config = _config(args.config)
            manifest = _read_json(args.manifest)
            result = run_campaign(
                manifest, args.output, args.executor,
                cases=config["cases"],
                stable_samples=config["sampling"]["stable_samples"],
                max_samples=config["sampling"]["max_samples"],
                variability_threshold=config["sampling"]["relative_range_threshold"],
                require_provenance=config.get("require_provenance", True),
                max_actions=args.max_actions,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                campaign_revision=config["campaign_revision"],
            )
            counts: dict[str, int] = {}
            for item in result["executions"].values():
                counts[item["status"]] = counts.get(item["status"], 0) + 1
            print(json.dumps(counts, sort_keys=True))
        elif args.command == "merge-results":
            config = _config(args.config)
            manifest = _read_json(args.manifest)
            expected = _run_contract(
                config["cases"],
                config["sampling"]["stable_samples"],
                config["sampling"]["max_samples"],
                config["sampling"]["relative_range_threshold"],
                config.get("require_provenance", True),
                config["environment"],
                _executor_identity(args.executor, config["campaign_revision"]),
            )
            result = merge_results(manifest, args.input, expected_contract=expected)
            _atomic_json(args.output, result)
            print(json.dumps({
                "present_shards": result["present_shards"],
                "missing_shards": result["missing_shards"],
                "missing_executions": len(result["missing_execution_ids"]),
            }, sort_keys=True))
        else:
            result = _read_json(args.results)
            counts: dict[str, int] = {}
            for item in result["executions"].values():
                counts[item["status"]] = counts.get(item["status"], 0) + 1
            print(json.dumps(counts, sort_keys=True))
        return 0
    except (InvalidArtifact, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
