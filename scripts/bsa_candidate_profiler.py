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
import itertools
import json
import math
import os
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
RESULT_SCHEMA = "profiling-skill/bsa-candidate-results/v1"


class InvalidArtifact(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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


def inventory_campaign(root: Path, *, environment: dict[str, Any]) -> dict[str, Any]:
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
                            "declared_kernel_name": request.get(
                                "expected_declared_kernel_name",
                                "block_sparse_attention_kernel",
                            ),
                            "occurrence_ids": [],
                        }
                    executions[execution_id]["occurrence_ids"].append(occurrence_id)
                occurrences.append(occurrence)
        treatment_summary[treatment] = {
            "occurrences": count,
            "recoverable_occurrences": recoverable,
            "unrecoverable_occurrences": count - recoverable,
            "unique_executions": len(treatment_executions),
        }

    recoverable_count = sum(
        item["recoverability"] == "recoverable" for item in occurrences
    )
    return {
        "schema": SCHEMA,
        "campaign_root": str(root),
        "environment": environment,
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


def _initial_results(manifest: dict[str, Any], cases: list[int]) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "manifest_sha256": _sha(manifest),
        "cases": cases,
        "executions": {
            item["execution_id"]: {
                "status": "pending",
                "kernel_name": None,
                "basic_info": None,
                "cases": {str(case): {"status": "pending", "samples": []} for case in cases},
                "infrastructure_failures": [],
            }
            for item in manifest["executions"]
        },
    }


def _load_results(path: Path, manifest: dict[str, Any], cases: list[int]) -> dict[str, Any]:
    if not path.exists():
        return _initial_results(manifest, cases)
    value = _read_json(path)
    if value.get("schema") != RESULT_SCHEMA:
        raise InvalidArtifact("result schema mismatch")
    if value.get("manifest_sha256") != _sha(manifest):
        raise InvalidArtifact("result manifest identity mismatch")
    if value.get("cases") != cases:
        raise InvalidArtifact("result cases do not match requested cases")
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


def _stable_subset(samples: list[dict[str, Any]], wanted: int, threshold: float):
    best = None
    for indexes in itertools.combinations(range(len(samples)), wanted):
        values = [float(samples[index]["duration_us"]) for index in indexes]
        median = statistics.median(values)
        relative_range = (max(values) - min(values)) / median
        candidate = (relative_range, indexes, median)
        if best is None or candidate < best:
            best = candidate
    if best is not None and best[0] <= threshold:
        return best
    return None


def _failure(response: dict[str, Any]) -> dict[str, Any]:
    kind = response.get("status", "infrastructure_error")
    return {
        "kind": kind,
        "category": "candidate" if kind in CANDIDATE_FAILURES else "infrastructure",
        "diagnostics": str(response.get("diagnostics", "")),
        "evidence_sha256": response.get("evidence_sha256"),
        "provenance": response.get("provenance"),
    }


def _valid_provenance(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("target"), str)
        and isinstance(value.get("runtime"), str)
        and isinstance(value.get("physical_device"), int)
        and isinstance(value.get("command"), (str, list))
        and isinstance(value.get("handle"), str)
        and value["handle"].startswith("remote:")
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
) -> dict[str, Any]:
    if manifest.get("schema") != SCHEMA:
        raise InvalidArtifact("manifest schema mismatch")
    if stable_samples < 1 or max_samples < stable_samples:
        raise InvalidArtifact("sample limits are invalid")
    results = _load_results(output, manifest, cases)
    protocol_dir = output.parent / (output.stem + ".protocol")
    actions = 0

    for candidate in manifest["executions"]:
        state = results["executions"][candidate["execution_id"]]
        if state["status"] in {"completed", "candidate_failure", "measurement_unstable"}:
            continue
        state["status"] = "running"
        if state["kernel_name"] is None:
            if max_actions is not None and actions >= max_actions:
                _atomic_json(output, results)
                return results
            response = _invoke_executor(executor, protocol_dir, {
                "schema": "profiling-skill/bsa-candidate-executor-request/v1",
                "action": "discover",
                "candidate": candidate,
                "correctness_cases": cases,
                "metric": "BasicInfo",
            })
            actions += 1
            status = response.get("status")
            if status in CANDIDATE_FAILURES:
                state["status"] = "candidate_failure"
                state["failure"] = _failure(response)
                _atomic_json(output, results)
                continue
            if status != "ok":
                state["status"] = "retryable_infrastructure"
                state["infrastructure_failures"].append(_failure(response))
                _atomic_json(output, results)
                continue
            if require_provenance and not _valid_provenance(response.get("provenance")):
                state["status"] = "retryable_infrastructure"
                state["infrastructure_failures"].append({
                    "kind": "invalid_provenance",
                    "category": "infrastructure",
                    "diagnostics": "BasicInfo response lacks target/runtime/device/command/handle",
                })
                _atomic_json(output, results)
                continue
            names = response.get("kernel_names")
            if not isinstance(names, list) or not names or not all(
                isinstance(name, str) and name for name in names
            ):
                state["status"] = "retryable_infrastructure"
                state["infrastructure_failures"].append({
                    "kind": "invalid_basic_info",
                    "diagnostics": "BasicInfo returned no valid kernel names",
                })
                _atomic_json(output, results)
                continue
            declared = candidate["declared_kernel_name"]
            state["kernel_name"] = declared if declared in names else (names[0] if len(names) == 1 else None)
            if state["kernel_name"] is None:
                state["status"] = "candidate_failure"
                state["failure"] = {
                    "kind": "kernel_selector_ambiguous",
                    "category": "candidate",
                    "diagnostics": f"declared {declared!r}; BasicInfo observed {names!r}",
                }
                _atomic_json(output, results)
                continue
            state["basic_info"] = {
                "kernel_names": names,
                "compact": response.get("basic_info"),
                "evidence_sha256": response.get("evidence_sha256"),
                "provenance": response.get("provenance"),
            }
            _atomic_json(output, results)

        interrupted = False
        for case in cases:
            case_state = state["cases"][str(case)]
            if case_state["status"] == "completed":
                continue
            while len(case_state["samples"]) < max_samples:
                stable = _stable_subset(
                    case_state["samples"], stable_samples, variability_threshold
                ) if len(case_state["samples"]) >= stable_samples else None
                if stable is not None:
                    spread, indexes, median = stable
                    case_state.update({
                        "status": "completed",
                        "accepted_sample_indexes": list(indexes),
                        "median_us": median,
                        "relative_range": spread,
                    })
                    break
                if max_actions is not None and actions >= max_actions:
                    interrupted = True
                    break
                response = _invoke_executor(executor, protocol_dir, {
                    "schema": "profiling-skill/bsa-candidate-executor-request/v1",
                    "action": "capture",
                    "candidate": candidate,
                    "case": case,
                    "sample_index": len(case_state["samples"]),
                    "kernel_name": state["kernel_name"],
                    "metric": "PipeUtilization",
                })
                actions += 1
                status = response.get("status")
                if status in CANDIDATE_FAILURES:
                    state["status"] = "candidate_failure"
                    state["failure"] = _failure(response)
                    interrupted = True
                    break
                if status != "ok":
                    state["status"] = "retryable_infrastructure"
                    state["infrastructure_failures"].append(_failure(response))
                    interrupted = True
                    break
                if require_provenance and not _valid_provenance(response.get("provenance")):
                    state["status"] = "retryable_infrastructure"
                    state["infrastructure_failures"].append({
                        "kind": "invalid_provenance",
                        "category": "infrastructure",
                        "diagnostics": "PipeUtilization response lacks target/runtime/device/command/handle",
                    })
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
                    state["infrastructure_failures"].append({
                        "kind": "invalid_pipe_evidence",
                        "diagnostics": "capture lacks positive duration or compact PipeUtilization",
                    })
                    interrupted = True
                    break
                case_state["samples"].append({
                    "duration_us": float(duration),
                    "pipe_utilization": response["pipe_utilization"],
                    "evidence_sha256": response.get("evidence_sha256"),
                    "provenance": response.get("provenance"),
                })
                _atomic_json(output, results)
            if interrupted:
                break
            if case_state["status"] != "completed":
                state["status"] = "measurement_unstable"
                state["failure"] = {
                    "kind": "measurement_instability",
                    "category": "infrastructure",
                    "diagnostics": (
                        f"no {stable_samples}-sample subset within "
                        f"{variability_threshold:.1%} after {max_samples} valid samples"
                    ),
                }
                break
        if not interrupted and all(
            item["status"] == "completed" for item in state["cases"].values()
        ):
            state["status"] = "completed"
            state["geomean_us"] = math.exp(
                sum(math.log(item["median_us"]) for item in state["cases"].values())
                / len(state["cases"])
            )
        _atomic_json(output, results)
    return results


def _config(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != "profiling-skill/bsa-profile-candidates-config/v1":
        raise InvalidArtifact("configuration schema mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--campaign", type=Path, required=True)
    inventory.add_argument("--config", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--executor", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-actions", type=int)
    status = subparsers.add_parser("status")
    status.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "inventory":
            config = _config(args.config)
            manifest = inventory_campaign(args.campaign, environment=config["environment"])
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
            )
            counts: dict[str, int] = {}
            for item in result["executions"].values():
                counts[item["status"]] = counts.get(item["status"], 0) + 1
            print(json.dumps(counts, sort_keys=True))
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
