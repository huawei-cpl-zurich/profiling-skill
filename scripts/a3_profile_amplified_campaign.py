#!/usr/bin/env python3
"""Paired acquisitions without scheduling: a caller owns concurrency and devices.

BasicInfo measures kernel duration independently of PipeUtilization. Source
bundles and cell arguments are immutable across resume; previous campaigns are
never imported. The same pair interface also accepts original/timeline probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from a3_profile_microbench_campaign import canonical, invoke_acquisition

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = {"scalar-mask-decision": "blocks-32", "load-masking": "irregular-25",
          "sparse-mode-control": "dense-mode"}
INFRA = {"host_environment", "device_unavailable"}
PENDING = {"transport_error", "observation_unavailable"}


def manifest() -> list[dict]:
    """Nine derived pairs, each using its original sweep and seeded inputs."""
    result = []
    for experiment, sweep in SWEEPS.items():
        for repetitions in (1, 32, 128):
            pair_id = f"amplified/{experiment}/{repetitions}"
            cells = []
            for variant in ("before", "after"):
                cells.append({"id": f"{pair_id}/{variant}", "variant": variant,
                              "kind": "amplified", "experiment": experiment,
                              "sweep": sweep, "repetitions": repetitions,
                              "entrypoint": "benchmarks/a3_profile_amplified.py",
                              "arguments": ["--experiment", experiment, "--sweep", sweep,
                                            "--variant", variant, "--repetitions", str(repetitions)],
                              "kernel_names": ["a3_amplified_probe"]})
            result.append({"id": pair_id, "cells": cells})
    return result


def prepare_bundle(destination: Path) -> Path:
    """Stage only the four runtime dependencies, excluding caches and results."""
    for relative in ("benchmarks/a3_profile_amplified.py", "benchmarks/a3_profile_microbench.py",
                     "benchmarks/compile_evidence.py", "experiments/a3-profile-hypotheses.json"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (ROOT / relative).read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"refusing to replace frozen bundle member {relative}")
        if not target.exists():
            target.write_bytes(content)
    return destination


def bundle_identity(bundle: Path) -> str:
    rows = [(str(path.relative_to(bundle)), hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(bundle.rglob("*")) if path.is_file()]
    return hashlib.sha256(canonical(rows)).hexdigest()


def save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def stable(samples: list[float]) -> dict:
    for start in range(len(samples) - 2):
        window = samples[start:start + 3]
        median = statistics.median(window)
        spread = (max(window) - min(window)) / median
        if spread <= .05:
            return {"stable": True, "accepted_sample_indexes": list(range(start, start + 3)),
                    "median_us": median, "relative_range": spread}
    return {"stable": False, "accepted_sample_indexes": [], "median_us": None}


def basic_duration(evidence: dict, selectors: list[str]) -> float:
    """Sum sequential declared kernel medians, never per-core activity times."""
    grouped = {selector: [] for selector in selectors}
    for row in evidence["capture"].get("rows", []):
        name = next((row[key] for key in ("Op Name", "Kernel Name", "Name", "kernel_name")
                     if key in row), None)
        if name not in grouped:
            continue
        raw = row.get("Task Duration(us)", row.get("Duration(us)"))
        try:
            duration = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("BasicInfo lacks numeric duration") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("BasicInfo duration must be finite and positive")
        grouped[name].append(duration)
    if not grouped or any(not values for values in grouped.values()):
        raise ValueError("BasicInfo lacks exact-selector duration rows")
    return sum(statistics.median(values) for values in grouped.values())


def run_pair(spec: dict, output: Path, acquire, *, campaign: str) -> dict:
    """Run/resume an interleaved pair; acquire owns exact-handle observation.

    acquire(phase, cell, attempt, evidence_path, selector, basic_path) returns
    invoke_acquisition's {ok, evidence, handle} or classified failure. Transport
    uncertainty yields pending without incrementing attempts or redispatching.
    """
    output = Path(output)
    spec = {**spec, "bundle": str(spec["bundle"])}
    if len(spec["cells"]) != 2 or [cell["variant"] for cell in spec["cells"]] != ["before", "after"]:
        raise ValueError("pair must contain exactly before and after cells in that order")
    contract = {"campaign": campaign, "spec": spec, "source": bundle_identity(Path(spec["bundle"])),
                "timing": {"required": 3, "max_valid": 7, "range": .05}, "pipe_samples": 3}
    checkpoint = output / "pair-result.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_bytes())
        if state["contract"] != contract:
            raise ValueError("resume contract/source differs from frozen pair")
    else:
        state = {"contract": contract, "cells": []}
        for cell in spec["cells"]:
            state["cells"].append({"cell": cell, "status": "running", "attempts": [],
                                   "discovery": None, "timing_samples_us": [], "pipe": {},
                                   "timeline": None, "errors": {}})
        save(checkpoint, state)
    # Each iteration performs at most one acquisition per side, interleaving
    # discovery, timing and metric replays even when one side is unstable.
    while True:
        progress = False
        for row in state["cells"]:
            if row["status"] not in {"running", "pending"}:
                continue
            cell = row["cell"]
            selectors = cell["kernel_names"]
            timing = stable(row["timing_samples_us"])
            if row["discovery"] is None:
                phase, token, selector = "basic", "discovery", None
            elif not timing["stable"] and len(row["timing_samples_us"]) < 7:
                phase, token, selector = "basic", f"timing-{len(row['timing_samples_us'])}", None
            else:
                missing = [(name, sample) for sample in range(3) for name in selectors
                           if f"{name}/{sample}" not in row["pipe"]]
                if missing:
                    selector, sample = missing[0]
                    phase, token = "pipe", f"pipe-{selectors.index(selector)}-{sample}"
                else:
                    row["status"] = "success" if timing["stable"] else "inconclusive"
                    row["timing"] = timing
                    save(checkpoint, state)
                    progress = True
                    continue
            errors = row["errors"].get(token, [])
            # A transport interruption must reuse exactly the same path/key.
            attempt = len(errors)
            folder = output / cell["variant"]
            path = folder / f"{token}-attempt-{attempt}.json"
            basic = Path(row["discovery"]["path"]) if row["discovery"] else None
            result = acquire(phase, cell, attempt, path, selector, basic)
            if not result.get("ok"):
                classification = result.get("classification", "controller_failure")
                if classification in PENDING:
                    row["status"] = "pending"
                    row["pending"] = result
                    save(checkpoint, state)
                    continue
                row.pop("pending", None)
                errors.append(result)
                row["errors"][token] = errors
                row["attempts"].append({"token": token, **result})
                if classification in INFRA:
                    terminal = len(errors) >= 3
                else:
                    terminal = classification not in {"evidence_failure", "profiler_failure"} or len(errors) >= 2
                if terminal:
                    row["status"] = "infrastructure_exhausted" if classification in INFRA else "failure"
                else:
                    row["status"] = "running"
                save(checkpoint, state)
                progress = True
                continue
            row["status"] = "running"
            row.pop("pending", None)
            entry = {**result, "path": str(path)}
            try:
                if token == "discovery":
                    observed = result["evidence"]["capture"]["observed_kernel_names"]
                    if not set(selectors) <= set(observed):
                        raise ValueError("discovery missing expected exact selectors")
                    if cell.get("kind") == "amplified":
                        workload = result["evidence"].get("normal_run", {}).get("workload_result", {})
                        if not isinstance(workload, dict):
                            raise ValueError("amplified discovery lacks a structured workload result")
                        artifacts = workload.get("compiler", [])
                        if not isinstance(artifacts, list) or any(
                            not isinstance(artifact, dict)
                            or not isinstance(artifact.get("loop_evidence", []), list)
                            or any(not isinstance(line, str) for line in artifact.get("loop_evidence", []))
                            for artifact in artifacts
                        ):
                            raise ValueError("amplified discovery has malformed compiler proof")
                        loops = [line for artifact in artifacts for line in artifact.get("loop_evidence", [])
                                 if any(token in line for token in ("scf.for", "scf.while", "cf.cond_br", "llvm.cond_br", "br i1"))]
                        if workload.get("correctness") is not True or not loops:
                            raise ValueError("amplified discovery lacks correctness/runtime-loop compiler proof")
                        row["compiler_loop_proof"] = artifacts
                    row["discovery"] = entry
                elif phase == "basic":
                    row["timing_samples_us"].append(basic_duration(result["evidence"], selectors))
                elif phase == "pipe":
                    if not result["evidence"]["capture"].get("rows"):
                        raise ValueError("PipeUtilization has no exact-selector rows")
                    row["pipe"][f"{selector}/{sample}"] = entry
            except ValueError as exc:
                errors.append({"classification": "evidence_failure", "message": str(exc),
                               "handle": result.get("handle"), "path": str(path)})
                row["errors"][token] = errors
                row["status"] = "failure" if len(errors) >= 2 else "running"
                row["diagnostic"] = str(exc)
            row["attempts"].append({"token": token, **entry})
            save(checkpoint, state)
            progress = True
        if not progress or all(row["status"] not in {"running", "pending"} for row in state["cells"]):
            state["terminal"] = all(row["status"] not in {"running", "pending"} for row in state["cells"])
            save(checkpoint, state)
            return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifest", "pair", "summary"))
    parser.add_argument("--pair-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--campaign")
    parser.add_argument("--cpl-remote", type=Path)
    args = parser.parse_args(argv)
    if args.action == "manifest":
        print(json.dumps(manifest(), indent=2))
        return 0
    if args.output is None:
        parser.error("--output required")
    if args.action == "summary":
        results = [json.loads(path.read_bytes()) for path in sorted(args.output.rglob("pair-result.json"))]
        print(json.dumps({"pairs": results, "cell_count": sum(len(row["cells"]) for row in results)}))
        return 0
    if not args.campaign:
        parser.error("--campaign required")
    pair = next((pair for pair in manifest() if pair["id"] == args.pair_id), None)
    if pair is None:
        parser.error("unknown --pair-id")
    bundle = prepare_bundle(args.output / "bundle")
    pair.update(bundle=str(bundle.resolve()), target="bz-a3-1")
    def acquire(phase, cell, attempt, path, selector, basic):
        key = hashlib.sha256(canonical([args.campaign, cell, str(path), phase])).hexdigest()
        return invoke_acquisition(phase=phase, target="bz-a3-1", dispatch_key=key,
                                  bundle=bundle, entrypoint=cell["entrypoint"],
                                  workload_args=cell["arguments"], evidence_path=path,
                                  selector=selector, basic_evidence=basic,
                                  cpl_remote=args.cpl_remote,
                                  launch_count=len(cell["kernel_names"]) if phase == "basic" else 1)
    result = run_pair(pair, args.output, acquire, campaign=args.campaign)
    print(json.dumps(result))
    return 0 if result["terminal"] else 75


if __name__ == "__main__":
    raise SystemExit(main())
