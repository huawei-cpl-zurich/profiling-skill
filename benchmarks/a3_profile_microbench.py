#!/usr/bin/env python3
"""Controlled before/after Triton probes for A3 profiler hypotheses.

Contract discovery and reference checks work without Triton. Device execution
imports torch-npu lazily and reports unsupported compiler constructs explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "a3-profile-hypotheses.json"
SEED = 20261013
ATOL = 2e-2
RTOL = 2e-2


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def experiments() -> list[dict]:
    manifest = _manifest()
    return manifest["hypotheses"] + manifest["timeline_probes"]


def _experiment(experiment_id: str) -> dict:
    try:
        return next(item for item in experiments() if item["id"] == experiment_id)
    except StopIteration as exc:
        raise ValueError(f"unknown experiment {experiment_id!r}") from exc


def _sweep(experiment: dict, sweep_id: str) -> dict:
    try:
        return next(item for item in experiment["sweeps"] if item["id"] == sweep_id)
    except StopIteration as exc:
        raise ValueError(
            f"unknown sweep {sweep_id!r} for experiment {experiment['id']!r}"
        ) from exc


def contract() -> dict:
    return {
        "schema": "profiling-skill/a3-profile-contract/v1",
        "seed": SEED,
        "dtype": "float32",
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "variants": ["before", "after"],
        "experiments": experiments(),
        "profile_contract": {
            "mode": "msprof-op",
            "required_exports": ["BasicInfo", "PipeUtilization"],
            "timeline_export": "TimelineDetail when present; absence is evidence",
        },
    }


def generate_inputs(experiment_id: str, sweep_id: str) -> dict[str, torch.Tensor]:
    """Generate repeatable CPU inputs shared by both variants of a pair."""
    experiment = _experiment(experiment_id)
    sweep = _sweep(experiment, sweep_id)
    generator = torch.Generator().manual_seed(SEED)
    n, work = sweep["n"], sweep["work"]
    if experiment_id in {"row-reduction-vs-dot", "query-reuse"}:
        size = n
        return {
            "lhs": torch.randn((size, size), generator=generator, dtype=torch.float32),
            "rhs": torch.randn((size, size), generator=generator, dtype=torch.float32),
        }
    x = torch.randn((n,), generator=generator, dtype=torch.float32)
    selected = torch.arange(n) < min(work, n)
    if experiment_id == "scalar-mask-decision":
        selected = (torch.arange(n) // 128) < max(1, work // 2)
    # Irregular load masking deliberately differs from the block-prefix probes.
    if experiment_id == "load-masking" and sweep_id.startswith("irregular"):
        selected = (torch.arange(n) * 17 % 4) == 0
    head_flags = torch.zeros((max(work, 1),), dtype=torch.int32)
    head_flags[::4] = 1
    return {
        "x": x, "selected": selected, "work": torch.tensor(work),
        "head_flags": head_flags, "rank": head_flags.sum().reshape(1),
    }


def reference_output(
    experiment_id: str, variant: str, inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    if variant not in {"before", "after"}:
        raise ValueError(f"unknown variant {variant!r}")
    if experiment_id in {"row-reduction-vs-dot", "query-reuse"}:
        return inputs["lhs"] @ inputs["rhs"]
    x = inputs["x"]
    if experiment_id == "head-rank":
        return torch.roll(x, shifts=-int(inputs["rank"][0])) * 1.5 + 2.0
    if experiment_id == "key-tile-size":
        return x * 1.5 + 2.0 + x.sum() * 1e-6
    if experiment_id == "loop-bound":
        return x * (1.0 + int(inputs["work"]) * 1e-6) * 1.5 + 2.0
    if experiment_id == "scalar-mask-decision":
        adjusted = torch.where(inputs["selected"], x + 0.125, x - 0.125)
        return adjusted * 1.5 + 2.0
    if experiment_id in {
        "empty-tile-skip", "load-masking", "sparse-mode-control"
    }:
        return torch.where(inputs["selected"], x * 1.5 + 2.0, torch.zeros_like(x))
    # Timeline controls intentionally have identical final values even when
    # internal phase order, repetition, dependency, or launch topology differs.
    return x * 1.5 + 2.0


def outputs_close(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left.float(), right.float(), rtol=RTOL, atol=ATOL))


if triton is not None:
    @triton.jit
    def a3_vector_probe(
        x, selected, work_ptr, output, n: tl.constexpr, work: tl.constexpr,
        block: tl.constexpr, mode: tl.constexpr, loops: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < n
        chosen = tl.load(selected + offsets, mask=valid, other=0).to(tl.int1)
        if mode == 4:
            values = tl.load(x + offsets, mask=valid & chosen, other=0.0)
        else:
            values = tl.load(x + offsets, mask=valid, other=0.0)
        if mode == 1:  # explicit fixed-bound control work
            runtime_work = tl.load(work_ptr)
            state = values * 0.0
            for index in range(0, loops):
                state += tl.where(index < runtime_work, values * 0.000001, values * 0.0)
            values += state
        if mode == 2:  # replicated lane decision
            decision = tl.load(selected + offsets, mask=valid, other=0).to(tl.int1)
            values = tl.where(decision, values + 0.125, values - 0.125)
        if mode == 3:  # scalar decision broadcast over a block
            decision = tl.load(selected + tl.program_id(0) * block, mask=True, other=0)
            values = tl.where(decision, values + 0.125, values - 0.125)
        result = values * 1.5 + 2.0
        if mode == 4:  # only this variant executes the final-predicate load
            result = values * 1.5 + 2.0
        if mode == 5:  # runtime-generic mode predicate
            runtime_work = tl.load(work_ptr)
            chosen = offsets < runtime_work
        if mode == 6:  # compile-time specialized equivalent
            chosen = offsets < work
        if mode >= 4:
            result = tl.where(chosen, result, 0.0)
        tl.store(output + offsets, result, mask=valid)

    @triton.jit
    def a3_head_rank_probe(
        x, flags, rank_ptr, output, n: tl.constexpr, heads: tl.constexpr,
        block: tl.constexpr, direct: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        if direct:
            rank = tl.load(rank_ptr)
        else:
            rank = 0
            for index in range(0, heads):
                rank += tl.load(flags + index)
        source = (offsets + rank) % n
        values = tl.load(x + source, mask=offsets < n, other=0.0)
        tl.store(output + offsets, values * 1.5 + 2.0, mask=offsets < n)

    @triton.jit
    def a3_key_tile_probe(
        x, output, n: tl.constexpr, block: tl.constexpr, key_tile: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        total = 0.0
        for start in range(0, n, key_tile):
            keys = start + tl.arange(0, key_tile)
            total += tl.sum(tl.load(x + keys, mask=keys < n, other=0.0))
        values = tl.load(x + offsets, mask=offsets < n, other=0.0)
        tl.store(output + offsets, values * 1.5 + 2.0 + total * 1e-6,
                 mask=offsets < n)

    @triton.jit
    def a3_row_reduce_probe(lhs, rhs, output, size: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, size)
        left = tl.load(lhs + row * size + cols)
        for column in range(0, size):
            right = tl.load(rhs + cols * size + column)
            tl.store(output + row * size + column, tl.sum(left * right))

    @triton.jit
    def a3_empty_skip_probe(x, selected, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < n
        # The first lane is a deliberately block-aligned occupancy flag.
        has_work = tl.load(selected + tl.program_id(0) * block, mask=True, other=0)
        if has_work:
            chosen = tl.load(selected + offsets, mask=valid, other=0).to(tl.int1)
            values = tl.load(x + offsets, mask=valid, other=0.0)
            result = tl.where(chosen, values * 1.5 + 2.0, 0.0)
            tl.store(output + offsets, result, mask=valid)
        else:
            tl.store(output + offsets, 0.0, mask=valid)

    @triton.jit
    def a3_dot_probe(lhs, rhs, output, size: tl.constexpr, rows_per_program: tl.constexpr):
        base = tl.program_id(0) * rows_per_program
        rows = base + tl.arange(0, rows_per_program)[:, None]
        cols = tl.arange(0, size)[None, :]
        reduction = tl.arange(0, size)
        a = tl.load(lhs + rows * size + reduction[None, :])
        b = tl.load(rhs + reduction[:, None] * size + cols)
        tl.store(output + rows * size + cols, tl.dot(a, b))

    @triton.jit
    def a3_phase_probe(
        x, scratch, output, n: tl.constexpr, block: tl.constexpr,
        compute_loops: tl.constexpr, phase_mode: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < n
        if phase_mode == 3:  # expose an independent load before compute
            independent = tl.load(scratch + offsets, mask=valid, other=0.0)
        if phase_mode == 1:  # independent compute first, then the visible load
            state = offsets.to(tl.float32)
            for _ in range(0, compute_loops):
                state = state * 1.000001 + 0.000001
            first = tl.load(x + offsets, mask=valid, other=0.0)
        else:
            first = tl.load(x + offsets, mask=valid, other=0.0)
            state = first
            for _ in range(0, compute_loops):
                state = state * 1.000001 + 0.000001
        if phase_mode == 2:  # repeated load-compute-load
            again = tl.load(scratch + offsets, mask=valid, other=0.0)
            state += again * 0.0000001
        if phase_mode == 4:  # source-serialized independent second stream
            independent = tl.load(scratch + offsets, mask=valid, other=0.0)
        if phase_mode >= 3:
            state += independent * 0.0000001
        # Keep phase work live while holding numerical drift well inside the
        # pair's declared tolerance. A compiler may still transform phases;
        # that is an empirical result, not something this probe assumes away.
        tl.store(output + offsets, first * 1.5 + 2.0 + state * 0.0000001, mask=valid)

    @triton.jit
    def a3_phase_copy(x, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < n
        tl.store(output + offsets, tl.load(x + offsets, mask=valid), mask=valid)

    @triton.jit
    def a3_phase_compute(x, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < n
        value = tl.load(x + offsets, mask=valid)
        tl.store(output + offsets, value * 1.5 + 2.0, mask=valid)

    @triton.jit
    def a3_phase_store(x, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < n
        tl.store(output + offsets, tl.load(x + offsets, mask=valid), mask=valid)
else:
    a3_vector_probe = a3_head_rank_probe = a3_key_tile_probe = None
    a3_row_reduce_probe = a3_empty_skip_probe = a3_dot_probe = None
    a3_phase_probe = a3_phase_copy = a3_phase_compute = a3_phase_store = None


def _classify_exception(exc: Exception) -> str:
    identity = f"{type(exc).__module__}.{type(exc).__name__}".lower()
    message = str(exc).lower()
    if any(word in identity + message for word in ("compile", "semantic", "lowering")):
        return "unsupported"
    return "failure"


def _launch(experiment_id: str, variant: str, inputs: dict, output: torch.Tensor) -> list[str]:
    n = int(inputs.get("x", inputs.get("lhs")).shape[0])
    if experiment_id in {"row-reduction-vs-dot", "query-reuse"}:
        lhs, rhs = inputs["lhs"], inputs["rhs"]
        if experiment_id == "row-reduction-vs-dot" and variant == "before":
            a3_row_reduce_probe[(n,)](lhs, rhs, output, size=n)
            return ["a3_row_reduce_probe"]
        rows = 1 if variant == "before" else (n if experiment_id == "row-reduction-vs-dot" else 4)
        a3_dot_probe[(triton.cdiv(n, rows),)](
            lhs, rhs, output, size=n, rows_per_program=rows
        )
        return ["a3_dot_probe"]

    x, selected = inputs["x"], inputs["selected"]
    work_ptr = inputs["work"]
    work, block, mode, loops = int(inputs["work"]), 128, 0, 1
    if experiment_id == "loop-bound":
        mode, loops = 1, (256 if variant == "before" else work)
    elif experiment_id == "head-rank":
        a3_head_rank_probe[(triton.cdiv(n, block),)](
            x, inputs["head_flags"], inputs["rank"], output,
            n=n, heads=work, block=block, direct=(variant == "after"),
        )
        return ["a3_head_rank_probe"]
    elif experiment_id == "key-tile-size":
        key_tile = 64 if variant == "before" else 128
        a3_key_tile_probe[(triton.cdiv(n, block),)](
            x, output, n=n, block=block, key_tile=key_tile,
        )
        return ["a3_key_tile_probe"]
    elif experiment_id == "empty-tile-skip":
        if variant == "after":
            a3_empty_skip_probe[(triton.cdiv(n, block),)](
                x, selected, output, n=n, block=block
            )
            return ["a3_empty_skip_probe"]
        mode = 6
    elif experiment_id == "scalar-mask-decision":
        mode = 2 if variant == "before" else 3
    elif experiment_id == "load-masking":
        mode = 7 if variant == "before" else 4
    elif experiment_id == "sparse-mode-control":
        mode = 5 if variant == "before" else 6

    timeline = experiment_id in {item["id"] for item in _manifest()["timeline_probes"]}
    if timeline:
        phase_mode, compute_loops = 0, work
        if experiment_id == "reversed-phases" and variant == "after":
            phase_mode = 1  # compiler may reorder it: that is a valid finding
        elif experiment_id == "repeated-phases" and variant == "after":
            phase_mode = 2
        elif experiment_id == "duration-scaled-phases" and variant == "after":
            compute_loops = work * 4
        elif experiment_id == "serialized-vs-overlapped":
            phase_mode = 4 if variant == "before" else 3
        if experiment_id == "fused-vs-split" and variant == "after":
            loaded = torch.empty_like(x)
            computed = torch.empty_like(x)
            grid = (triton.cdiv(n, block),)
            a3_phase_copy[grid](x, loaded, n=n, block=block)
            a3_phase_compute[grid](loaded, computed, n=n, block=block)
            a3_phase_store[grid](computed, output, n=n, block=block)
            return ["a3_phase_copy", "a3_phase_compute", "a3_phase_store"]
        # Reuse the deterministic input as the second stream. Allocating or
        # zeroing scratch here would add a setup operator ahead of the probe,
        # breaking the BasicInfo launch contract.
        scratch = x
        a3_phase_probe[(triton.cdiv(n, block),)](
            x, scratch, output, n=n, block=block,
            compute_loops=compute_loops, phase_mode=phase_mode,
        )
        return ["a3_phase_probe"]

    kwargs = {}
    if experiment_id == "schedule-knobs":
        kwargs = {"num_warps": 4 if variant == "before" else 8,
                  "num_stages": 1 if variant == "before" else 2}
    a3_vector_probe[(triton.cdiv(n, block),)](
        x, selected, work_ptr, output, n=n, work=work, block=block,
        mode=mode, loops=loops, **kwargs,
    )
    return ["a3_vector_probe"]


def run_variant(experiment_id: str, variant: str, sweep_id: str, launches: int) -> dict:
    if triton is None:
        return {"status": "unsupported", "reason": "triton is not installed"}
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError:
        return {"status": "unsupported", "reason": "torch_npu is not installed"}
    cpu_inputs = generate_inputs(experiment_id, sweep_id)
    device_inputs = {key: value.npu() for key, value in cpu_inputs.items()}
    expected = reference_output(experiment_id, variant, cpu_inputs)
    output = torch.empty_like(expected, device="npu")
    try:
        kernel_names = _launch(experiment_id, variant, device_inputs, output)
        torch.npu.synchronize()
        samples = []
        for _ in range(launches):
            started = time.perf_counter_ns()
            _launch(experiment_id, variant, device_inputs, output)
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - started) / 1000)
    except Exception as exc:
        return {"status": _classify_exception(exc), "reason": str(exc)}
    observed = output.cpu()
    if not outputs_close(observed, expected):
        return {
            "status": "failure", "reason": "output differs from CPU reference",
            "max_abs_error": float((observed - expected).abs().max()),
        }
    return {
        "status": "success", "variant": variant, "kernel_names": kernel_names,
        "host_observed_us": samples, "output": observed.flatten().tolist(),
    }


def run_pair(experiment_id: str, sweep_id: str, launches: int) -> dict:
    experiment = _experiment(experiment_id)
    _sweep(experiment, sweep_id)
    results = {
        variant: run_variant(experiment_id, variant, sweep_id, launches)
        for variant in ("before", "after")
    }
    if any(result["status"] != "success" for result in results.values()):
        status, equivalent = "inconclusive", None
    else:
        left = torch.tensor(results["before"].pop("output"))
        right = torch.tensor(results["after"].pop("output"))
        equivalent = outputs_close(left, right)
        status = "success" if equivalent else "failure"
    kernel_names = []
    for result in results.values():
        kernel_names.extend(result.get("kernel_names", []))
    return {
        "schema": "profiling-skill/a3-profile-result/v1",
        "status": status, "experiment": experiment_id, "sweep": sweep_id,
        "launches": launches, "equivalent": equivalent, "results": results,
        "expected_evidence": experiment["expected_evidence"],
        "falsifier": experiment["falsifier"],
        "profile": {
            "mode": "msprof-op", "kernel_names": kernel_names,
            "required_exports": ["BasicInfo", "PipeUtilization"],
            "timeline_export": "TimelineDetail when present",
        },
    }


def _emit(payload: dict, path: Path | None) -> None:
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--experiment", choices=[item["id"] for item in experiments()])
    parser.add_argument("--sweep")
    parser.add_argument("--variant", choices=("before", "after"))
    parser.add_argument("--pair", action="store_true")
    parser.add_argument("--launches", type=int, default=3)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    if args.contract == (args.experiment is not None):
        parser.error("choose exactly one of --contract or --experiment")
    if args.contract and (args.sweep or args.variant or args.pair):
        parser.error("--contract cannot be combined with execution options")
    if args.experiment:
        experiment = _experiment(args.experiment)
        if not args.sweep:
            parser.error("--sweep is required")
        try:
            _sweep(experiment, args.sweep)
        except ValueError as exc:
            parser.error(str(exc))
        if args.pair == (args.variant is not None):
            parser.error("choose exactly one of --pair or --variant")
    if args.launches < 1:
        parser.error("--launches must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.contract:
        payload = contract()
    elif args.pair:
        payload = run_pair(args.experiment, args.sweep, args.launches)
    else:
        payload = run_variant(args.experiment, args.variant, args.sweep, args.launches)
        # Pair execution consumes raw values to prove equivalence. A single
        # profiled variant keeps logs compact because msprof already captures
        # the operator evidence and large tensor dumps obscure diagnostics.
        payload.pop("output", None)
    _emit(payload, args.json_output)
    return 0 if payload.get("status", "success") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
