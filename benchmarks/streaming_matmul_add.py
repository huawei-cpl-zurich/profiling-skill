#!/usr/bin/env python3
"""Deterministic K-tiled Triton matmul-add control for A2/A3 experiments."""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


# Triton resolves JIT annotations from the function's module globals.  Keep the
# optional import at module scope, while still allowing contract discovery on
# controller hosts where Triton is not installed.
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


KERNEL_NAME = "streaming_matmul_add_kernel"
SEED = 20260925
RTOL = 2e-2
ATOL = 2e-2


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    n: int
    k: int
    kind: str


# Correctness cases deliberately cover tails in every dimension, skinny and
# rectangular matrices, and more than one K tile. Performance cases are kept
# separate so the experiment controller cannot silently change its workload.
CASES = (
    Case("correctness-00-tiny", 1, 1, 1, "correctness"),
    Case("correctness-01-square-tail", 33, 33, 31, "correctness"),
    Case("correctness-02-m-tail", 65, 32, 64, "correctness"),
    Case("correctness-03-n-tail", 32, 71, 64, "correctness"),
    Case("correctness-04-k-tail", 64, 64, 97, "correctness"),
    Case("correctness-05-wide", 37, 113, 79, "correctness"),
    Case("correctness-06-tall", 129, 29, 131, "correctness"),
    Case("performance-small", 256, 256, 256, "performance"),
    Case("performance-medium", 1024, 1024, 1024, "performance"),
    Case("performance-large", 2048, 2048, 2048, "performance"),
)
CASE_BY_NAME = {case.name: case for case in CASES}


if triton is not None:
    @triton.jit
    def streaming_matmul_add_kernel(
        lhs,
        rhs,
        bias,
        output,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        reduction = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Streaming over K bounds temporary storage to two tiles while retaining
        # fp32 accumulation. Static range makes the reduction pipeline explicit.
        for start in range(0, k, BLOCK_K):
            lhs_offsets = rows[:, None] * k + start + reduction[None, :]
            rhs_offsets = (start + reduction[:, None]) * n + cols[None, :]
            lhs_tile = tl.load(
                lhs + lhs_offsets,
                mask=(rows[:, None] < m) & (start + reduction[None, :] < k),
                other=0.0,
            )
            rhs_tile = tl.load(
                rhs + rhs_offsets,
                mask=(start + reduction[:, None] < k) & (cols[None, :] < n),
                other=0.0,
            )
            accumulator += tl.dot(lhs_tile, rhs_tile)
        result = accumulator + tl.load(bias + cols, mask=cols < n, other=0.0)[None, :]
        offsets = rows[:, None] * n + cols[None, :]
        tl.store(output + offsets, result, mask=(rows[:, None] < m) & (cols[None, :] < n))
else:
    streaming_matmul_add_kernel = None


def contract() -> dict:
    return {
        "schema_version": 1,
        "benchmark": "streaming-matmul-add",
        "operation": "output = lhs @ rhs + bias",
        "kernel_name": KERNEL_NAME,
        "seed": SEED,
        "dtype": "float16",
        "accumulator_dtype": "float32",
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "cases": [asdict(case) for case in CASES],
    }


def classify_exception(exc: BaseException) -> str:
    """Classify candidate failures without depending on Triton internals."""
    identity = f"{type(exc).__module__}.{type(exc).__name__}".lower()
    text = str(exc).lower()
    compile_words = ("compile", "compiler", "codegen", "lowering", "semantic")
    if any(word in identity or word in text for word in compile_words):
        return "compilation"
    return "runtime"


def emit(payload: dict, output: Path | None) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def run(case: Case, launches: int) -> dict:
    # Torch imports remain delayed so contract discovery works on controllers.
    import torch
    import torch_npu  # noqa: F401

    if streaming_matmul_add_kernel is None:
        raise RuntimeError("triton is not installed")

    torch.manual_seed(SEED)
    lhs_cpu = torch.randn((case.m, case.k), dtype=torch.float16)
    rhs_cpu = torch.randn((case.k, case.n), dtype=torch.float16)
    bias_cpu = torch.randn((case.n,), dtype=torch.float16)
    expected = lhs_cpu.float().matmul(rhs_cpu.float()) + bias_cpu.float()
    lhs = lhs_cpu.npu()
    rhs = rhs_cpu.npu()
    bias = bias_cpu.npu()
    output = torch.empty((case.m, case.n), device="npu", dtype=torch.float32)
    block_m, block_n, block_k = 32, 32, 32
    grid = (triton.cdiv(case.m, block_m), triton.cdiv(case.n, block_n))

    try:
        # First launch owns compilation and is excluded from latency samples.
        streaming_matmul_add_kernel[grid](
            lhs, rhs, bias, output, case.m, case.n, case.k,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        torch.npu.synchronize()
    except BaseException as exc:
        return {
            "status": "failure",
            "failure": {"kind": classify_exception(exc), "message": str(exc)},
        }

    observed = output.cpu()
    difference = (observed - expected).abs()
    max_abs = float(difference.max())
    denominator = expected.abs().clamp_min(ATOL)
    max_rel = float((difference / denominator).max())
    if not torch.allclose(observed, expected, rtol=RTOL, atol=ATOL):
        return {
            "status": "failure",
            "failure": {
                "kind": "correctness",
                "message": "output differs from fp32 CPU reference",
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
            },
        }

    samples = []
    try:
        for _ in range(launches):
            torch.npu.synchronize()
            started = time.perf_counter_ns()
            streaming_matmul_add_kernel[grid](
                lhs, rhs, bias, output, case.m, case.n, case.k,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            )
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - started) / 1_000)
    except BaseException as exc:
        return {
            "status": "failure",
            "failure": {"kind": "runtime", "message": str(exc)},
        }
    return {
        "status": "success",
        "kernel_name": KERNEL_NAME,
        "case": asdict(case),
        "launches": launches,
        "host_observed_us": samples,
        "correctness": {"max_abs_error": max_abs, "max_rel_error": max_rel},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--case", choices=sorted(CASE_BY_NAME))
    parser.add_argument("--launches", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    if not args.contract and not args.case:
        parser.error("one of --contract or --case is required")
    if args.contract and args.case:
        parser.error("--contract and --case are mutually exclusive")
    if args.launches < 1:
        parser.error("--launches must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.contract:
        emit(contract(), args.json_output)
        return 0
    case = CASE_BY_NAME[args.case]
    base = {"schema_version": 1, "benchmark": "streaming-matmul-add"}
    try:
        result = run(case, args.launches)
    except BaseException as exc:
        result = {
            "status": "failure",
            "failure": {"kind": classify_exception(exc), "message": str(exc)},
        }
    payload = base | result
    emit(payload, args.json_output)
    return 0 if payload["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
