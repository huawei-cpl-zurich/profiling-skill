#!/usr/bin/env python3
"""Synthetic correctness and timing smoke for two specific ACLNN operators."""
import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import torch


def summary(samples):
    if not samples:
        raise ValueError("at least one sample required")
    return {"samples_us": samples, "mean_us": statistics.mean(samples),
            "sample_sd_us": statistics.stdev(samples) if len(samples) > 1 else None}


def execute(call, synchronize, event_factory, warmup, repeats, replay=False):
    """Keep wall and event measurements separate; replay emits no timing events."""
    for _ in range(warmup):
        call()
    synchronize()
    if replay:
        result = call()
        synchronize()
        return result, {}
    wall, device = [], []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter_ns()
        result = call()
        synchronize()
        wall.append((time.perf_counter_ns() - start) / 1000)
    for _ in range(repeats):
        first, last = event_factory(), event_factory()
        first.record()
        result = call()
        last.record()
        synchronize()
        device.append(first.elapsed_time(last) * 1000)
    return result, {"synchronized_wall": summary(wall), "npu_event": summary(device)}


def bsa_reference(q, k, v, indices, counts, block=128):
    q, k, v = q.float(), k.float(), v.float()
    out = torch.empty_like(q)
    for qb in range(indices.shape[0]):
        lo, hi = qb * block, min((qb + 1) * block, q.shape[0])
        for head in range(q.shape[1]):
            selected = indices[qb, head, :int(counts[qb, head])].tolist()
            rows = [i for b in selected for i in range(b * block, min((b + 1) * block, k.shape[0]))]
            scores = q[lo:hi, head] @ k[rows, head].T * q.shape[-1] ** -0.5
            out[lo:hi, head] = scores.softmax(-1) @ v[rows, head]
    return (out,)


def gdn_reference(k, w, u, initial, chunk_size):
    """Zero scalar gate: chunk recurrence, not the complete GDN pipeline."""
    state = initial.float().clone()
    states, values = [], []
    for lo in range(0, k.shape[2], chunk_size):
        states.append(state.clone())
        sl = slice(lo, lo + chunk_size)
        value = u[:, :, sl].float() - w[:, :, sl].float() @ state
        values.append(value)
        state = state + k[:, :, sl].float().transpose(-1, -2) @ value
    return torch.stack(states, dim=2), torch.cat(values, dim=2), state


def check_outputs(actual, expected, atol, rtol):
    if len(actual) != len(expected):
        raise AssertionError("unexpected output count")
    checks = []
    for got, want in zip(actual, expected):
        got = got.detach().cpu().float()
        torch.testing.assert_close(got, want.float(), atol=atol, rtol=rtol)
        checks.append({"shape": list(got.shape), "max_abs_error": (got - want).abs().max().item()})
    return checks


def build_case(args, device):
    generator = torch.Generator().manual_seed(2026)
    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).half()
    t, h, d = args.tokens, args.heads, args.dim
    if args.operator == "bsa":
        import mindiesd  # noqa: F401: registers torch.ops.mindiesd
        q, k, v = [rand((t, h, d)) for _ in range(3)]
        blocks = (t + 127) // 128
        idx = torch.full((blocks, h, blocks), -1, dtype=torch.int64)
        count = blocks if args.selected_blocks == "all" else 1
        idx[:, :, :count] = torch.arange(count)
        counts = torch.full((blocks, h), count, dtype=torch.int64)
        expected = bsa_reference(q, k, v, idx, counts)
        q, k, v, idx, counts = [x.to(device) for x in (q, k, v, idx, counts)]
        def call():
            output = torch.ops.mindiesd.rainfusionattention(
                q, k, v, idx, counts, [128, 128], actual_seq_qlen=[t],
                actual_seq_kvlen=[t], q_input_layout="TND", kv_input_layout="TND",
                head_num=h, mask_type=0, scale=d ** -0.5, inner_precise=0, block_size=0)
            return (output[0],)
    else:
        from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h
        k, w = [rand((1, h, t, d), 0.02) for _ in range(2)]
        u = rand((1, h, t, d))
        initial = rand((1, h, d, d), 0.02).float()
        g = torch.zeros((1, h, t), dtype=torch.float32)
        expected = gdn_reference(k, w, u, initial, args.chunk_size)
        k, w, u, g, initial = [x.to(device) for x in (k, w, u, g, initial)]
        def call():
            return chunk_gated_delta_rule_fwd_h(
                k, w, u, g=g, initial_state=initial, output_final_state=True,
                chunk_size=args.chunk_size, cu_seqlens=None, state_v_first=False)
    return call, expected


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator", choices=("bsa", "gdn"), required=True)
    p.add_argument("--device", type=int, default=0, help="logical NPU device")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    p.add_argument("--selected-blocks", choices=("all", "one"), default="one")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.tokens, args.heads, args.dim, args.repeats) <= 0 or args.warmup < 0 or args.device < 0:
        p.error("dimensions/repeats must be positive; warmup/device must be nonnegative")
    if args.operator == "gdn" and (args.dim not in (128, 256) or args.tokens % args.chunk_size):
        p.error("GDN smoke requires dim 128/256 and full chunks")
    import torch_npu
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    call, expected = build_case(args, device)
    with torch.inference_mode():
        # This is the sole execution after warmups in replay mode.
        output, timing = execute(call, torch.npu.synchronize,
                                 lambda: torch.npu.Event(enable_timing=True),
                                 args.warmup, args.repeats, args.replay)
    checks = check_outputs(output, expected, atol=0.02, rtol=0.02)
    versions = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
    for package in ("mindiesd", "fla-npu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "distribution metadata unavailable"
    result = {"scope": "synthetic-smoke; not full50 or frozen-baseline matching",
              "operator": args.operator, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "dtype": "float16 inputs; float32 GDN gate/state", "seed": 2026,
              "gdn_gate": "zero cumulative natural-log gate" if args.operator == "gdn" else None,
              "versions": versions, "correctness": {"passed": True, "atol": 0.02, "rtol": 0.02, "outputs": checks},
              "timing": timing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
