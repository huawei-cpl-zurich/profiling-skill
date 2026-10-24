"""Independent CPU contract for E05 selected-block noncausal GQA attention."""
from pathlib import Path
import math

import torch

BM, BN = 16, 32
RTOL = ATOL = 1e-3
CASE_SOURCES = [Path(__file__)]


def check_spec(spec):
    keys = ('hq', 'hkv', 'q', 'k', 'd', 'slots', 'selected_count', 'seed')
    if any(type(spec.get(key)) is not int for key in keys):
        raise ValueError('head counts, lengths, dimension, slots, count and seed must be integers')
    hq, hkv, q, k, d, slots, count, seed = (spec[key] for key in keys)
    blocks = (k + BN - 1) // BN
    if (spec.get('family') != 'E05' or not 1 <= hkv <= hq <= 8 or hq % hkv
            or not 1 <= q <= 2048 or not 1 <= k <= 8192 or d not in (16, 32, 64, 128)
            or not 1 <= slots <= 256 or not 0 <= count <= min(slots, blocks)
            or not 0 <= seed < 2**64 or hq*q*k > 2**26):
        raise ValueError('case exceeds the fixed sparse attention envelope')
    return hq, hkv, q, k, d, slots


def reference(inputs):
    """Dense FP32 softmax and explicit Boolean mask, not device online algebra."""
    q, k, v, selected = (inputs[key] for key in ('q', 'k', 'v', 'selected'))
    hq, nq, d = q.shape
    mask = torch.zeros(hq, nq, k.shape[1], dtype=torch.bool)
    for head in range(hq):
        for tile in range(selected.shape[1]):
            for block in selected[head, tile].tolist():
                if block >= 0:
                    mask[head, tile*BM:(tile+1)*BM, block*BN:(block+1)*BN] = True
    mapping = torch.arange(hq) // (hq // k.shape[0])
    scores = (q.float() @ k[mapping].float().transpose(-1, -2)) / math.sqrt(d)
    present = mask.any(-1, keepdim=True)
    safe = torch.where(present, scores.masked_fill(~mask, -float('inf')), 0)
    return torch.where(present, torch.softmax(safe, -1) @ v[mapping].float(), 0)


def validate_inputs(spec, inputs):
    hq, hk, nq, nk, d, slots = check_spec(spec)
    shapes = dict(q=(hq,nq,d), k=(hk,nk,d), v=(hk,nk,d),
                  selected=(hq,(nq+BM-1)//BM,slots))
    if set(inputs) != set(shapes):
        raise ValueError('q/k/v/selected inputs required')
    for key, shape in shapes.items():
        value = inputs[key]
        dtype = torch.int32 if key=='selected' else torch.float16
        if (not isinstance(value, torch.Tensor) or value.device.type!='cpu'
                or value.shape!=shape or value.dtype!=dtype or not value.is_contiguous()
                or not torch.isfinite(value).all()):
            raise ValueError('contiguous finite CPU inputs with declared shape/dtype required')
    for row in inputs['selected'].reshape(-1,slots):
        valid = row[row>=0]
        if (bool((row < -1).any()) or bool((valid >= (nk+BN-1)//BN).any())
                or len(valid)!=len(torch.unique(valid))):
            raise ValueError('selection requires unique in-range block IDs or -1 holes')


def make_case(spec):
    hq, hkv, nq, nk, d, slots = check_spec(spec)
    rng = torch.Generator(device='cpu').manual_seed(spec['seed'])
    inputs = dict(q=(torch.randn(hq, nq, d, generator=rng)*.5).half(),
                  k=(torch.randn(hkv, nk, d, generator=rng)*.5).half(),
                  v=torch.randn(hkv, nk, d, generator=rng).half())
    selected = torch.full((hq, (nq+BM-1)//BM, slots), -1, dtype=torch.int32)
    count = spec['selected_count']
    for head in range(hq):
        for tile in range(selected.shape[1]):
            positions = torch.randperm(slots, generator=rng)[:count]
            blocks = torch.randperm((nk+BN-1)//BN, generator=rng)[:count].int()
            selected[head, tile, positions] = blocks
    # A mandatory empty tile exercises finite zero output in both implementations.
    selected[0, 0] = -1
    inputs['selected'] = selected
    return dict(inputs=inputs, expected=reference(inputs))


def diagnostics(actual, fixture, device_inputs=None):
    expected = fixture['expected']
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != torch.float32 or not torch.isfinite(actual).all()):
        return dict(passed=False, reason='wrong output shape/dtype or nonfinite output')
    error = (actual.cpu() - expected).abs()
    if device_inputs is not None and any(not torch.equal(value.cpu(), fixture['inputs'][key])
                                         for key, value in device_inputs.items()):
        return dict(passed=False, reason='input mutation')
    return dict(passed=bool((error <= ATOL+RTOL*expected.abs()).all()), rtol=RTOL, atol=ATOL,
                max_absolute_error=float(error.max()))
