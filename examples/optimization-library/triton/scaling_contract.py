"""Independent CPU contract for lower-triangular, row-scaled K K^T."""
from pathlib import Path

import torch

RTOL = ATOL = 1e-2


def dimensions(spec):
    values = tuple(spec.get(key) for key in ('groups', 'bt', 'dk', 'seed'))
    if any(type(value) is not int for value in values):
        raise ValueError('groups, bt, dk and seed must be integers')
    groups, bt, dk, seed = values
    if not 1 <= groups <= 1024 or bt not in (32, 64) or dk not in (16, 32, 128, 256):
        raise ValueError('supported envelope: groups 1..1024, BT 32/64, DK 16/32/128/256')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    return groups, bt, dk, seed


def reference(k, beta):
    """FP32 CPU oracle; neither candidate's conversion order is reproduced."""
    return torch.tril(torch.bmm(k.float(), k.float().transpose(1, 2)) * beta[..., None], diagonal=-1)


def make_case(spec):
    groups, bt, dk, seed = dimensions(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    k = torch.randn(groups, bt, dk, generator=rng)
    # Near-opposite rows stress cancellation and sign handling without NaNs.
    k[:, 1::2] = -k[:, ::2] + 0.001 * k[:, 1::2]
    k = (k / torch.linalg.vector_norm(k, dim=-1, keepdim=True)).to(torch.bfloat16)
    beta = torch.rand(groups, bt, generator=rng, dtype=torch.float32)
    beta[:, 0], beta[:, 1] = 0.0, 1.0
    beta[:, 2] = torch.finfo(torch.bfloat16).tiny
    beta[:, 3] = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    return dict(k=k, beta=beta, expected=reference(k, beta))


def diagnostics(actual, fixture):
    expected = fixture['expected']
    if actual.device.type != 'cpu' or actual.shape != expected.shape or actual.dtype != torch.float32:
        return dict(passed=False, reason='expected CPU FP32 output with declared shape')
    finite = bool(torch.isfinite(actual).all())
    if not finite:
        return dict(passed=False, reason='nonfinite output')
    error = (actual - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    return dict(passed=bool((error <= allowed).all()), rtol=RTOL, atol=ATOL,
                max_absolute_error=error.max().item(),
                max_relative_error=(error / expected.abs().clamp_min(1e-12)).max().item(),
                max_tolerance_fraction=(error / allowed).max().item())


CASE_SOURCES = [Path(__file__)]
