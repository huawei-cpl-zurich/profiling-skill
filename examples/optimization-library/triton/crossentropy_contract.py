"""Independent oracle for unreduced cross-entropy and its complete sum-VJP."""
from pathlib import Path
import torch

CASE_SOURCES = [Path(__file__)]
RTOL, ATOL = 2e-4, 2e-4


def check_spec(spec):
    rows, width, seed = (spec.get(k) for k in ('rows', 'width', 'seed'))
    if spec.get('family') != 'E10' or any(type(v) is not int for v in (rows, width, seed)):
        raise ValueError('E10 integer rows, width and seed required')
    if not (1 <= rows <= 4096 and 1 <= width <= 8192 and rows * width <= 8388608
            and 0 <= seed < 2**64):
        raise ValueError('bounded positive logits dimensions and uint64 seed required')
    return rows, width, seed


def make_case(spec):
    rows, width, seed = check_spec(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    logits = torch.randn(rows, width, generator=rng) * 1.5
    logits[::13].zero_()
    logits[0, 0], logits[0, -1] = 80., -80.
    targets = torch.randint(width, (rows,), generator=rng, dtype=torch.int64)
    targets[0] = width - 1
    precise = logits.double().requires_grad_()
    loss = -torch.log_softmax(precise, -1).gather(1, targets[:, None]).squeeze(1)
    gradient, = torch.autograd.grad(loss.sum(), precise)
    return dict(inputs=dict(logits=logits, targets=targets),
                expected=(loss.detach().float(), gradient.float()))


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        return dict(passed=False, reason='loss and complete logits gradient required')
    outputs = []
    for got, expected in zip(actual, fixture['expected']):
        if not isinstance(got, torch.Tensor) or got.device.type != 'cpu' or (
                got.shape != expected.shape or got.dtype != torch.float32 or not got.is_contiguous()):
            return dict(passed=False, reason='contiguous CPU FP32 output shapes required')
        if not torch.isfinite(got).all():
            return dict(passed=False, reason='nonfinite output')
        error = (got - expected).abs()
        allowed = ATOL + RTOL * expected.abs()
        outputs.append(dict(passed=bool((error <= allowed).all()),
                            max_absolute_error=error.max().item(),
                            max_tolerance_fraction=(error / allowed).max().item()))
    return dict(passed=all(row['passed'] for row in outputs), outputs=outputs,
                rtol=RTOL, atol=ATOL)
