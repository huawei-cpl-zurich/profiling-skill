"""Bounded FP16 GEMM with an independent FP32 CPU oracle and fixed numeric gate."""
from pathlib import Path
import torch

CASE_SOURCES = [Path(__file__)]


def check_spec(spec):
    for key in ('m', 'n', 'k'):
        if type(spec.get(key)) is not int or not 0 < spec[key] <= 32768:
            raise ValueError('positive integer dimensions at most 32768 are required')
    if spec['m'] % 16 or spec['n'] % 16 or spec['k'] % 256:
        raise ValueError('M/N must align to16; K must align to256')
    if type(spec.get('seed')) is not int:
        raise ValueError('an integer seed is required')


def geometry(spec, cores=20):
    check_spec(spec)
    if cores != 20:
        raise ValueError('the frozen matrix requires exactly20 qualified Cube cores')
    tiles = ((spec['m'] + 127) // 128) * ((spec['n'] + 255) // 256)
    tail = tiles % cores
    work = tail * (spec['k'] // 256)
    return dict(output_tiles=tiles, normal_tiles=tiles-tail, tail_tiles=tail,
                tail_k_blocks=work, tail_workers=min(work, cores),
                tail_k_blocks_per_worker=work//cores, tail_extra_workers=work % cores)


def make_case(spec):
    check_spec(spec)
    generator = torch.Generator().manual_seed(spec['seed'])
    a = (torch.rand(spec['m'], spec['k'], generator=generator) * .25 - .125).half()
    b = (torch.rand(spec['k'], spec['n'], generator=generator) * .25 - .125).half()
    expected = a.float() @ b.float()
    return dict(inputs=dict(a=a, b=b), expected=expected)


def validate(actual, fixture, inputs=None):
    result = actual.detach().cpu()
    expected = fixture['expected']
    if result.shape != expected.shape or result.dtype != torch.float16:
        raise ValueError('output must have the exact FP16 dense GEMM shape')
    delta = (result.float() - expected).abs()
    passed = bool(torch.isfinite(result).all() and
                  torch.all(delta <= .02 + .02 * expected.abs()))
    if inputs is not None:
        for name, original in fixture['inputs'].items():
            if not torch.equal(inputs[name].cpu(), original):
                raise ValueError('input mutation: ' + name)
    return dict(passed=passed, max_absolute_error=float(delta.max()), atol=.02, rtol=.02)
