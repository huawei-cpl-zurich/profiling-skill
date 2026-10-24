"""L02: bounded represented FP16 GEMM; independent CPU FP64 oracle."""
from pathlib import Path
import torch
CASE_SOURCES = [Path(__file__)]


def check_spec(spec, cores=None):
    for key in ('m','n','k'):
        value = spec.get(key)
        if type(value) is not int or not 0 < value <= 32768 or value % 16:
            raise ValueError('positive dimensions aligned to 16 required')
    if spec['k'] > 256 or spec.get('b_layout','row') != 'row':
        raise ValueError('single L1 stage and dense row-major operands required')
    if spec.get('consecutive_distinct_calls',1) not in (1,2):
        raise ValueError('only one GEMM or the two-GEMM composite guard is supported')
    if spec.get('consecutive_distinct_calls',1) == 2 and spec.get('role') != 'guard':
        raise ValueError('composite calls are correctness-only guards')
    if type(spec.get('seed')) is not int:
        raise ValueError('an explicit integer seed is required')
    tasks = ((spec['m']+127)//128)*((spec['n']+255)//256)
    if cores is not None and (type(cores) is not int or cores <= 0 or tasks > cores):
        raise ValueError('task grid exceeds observed Cube-core count')
    return tasks


def make_case(spec):
    check_spec(spec)
    generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    fixtures = []
    for _ in range(spec.get('consecutive_distinct_calls',1)):
        m,n,k = (spec[key] for key in ('m','n','k'))
        a = (torch.rand((m,k),generator=generator,dtype=torch.float32,device='cpu')*.5-.25).to(torch.float16)
        b = (torch.rand((k,n),generator=generator,dtype=torch.float32,device='cpu')*.5-.25).to(torch.float16)
        fixtures.append(dict(inputs=dict(a=a,b=b), expected=a.to(torch.float64)@b.to(torch.float64)))
    # The composite guard is one cached payload containing two independently drawn GEMMs.
    return fixtures[0] if len(fixtures)==1 else dict(components=tuple(fixtures),
        expected=tuple(fixture['expected'] for fixture in fixtures))


def validate_actual(actual, fixture, inputs):
    expected = fixture['expected']
    result = actual.detach().cpu()
    if result.shape != expected.shape or result.dtype != torch.float16 or not torch.isfinite(result).all():
        raise ValueError('finite FP16 output with exact logical shape required')
    error = (result.to(torch.float64)-expected).abs()
    for name,original in fixture['inputs'].items():
        if not torch.equal(inputs[name].detach().cpu(),original):
            raise ValueError('input mutated: '+name)
    return dict(passed=bool(torch.all(error <= .002+.002*expected.abs())),
                max_absolute_error=float(error.max()), atol=.002, rtol=.002)
