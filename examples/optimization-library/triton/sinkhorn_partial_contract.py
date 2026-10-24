"""Fixed FP64 oracle and conservative serial/tree summation error bounds."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    n = p.get('num_blocks')
    if (type(n) is not int or not 1 <= n <= 4096
            or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('num_blocks must be an integer in [1,4096], with a nonnegative seed')
    return p, n


def validate_inputs(inputs, spec):
    _, n = dimensions(spec)
    shapes = dict(partials_scale=(n,2), partials_base=(n,8), scale=(3,), base=(24,))
    if set(inputs) != set(shapes):
        raise ValueError('four partial and caller-state buffers required')
    pointers = set()
    for key, shape in shapes.items():
        v = inputs[key]
        if (not isinstance(v, torch.Tensor) or v.device.type != 'cpu'
                or v.dtype != torch.float32 or tuple(v.shape) != shape
                or not v.is_contiguous() or v.storage_offset() != 0
                or v.untyped_storage().nbytes() != v.numel()*4
                or not torch.isfinite(v).all()):
            raise ValueError('owned finite contiguous FP32 CPU buffer required: '+key)
        if key.startswith('partials'):
            a = v.abs()
            if not bool(((a == 0) | ((a >= 2**-20) & (a <= 1))).all()):
                raise ValueError('partial values outside fixed numerical domain')
        ptr = v.untyped_storage().data_ptr()
        if ptr in pointers:
            raise ValueError('aliased buffers')
        pointers.add(ptr)


def reference(inputs):
    n = inputs['partials_scale'].shape[0]
    gamma = sum(n*u/(1-n*u) for u in (2**-24, 2**-53))
    expected, bounds = [], []
    for name, count in (('scale',2), ('base',8)):
        v = inputs['partials_'+name].double()
        output = inputs[name].double().clone()
        output[:count] = v.sum(0)
        expected.append(output)
        bounds.append(1e-7 + gamma*v.abs().sum(0))
    return tuple(expected), tuple(bounds)


def make_case(spec):
    p, n = dimensions(spec)
    g = torch.Generator().manual_seed(spec['seed'])
    inputs = {k: torch.rand((n,c),generator=g)*2-1
              for k,c in (('partials_scale',2),('partials_base',8))}
    for v in inputs.values():
        v[(v != 0) & (v.abs() < 2**-20)] = 2**-20
    if p.get('cancellation_and_zeros'):
        pattern = torch.tensor([1.,-1.,0.,2**-16,-2**-16,-0.])
        for v in inputs.values():
            for j in range(v.shape[1]):
                v[:,j] = pattern[(torch.arange(n)+j)%len(pattern)]
    inputs['scale'] = torch.tensor([.125,-.25,-0.])
    inputs['base'] = torch.arange(24,dtype=torch.float32)*.125-2
    inputs['base'][8] = -0.
    validate_inputs(inputs,spec)
    expected,bounds = reference(inputs)
    return dict(inputs=inputs,expected=expected,bounds=bounds)


def check_outputs(actual, fixture):
    if not isinstance(actual,tuple) or len(actual) != 2:
        return dict(passed=False,reason='expected exact (scale,base) tuple')
    rows = []
    for got,name,count,wanted,bound in zip(actual,('scale','base'),(2,8),fixture['expected'],fixture['bounds']):
        if (not isinstance(got,torch.Tensor) or got.dtype != torch.float32
                or got.shape != wanted.shape):
            return dict(passed=False,reason='output dtype/shape mismatch')
        cpu = got.detach().cpu()
        error = (cpu[:count].double()-wanted[:count]).abs()
        untouched = torch.equal(cpu[count:].view(torch.int32),fixture['inputs'][name][count:].view(torch.int32))
        rows.append(dict(name=name,errors=error.tolist(),bounds=bound.tolist(),
                         untouched_bitwise=untouched,negative_zero_count=int(((cpu==0)&torch.signbit(cpu)).sum()),
                         passed=bool(torch.isfinite(cpu).all() and (error<=bound).all()) and untouched))
    return dict(passed=all(r['passed'] for r in rows),outputs=rows,
                determinism='structural fixed order; no empirical repeat-bit claim')


CASE_SOURCES = [Path(__file__)]
