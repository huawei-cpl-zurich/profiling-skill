"""Independent CPU fixture provider for bounded dual BF16 projections."""
import torch

K, N = 5120, 512
CONFIG = dict(BM=32, BN=64, BK=128, GROUP=8)


def check_spec(spec):
    p = spec.get('parameters', spec)
    if type(p.get('M')) is not int or not 0 <= p['M'] <= 256:
        raise ValueError('M must be an integer in [0,256]')
    for key, value in dict(IN_DIM=K, OUT_DIM=N, **CONFIG).items():
        if p.get(key, value) != value:
            raise ValueError('unsupported ' + key)
    if type(spec.get('seed', 0)) is not int:
        raise ValueError('integer seed required')
    return p['M']


def make_case(spec):
    m = check_spec(spec)
    rng = torch.Generator().manual_seed(spec.get('seed', 0))
    q = {name: torch.randint(-16, 17, shape, generator=rng).to(torch.float64)
         for name, shape in [('x', (m, K)), ('w0', (N, K)), ('w1', (N, K))]}
    if spec.get('parameters', spec).get('signed_cancelled_distinct_weights') and m >= 2:
        q['x'][0].zero_()
        q['x'][1] = torch.arange(K).remainder(2).to(torch.float64).mul(32).sub(16)
    bound = K * 256
    gamma = (K * 2**-53) / (1 - K * 2**-53)
    assert gamma * bound < .5 and bound < 2**24
    expected = []
    for name in ('w0', 'w1'):
        dot = q['x'] @ q[name].T if m else torch.empty((0, N), dtype=torch.float64)
        integer = dot.round()
        if dot.numel() and (not torch.all((dot - integer).abs() <= gamma * bound)
                            or not torch.all(integer.abs() <= bound)):
            raise ValueError('CPU exact-integer recovery certificate failed')
        expected.append((integer / 16384).float())
    return dict(inputs={name: (v / 128).to(torch.bfloat16) for name, v in q.items()},
                expected=expected, certificate=dict(gamma=gamma, sumabs_bound=bound,
                integer_recovery_error_bound=gamma * bound, denominator=16384))


def check_inputs(inputs, m, device):
    if set(inputs) != {'x', 'w0', 'w1'}:
        raise ValueError('exactly x,w0,w1 required')
    for name, t in inputs.items():
        shape = (m, K) if name == 'x' else (N, K)
        if (not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != torch.bfloat16
                or t.device.type != device or not t.is_contiguous() or t.storage_offset() != 0):
            raise ValueError('invalid tensor layout/dtype/device: ' + name)
    storage = [t.untyped_storage().data_ptr() for t in inputs.values() if t.numel()]
    if len(storage) != len(set(storage)):
        raise ValueError('inputs must not alias')


def validate_actual(actual, fixture, device_inputs=None, buffers=None):
    expected = fixture['expected']
    if not isinstance(actual, (tuple, list)) or len(actual) != 2 or len(expected) != 2:
        raise ValueError('exactly two outputs required')
    ptrs = []
    for i, (a, e) in enumerate(zip(actual, expected)):
        if (not isinstance(a, torch.Tensor) or a.dtype != torch.float32 or a.shape != e.shape
                or not a.is_contiguous()):
            raise ValueError('output shape/dtype/layout')
        if device_inputs is not None and a.device != device_inputs['x'].device:
            raise ValueError('output device')
        if a.numel():
            ptrs.append(a.untyped_storage().data_ptr())
        if not torch.equal(a.detach().cpu().view(torch.int32), e.view(torch.int32)):
            raise ValueError('exact FP32 output bits differ: ' + str(i))
    if device_inputs is not None:
        ptrs += [v.untyped_storage().data_ptr() for v in device_inputs.values() if v.numel()]
        for name, value in device_inputs.items():
            if not torch.equal(value.cpu(), fixture['inputs'][name]):
                raise ValueError('input mutation: ' + name)
    if len(ptrs) != len(set(ptrs)):
        raise ValueError('output/input storage alias')
    for b in buffers or []:
        host = b.cpu()
        if not torch.all(host[:32] == 12345) or not torch.all(host[-32:] == 12345):
            raise ValueError('output canary changed')
    return dict(passed=True, exact_bits=True, outputs=2, rtol=0, atol=0)
