"""Independent no-pivot Thomas contract; dense FP64 oracle is separate."""
from pathlib import Path
import torch

SENTINEL = 98765.25


def dimensions(spec):
    p = spec.get('parameters', spec)
    if any(type(p.get(k)) is not int for k in ('m', 'R', 'ldb')):
        raise ValueError('integer m,R,ldb required')
    m, r, ldb = (p[k] for k in ('m', 'R', 'ldb'))
    if not (1 <= m <= 1024 and 0 <= r <= 128 and m <= ldb <= m+64):
        raise ValueError('unsupported dimensions')
    return m, r, ldb


def matrix(inputs):
    return (torch.diag(inputs['d'].double()) + torch.diag(inputs['dl'][1:].double(), -1)
            + torch.diag(inputs['du'][:-1].double(), 1))


def dense_solve(inputs):
    m = inputs['d'].numel()
    out = inputs['b'].clone()
    out[:, :m] = torch.linalg.solve(matrix(inputs), inputs['b'][:, :m].double().T).T.float()
    return out


def make_case(spec):
    m, r, ldb = dimensions(spec)
    g = torch.Generator().manual_seed(spec['seed'])
    inputs = dict(dl=torch.rand(m, generator=g)*.5-.25, d=torch.full((m,), 4.),
                  du=torch.rand(m, generator=g)*.5-.25, b=torch.full((r, ldb), SENTINEL))
    inputs['dl'][0] = inputs['du'][-1] = 0.
    inputs['b'][:, :m] = torch.rand((r, m), generator=g)*.25-.125
    if spec.get('parameters', spec).get('distribution') == 'nonsingular-zero-first-pivot':
        inputs['d'][0] = 0.
        inputs['dl'][1] = inputs['du'][0] = .25
    return dict(inputs=inputs, expected=dense_solve(inputs))


def validate_inputs(inputs, spec):
    m, r, ldb = dimensions(spec)
    for k, shape in [('dl', (m,)), ('d', (m,)), ('du', (m,)), ('b', (r, ldb))]:
        x = inputs[k]
        if x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != shape or not x.is_contiguous():
            raise ValueError('contiguous CPU FP32 input required: '+k)
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite input')
    if inputs['dl'][0] != 0 or inputs['du'][-1] != 0:
        raise ValueError('boundary coefficient must be zero')


def eligible(inputs):
    # A sufficient host-side no-pivot guard, not an oracle/pivot lookup.
    return bool((inputs['d'].abs() > inputs['dl'].abs()+inputs['du'].abs()).all())


def thomas_cpu(inputs, shared):
    """Functional FP32 model for both ownership schedules, including empty RHS."""
    if not eligible(inputs):
        return dense_solve(inputs)
    dl, d, du, b = (inputs[k] for k in ('dl', 'd', 'du', 'b'))
    m, r = d.numel(), b.shape[0]
    out = b.clone()
    def factors():
        cp, inv = torch.empty_like(d), torch.empty_like(d)
        previous = torch.tensor(0., dtype=torch.float32)
        for i in range(m):
            inv[i] = 1. / (d[i]-dl[i]*previous)
            cp[i] = du[i]*inv[i]
            previous = cp[i]
        return cp, inv
    common = factors() if shared and r else None
    for rhs in range(r):
        cp, inv = common if shared else factors()
        previous = torch.tensor(0., dtype=torch.float32)
        for i in range(m):
            out[rhs, i] = (b[rhs, i]-dl[i]*previous)*inv[i]
            previous = out[rhs, i].clone()
        following = torch.tensor(0., dtype=torch.float32)
        for i in range(m-1, -1, -1):
            out[rhs, i] = out[rhs, i]-cp[i]*following
            following = out[rhs, i].clone()
    return out


def diagnostics(actual, fixture):
    x = actual.detach().cpu()
    inputs, expected = fixture['inputs'], fixture['expected']
    m = inputs['d'].numel()
    if x.shape != expected.shape or x.dtype != torch.float32 or not torch.isfinite(x).all():
        return dict(passed=False, reason='shape/dtype/nonfinite')
    pad = torch.equal(x[:, m:].view(torch.int32), inputs['b'][:, m:].contiguous().view(torch.int32))
    residual = matrix(inputs) @ x[:, :m].double().T-inputs['b'][:, :m].double().T
    relative = float(residual.abs().max()/inputs['b'][:, :m].abs().double().max().clamp_min(1e-30)) if x.shape[0] else 0.
    close = bool(torch.allclose(x[:, :m], expected[:, :m], atol=2e-5, rtol=2e-4))
    return dict(passed=close and pad and relative <= 2e-5, padding_bitwise_unchanged=pad,
                relative_inf_residual=relative, residual_limit=2e-5, atol=2e-5, rtol=2e-4)


CASE_SOURCES = [Path(__file__)]
