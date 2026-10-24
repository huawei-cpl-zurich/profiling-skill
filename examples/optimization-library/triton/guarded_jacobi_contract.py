"""Independent moderate-range FP32 Jacobi contract, not native device code."""
from pathlib import Path
import torch
from tridiagonal_contract import matrix, SENTINEL


def dimensions(spec):
    p = spec.get('parameters', spec)
    m, r, ldb = (p.get(k) for k in ('m', 'R', 'ldb'))
    if any(type(x) is not int for x in (m, r, ldb)) or not (1 <= m <= 6144 and r == 1 and m <= ldb <= m+64):
        raise ValueError('single RHS, m in 1..6144 and bounded padding required')
    return m, r, ldb


def make_case(spec):
    m, r, ldb = dimensions(spec)
    g = torch.Generator().manual_seed(spec['seed'])
    dl = torch.rand(m, generator=g, dtype=torch.float32)*.5-.25
    du = torch.rand(m, generator=g, dtype=torch.float32)*.5-.25
    dl[0] = du[-1] = 0.
    d = 2+dl.abs()+du.abs()
    b = torch.full((r, ldb), SENTINEL, dtype=torch.float32)
    b[:, :m] = torch.rand((r, m), generator=g, dtype=torch.float32)*.25-.125
    regime = spec.get('parameters', spec).get('regime')
    if regime == 'exact-dominance-boundary':
        dl[1:] = .125; du[:-1] = -.125
        d = 2*(dl.abs()+du.abs())
    if regime == 'boundary-impulses':
        b[:, :m] = 0.; b[0, 0] = .125; b[0, -1 if ldb == m else m-1] = -.125
    inputs = dict(dl=dl, d=d, du=du, b=b)
    expected = b.double()
    expected[:, :m] = torch.linalg.solve(matrix(inputs), b[:, :m].double().T).T
    return dict(inputs=inputs, expected=expected)


def validate_inputs(inputs, spec):
    m, r, ldb = dimensions(spec)
    for k, shape in [('dl', (m,)), ('d', (m,)), ('du', (m,)), ('b', (r, ldb))]:
        x = inputs[k]
        if x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != shape or not x.is_contiguous():
            raise ValueError('contiguous CPU FP32 input required: '+k)
    if inputs['dl'][0] != 0 or inputs['du'][-1] != 0:
        raise ValueError('zero off-domain coefficients required')


def certificate(inputs):
    dl, d, du, b = (inputs[k] for k in ('dl', 'd', 'du', 'b'))
    a = d.abs(); off = dl.abs()+du.abs(); active = b[:, :d.numel()].abs()
    return bool(128 <= d.numel() <= 6144 and all(torch.isfinite(x).all() for x in (dl,d,du,active))
                and (a > 2*off).all() and a.min() >= 2**-24 and a.max() <= 2**24
                and active.max() <= 2**24 and active.max()/a.min() <= 4
                and (off <= 2**23).all())


def model(inputs, after):
    from tridiagonal_contract import thomas_cpu
    if not after or not certificate(inputs):
        return thomas_cpu(inputs, False)
    dl, d, du, b = (inputs[k] for k in ('dl', 'd', 'du', 'b'))
    m = d.numel(); out = b.clone(); x = torch.zeros(m, dtype=torch.float32)
    for _ in range(40):
        left = torch.cat((x.new_zeros(1), x[:-1]))
        right = torch.cat((x[1:], x.new_zeros(1)))
        x = ((b[0,:m]-dl*left)-du*right)/d
    out[0,:m] = x
    return out


def diagnostics(actual, fixture):
    x = actual.detach().cpu(); inputs = fixture['inputs']; expected = fixture['expected']
    m = inputs['d'].numel(); b = inputs['b'][:, :m].double(); y = x[:, :m].double()
    if x.shape != expected.shape or x.dtype != torch.float32 or not torch.isfinite(x).all():
        return dict(passed=False, reason='shape/dtype/nonfinite')
    error = y-expected[:,:m]
    a = matrix(inputs); residual = b.T-a@y.T
    denom = a.abs()@y.T.abs()+b.T.abs()
    padding = torch.equal(x[:,m:].contiguous().view(torch.int32), inputs['b'][:,m:].contiguous().view(torch.int32))
    return dict(passed=bool(torch.allclose(y,expected[:,:m],atol=1e-5,rtol=1e-4)) and padding,
        forward_max_abs_error=float(error.abs().max()), relative_l2_error=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected[:,:m]).clamp_min(1e-300)),
        fp64_residual_inf=float(residual.abs().max()), componentwise_backward_error=float((residual.abs()/denom.clamp_min(1e-300)).max()),
        padding_bitwise_unchanged=padding,atol=1e-5,rtol=1e-4)


CASE_SOURCES = [Path(__file__), Path(__file__).with_name('tridiagonal_contract.py')]
