"""FP32 two-correction model and FP64 forward-error oracle; no extended range claim."""
from pathlib import Path
import torch
from tridiagonal_contract import dimensions, matrix, validate_inputs


def make_case(spec):
    p = spec.get('parameters', spec)
    if p.get('component'):
        spec = dict(spec, parameters=dict(p, R=1, ldb=p['m']+8))
    m, r, ldb = dimensions(spec)
    p = spec.get('parameters', spec)
    g = torch.Generator().manual_seed(spec['seed'])
    inputs = dict(dl=torch.rand(m, generator=g)*.5-.25, d=torch.full((m,), 4., dtype=torch.float32),
                  du=torch.rand(m, generator=g)*.5-.25,
                  b=torch.full((r, ldb), 98765.25, dtype=torch.float32))
    inputs['b'][:, :m] = torch.rand((r, m), generator=g)*.25-.125
    if p.get('regime') == 'moderate-cancellation':
        inputs['dl'].fill_(-1.)
        inputs['du'].fill_(-1.)
        inputs['d'].fill_(2.+1./64)
    inputs['dl'][0] = inputs['du'][-1] = 0.
    if p.get('regime') == 'diagonal-power-two-exact':
        inputs['dl'].zero_(); inputs['du'].zero_(); inputs['d'].fill_(4.)
        inputs['b'][:, :m] = torch.arange(m, dtype=torch.float32)/16
    if p.get('component') == 'whole-finite-update-rollback':
        inputs['b'][:, :m] = torch.arange(m, dtype=torch.float32)+1
        inputs['b'][0, 0] = torch.finfo(torch.float32).max
        delta = torch.ones((r, m), dtype=torch.float32)
        delta[0, 0] = torch.finfo(torch.float32).max
        inputs['delta'] = delta
        return dict(inputs=inputs, expected=inputs['b'].clone())
    expected = inputs['b'].double().clone()
    expected[:, :m] = torch.linalg.solve(matrix(inputs), inputs['b'][:, :m].double().T).T
    return dict(inputs=inputs, expected=expected)


def factors(inputs):
    dl, d, du = (inputs[k] for k in ('dl', 'd', 'du'))
    cp, inv = torch.empty_like(d), torch.empty_like(d)
    previous = torch.tensor(0., dtype=torch.float32)
    for i in range(d.numel()):
        inv[i] = 1./(d[i]-dl[i]*previous)
        cp[i] = du[i]*inv[i]
        previous = cp[i]
    return cp, inv


def solve(inputs, b, cp, inv):
    out = b.clone()
    for rhs in range(b.shape[0]):
        previous = torch.tensor(0., dtype=torch.float32)
        for i in range(cp.numel()):
            out[rhs, i] = (b[rhs, i]-inputs['dl'][i]*previous)*inv[i]
            previous = out[rhs, i].clone()
        following = torch.tensor(0., dtype=torch.float32)
        for i in range(cp.numel()-1, -1, -1):
            out[rhs, i] -= cp[i]*following
            following = out[rhs, i].clone()
    return out


def commit(x, delta, m):
    proposal = x[:, :m]+delta[:, :m]
    # One invalid component rejects the entire RHS vector.
    return torch.where(torch.isfinite(proposal).all(dim=1)[:, None], proposal, x[:, :m])


def refinement_cpu(inputs, after):
    m = inputs['d'].numel()
    if 'delta' in inputs:
        out = inputs['b'].clone()
        if after:
            out[:, :m] = commit(out, inputs['delta'], m)
        return out
    cp, inv = factors(inputs)
    out = solve(inputs, inputs['b'], cp, inv)
    if after:
        for _ in range(2):
            x = out[:, :m]
            lower = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
            upper = torch.cat((x[:, 1:], torch.zeros_like(x[:, :1])), dim=1)
            ax = (inputs['d']*x+inputs['dl']*lower)+inputs['du']*upper
            residual = inputs['b'][:, :m]-ax
            delta = solve(inputs, residual, cp, inv)
            out[:, :m] = commit(out, delta, m)
    return out


def diagnostics(actual, fixture):
    x = actual.detach().cpu()
    inputs, expected = fixture['inputs'], fixture['expected']
    m = inputs['d'].numel()
    if x.shape != expected.shape or x.dtype != torch.float32 or not torch.isfinite(x).all():
        return dict(passed=False, reason='shape/dtype/nonfinite')
    padding = torch.equal(x[:, m:].view(torch.int32), inputs['b'][:, m:].view(torch.int32))
    if 'delta' in inputs:
        equal = torch.equal(x, expected)
        return dict(passed=equal and padding, whole_vector_rollback=equal, component_guard=True)
    active, reference = x[:, :m].double(), expected[:, :m].double()
    error = active-reference
    a, b = matrix(inputs), inputs['b'][:, :m].double()
    residual = b-active@a.T
    denominator = b.abs()+active.abs()@a.abs().T
    backward = (residual.abs()/denominator.clamp_min(1e-300)).max()
    return dict(passed=bool(torch.allclose(active, reference, atol=1e-5, rtol=1e-4)) and padding,
                padding_bitwise_unchanged=padding, atol=1e-5, rtol=1e-4,
                max_forward_error=float(error.abs().max()),
                relative_l2_forward_error=float(error.norm()/reference.norm().clamp_min(1e-300)),
                max_abs_residual=float(residual.abs().max()), componentwise_backward_error=float(backward))


CASE_SOURCES = [Path(__file__), Path(__file__).with_name('tridiagonal_contract.py')]
