"""Independent CPU given-saved-state derivative; never executes a GPU forward."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    rows, d = p.get('rows'), p.get('D')
    if (type(rows) is not int or not 1 <= rows <= 4096 or type(d) is not int
            or not 1 <= d <= 2048 or p.get('offset') not in (0, 1)
            or p.get('eps') != 1e-6 or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('unsupported provided-state RMS backward case')
    return p, rows, d


def validate_inputs(inputs, spec):
    _, rows, d = dimensions(spec)
    shapes = dict(X=(rows, d), dY=(rows, d), W=(d,), RSTD=(rows,))
    if set(inputs) != set(shapes):
        raise ValueError('expected X,dY,W,RSTD inputs')
    pointers = set()
    for key, shape in shapes.items():
        v = inputs[key]
        if (not isinstance(v, torch.Tensor) or v.device.type != 'cpu'
                or v.dtype != torch.float32 or tuple(v.shape) != shape
                or not v.is_contiguous() or v.storage_offset() != 0
                or v.untyped_storage().nbytes() != v.numel()*4
                or not torch.isfinite(v).all()
                or (key != 'RSTD' and not (v.abs() <= 1).all())
                or (key == 'RSTD' and not ((v > 0) & (v <= 1000.001)).all())):
            raise ValueError('invalid owned contiguous finite FP32 input: ' + key)
        pointer = v.untyped_storage().data_ptr()
        if pointer in pointers:
            raise ValueError('aliased inputs')
        pointers.add(pointer)


def make_case(spec):
    p, rows, d = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    inputs = {k: torch.rand(shape, generator=rng)*2-1
              for k, shape in dict(X=(rows,d), dY=(rows,d), W=(d,)).items()}
    if p.get('zero_cancellation_poison_tail'):
        if rows < 3:
            raise ValueError('tail guard requires at least three rows')
        inputs['X'][0].zero_()
        inputs['X'][1].fill_(.25); inputs['X'][2].fill_(-.25)
        inputs['dY'][1:3].fill_(.25)
    x, dy, w = (inputs[k].double() for k in ('X','dY','W'))
    ideal = (x.square().mean(-1) + p['eps']).rsqrt()
    inputs['RSTD'] = ideal.float()
    r = inputs['RSTD'].double()[:,None]
    m = dy * (w + p['offset'])
    dx = r*m - r.pow(3)*x*(m*x).mean(-1, keepdim=True)
    dw = (dy*x*r).sum(0)
    validate_inputs(inputs, spec)
    return dict(inputs=inputs, expected=(dx, dw), oracle='FP64 given rounded saved RSTD')


def check_outputs(actual, expected):
    if not isinstance(actual, tuple) or len(actual) != 2:
        return dict(passed=False, reason='output must be (dX,dW) tuple')
    rows = []
    pointers = set()
    for got, wanted, atol, rtol in zip(actual, expected, (1e-4,2e-4), (1e-5,2e-5)):
        if (not isinstance(got, torch.Tensor) or got.shape != wanted.shape
                or got.dtype != torch.float32 or not got.is_contiguous()):
            return dict(passed=False, reason='output shape/dtype/layout mismatch')
        pointers.add(got.untyped_storage().data_ptr())
        cpu = got.detach().cpu().double()
        rows.append(dict(passed=bool(torch.isfinite(cpu).all() and torch.allclose(cpu,wanted,atol=atol,rtol=rtol)),
                         max_abs_error=float((cpu-wanted).abs().max()), atol=atol, rtol=rtol))
    return dict(passed=len(pointers)==2 and all(r['passed'] for r in rows), outputs=rows,
                cross_arm_dw_bitwise_guarantee=False)


CASE_SOURCES = [Path(__file__)]
