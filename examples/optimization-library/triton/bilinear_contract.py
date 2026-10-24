"""Standalone NHWC bilinear resize: fixed asymmetric coordinates and FP64 oracle."""
from pathlib import Path
import torch


def parameters(spec, *, modes=True):
    p = dict(spec.get('parameters', spec))
    for key in ('N', 'H', 'W', 'C', 'OH', 'OW'):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError('positive integer ' + key + ' required')
    for key in ('align_corners', 'half_pixel_centers', 'antialias'):
        if type(p.get(key, False)) is not bool or (modes and p.get(key, False)):
            raise ValueError('only asymmetric coordinates without antialias are supported')
    return p


def checked_input(x, p):
    if x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != tuple(p[k] for k in ('N','H','W','C')) or not x.is_contiguous():
        raise ValueError('CPU contiguous FP32 NHWC input required')
    if not bool(torch.isfinite(x).all()) or not bool((x.abs() <= 1).all()):
        raise ValueError('finite input in [-1,1] required')


def reference(x, p):
    """Independent FP64 coordinates, four-corner interpolation and border clamp."""
    checked_input(x, p)
    sy = torch.arange(p['OH'], dtype=torch.float64, device='cpu') * (p['H']/p['OH'])
    sx = torch.arange(p['OW'], dtype=torch.float64, device='cpu') * (p['W']/p['OW'])
    y0, x0 = sy.floor().long(), sx.floor().long()
    wy, wx = sy-y0, sx-x0
    y0, x0 = y0.clamp(0,p['H']-1), x0.clamp(0,p['W']-1)
    y1, x1 = (y0+1).clamp(max=p['H']-1), (x0+1).clamp(max=p['W']-1)
    a = x.double()
    upper = a[:,y0[:,None],x0[None,:],:]*(1-wx)[None,None,:,None] + a[:,y0[:,None],x1[None,:],:]*wx[None,None,:,None]
    lower = a[:,y1[:,None],x0[None,:],:]*(1-wx)[None,None,:,None] + a[:,y1[:,None],x1[None,:],:]*wx[None,None,:,None]
    return upper*(1-wy)[None,:,None,None] + lower*wy[None,:,None,None]


def make_case(spec):
    p = parameters(spec, modes=False)
    if any(p.get(k, False) for k in ('align_corners','half_pixel_centers','antialias')):
        return {'expected_host_rejection': True}
    rng = torch.Generator(device='cpu').manual_seed(spec.get('seed',0))
    x = torch.rand(tuple(p[k] for k in ('N','H','W','C')), generator=rng, dtype=torch.float32, device='cpu')*2-1
    return {'x': x, 'expected': reference(x,p)}

CASE_SOURCES = [Path(__file__)]


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.save(make_case(json.loads(args.case.read_text())), args.output)

if __name__ == '__main__':
    main()
