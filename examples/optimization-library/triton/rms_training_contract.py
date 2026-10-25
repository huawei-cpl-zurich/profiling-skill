"""Independent full RMSNorm FP64 oracle; fixed quantized FP32 input domain."""
from pathlib import Path
import torch


def dimensions(spec):
    p=spec.get('parameters',spec);rows,d=p.get('rows'),p.get('D')
    if (type(rows) is not int or not 1<=rows<=32768 or type(d) is not int
            or not 1<=d<=2048 or type(p.get('offset')) is not int or p['offset'] not in (0,1)
            or p.get('eps',1e-6)!=1e-6 or type(spec.get('seed')) is not int or spec['seed']<0):
        raise ValueError('unsupported full RMS training case')
    if p.get('zero_cancel') and rows<3:raise ValueError('zero/cancellation guard requires three rows')
    return p,rows,d


def validate_inputs(inputs,spec):
    _,rows,d=dimensions(spec)
    shapes=dict(X=(rows,d),DY=(rows,d),W=(d,))
    if set(inputs)!=set(shapes):raise ValueError('X,DY,W inputs required')
    pointers=set()
    for key,shape in shapes.items():
        v=inputs[key]
        if (not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float32
                or tuple(v.shape)!=shape or not v.is_contiguous() or v.storage_offset()!=0
                or v.untyped_storage().nbytes()!=v.numel()*4 or not torch.isfinite(v).all()
                or not (v.abs()<=1).all() or not ((v*128)==(v*128).trunc()).all()):
            raise ValueError('owned finite contiguous quantized1/128 FP32 input required: '+key)
        pointer=v.untyped_storage().data_ptr()
        if pointer in pointers:raise ValueError('aliased inputs')
        pointers.add(pointer)


def make_case(spec):
    p,rows,d=dimensions(spec);rng=torch.Generator().manual_seed(spec['seed'])
    inputs={k:torch.randint(-128,129,shape,generator=rng).float()/128
            for k,shape in dict(X=(rows,d),DY=(rows,d),W=(d,)).items()}
    if p.get('zero_cancel'):
        inputs['X'][0].zero_();inputs['X'][1].fill_(.25);inputs['X'][2].fill_(-.25)
        inputs['DY'][1:3].fill_(.25)
    validate_inputs(inputs,spec)
    x,dy,w=(inputs[k].double() for k in ('X','DY','W'))
    r=(x.square().mean(-1,keepdim=True)+1e-6).rsqrt();m=dy*(w+p['offset'])
    y=x*r*(w+p['offset']);dx=r*m-r.pow(3)*x*(m*x).mean(-1,keepdim=True)
    dw=(dy*x*r).sum(0)
    return dict(inputs=inputs,expected=(y,dx,dw))


def check_outputs(actual,expected):
    if not isinstance(actual,tuple) or len(actual)!=3:
        return dict(passed=False,reason='output must be (Y,DX,DW) tuple')
    rows=[];pointers=set()
    for got,wanted,atol,rtol in zip(actual,expected,(2e-5,1e-4,5e-4),(2e-5,1e-5,3e-5)):
        if (not isinstance(got,torch.Tensor) or got.dtype!=torch.float32
                or got.shape!=wanted.shape or not got.is_contiguous()):
            return dict(passed=False,reason='output FP32 shape/layout mismatch')
        pointers.add(got.untyped_storage().data_ptr());cpu=got.detach().cpu().double()
        rows.append(dict(passed=bool(torch.isfinite(cpu).all() and torch.allclose(cpu,wanted,atol=atol,rtol=rtol)),
                         max_abs_error=float((cpu-wanted).abs().max()),atol=atol,rtol=rtol))
    return dict(passed=len(pointers)==3 and all(r['passed'] for r in rows),outputs=rows,
                internal_rstd_observed=False,dx_dw_bitwise_guarantee=False)


CASE_SOURCES=[Path(__file__)]
