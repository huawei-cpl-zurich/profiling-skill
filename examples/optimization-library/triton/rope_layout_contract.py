"""Independent exact-dyadic FP64 half-rotation and physical producer contract."""
import torch,math


def parameters(spec):
    p=dict(spec.get('parameters',spec))
    if any(type(p.get(k)) is not int for k in ('B','S','Hq','Hk','D','cos_B')):raise ValueError('integer shape required')
    if not (1<=p['B']<=2 and 2<=p['S']<=256 and 2<=p['Hq']<=16 and 1<=p['Hk']<=4 and 2<=p['D']<=64 and p['D']%2==0 and p['cos_B'] in (1,p['B']) and p.get('producer') in ('BHSD','BSHD-backing')):
        raise ValueError('outside admitted forward layout domain')
    return p


def strides(p,h):
    b,s,d=p['B'],p['S'],p['D']
    return (h*s*d,s*d,d,1) if p['producer']=='BHSD' else (s*h*d,d,h*d,1)


def check_inputs(inputs,p):
    if set(inputs)!={'q','k','cos','sin'}:raise ValueError('four inputs required')
    pointers=[]
    for name,t in inputs.items():
        if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=torch.float32 or t.requires_grad or not torch.isfinite(t).all():raise ValueError('finite CPU FP32 inputs required')
        pointers.append(t.untyped_storage().data_ptr())
        if name in ('q','k'):
            h=p['Hq'] if name=='q' else p['Hk']
            if t.shape!=(p['B'],h,p['S'],p['D']) or t.stride()!=strides(p,h):raise ValueError('producer layout mismatch')
            scale=128
        else:
            if t.shape!=(p['cos_B'],p['S'],p['D']) or not t.is_contiguous():raise ValueError('cos/sin layout mismatch')
            half=p['D']//2
            if not torch.equal(t[...,:half].view(torch.int32),t[...,half:].view(torch.int32)):raise ValueError('duplicated trig halves required')
            scale=256
        if (t.abs()>1).any() or not torch.equal(t*scale,(t*scale).round()):raise ValueError('exact dyadic input domain required')
    if len(set(pointers))!=4:raise ValueError('overlapping input storage')


def input_case(spec):
    p=parameters(spec);seed=spec.get('seed',p.get('seed'))
    if type(seed) is not int or seed<0:raise ValueError('nonnegative seed required')
    g=torch.Generator().manual_seed(seed);inputs={}
    for name,h in (('q',p['Hq']),('k',p['Hk'])):
        shape=(p['B'],h,p['S'],p['D']) if p['producer']=='BHSD' else (p['B'],p['S'],h,p['D'])
        t=torch.randint(-128,129,shape,generator=g).float()/128
        inputs[name]=t if p['producer']=='BHSD' else t.transpose(1,2)
    angles=torch.rand((p['cos_B'],p['S'],p['D']//2),generator=g,dtype=torch.float64)*2*math.pi
    for name,v in (('cos',angles.cos()),('sin',angles.sin())):
        half=(v*256).round().float()/256;inputs[name]=torch.cat((half,half),-1)
    return inputs


def reference(inputs,p):
    check_inputs(inputs,p);c=inputs['cos'][...,:p['D']//2].double()[:,None];s=inputs['sin'][...,:p['D']//2].double()[:,None];out=[]
    for name in ('q','k'):
        left,right=inputs[name].double().chunk(2,-1)
        out.append(torch.cat((left*c-right*s,right*c+left*s),-1).float())
    return tuple(out)


def make_case(spec):
    inputs=input_case(spec)
    return dict(inputs=inputs,expected=reference(inputs,parameters(spec)))


def equal(a,b,bits=False):
    if not isinstance(a,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:return False
    a=a.detach().cpu()
    return torch.equal(a.view(torch.int32),b.view(torch.int32)) if bits else torch.equal(a,b)
