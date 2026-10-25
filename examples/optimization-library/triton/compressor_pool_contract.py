"""Independent bounded projected-KV pooling fixtures and BF16-boundary oracle."""
from pathlib import Path
import torch

ATOL, RTOL = .03125, .015625


def dimensions(spec):
    p=spec.get('parameters',spec)
    g,ratio,start,seed=(p.get('G'),p.get('ratio'),p.get('start_pos'),spec.get('seed'))
    if any(type(x) is not int for x in (g,ratio,start,seed)) or seed<0:
        raise ValueError('integer G/ratio/start_pos and nonnegative seed required')
    residual=p.get('residual',False)
    if type(residual) is not bool or ratio not in (2,3) or not 1<=g<=1024:
        raise ValueError('G1..1024, ratio2/3 and boolean residual required')
    used=p.get('used_len',g*ratio)
    if type(used) is not int:raise ValueError('integer used_len required')
    if residual:
        if (g,ratio,start,used)!=(1,2,1,1):raise ValueError('residual envelope: exactly one cached and one current token')
    elif start!=0 or used!=g*ratio:
        raise ValueError('nonresidual segment must contain exactly G complete groups at start0')
    return g,ratio,start,used,residual,seed


def reference(kv,score,weight,cache,shape):
    g,r,_,_,res,_=shape
    if res:
        values=torch.stack((cache[0,0,:512],kv[0]))[None].double()
        scores=torch.stack((cache[0,0,512:],score[0]))[None].double()
    else:
        values=kv.reshape(g,r,512).double();scores=score.reshape(g,r,512).double()
    probs=torch.softmax(scores,dim=1)
    rounded=(probs*values).sum(1).to(torch.bfloat16).double()
    return (rounded*torch.rsqrt(rounded.square().mean(1,keepdim=True)+1e-6)*weight.double()).to(torch.bfloat16)


def make_case(spec):
    shape=dimensions(spec);g,r,_,used,res,seed=shape
    rng=torch.Generator().manual_seed(seed)
    kv=torch.randint(-128,129,(used,512),generator=rng).float()/128
    score=torch.randint(-192,193,(used,512),generator=rng).float()/64
    weight=(.5+torch.randint(0,65,(512,),generator=rng).float()/128).bfloat16()
    cache=torch.empty((1,8,1024),dtype=torch.float32) if res else None
    if res:
        cache[:,:,:512]=torch.randint(-128,129,(1,8,512),generator=rng).float()/128
        cache[:,:,512:]=torch.randint(-192,193,(1,8,512),generator=rng).float()/64
    return dict(kv=kv,score=score,weight=weight,cache=cache,expected=reference(kv,score,weight,cache,shape))


def check_fixture(spec,data):
    shape=dimensions(spec);g,r,_,used,res,_=shape
    if set(data)!={'kv','score','weight','cache','expected'}:raise ValueError('exact fixture arity required')
    names=dict(kv=((used,512),torch.float32),score=((used,512),torch.float32),
               weight=((512,),torch.bfloat16),expected=((g,512),torch.bfloat16))
    if res:names['cache']=((1,8,1024),torch.float32)
    elif data['cache'] is not None:raise ValueError('nonresidual cache must be None')
    pointers=[]
    for name,(size,dtype) in names.items():
        x=data[name]
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=size or not x.is_contiguous() or not torch.isfinite(x).all():
            raise ValueError('finite CPU contiguous fixture shape/dtype required: '+name)
        pointers.append(x.untyped_storage().data_ptr())
    if len(set(pointers))!=len(pointers):raise ValueError('independent input/output storages required')
    for name,bound,denom in [('kv',1,128),('score',3,64)]:
        x=data[name]
        if not (x.abs()<=bound).all() or not torch.equal(x*denom,(x*denom).round()):raise ValueError('bounded dyadic '+name+' required')
    w=data['weight'].float()
    if not ((w>=.5)&(w<=1)).all() or not torch.equal(w*128,(w*128).round()):raise ValueError('bounded dyadic norm weights required')
    if res:
        for x,bound,denom in [(data['cache'][...,:512],1,128),(data['cache'][...,512:],3,64)]:
            if not (x.abs()<=bound).all() or not torch.equal(x*denom,(x*denom).round()):raise ValueError('bounded dyadic cache required')
    return shape


CASE_SOURCES=[Path(__file__)]
