"""Independent CPU FP64 dyadic DSA score oracle; no backend imports."""
import torch


def parameters(spec):
    p=spec.get('parameters',spec)
    for k,limit in [('B',2),('M',128),('H',64),('D',128),('N',4096)]:
        if type(p.get(k)) is not int or not 1<=p[k]<=limit:raise ValueError('bounded positive B/M/H/D/N required')
    if p['H']<16 or p['H']&(p['H']-1) or p['D']<32 or p['D']&(p['D']-1) or p.get('dtype','float16')!='float16':raise ValueError('FP16; power-of-two H16..64/D32..128 required')
    return p


def shapes(p):return dict(q=(p['B'],p['M'],p['H'],p['D']),k=(p['B'],p['N'],p['D']),weights=(p['B'],p['M'],p['H']))


def check_inputs(inputs,p):
    if not isinstance(inputs,dict) or set(inputs)!=set(shapes(p)):raise ValueError('exact Q/K/weights input mapping required')
    for name,shape in shapes(p).items():
        x=inputs[name];dtype=torch.float32 if name=='weights' else torch.float16;scale=16 if name=='weights' else 64
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or x.shape!=shape or not x.is_contiguous() or x.requires_grad:raise ValueError('contiguous CPU FP16 Q/K and FP32 weights with exact shapes required')
        if not torch.isfinite(x).all() or (x.abs()>4/scale).any() or not torch.equal(x.float()*scale,(x.float()*scale).round()):raise ValueError('finite bounded integer/64 Q/K and integer/16 weights required')
    if len({x.untyped_storage().data_ptr() for x in inputs.values()})!=3:raise ValueError('input storage aliases forbidden')


def make_case(spec):
    p=parameters(spec);seed=spec.get('seed',p.get('seed'))
    if type(seed) is not int or seed<0:raise ValueError('nonnegative integer seed required')
    g=torch.Generator().manual_seed(seed);inputs={n:(torch.randint(-4,5,s,generator=g).float()/(16 if n=='weights' else 64)).to(torch.float32 if n=='weights' else torch.float16) for n,s in shapes(p).items()}
    if spec.get('weight_pattern')=='materialized_broadcast_heads_rows_0_and_minus4_over16':
        if p['B']!=1 or p['M']!=2:raise ValueError('broadcast-valued guard requires B1/M2')
        inputs['weights']=torch.tensor([0.,-.25],dtype=torch.float32).reshape(1,2,1).expand(1,2,p['H']).contiguous()
    check_inputs(inputs,p)
    q=inputs['q'].reshape(p['B'],p['M']*p['H'],p['D']).double();k=inputs['k'].double().transpose(1,2)
    scores=torch.matmul(q,k).reshape(p['B'],p['M'],p['H'],p['N'])
    exact=(scores.clamp_min(0)*inputs['weights'].double().unsqueeze(-1)).sum(2)
    return dict(inputs=inputs,expected=exact.float(),expected_fp64=exact)


def bits(a,b):
    dtype=torch.int32 if a.dtype==torch.float32 else torch.int16
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().view(dtype),b.detach().cpu().view(dtype))
