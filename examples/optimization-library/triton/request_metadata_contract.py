"""Exact CPU integer request-index mapping and overflow-safe host admission."""
from pathlib import Path
import torch
CASE_SOURCES=[Path(__file__)]

def dimensions(spec):
    p=spec.get('parameters',spec);t,w,m,r=(p.get(x) for x in ('tokens','topK','max_blocks','requests'))
    if (any(type(x) is not int or x<=0 for x in (t,w,m,r)) or w%128
            or type(p.get('HAS_PREFILL')) is not bool or type(spec.get('seed')) is not int or spec['seed']<0
            or any(type(p.get(x,False)) is not bool for x in ('invalid_boundaries','large_integer_indices'))):
        raise ValueError('positive dimensions, topK multiple128, bool prefill and seed required')
    if max(t*w*4,r*m*4,m*64+2097152)>=2**31:raise ValueError('shape or generated workspace arithmetic overflows int32')
    if p.get('large_integer_indices') and m<262146:raise ValueError('real large-index table required')
    return t,w,m,r

def validate_inputs(v,spec):
    t,w,m,r=dimensions(spec);pre=spec.get('parameters',spec)['HAS_PREFILL']
    shapes={'R':(t,),'B':(r,m),'I':(t,w)}
    if pre:shapes.update(P=(t,),W=(3,))
    if not isinstance(v,dict) or set(v)!=set(shapes):raise ValueError('request/table/index and optional P/W inputs required')
    for n,x in v.items():
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.int32 or tuple(x.shape)!=shapes[n]
                or not x.is_contiguous() or x.storage_offset()!=0 or x.untyped_storage().nbytes()!=x.numel()*4):
            raise ValueError('owned contiguous CPU int32 inputs required')
    if len({x.untyped_storage().data_ptr() for x in v.values()})!=len(v):raise ValueError('input alias')
    if not bool(((v['R']>=0)&(v['R']<r)).all()):raise ValueError('request range')
    if not bool(((v['B']>=0)&(v['B']<=100000)).all()):raise ValueError('page domain')
    if pre:
        if not bool(((v['P']>=-1)&(v['P']<=2)).all()) or not torch.equal(v['W'],torch.tensor([0,1048576,2097152],dtype=torch.int32)):raise ValueError('prefill format')
    lo,hi=int(v['I'].min()),int(v['I'].max())
    if hi+2097152>2**31-1 or ((r-1)*m+hi//64)*4>2**31-1 or (lo//64)*4 < -2**31:
        raise ValueError('masked pointers or workspace addition overflow int32')

def oracle(v):
    # Independent int64 CPU indexing: gather ONLY valid blocks, then prefill/decode.
    idx=v['I'].long();expected=torch.full_like(v['I'],-1)
    for row in range(idx.shape[0]):
        tok=idx[row];valid=(tok>=0)&(tok//64<v['B'].shape[1])
        pid=int(v['P'][row]) if 'P' in v else -1
        value=(int(v['W'][pid])+tok[valid] if pid>=0 else
               v['B'][int(v['R'][row]),tok[valid]//64].long()*64+tok[valid]%64)
        expected[row,valid]=value.to(torch.int32)
    return expected

def make_case(spec):
    t,w,m,r=dimensions(spec);p=spec.get('parameters',spec);rng=torch.Generator().manual_seed(spec['seed'])
    v=dict(R=torch.randint(0,r,(t,),dtype=torch.int32,generator=rng),
        B=torch.randint(0,100001,(r,m),dtype=torch.int32,generator=rng),
        I=torch.randint(0,m*64,(t,w),dtype=torch.int32,generator=rng))
    flat=v['I'].view(-1);flat[::17]=-1;flat[::29]=-7;flat[::31]=m*64
    if p.get('invalid_boundaries'):
        pattern=torch.tensor([-1,-7,m*64-1,m*64,0,63,64],dtype=torch.int32)
        flat.copy_(pattern.repeat((flat.numel()+6)//7)[:flat.numel()])
    if p.get('large_integer_indices'):
        pattern=torch.tensor([16777215,16777216,16777217,16777343,-1,-7,m*64],dtype=torch.int32)
        flat.copy_(pattern.repeat((flat.numel()+6)//7)[:flat.numel()])
    if p['HAS_PREFILL']:
        row=torch.arange(t,dtype=torch.int32)
        v.update(P=torch.where(row%3==0,(row//3)%3,-1),W=torch.tensor([0,1048576,2097152],dtype=torch.int32))
    validate_inputs(v,spec)
    return dict(inputs=v,expected=oracle(v))

def check_output(actual,expected):
    if not isinstance(actual,torch.Tensor) or actual.dtype!=torch.int32 or actual.shape!=expected.shape or not actual.is_contiguous():
        return dict(passed=False,reason='one contiguous int32 output required')
    same=torch.equal(actual.detach().cpu(),expected)
    return dict(passed=same,exact_integer_bits=same)
