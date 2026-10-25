"""Independent initial-state FP64 scalar causal convolution; no kernel/provider reuse."""
import torch,math


def parameters(spec):
    p=dict(spec.get('parameters',spec))
    if type(p.get('B')) is not int or type(p.get('D')) is not int or not 1<=p['B']<=64 or not 16<=p['D']<=4096 or p['D']%16:raise ValueError('B1..64 and D16-multiple <=4096 required')
    if any(type(p.get(k)) is not bool for k in ('bias','silu','indexed')) or p.get('seq_len',1)!=1 or p.get('width',4)!=4 or p.get('state_len',3)!=3:raise ValueError('seq1 width4 state3 boolean flags only')
    p['slots']=p.get('slots',p['B']+2 if p['indexed'] else p['B'])
    if type(p['slots']) is not int or not 1<=p['slots']<=66 or (not p['indexed'] and p['slots']!=p['B']):raise ValueError('invalid state slots')
    if 'indices' in p:check_indices(torch.tensor(p['indices']),p,require_dtype=False)
    return p


def block(dim):
    value=dim&-dim;threshold=2048 if value<=2048 else 1024
    while value>threshold:value//=2
    return value


def check_indices(t,p,require_dtype=True):
    if not p['indexed'] or not isinstance(t,torch.Tensor) or t.shape!=(p['B'],) or t.device.type!='cpu' or not t.is_contiguous() or (require_dtype and t.dtype!=torch.int32) or t.dtype not in (torch.int32,torch.int64):raise ValueError('contiguous integer indices required')
    values=t.tolist();active=[v for v in values if v>=0]
    if any(v < -1 or v>=p['slots'] for v in values) or len(active)!=len(set(active)):raise ValueError('unique active in-range indices, only -1 skipped')
    if 'indices' in p and values!=p['indices']:raise ValueError('specified indices mismatch')


def check_inputs(inputs,p):
    shapes=dict(x=(p['B'],p['D']),state=(p['slots'],p['D'],3),weight=(p['D'],4))
    if p['bias']:shapes['bias']=(p['D'],)
    if p['indexed']:shapes['indices']=(p['B'],)
    if set(inputs)!=set(shapes):raise ValueError('exact source input set required')
    pointers=[]
    for name,shape in shapes.items():
        value=inputs[name];dtype=torch.int32 if name=='indices' else torch.float32
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=dtype or tuple(value.shape)!=shape or not value.is_contiguous() or value.requires_grad:raise ValueError('source contiguous CPU shape/dtype required')
        pointers.append(value.untyped_storage().data_ptr())
        if name=='indices':check_indices(value,p);continue
        scale,limit=(128,.5) if name in ('x','state') else (256,.125)
        if not torch.isfinite(value).all() or (value.abs()>limit).any() or not torch.equal(value*scale,(value*scale).round()):raise ValueError('finite bounded dyadic initial inputs required')
    if len(set(pointers))!=len(pointers):raise ValueError('input storage overlaps')


def input_case(spec):
    p=parameters(spec);seed=spec.get('seed',p.get('seed'))
    if type(seed) is not int or seed<0:raise ValueError('nonnegative seed required')
    g=torch.Generator().manual_seed(seed);b,d=p['B'],p['D']
    inputs=dict(x=torch.randint(-64,65,(b,d),generator=g).float()/128,state=torch.randint(-64,65,(p['slots'],d,3),generator=g).float()/128,weight=torch.randint(-32,33,(d,4),generator=g).float()/256)
    if p['bias']:inputs['bias']=torch.randint(-32,33,(d,),generator=g).float()/256
    if p['indexed']:inputs['indices']=torch.tensor(p['indices'],dtype=torch.int32) if 'indices' in p else (torch.randperm(b,generator=g)+1).to(torch.int32)
    check_inputs(inputs,p);return inputs


def reference(inputs,p):
    out=inputs['x'].clone();new_state=inputs['state'].clone();indices=inputs['indices'].tolist() if p['indexed'] else list(range(p['B']))
    for row,slot in enumerate(indices):
        if slot==-1:continue
        for channel in range(p['D']):
            old=[float(inputs['state'][slot,channel,t]) for t in range(3)]+[float(inputs['x'][row,channel])]
            value=sum(old[t]*float(inputs['weight'][channel,t]) for t in range(4))
            if p['bias']:value+=float(inputs['bias'][channel])
            if p['silu']:value=value/(1+math.exp(-value))
            out[row,channel]=value;new_state[slot,channel]=torch.tensor(old[1:],dtype=torch.float32)
    return out,new_state


def make_case(spec):
    inputs=input_case(spec);return dict(inputs=inputs,expected=reference(inputs,parameters(spec)))


def equal(a,b,bits=False):
    if not isinstance(a,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape:return False
    a=a.detach().cpu();return torch.equal(a.view(torch.int32),b.view(torch.int32)) if bits else torch.equal(a,b)
