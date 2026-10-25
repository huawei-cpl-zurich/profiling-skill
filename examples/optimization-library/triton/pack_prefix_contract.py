"""Independent CPU serial-prefix packing; no device work or cached GPU metadata."""
import torch


def parameters(spec):
    p=dict(spec.get('parameters',spec))
    for key in ('B','Lmax','D'):
        if type(p.get(key)) is not int:raise ValueError('integer B/Lmax/D required')
    if not (1<=p['B']<=256 and 1<=p['Lmax']<=128 and 1<=p['D']<=256):
        raise ValueError('outside verified pack domain')
    lengths=p.get('lengths')
    if lengths is not None and (not isinstance(lengths,list) or len(lengths)!=p['B'] or
            any(type(v) is not int or not 0<=v<=p['Lmax'] for v in lengths) or max(lengths)!=p['Lmax']):
        raise ValueError('invalid lengths or Lmax')
    return p


def check_inputs(inputs,p):
    if set(inputs)!={'x','lengths'}:raise ValueError('X and lengths required')
    x,l=inputs['x'],inputs['lengths']
    for t,dtype in ((x,torch.float32),(l,torch.int32)):
        if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=dtype or not t.is_contiguous():
            raise ValueError('contiguous CPU FP32 X/int32 lengths required')
    if l.shape!=(p['B'],) or bool((l<0).any()) or bool((l>p['Lmax']).any()) or int(l.max())!=p['Lmax']:
        raise ValueError('invalid lengths/max')
    n=int(l.sum())
    if not 0<n<=32768 or x.shape!=(n,p['D']):raise ValueError('sum lengths/X mismatch')
    if 'lengths' in p and l.tolist()!=p['lengths']:raise ValueError('specified lengths mismatch')
    if not bool(torch.isfinite(x).all()) or bool((x.abs()>2).any()) or not bool((x*64==(x*64).round()).all()):
        raise ValueError('X must be finite q/64 in [-2,2]')
    return n


def input_case(spec):
    p=parameters(spec);seed=spec.get('seed',p.get('seed'))
    if type(seed) is not int or seed<0:raise ValueError('nonnegative integer seed required')
    g=torch.Generator().manual_seed(seed)
    lengths=p.get('lengths')
    if lengths is None:
        lengths=torch.randint(0,p['Lmax']+1,(p['B'],),generator=g).tolist()
        lengths[0]=0;lengths[-1]=p['Lmax']
    l=torch.tensor(lengths,dtype=torch.int32);n=int(l.sum())
    x=torch.randint(-128,129,(n,p['D']),generator=g).float()/64
    if p.get('signed_zero'):x.fill_(-0.)
    return dict(x=x,lengths=l)


def reference(inputs,p):
    n=check_inputs(inputs,p);starts=[0]
    packed=torch.full((p['B'],p['Lmax'],p['D']),-9.5,dtype=torch.float32)
    roundtrip=torch.empty_like(inputs['x'])
    for b,length in enumerate(inputs['lengths'].tolist()):
        start=starts[-1];starts.append(start+length)
        for t in range(length):
            packed[b,t].copy_(inputs['x'][start+t]);roundtrip[start+t].copy_(packed[b,t])
    assert starts[-1]==n
    return (packed,roundtrip),torch.tensor(starts,dtype=torch.int32)


def make_case(spec):
    inputs=input_case(spec);expected,prefix=reference(inputs,parameters(spec))
    return dict(inputs=inputs,expected=expected,expected_prefix=prefix)


def bits_equal(a,b):
    return isinstance(a,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().cpu().view(torch.int32),b.view(torch.int32))
