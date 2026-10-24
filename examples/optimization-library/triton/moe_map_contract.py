"""CPU-only supplied-metadata map admission and integer oracle."""
from pathlib import Path
import torch
MAX_N = 2**23-128


def dimensions(spec):
    p = spec.get('parameters', spec)
    n, s = p.get('N'), p.get('num_splits')
    if (type(n) is not int or not 0 < n <= MAX_N or type(s) is not int or not 0 < s <= 128
            or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('unsupported bounded supplied-metadata map case')
    return p,n,s


def validate_inputs(inputs,spec):
    _,n,s = dimensions(spec)
    shapes = dict(split_sizes=(s,),input_cumsum=(s+1,),inverse_sorted_indices=(s,),output_cumsum=(s+1,))
    if set(inputs) != set(shapes): raise ValueError('metadata keys disagree')
    pointers=set()
    for key,shape in shapes.items():
        v=inputs[key]
        if (not isinstance(v,torch.Tensor) or v.dtype!=torch.int32 or v.device.type!='cpu'
                or tuple(v.shape)!=shape or not v.is_contiguous() or v.storage_offset()!=0
                or v.untyped_storage().nbytes()!=v.numel()*4):
            raise ValueError('owned contiguous CPU int32 metadata required: '+key)
        ptr=v.untyped_storage().data_ptr()
        if ptr in pointers: raise ValueError('metadata alias')
        pointers.add(ptr)
    sizes=inputs['split_sizes'].long(); inv=inputs['inverse_sorted_indices'].long()
    if bool((sizes<0).any()) or int(sizes.sum())!=n: raise ValueError('split sizes must sum to N')
    if not torch.equal(torch.sort(inv).values,torch.arange(s)): raise ValueError('inverse must be a permutation')
    starts=torch.cat((torch.zeros(1,dtype=torch.int64),sizes.cumsum(0)))
    order=torch.argsort(inv)
    output=torch.cat((torch.zeros(1,dtype=torch.int64),sizes[order].cumsum(0)))
    if not torch.equal(starts,inputs['input_cumsum'].long()) or not torch.equal(output,inputs['output_cumsum'].long()):
        raise ValueError('inconsistent supplied prefixes')


def reference(inputs):
    # Vectorized per chunk; the precision guard has one ~32 MB int32 payload.
    sizes=inputs['split_sizes'].tolist(); inv=inputs['inverse_sorted_indices'].tolist()
    output=inputs['output_cumsum'].tolist()
    return torch.cat([torch.arange(output[inv[i]],output[inv[i]]+size,dtype=torch.int32)
                      for i,size in enumerate(sizes)])


def make_case(spec):
    p,n,s=dimensions(spec)
    rng=torch.Generator().manual_seed(spec['seed'])
    sizes=p.get('split_sizes'); order=p.get('sorted_indices')
    if sizes is None:
        kind=p.get('map_kind','identity')
        if kind not in ('identity','reverse','uneven_zero_shuffle'): raise ValueError('unsupported map kind')
        if kind=='uneven_zero_shuffle':
            weights=torch.randint(0,17,(s,),generator=rng,dtype=torch.int64);weights[0]=1
            counts=weights*n//int(weights.sum());counts[0]+=n-int(counts.sum());sizes=counts.tolist()
        else: sizes=[n//s+int(i<n%s) for i in range(s)]
        if order is None:
            order=(torch.randperm(s,generator=rng).tolist() if kind=='uneven_zero_shuffle'
                   else list(reversed(range(s))) if kind=='reverse' else list(range(s)))
    if (not isinstance(sizes,list) or len(sizes)!=s or any(type(x) is not int or x<0 or x>n for x in sizes)
            or sum(sizes)!=n or not isinstance(order,list) or any(type(x) is not int for x in order)
            or sorted(order)!=list(range(s))): raise ValueError('invalid split/permutation specification')
    sizes=torch.tensor(sizes,dtype=torch.int32); order=torch.tensor(order)
    inv=torch.empty(s,dtype=torch.int32);inv[order]=torch.arange(s,dtype=torch.int32)
    inputs=dict(split_sizes=sizes,input_cumsum=torch.cat((torch.zeros(1,dtype=torch.int32),sizes.cumsum(0,dtype=torch.int32))),
        inverse_sorted_indices=inv,output_cumsum=torch.cat((torch.zeros(1,dtype=torch.int32),sizes[order].cumsum(0,dtype=torch.int32))))
    validate_inputs(inputs,spec)
    return dict(inputs=inputs,expected=reference(inputs))


CASE_SOURCES=[Path(__file__)]
