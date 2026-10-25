"""Independent integer list-expansion reference and CPU-only prepared domain."""
import torch
CANARY=0x13572468


def i32(x):
    if type(x) is not int or not -2**31<=x<2**31:raise ValueError('exact signed int32 scalar required')
    return x


def dimensions(spec):
    p=spec.get('parameters',spec);b=p.get('batch');mode=p.get('length_mode');n=p.get('num_tokens')
    if type(b) is not int or not 1<=b<=65536:raise ValueError('nonempty bounded batch required')
    if mode=='one':lengths=[1]*b
    elif mode=='full':lengths=[32]*b
    elif mode=='sparse128':lengths=[int(i%128==0) for i in range(b)]
    elif mode=='mixed' and b==8:lengths=[0,1,31,32,0,1,32,0]
    elif mode=='explicit':lengths=p.get('lengths')
    else:raise ValueError('bounded length mode required')
    if (not isinstance(lengths,list) or len(lengths)!=b or any(type(v) is not int or not 0<=v<=32 for v in lengths)
            or type(n) is not int or n!=sum(lengths) or not 1<=n<=2097152 or n+31>=2**31):raise ValueError('interval0..32/nonempty/final token extent required')
    rf,rt=i32(p.get('replace_from')),i32(p.get('replace_to'))
    if rf==rt:raise ValueError('distinct replacement scalars required')
    return b,n,lengths,rf,rt


def backing(t):return t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0)


def check_fixture(spec,f):
    b,n,lengths,rf,rt=dimensions(spec)
    if not isinstance(f,dict) or set(f)!={'inputs','expected'} or not isinstance(f['inputs'],dict) or set(f['inputs'])!={'x','ends'}:raise ValueError('x/endpoints/expected fixture required')
    tensors=[]
    for name in ('x','ends'):
        t=f['inputs'][name]
        if (not isinstance(t,torch.Tensor) or t.dtype!=torch.int32 or t.device.type!='cpu' or t.shape!=(b,) or t.stride()!=(1,)
                or t.storage_offset()!=32 or backing(t).numel()!=b+64 or t.requires_grad):raise ValueError('owned contiguous CPU int32 with32-element halos required')
        raw=backing(t)
        if not bool((raw[:32]==CANARY).all() and (raw[-32:]==CANARY).all()):raise ValueError('input halo canary')
        tensors.append(t)
    ends=tensors[1].tolist();last=0;actual=[]
    for end in ends:
        if end<last or end<0 or end-last>32:raise ValueError('nonnegative monotone endpoints interval<=32 required')
        actual.append(end-last);last=end
    if last!=n or actual!=lengths:raise ValueError('final extent/intervals disagree with case')
    e=f['expected']
    if not isinstance(e,torch.Tensor) or e.dtype!=torch.int32 or e.device.type!='cpu' or e.shape!=(n,) or not e.is_contiguous():raise ValueError('exact int32 CPU expected required')
    if len({t.untyped_storage().data_ptr() for t in (*tensors,e)})!=3:raise ValueError('fixture alias')


def make_case(spec):
    b,n,lengths,rf,rt=dimensions(spec);seed=i32(spec.get('seed',0));inputs={}
    vals=[];ends=[];end=0;expected=[]
    for i,length in enumerate(lengths):
        value=rf if i%3==0 else (i*17+seed)%101-50
        if i%3 and value==rf:value=14 if rf!=14 else 15
        vals.append(value);end+=length;ends.append(end)
        expected.extend([rt if value==rf else value]*length)
    for name,values in (('x',vals),('ends',ends)):
        store=torch.full((b+64,),CANARY,dtype=torch.int32,device='cpu');t=store[32:-32]
        t.copy_(torch.tensor(values,dtype=torch.int32,device='cpu'));inputs[name]=t
    f=dict(inputs=inputs,expected=torch.tensor(expected,dtype=torch.int32,device='cpu'));check_fixture(spec,f);return f
