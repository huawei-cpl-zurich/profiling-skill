"""Bounded featurewise KKT inputs and independent ideal/typed CPU references."""
import torch
CANARY=1000.;ATOL=1e-4;RTOL=1e-3


def dimensions(spec):
    p=spec.get('parameters',spec);t=p.get('T');k=p.get('K');bt=p.get('BT')
    if type(t)is not int or not 1<=t<=1024 or type(k)is not int or k not in (64,128) or type(bt)is not int or bt not in (32,64,128,256):raise ValueError('bounded T/K/BT required')
    for name,value in dict(B=1,H=1,BC=16).items():
        if type(p.get(name,value))is not int or p.get(name,value)!=value:raise ValueError('B=H1/BC16 required')
    if p.get('featurewise')is not True or p.get('varlen')is not False or p.get('precision')!='ieee':raise ValueError('dense featurewise IEEE component only')
    return t,k,bt


def backing(t):return t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0)
def guarded(v):
    raw=torch.full((v.numel()+64,),CANARY,dtype=torch.float32,device='cpu');out=raw[32:-32].view(v.shape);out.copy_(v);return out


def gate(t,k,bt):
    row=torch.arange(t,device='cpu')[:,None];col=torch.arange(k,device='cpu')[None,:]
    return (((row%bt)*((col%7)-3)+((row%5)-2)*((col%3)-1)).double()/8192).float().view(1,t,1,k)


def shapes(t,k):return dict(k=(1,1,t,k),gk=(1,t,1,k),beta=(1,t,1))


def check_fixture(spec,f):
    t,k,bt=dimensions(spec)
    if not isinstance(f,dict) or set(f)!={'inputs','expected'} or set(f['inputs'])!={'k','gk','beta'} or set(f['expected'])!={'ideal','typed'}:raise ValueError('all inputs and ideal/typed independent expected required')
    tensors=[]
    for name,shape in shapes(t,k).items():
        x=f['inputs'][name]
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.shape!=shape or not x.is_contiguous() or x.requires_grad
                or x.storage_offset()!=32 or backing(x).numel()!=x.numel()+64):raise ValueError('owned dense CPU F32/128B halos required')
        raw=backing(x)
        if not bool(torch.isfinite(x).all() and (raw[:32]==CANARY).all() and (raw[-32:]==CANARY).all()):raise ValueError('input finite/halo')
        tensors.append(x)
    z=f['inputs'];x=z['k'];beta=z['beta']
    if not bool((x.abs()<=.25).all() and (x*32==torch.round(x*32)).all()):raise ValueError('K dyadic domain')
    if not bool((beta>=.25).all() and (beta<=.75).all() and (beta*16==torch.round(beta*16)).all()):raise ValueError('beta dyadic domain')
    if not torch.equal(z['gk'],gate(t,k,bt)):raise ValueError('bounded featuregate pattern')
    for e in f['expected'].values():
        if not isinstance(e,torch.Tensor) or e.device.type!='cpu' or e.dtype!=torch.float32 or e.shape!=(1,t,1,bt) or not e.is_contiguous() or not bool(torch.isfinite(e).all()):raise ValueError('both independent F32 expected tensors required')
        tensors.append(e)
    if len({x.untyped_storage().data_ptr() for x in tensors})!=len(tensors):raise ValueError('fixture aliases')


def reference(z,t,k,bt):
    ideal=torch.zeros((t,bt),dtype=torch.float64,device='cpu');typed=ideal.clone();keys=z['k'].view(t,k).double();g=z['gk'].view(t,k).double();beta=z['beta'].view(t).double()
    for start in range(0,t,bt):
        n=min(bt,t-start);a=keys[start:start+n];gs=g[start:start+n];be=beta[start:start+n]
        exp=(gs[:,None,:]-gs[None,:,:]).exp();values=(a[:,None,:]*a[None,:,:]*exp).sum(-1)*be[:,None]
        mask=torch.arange(n)[:,None]>torch.arange(n)[None,:];ideal[start:start+n,:n]=torch.where(mask,values,0.)
        for row in range(0,n,16):
            nr=min(16,n-row)
            for col in range(0,row,16):
                nc=min(16,n-col);pivot=gs[row]
                left=(a[row:row+nr]*(gs[row:row+nr]-pivot).exp().float().double()).float().double()
                right=(a[col:col+nc]*(pivot-gs[col:col+nc]).exp().float().double()).float().double()
                dot=(left@right.T).float().double();typed[start+row:start+row+nr,col:col+nc]=(dot*be[row:row+nr,None]).float().double()
            kb=(a[row:row+nr]*be[row:row+nr,None]).float().double()
            for col in range(row,row+nr):
                product=(kb*a[col]).float().double();factor=(gs[row:row+nr]-gs[col]).exp().float().double()
                value=(product*factor).float().double().sum(-1).float().double();rows=torch.arange(row,row+nr)
                typed[start+row:start+row+nr,col]=torch.where(rows>col,value,0.)
    return dict(ideal=ideal.float().view(1,t,1,bt),typed=typed.float().view(1,t,1,bt))


def make_case(spec):
    t,k,bt=dimensions(spec);rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',0))
    z=dict(k=guarded((torch.randint(-8,9,(1,1,t,k),generator=rng,device='cpu').double()/32).float()),gk=guarded(gate(t,k,bt)),beta=guarded((torch.randint(4,13,(1,t,1),generator=rng,device='cpu').double()/16).float()))
    f=dict(inputs=z,expected=reference(z,t,k,bt));check_fixture(spec,f);return f
