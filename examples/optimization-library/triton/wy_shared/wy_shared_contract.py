"""Bounded packed-single-sequence WY inputs and independent typed FP64 oracle."""
import torch
CANARY=1000
ATOL=2e-4
RTOL=2e-3


def dimensions(spec):
    p=spec.get('parameters',spec);t=p.get('T')
    if type(t) is not int or not 1<=t<=2048:raise ValueError('positive bounded T required')
    for k,v in dict(B=1,H=8,Hg=8,K=128,V=128,BT=64).items():
        if type(p.get(k,v)) is not int or p.get(k,v)!=v:raise ValueError('supported B1/H=Hg8/K=V128/BT64 only')
    cu=p.get('cu_seqlens')
    if (p.get('packed') is not True or p.get('sequence_count',1)!=1 or not isinstance(cu,list)
            or any(type(x) is not int for x in cu) or cu!=[0,t]):raise ValueError('ONLY packed SINGLE nonempty cu=[0,T]; dense/batch/empty general source hazards remain open')
    if p.get('precision','ieee')!='ieee':raise ValueError('common IEEE policy required')
    return t,(t+63)//64


def layout(t):return dict(k=((1,t,8,128),torch.float16),v=((1,t,8,128),torch.float16),A=((1,t,8,64),torch.float16),beta=((1,8,t),torch.float16),g=((1,8,t),torch.float32),cu=((2,),torch.int64),indices=(((t+63)//64,2),torch.int64))
def backing(t):return t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0)
def halo(t):return 128//t.element_size()


def guarded(values):
    h=128//values.element_size();store=torch.full((values.numel()+2*h,),CANARY,dtype=values.dtype,device='cpu');out=store[h:-h].view(values.shape);out.copy_(values);return out


def patterns(t):
    row=torch.arange(t,device='cpu')%64;col=torch.arange(64,device='cpu');head=torch.arange(8,device='cpu')
    a=torch.where(col[None,None,:]==row[:,None,None],1.,torch.where(col[None,None,:]<row[:,None,None],((row[:,None,None]+col[None,None,:]+head[None,:,None])%3-1).double()/512,0.)).to(torch.float16).unsqueeze(0)
    g=(-(row[None,:]+1)*(head[:,None]+1)/1024).to(torch.float32).unsqueeze(0)
    return a,g


def check_fixture(spec,f):
    t,nt=dimensions(spec);shapes=layout(t)
    if not isinstance(f,dict) or set(f)!={'inputs','expected'} or set(f['inputs'])!=set(shapes) or set(f['expected'])!={'w','u'}:raise ValueError('all WY inputs and BOTH expected outputs required')
    tensors=[]
    for name,(shape,dtype) in shapes.items():
        x=f['inputs'][name]
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or x.shape!=shape or not x.is_contiguous()
                or x.storage_offset()!=halo(x) or backing(x).numel()!=x.numel()+2*halo(x) or x.requires_grad):raise ValueError('exact owned dense typed input/128B halos required: '+name)
        raw=backing(x);h=halo(x)
        if not bool((raw[:h]==CANARY).all() and (raw[-h:]==CANARY).all() and torch.isfinite(x).all()):raise ValueError('input canaries/finiteness')
        tensors.append(x)
    z=f['inputs'];a,g=patterns(t)
    if not torch.equal(z['A'],a) or not torch.equal(z['g'],g):raise ValueError('wellconditioned full64-column A/negative chunk-cumulative g certificate')
    for k in ('k','v'):
        x=z[k].float()
        if not bool((x.abs()<=.25).all() and (x*32==torch.round(x*32)).all()):raise ValueError('dyadic K/V domain')
    x=z['beta'].float()
    if not bool((x>=.25).all() and (x<=.75).all() and (x*16==torch.round(x*16)).all()):raise ValueError('dyadic beta domain')
    if z['cu'].tolist()!=[0,t] or z['indices'].tolist()!=[[0,i] for i in range(nt)]:raise ValueError('exact single-sequence source-equivalent chunk metadata required')
    for e in f['expected'].values():
        if not isinstance(e,torch.Tensor) or e.device.type!='cpu' or e.dtype!=torch.float16 or e.shape!=(1,t,8,128) or not e.is_contiguous() or not bool(torch.isfinite(e).all()):raise ValueError('BOTH final FP16 expected matrices required')
        tensors.append(e)
    if len({x.untyped_storage().data_ptr() for x in tensors})!=len(tensors):raise ValueError('input/output fixture aliases')


def reference(z,t):
    result={k:torch.empty((1,t,8,128),dtype=torch.float16,device='cpu') for k in ('w','u')}
    for start in range(0,t,64):
        end=min(start+64,t);n=end-start
        a=torch.zeros((8,64,64),dtype=torch.float64,device='cpu');a[:,:n,:]=z['A'][0,start:end].permute(1,0,2).double()
        beta=z['beta'][0,:,start:end].double();gate=z['g'][0,:,start:end].double().exp().float().double()
        for name,key in (('w','k'),('u','v')):
            data=z[key][0,start:end].permute(1,0,2).double();product=(data*beta[:,:,None]).float()
            if name=='w':product=(product.double()*gate[:,:,None]).float()
            rhs=torch.zeros((8,64,128),dtype=torch.float64,device='cpu');rhs[:,:n,:]=product.double()
            result[name][0,start:end].copy_((a@rhs)[:,:n,:].permute(1,0,2).to(torch.float16))
    return result


def make_case(spec):
    t,nt=dimensions(spec);rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',0));z={}
    for k in ('k','v'):z[k]=guarded((torch.randint(-8,9,(1,t,8,128),generator=rng,device='cpu').double()/32).to(torch.float16))
    z['beta']=guarded((torch.randint(4,13,(1,8,t),generator=rng,device='cpu').double()/16).to(torch.float16));a,g=patterns(t);z['A']=guarded(a);z['g']=guarded(g)
    z['cu']=guarded(torch.tensor([0,t],dtype=torch.int64,device='cpu'));z['indices']=guarded(torch.tensor([[0,i] for i in range(nt)],dtype=torch.int64,device='cpu'))
    f=dict(inputs=z,expected=reference(z,t));check_fixture(spec,f);return f
