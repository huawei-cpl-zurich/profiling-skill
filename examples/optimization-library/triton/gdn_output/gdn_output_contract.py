"""Bounded prepared GDN output inputs and independent typed FP64 oracle."""
import math
import torch
CANARY=1000
ATOL=2e-4
RTOL=2e-3


def parameters(spec):
    p=spec.get('parameters',spec);t=p.get('T')
    if type(t) is not int or not 1<=t<=2048:raise ValueError('bounded positive T required')
    for k,v in dict(B=1,H=8,Hg=8,K=128,V=128,BT=64,sequence_count=1).items():
        if type(p.get(k,v)) is not int or p.get(k,v)!=v:raise ValueError('B1/single/H=Hg8/K=V128/BT64 required')
    if p.get('packed') is not True or p.get('cu_seqlens')!=[0,t] or any(type(x) is not int for x in p['cu_seqlens']):raise ValueError('ONLY packed single positive cu=[0,T]')
    scale=p.get('scale')
    if type(scale) not in (int,float) or scale!=.5:raise ValueError('explicit scale .5 required')
    return t,(t+63)//64


def layout(t):return dict(q=((1,t,8,128),torch.float16),k=((1,t,8,128),torch.float16),v=((1,t,8,128),torch.float16),h=((1,(t+63)//64,8,128,128),torch.float16),g=((1,8,t),torch.float32),cu=((2,),torch.int64),offsets=((2,),torch.int64))
def backing(x):return x.as_strided((x.untyped_storage().nbytes()//x.element_size(),),(1,),0)
def halo(x):return 128//x.element_size()


def guarded(x):
    h=halo(x);store=torch.full((x.numel()+2*h,),CANARY,dtype=x.dtype,device='cpu');out=store[h:-h].view(x.shape);out.copy_(x);return out


def gate(t):return (-((torch.arange(t,device='cpu')%64)[None,:]+1)*(torch.arange(8,device='cpu')[:,None]+1)/1024).to(torch.float32).unsqueeze(0)


def check_fixture(spec,fixture):
    t,nt=parameters(spec);shapes=layout(t)
    if not isinstance(fixture,dict) or set(fixture)!={'inputs','expected'} or set(fixture['inputs'])!=set(shapes):raise ValueError('all component inputs and independent expected required')
    z=fixture['inputs'];owned=[]
    for name,(shape,dtype) in shapes.items():
        x=z[name]
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or x.shape!=shape or x.requires_grad or not x.is_contiguous() or x.storage_offset()!=halo(x) or backing(x).numel()!=x.numel()+2*halo(x):raise ValueError('owned dense CPU input/128B halos: '+name)
        raw=backing(x);h=halo(x)
        if not bool(torch.isfinite(x).all() and (raw[:h]==CANARY).all() and (raw[-h:]==CANARY).all()):raise ValueError('finite inputs and intact canaries required')
        if name in ('q','k','v','h'):
            f=x.float()
            if not bool((f.abs()<=.125).all() and (f*32==f.mul(32).round()).all()):raise ValueError('certified dyadic input domain')
        owned.append(x)
    if not torch.equal(z['g'],gate(t)) or z['cu'].tolist()!=[0,t] or z['offsets'].tolist()!=[0,nt]:raise ValueError('exact nontrivial gates/single-sequence offsets required')
    e=fixture['expected']
    if not isinstance(e,torch.Tensor) or e.device.type!='cpu' or e.dtype!=torch.float16 or e.shape!=(1,t,8,128) or e.requires_grad or not e.is_contiguous() or not bool(torch.isfinite(e).all()):raise ValueError('independent finite FP16 component expected required')
    owned.append(e)
    if len({x.untyped_storage().data_ptr() for x in owned})!=len(owned):raise ValueError('fixture aliases')


def reference(z,t):
    out=torch.empty((1,t,8,128),dtype=torch.float16,device='cpu')
    for start in range(0,t,64):
        end=min(start+64,t);n=end-start
        q=z['q'][0,start:end].permute(1,0,2).double();k=z['k'][0,start:end].permute(1,0,2).double();v=z['v'][0,start:end].permute(1,0,2).double();h=z['h'][0,start//64].double()
        state=(q@h).float();score=(q@k.transpose(1,2)).float();g=z['g'][0,:,start:end]
        state=(state.double()*g.double().exp().float().double()[:,:,None]).float()
        diff=(g[:,:,None]-g[:,None,:]).float();exp=torch.where(diff<=0,diff.double().exp(),0.).float()
        score=(score.double()*exp.double()).float();mask=torch.arange(n,device='cpu')[:,None]>=torch.arange(n,device='cpu')[None,:]
        score=torch.where(mask,score,0.).to(torch.float16)
        local=(score.double()@v).float();state=(state.double()*.5).float();local=(local.double()*.5).float()
        out[0,start:end].copy_((state.double()+local.double()).float().permute(1,0,2).to(torch.float16))
    return out


def make_case(spec):
    t,nt=parameters(spec);rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',0));z={}
    for name in ('q','k','v','h'):
        shape=layout(t)[name][0];z[name]=guarded((torch.randint(-4,5,shape,generator=rng,device='cpu').double()/32).to(torch.float16))
    z['g']=guarded(gate(t));z['cu']=guarded(torch.tensor([0,t],dtype=torch.int64,device='cpu'));z['offsets']=guarded(torch.tensor([0,nt],dtype=torch.int64,device='cpu'))
    result=dict(inputs=z,expected=reference(z,t));check_fixture(spec,result);return result
