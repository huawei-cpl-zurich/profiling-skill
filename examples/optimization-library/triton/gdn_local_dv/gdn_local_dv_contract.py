"""Dense local-dV contract and independent typed FP64 CPU reference."""
import math
import torch
CANARY=1000
ATOL=2e-4
RTOL=2e-3


def parameters(spec):
    p=spec.get('parameters',spec);z={}
    for key,allowed in [('B',(1,2)),('H',(4,8)),('K',(64,128)),('V',(64,128))]:
        n=p.get(key)
        if type(n)is not int or n not in allowed:raise ValueError('bounded dense '+key+' required')
        z[key]=n
    t=p.get('T')
    if type(t)is not int or not 1<=t<=2048:raise ValueError('bounded positive T required')
    z['T']=t;z['BT']=min(64,max(16,1<<(t-1).bit_length()))
    if p.get('BT')!=z['BT'] or p.get('dense')is not True or p.get('USE_G')is not True or p.get('USE_G_GAMMA')is not False or type(p.get('scale'))not in(int,float) or p['scale']!=.5:raise ValueError('source BT/dense/gate/scale policy required')
    if p.get('g_mode')not in('cumulative','zigzag'):raise ValueError('frozen nontrivial gate pattern required')
    z['g_mode']=p['g_mode'];return z


def layout(p):
    b,t,h,k,v=(p[x]for x in('B','T','H','K','V'))
    return dict(q=((b,t,h,k),torch.float16),k=((b,t,h,k),torch.float16),do=((b,t,h,v),torch.float16),g=((b,h,t),torch.float32))
def backing(x):return x.as_strided((x.untyped_storage().nbytes()//x.element_size(),),(1,),0)
def halo(x):return 128//x.element_size()


def guarded(x):
    n=halo(x);store=torch.full((x.numel()+2*n,),CANARY,dtype=x.dtype,device='cpu');out=store[n:-n].view(x.shape);out.copy_(x);return out


def gate(p):
    t=torch.arange(p['T'],device='cpu')[None,:];h=torch.arange(p['H'],device='cpu')[:,None]+1
    raw=-(t%p['BT']+1)*h/1024 if p['g_mode']=='cumulative'else((t%17)-8)*h/256
    return raw.float().unsqueeze(0).repeat(p['B'],1,1)


def check_fixture(spec,fixture):
    p=parameters(spec);shapes=layout(p)
    if not isinstance(fixture,dict)or set(fixture)!={'inputs','expected'}or set(fixture['inputs'])!=set(shapes):raise ValueError('complete component fixture required')
    z=fixture['inputs'];owned=[]
    for name,(shape,dtype)in shapes.items():
        x=z[name]
        if not isinstance(x,torch.Tensor)or x.device.type!='cpu'or x.dtype!=dtype or x.shape!=shape or x.requires_grad or not x.is_contiguous()or x.storage_offset()!=halo(x)or backing(x).numel()!=x.numel()+2*halo(x):raise ValueError('owned CPU dense/128B halos: '+name)
        raw=backing(x);n=halo(x)
        if not bool(torch.isfinite(x).all()and(raw[:n]==CANARY).all()and(raw[-n:]==CANARY).all()):raise ValueError('finite intact input storage required')
        if name!='g':
            f=x.float()
            if not bool((f.abs()<=.125).all()and(f*32==f.mul(32).round()).all()):raise ValueError('certified dyadic domain required')
        owned.append(x)
    if not torch.equal(z['g'],gate(p)):raise ValueError('exact head-major gate pattern required')
    e=fixture['expected'];shape=shapes['do'][0]
    if not isinstance(e,torch.Tensor)or e.device.type!='cpu'or e.dtype!=torch.float16 or e.shape!=shape or e.requires_grad or not e.is_contiguous()or not bool(torch.isfinite(e).all()):raise ValueError('independent finite CPU F16 local-dV required')
    owned.append(e)
    if len({x.untyped_storage().data_ptr()for x in owned})!=len(owned):raise ValueError('fixture aliases')


def reference(z,p):
    b,t,h,v=(p[x]for x in('B','T','H','V'));bt=p['BT'];out=torch.empty((b,t,h,v),dtype=torch.float16,device='cpu')
    for start in range(0,t,bt):
        end=min(start+bt,t);n=end-start
        k=z['k'][:,start:end].permute(0,2,1,3).double();q=z['q'][:,start:end].permute(0,2,1,3).double();do=z['do'][:,start:end].permute(0,2,1,3).double()
        a=(k@q.transpose(-1,-2)).float();g=z['g'][:,:,start:end];diff=(g[:,:,None,:]-g[:,:,:,None]).float();exp=diff.double().exp().float()
        a=(a.double()*exp.double()).float();a=(a.double()*.5).float();mask=torch.arange(n,device='cpu')[:,None]<=torch.arange(n,device='cpu')[None,:]
        a=torch.where(mask,a,0.).to(torch.float16);value=(a.double()@do).float().to(torch.float16)
        out[:,start:end].copy_(value.permute(0,2,1,3))
    return out


def make_case(spec):
    p=parameters(spec);rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',0));z={}
    for name in('q','k','do'):
        z[name]=guarded((torch.randint(-4,5,layout(p)[name][0],generator=rng,device='cpu').double()/32).to(torch.float16))
    z['g']=guarded(gate(p));result=dict(inputs=z,expected=reference(z,p));check_fixture(spec,result);return result
