"""E11 complete linear/CE/VJP with full versus bounded logits lifetime.

Independent restricted example inspired by pinned fused_linear_cross_enrtopy.py
49-69,116-187. Logical allocation accounting is not a device peak-memory claim.
"""
from functools import lru_cache
from types import SimpleNamespace
import torch
import gemm
from loss_lifetime_contract import CASE_SOURCES, check_spec, diagnostics, make_case


def configuration(arm, spec):
    t, d, v, chunk = check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    rows = t if arm == 'before' else chunk
    return dict(tokens=t, hidden=d, vocab=v, rows=rows, chunks=(t+rows-1)//rows,
                logical_workspace_bytes=6*rows*v+4*t, block_m=16, block_n=32, block_k=32)


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def linear_loss_mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                       AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                       CM: tl.constexpr, CN: tl.constexpr, ADD: tl.constexpr,
                       BM: tl.constexpr, BN_: tl.constexpr, BK_: tl.constexpr):
        m = tl.program_id(0)*BM+tl.arange(0,BM)
        n = tl.program_id(1)*BN_+tl.arange(0,BN_)
        k = tl.arange(0,BK_)
        accumulator = tl.zeros((BM,BN_),tl.float32)
        for tile in range(tl.cdiv(K,BK_)):
            reduction = tile*BK_+k
            a = tl.load(A+m[:,None]*AM+reduction[None,:]*AK,
                        (m[:,None]<M)&(reduction[None,:]<K),0)
            b = tl.load(B+reduction[:,None]*BK+n[None,:]*BN,
                        (reduction[:,None]<K)&(n[None,:]<N),0)
            accumulator += tl.dot(a,b)
        if ADD:
            accumulator += tl.load(C+m[:,None]*CM+n[None,:]*CN,
                                   (m[:,None]<M)&(n[None,:]<N),0)
        tl.store(C+m[:,None]*CM+n[None,:]*CN,accumulator,
                 (m[:,None]<M)&(n[None,:]<N))

    @triton.jit
    def linear_loss_ce(Logits, Target, Gradient, Losses, V: tl.constexpr, BV: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0,BV)
        score = tl.load(Logits+row*V+col,col<V,-float('inf'))
        maximum = tl.max(score,0)
        probabilities = tl.exp(score-maximum)
        denominator = tl.sum(probabilities,0)
        target = tl.load(Target+row)
        target_score = tl.sum(tl.where(col==target,score,0.),0)
        tl.store(Losses+row,maximum+tl.log(denominator)-target_score)
        gradient = probabilities/denominator-tl.where(col==target,1.,0.)
        tl.store(Gradient+row*V+col,gradient.to(tl.float16),col<V)

    @triton.jit
    def linear_loss_reduce(Losses, Loss, T: tl.constexpr, BT: tl.constexpr):
        row = tl.arange(0,BT)
        tl.store(Loss,tl.sum(tl.load(Losses+row,row<T,0),0))

    return linear_loss_mm, linear_loss_ce, linear_loss_reduce


def execute(inputs, outputs, workspace, c, compiled):
    """Every forward writes all outputs and required workspace before reading them."""
    import triton
    loss, dx, dw = outputs
    logits, gradient, losses = workspace
    mm, ce, reduce = kernels()
    t,d,v,rows = (c[k] for k in ('tokens','hidden','vocab','rows'))
    for start in range(0,t,rows):
        count = min(rows,t-start)
        compiled['forward-'+str(count)] = mm[(triton.cdiv(count,16),triton.cdiv(v,32))](
            inputs['x'][start:],inputs['w'],logits,count,v,d,d,1,1,d,v,1,False,16,32,32)
        compiled['ce'] = ce[(count,)](logits,inputs['target'][start:],gradient,losses[start:],
                                     v,triton.next_power_of_2(v))
        compiled['dx-'+str(count)] = mm[(triton.cdiv(count,16),triton.cdiv(d,32))](
            gradient,inputs['w'],dx[start:],count,d,v,v,1,d,1,d,1,False,16,32,32)
        compiled['dw-'+str(count)+'-'+str(start>0)] = mm[(triton.cdiv(v,16),triton.cdiv(d,32))](
            gradient,inputs['x'][start:],dw,v,d,count,1,v,d,1,d,1,start>0,16,32,32)
    compiled['reduce'] = reduce[(1,)](losses,loss,t,triton.next_power_of_2(t))
    return outputs


def allocate(c, device):
    t,d,v,rows = (c[k] for k in ('tokens','hidden','vocab','rows'))
    outputs = (torch.empty(1,dtype=torch.float32,device=device),
               torch.empty((t,d),dtype=torch.float32,device=device),
               torch.empty((v,d),dtype=torch.float32,device=device))
    workspace = (torch.empty((rows,v),dtype=torch.float32,device=device),
                 torch.empty((rows,v),dtype=torch.float16,device=device),
                 torch.empty(t,dtype=torch.float32,device=device))
    return outputs,workspace


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        c = configuration(arm,spec)
        inputs = {name:value.npu() for name,value in fixture['inputs'].items()}
        for value in inputs.values():
            if not value.is_contiguous() or torch_npu.get_npu_format(value)!=2:
                raise ValueError('linear-loss requires physical dense ND inputs')
        self.inputs[arm] = inputs
        outputs,workspace = allocate(c,'npu:0')
        compiled = {}
        mm,ce,reduce = kernels()
        return SimpleNamespace(run=lambda:execute(inputs,outputs,workspace,c,compiled),
            expected_kernels=[mm.fn.__name__,ce.fn.__name__,reduce.fn.__name__],
            sources=[__file__,gemm.__file__,*CASE_SOURCES],
            metadata=dict(configuration=c,operation='summed linear CE loss and complete dX/dW',
                          precision='FP16 X/W; FP32 logits/softmax; FP16-rounded dlogits; FP32 GEMM sums',
                          workspace_bytes=c['logical_workspace_bytes'],
                          workspace_scope='logical allocated logits+gradient+rowloss; excludes common outputs; not physical peak or traffic',
                          input_strides={k:list(v.stride()) for k,v in inputs.items()}),
            compilation_state=lambda:gemm.compiled_identity(compiled))

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        for name,value in self.inputs[arm].items():
            if not torch.equal(value.cpu(),fixture['inputs'][name]):
                raise ValueError('input mutated: '+name)
        return diagnostics(actual,fixture)


def factory():
    return Adapter()
