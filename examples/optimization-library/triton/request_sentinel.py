"""Exploratory sentinel-aware expansion pair; controlled qualification is separate."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from request_sentinel_contract import dimensions,make_case,check_fixture,backing,CANARY


def identity(compiled):
    return [dict(hash=str(c.hash),binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
        for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


def load_json(path):
    def bad(x):raise ValueError('nonfinite JSON '+x)
    return json.loads(Path(path).read_text(),parse_constant=bad)


class Adapter:
    case_sources=[Path(__file__).with_name('request_sentinel_contract.py')]
    make_case=staticmethod(make_case)
    def __init__(self):self.items={};self.numeric={};self.current=None

    def compile_probe(self,arm,spec,output):
        if arm not in ('before','after'):raise ValueError('arm')
        b,*_=dimensions(spec)
        import torch_npu
        tensors=[torch.empty((b+64,),dtype=torch.int32,device='npu:0')[32:-32] for _ in range(2)]
        return self._prepare(arm,spec,*tensors,probe=True)

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('arm')
        check_fixture(spec,fixture) # CPU prefix/value/layout proof before cache/backend/transfers.
        import torch_npu
        x,ends=[backing(fixture['inputs'][k]).to('npu:0')[32:-32] for k in ('x','ends')]
        return self._prepare(arm,spec,x,ends,probe=False)

    def _prepare(self,arm,spec,x,ends,probe):
        from request_sentinel_kernels import expand_kernel,expand_unreplaced_kernel,replace_tokens_kernel
        b,n,lengths,rf,rt=dimensions(spec);compiled=[]
        store=(torch.empty if probe else torch.full)((n+64,),**({} if probe else {'fill_value':CANARY}),dtype=torch.int32,device=x.device);out=store[32:-32]
        import torch_npu
        physical=[]
        for t in (x,ends,out):
            fmt=int(torch_npu.get_npu_format(t));size=int(torch_npu.get_storage_size(t));span=t.storage_offset()+t.numel()
            if (t.dtype!=torch.int32 or t.stride()!=(1,) or t.data_ptr()%128 or fmt!=2 or size<span+32
                    or t.untyped_storage().nbytes()!=(t.numel()+64)*4):raise ValueError('actual aligned int32 ND physical storage required')
            physical.append(dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),storage_bytes=t.untyped_storage().nbytes(),format=fmt,physical_storage_numel=size))
        if len({t.untyped_storage().data_ptr() for t in (x,ends,out)})!=3:raise ValueError('device buffer alias')
        kernel=expand_unreplaced_kernel if arm=='before' else expand_kernel
        launches=[(kernel,(out,x,ends,rf,rt),dict(MAX_NUM_TOKENS=32),(b,))]
        if arm=='before':launches.append((replace_tokens_kernel,(out,rf,rt),dict(N=n,BLOCK=256,num_warps=4),((n+255)//256,)))
        def compile_only():
            if not compiled:
                pending=[]
                with torch.npu.device(x.device):
                    for k,args,kw,grid in launches:
                        c=k.warmup(*args,grid=grid,**kw);data=c.asm.get('npubin')
                        if not isinstance(data,bytes) or not data.startswith(b'\x7fELF'):raise ValueError('linked NPU ELF required')
                        pending.append(c)
                compiled.extend(pending) # Transaction commits only after ALL components qualify.
            return compiled
        def run():
            if probe:raise RuntimeError('compile prototype cannot execute')
            with torch.npu.device(x.device):
                compile_only()
                for k,args,kw,grid in launches:k[grid](*args,**kw)
            return out
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),expected_kernels=[k.fn.__name__ for k,*_ in launches],
            sources=[Path(__file__),*self.case_sources,Path(__file__).with_name('request_sentinel_kernels.py'),Path(__file__).with_name('request_sentinel_source.py')],
            metadata=dict(arm=arm,prototype=probe,batch=b,num_tokens=n,replace_from=rf,replace_to=rt,MAX_NUM_TOKENS=32,grids=[list(grid) for *_,grid in launches],physical_layouts=physical,prepared_output_reused=True,allocator_cost_excluded=True,api_kernels=len(launches),prefix_prepared_outside_API=True,prepared_device=str(x.device),device_context_preserves_caller_selection=True),
            x=x,ends=ends,out=out,store=store,spec=spec)
        if not probe:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.items:self.items={};self.numeric={};self.current=key
            self.items[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        item=self.items[arm];check_fixture(spec,fixture)
        if item.spec!=spec or not isinstance(actual,torch.Tensor) or actual.dtype!=torch.int32 or actual.shape!=fixture['expected'].shape or actual.device!=item.out.device or actual.data_ptr()!=item.out.data_ptr():raise ValueError('output shape/dtype/device/identity')
        raw=item.store.detach().cpu();host=raw[32:-32];expected=fixture['expected'];mask=host!=expected;coords=mask.nonzero().flatten();count=int(mask.sum())
        immutable=all(torch.equal(backing(getattr(item,k)).cpu(),backing(fixture['inputs'][k])) for k in ('x','ends'))
        canary=bool((raw[:32]==CANARY).all() and (raw[-32:]==CANARY).all());nonalias=len({t.untyped_storage().data_ptr() for t in (item.x,item.ends,item.out)})==3
        other=self.numeric.get('after' if arm=='before' else 'before');self.numeric[arm]=host.clone()
        return dict(passed=count==0 and immutable and canary and nonalias,mismatch_count=count,first_mismatches=[dict(coordinates=[int(i)],expected=int(expected[i]),actual=int(host[i])) for i in coords[:8]],max_absolute_error=int((host.to(torch.int64)-expected.to(torch.int64)).abs().max()),nonfinite_count=0,paired_bitwise_equal=None if other is None else bool(torch.equal(host,other)),inputs_immutable=immutable,canaries=canary,nonalias=nonalias)


def factory():return Adapter()


def main():
    import argparse,sys
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):p.add_argument('--'+name,required=True)
    p.add_argument('--correctness-only',action='store_true');a=p.parse_args();spec=load_json(a.case);dimensions(spec)
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    report=run_pair(Adapter(),spec,a.output,a.cache,load_json(a.provenance),correctness_only=a.correctness_only)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
