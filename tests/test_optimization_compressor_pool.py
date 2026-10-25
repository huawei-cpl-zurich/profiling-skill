from pathlib import Path
import os,sys,tempfile,subprocess,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
import torch
from compressor_pool_contract import make_case,reference,dimensions,check_fixture
from compressor_pool import Adapter
from optimization_pair import fixture


def spec(r=2,res=False):
    return dict(case_id='unit-pool',role='guard' if res or r!=2 else 'gallery',seed=9187,
        parameters=dict(G=1 if res else 2,ratio=r,start_pos=1 if res else 0,
                        **({'residual':True,'used_len':1} if res else {})))


class ContractTests(unittest.TestCase):
    def test_oracle_and_rounding_boundary(self):
        for s in (spec(),spec(3),spec(res=True)):
            d=make_case(s);check_fixture(s,d)
            shape=dimensions(s)
            if shape[4]:
                kv=torch.stack((d['cache'][0,0,:512],d['kv'][0]))[None].double()
                sc=torch.stack((d['cache'][0,0,512:],d['score'][0]))[None].double()
            else:kv=d['kv'].view(shape[0],shape[1],512).double();sc=d['score'].view_as(kv).double()
            ex=(sc-sc.max(1,keepdim=True).values).exp();p=(ex*kv).sum(1)/ex.sum(1)
            rounded=p.bfloat16().double();y=(rounded/(rounded.square().mean(1,keepdim=True)+1e-6).sqrt()*d['weight'].double()).bfloat16()
            self.assertTrue(torch.equal(y.view(torch.int16),d['expected'].view(torch.int16)))

    def test_strict_contract_and_cache(self):
        s=spec();d=make_case(s)
        for key,value in [('ratio',4),('G',0),('start_pos',1),('residual',1)]:
            bad={**s,'parameters':{**s['parameters'],key:value}}
            with self.assertRaises(ValueError):dimensions(bad)
        for key,value in [('kv',d['kv'].double()),('weight',d['weight'][:2]),('cache',torch.zeros(1))]:
            bad={**d,key:value}
            with self.assertRaises(ValueError):check_fixture(s,bad)
        with tempfile.TemporaryDirectory() as tmp:
            _,a=fixture(Adapter(),s,Path(tmp),torch);_,b=fixture(Adapter(),s,Path(tmp),torch)
            self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0))
        flat={**s['parameters'],'seed':s['seed']};self.assertEqual(dimensions(s),dimensions(flat))

    def test_cli_rejects_unsupported_before_cache(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);bad=spec();bad['parameters']['ratio']=4
            (folder/'case.json').write_text(json.dumps(bad))
            result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/compressor_pool.py'),
                '--case',str(folder/'case.json'),'--cache',str(folder/'cache'),'--output',str(folder/'out'),
                '--provenance',str(folder/'missing-provenance')],cwd=folder,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('ratio2/3',result.stderr)
            self.assertFalse((folder/'cache').exists());self.assertFalse((folder/'out').exists())

    def test_external_cwd(self):
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([sys.executable,'-c','from compressor_pool import Adapter; assert not Adapter().prepared'],
                cwd=cwd,env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')),check=True)

    def test_public_compile_gate_and_reuse(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        calls=[]
        artifact=SimpleNamespace(hash='a'*64,asm={'npubin':b'ELF'})
        class K:
            fn=SimpleNamespace(__name__='mock_pool')
            def warmup(self,*a,**k):calls.append('compile');return artifact
            def __getitem__(self,grid):
                def run(*a,**k):calls.append('launch');return artifact
                return run
        modules={'torch_npu':SimpleNamespace(),'triton':SimpleNamespace(next_power_of_2=lambda x:1<<(x-1).bit_length()),
                 'compressor_pool_kernel':SimpleNamespace(pool_general=K(),pool_ratio2=K())}
        with patch.dict(sys.modules,modules),patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True):
            adapter=Adapter();s=spec();d=make_case(s);item=adapter.prepare('after',s,d,Path('.'))
            self.assertEqual(calls,[]);self.assertIs(item.run(),item.run());self.assertEqual(calls,['compile','launch','launch'])
            item.result.copy_(d['expected']);self.assertTrue(adapter.validate('after',item.result,d,s)['passed'])
            item.store[0]=0;self.assertFalse(adapter.validate('after',item.result,d,s)['passed'])
            normal=make_case(spec());residual=make_case(spec(res=True))
            original_empty=torch.empty;default_dtype=torch.get_default_dtype()
            def cpu_empty(*args,**kwargs):
                if str(kwargs.get('device'))=='npu:0':kwargs['device']='cpu'
                return original_empty(*args,**kwargs)
            try:
                for dtype in (torch.float64,torch.float16):
                    torch.set_default_dtype(dtype)
                    for case,data in [(spec(),normal),(spec(res=True),residual)]:
                        for arm in ('before','after'):
                            changed=adapter.prepare(arm,case,data,Path('.'))
                            changed.result.copy_(data['expected'])
                            diagnostics=adapter.validate(arm,changed.result,data,case)
                            self.assertTrue(diagnostics['passed']);self.assertEqual(diagnostics['absolute_error_quantiles'],[0.]*5)
                    with patch.object(torch,'empty',cpu_empty):
                        prototype=adapter.compile_probe('after',spec(res=True),Path('.'))
                    self.assertEqual(prototype.inputs['cache'].dtype,torch.float32)
                    with self.assertRaises(RuntimeError):prototype.run()
            finally:torch.set_default_dtype(default_dtype)
            for order in [('before','after'),('after','before')]:
                guard=spec(3);data=make_case(guard)
                items={a:adapter.prepare(a,guard,data,Path('.')) for a in order}
                for a in order:
                    items[a].result.copy_(data['expected']);self.assertTrue(adapter.validate(a,items[a].result,data,guard)['passed'])
                items[order[-1]].result[0,0]+=1
                self.assertFalse(adapter.validate(order[-1],items[order[-1]].result,data,guard)['passed'])

if __name__=='__main__':unittest.main()
