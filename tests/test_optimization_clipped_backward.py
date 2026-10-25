from pathlib import Path
import os,sys,tempfile,unittest
from unittest.mock import patch
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import clipped_backward as op
import clipped_backward_contract as c


class BackwardTests(unittest.TestCase):
    def test_boundaries_strict_outside_and_independent_autograd(self):
        f=c.make_case(dict(M=2,K=19,pattern='exact_clamp_boundaries',seed=411))
        gu=f['inputs']['gup'].double().requires_grad_();g,u=gu.chunk(2,dim=1);gc=g.clamp(max=7);uc=u.clamp(-7,7)
        ((uc+1)*gc*torch.sigmoid(gc*1.702)).backward(f['inputs']['dy'].double())
        torch.testing.assert_close(f['expected'],gu.grad.float(),rtol=2e-5,atol=2e-5)
        self.assertTrue(c.validate_actual(f['expected'],f)['passed'])
        for a in (f['expected'].double(),f['expected'][:,:-1],[f['expected']]):
            with self.assertRaises(ValueError):c.validate_actual(a,f)
        wrong=f['expected'].clone();wrong[:,:19][f['inputs']['gup'][:,:19]>7]=-0.
        with self.assertRaises(ValueError):c.validate_actual(wrong,f)

    def test_all_patterns_nonfrozen_and_numeric_layout_guards(self):
        for pattern in ('saturated','unsaturated','saturated_half_up_clipped','mixed_every_tile','zeros_signed_cancellation'):
            f=c.make_case(dict(parameters=dict(M=2,K=21,pattern=pattern),seed=412))
            c.check_inputs(f['inputs'],dict(M=2,K=21),'cpu');self.assertTrue(c.validate_actual(f['expected'],f)['passed'])
        for bad in (dict(M=2,K=21,two_k=41),dict(M=2,K=21,limit=-1),dict(M=2,K=21,limit=float('nan'))):
            with self.assertRaises(ValueError):c.check_spec(bad)
            with patch.dict(sys.modules,{'torch_npu':None}):
                with self.assertRaises(ValueError):op.Adapter(compile_inspection=True).compile_probe('after',bad,None)
        f=c.make_case(dict(M=2,K=21,seed=413))
        for t in (f['inputs']['gup'].double(),f['inputs']['gup'][:,:-1],f['inputs']['gup'].T.contiguous().T,
                  torch.full_like(f['inputs']['gup'],float('inf'))):
            with self.assertRaises(ValueError):c.check_inputs(dict(f['inputs'],gup=t),dict(M=2,K=21),'cpu')
        b=torch.full((80,),12345.);b[-1]=0
        with self.assertRaises(ValueError):c.validate_actual(f['expected'],f,buffer=b)
        inputs={n:t.clone() for n,t in f['inputs'].items()};inputs['dy'][0,0]+=1
        with self.assertRaises(ValueError):c.validate_actual(f['expected'],f,inputs)

    def test_explicit_inspection_uninitialized_no_launch_and_atomic_cache(self):
        from types import SimpleNamespace as NS
        import struct
        data=bytearray(300);data[:6]=b'\x7fELF\x02\x01'
        struct.pack_into('<Q',data,40,64);struct.pack_into('<HHH',data,58,64,2,0)
        struct.pack_into('<IIQQQQIIQQ',data,64,0,3,0,0,192,7,0,0,1,0)
        struct.pack_into('<IIQQQQIIQQ',data,128,1,1,6,0,200,1,0,0,1,0)
        data[192:199]=b'\0.text\0';data[200]=1
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.compiles=0;self.launches=0;self.fail=False
            def warmup(self,*args,**kwargs):
                self.compiles+=1
                if self.fail:raise RuntimeError('unit compile failure')
                return NS(asm=dict(npubin=bytes(data)),hash='a'*64,name=self.fn.__name__)
            def __getitem__(self,grid):
                def launch(*args,**kwargs):self.launches+=1
                return launch
        before,after=Kernel('before'),Kernel('after');empty=torch.empty
        def host_empty(*args,**kwargs):kwargs.pop('device',None);return empty(*args,**kwargs)
        modules=dict(torch_npu=NS(),clipped_backward_kernels=NS(_clipped_swiglu_bwd_kernel=before,_saturated_clipped_bwd_kernel=after))
        with patch.dict(sys.modules,modules),patch.object(torch,'empty',host_empty):
            adapter=op.Adapter(compile_inspection=True);old_dtype=torch.get_default_dtype()
            try:
                for dtype in (torch.float64,torch.float16):
                    torch.set_default_dtype(dtype)
                    for order in [('before','after'),('after','before')]:
                        for arm in order:
                            kernel=before if arm=='before' else after;count=kernel.compiles
                            item=adapter.compile_probe(arm,dict(M=2,K=27),None)
                            self.assertTrue(all(x.dtype==torch.float32 for x in item.prototype_inputs.values()))
                            self.assertEqual(item.prototype_output.dtype,torch.float32)
                            with self.assertRaises(op.StudyUnsupported):item.run()
                            self.assertEqual((kernel.compiles,kernel.launches),(count,0))
                            kernel.fail=True
                            with self.assertRaises(RuntimeError):item.compile_only()
                            self.assertEqual(item.compilation_state(),[]);kernel.fail=False
                            item.compile_only();item.compile_only();self.assertEqual(kernel.compiles,count+2)
                            with self.assertRaises(op.StudyUnsupported):item.run()
                            self.assertEqual(kernel.launches,0)
            finally:torch.set_default_dtype(old_dtype)

    def test_default_workflows_reject_before_provider_backend_or_cache(self):
        sys.path.insert(0,str(ROOT/'scripts'));from optimization_pair import fixture
        calls=[];adapter=op.Adapter()
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'torch_npu':None,'triton':None}),patch.object(c,'make_case',side_effect=AssertionError('provider must not run')):
            for action in (op.factory,lambda:adapter.make_case(dict(M=2,K=23)),lambda:fixture(adapter,dict(M=2,K=23),Path(tmp)/'cache',torch),lambda:adapter.prepare('after',{},None,None),lambda:adapter.compile_probe('before',dict(M=2,K=23),None),lambda:adapter.capture(lambda:calls.append(1),None,None),lambda:adapter.validate('after',None,None,None)):
                with self.assertRaises(op.StudyUnsupported):action()
            self.assertFalse((Path(tmp)/'cache').exists());self.assertFalse(calls)

    def test_provider_cache_and_external_cwd_disabled_cli(self):
        import subprocess
        sys.path.insert(0,str(ROOT/'scripts'));from optimization_pair import fixture
        class Provider:
            case_sources=[Path(c.__file__)]
            make_case=staticmethod(c.make_case)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for spec in (dict(M=2,K=23,seed=414),dict(parameters=dict(M=1,K=25),seed=415,role='guard')):
                f,a=fixture(Provider(),spec,root/'unit-cache',torch);g,b=fixture(Provider(),spec,root/'unit-cache',torch)
                self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0));self.assertTrue(c.validate_actual(g['expected'],f)['passed'])
            for entry in ([sys.executable,op.__file__],[sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','clipped_backward:factory']):
                p=subprocess.run(entry+['--case',str(root/'missing-case'),'--output',str(root/'out'),'--cache',str(root/'numeric-cache'),'--provenance',str(root/'missing-provenance')],cwd=tmp,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')))
                self.assertNotEqual(p.returncode,0);self.assertIn('T04 failed study',p.stderr)
                self.assertFalse((root/'numeric-cache').exists());self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
