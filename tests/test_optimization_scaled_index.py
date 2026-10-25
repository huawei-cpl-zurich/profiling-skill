from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,os,subprocess,sys,tempfile,unittest
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
import torch
from scaled_index_bounds import index_audit
from scaled_index_contract import make_case,check_fixture,dimensions,dispatch_audit
from scaled_index import Adapter,StudyUnqualified
from optimization_pair import fixture


def spec(transpose=False,wide=False,bias=True):
    return dict(case_id='unit-index',role='guard' if wide else 'gallery',seed=92147,
                parameters=dict(M=5,N=9,K=128,transposed_b=transpose,metadata_wide_stride=wide,bias=bias))


class IndexTests(unittest.TestCase):
    def test_complete_synthetic_bounds(self):
        self.assertTrue(index_audit(5,9,128,(128,1),(9,1),(9,1))['fits_int32'])
        self.assertFalse(index_audit(1,1,128,(1<<28,1),(1,1),(1,1))['fits_int32'])
        post=index_audit(1,1,128,(1,1),(1<<24,1),(1,1));self.assertIn('b_sum',post['unsafe'])
        byte=index_audit(1,1,128,(1,1),(1,1),(1<<26,1));self.assertIn('c_byte_extent',byte['unsafe']);self.assertNotIn('c_sum',byte['unsafe'])
        for bad in [(0,1),(-1,1),(True,1)]:
            with self.assertRaises(ValueError):index_audit(1,1,128,bad,(1,1),(1,1))
        self.assertEqual(dispatch_audit(spec(wide=True))['candidate_bits'],64)

    def test_oracle_layout_cache_and_contract(self):
        for s in (spec(),spec(transpose=True),spec(bias=False)):
            d=make_case(s);check_fixture(s,d)
            for i,j in [(0,0),(4,8)]:
                dot=sum(int(d['a'][i,q])*int(d['b'][q,j]) for q in range(128))
                value=torch.tensor(dot,dtype=torch.float32)*.25;value=value*.5
                if d['bias'] is not None:value=value+d['bias'][j]
                self.assertEqual(value.view(torch.int32),d['expected'][i,j].view(torch.int32))
        s=spec(transpose=True);d=make_case(s);self.assertEqual(d['b'].stride(),(1,128))
        with self.assertRaises(ValueError):check_fixture(s,{**d,'b':d['b'].contiguous()})
        with self.assertRaises(ValueError):check_fixture(s,{**d,'a':d['a'].float()})
        flat={**s['parameters'],'seed':s['seed']};self.assertEqual(dimensions(flat)[:3],dimensions(s)[:3])
        with tempfile.TemporaryDirectory() as tmp:
            x,a=fixture(Adapter(),s,Path(tmp),torch);y,b=fixture(Adapter(),s,Path(tmp),torch)
            self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0));self.assertEqual(y['b'].stride(),(1,128))
            self.assertTrue(torch.equal(x['expected'],y['expected']))

    def test_cli_external_cwd_and_pre_cache_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);bad=spec();bad['parameters']['M']=257;(p/'case.json').write_text(json.dumps(bad))
            result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/scaled_index.py'),
                '--case',str(p/'case.json'),'--cache',str(p/'cache'),'--output',str(p/'out'),'--provenance',str(p/'missing')],
                cwd=p,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('T10 compiler-noop screen',result.stderr)
            self.assertFalse((p/'cache').exists());self.assertFalse((p/'out').exists())
            subprocess.run([sys.executable,'-c','from scaled_index import Adapter,StudyUnqualified; assert not Adapter().prepared'],cwd=p,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')),check=True)

    def test_default_interfaces_fail_before_work(self):
        with patch.dict(sys.modules,{'torch_npu':None,'triton':None}):
            with self.assertRaises(StudyUnqualified):Adapter().prepare('after',spec(),make_case(spec()),Path('.'))
            with self.assertRaises(StudyUnqualified):Adapter().compile_probe('after',spec(),Path('.'))
        calls=[]
        with self.assertRaises(StudyUnqualified):Adapter().capture(lambda:calls.append('forward'),None,None)
        self.assertFalse(calls)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','scaled_index:factory',
                '--case',str(p/'missing'),'--provenance',str(p/'missing-provenance'),'--cache',str(p/'cache'),'--output',str(p/'out')],
                cwd=p,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')))
            self.assertNotEqual(result.returncode,0);self.assertIn('T10 compiler-noop screen',result.stderr)
            self.assertFalse((p/'cache').exists());self.assertFalse((p/'out').exists())

    def test_transactional_compile_prototype_and_layout(self):
        calls=[];state={'fail':True};artifact=SimpleNamespace(hash='a'*64,asm={'npubin':b'unit'})
        class Kernel:
            def warmup(self,*a,**k):
                calls.append('compile')
                if state['fail']:state['fail']=False;raise RuntimeError('unit compile failure')
                return artifact
            def __getitem__(self,grid):
                def run(*a,**k):calls.append('launch');return artifact
                return run
        old_to=torch.Tensor.to;old_empty=torch.empty;old_strided=torch.empty_strided
        def move(x,*a,**k):
            return x.clone(memory_format=torch.preserve_format) if k.get('device')=='npu:0' else old_to(x,*a,**k)
        def empty(*a,**k):
            if k.get('device')=='npu:0':k['device']='cpu'
            return old_empty(*a,**k)
        def empty_strided(*a,**k):
            if k.get('device')=='npu:0':k['device']='cpu'
            return old_strided(*a,**k)
        language=SimpleNamespace(int32='i32',int64='i64')
        mods={'torch_npu':SimpleNamespace(),'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,language=language),
              'triton.language':language,'scaled_index_kernel':SimpleNamespace(scaled_mm_kernel=Kernel())}
        with patch.dict(sys.modules,mods),patch.object(torch.Tensor,'to',move),patch.object(torch,'empty',empty),patch.object(torch,'empty_strided',empty_strided):
            adapter=Adapter(compile_inspection=True);s=spec(transpose=True);prototype=adapter.compile_probe('after',s,Path('.'))
            self.assertEqual(prototype.inputs['b'].stride(),(1,128))
            with self.assertRaises(StudyUnqualified):prototype.run()
            self.assertFalse(calls)
            with self.assertRaises(RuntimeError):prototype.compile_only()
            self.assertEqual(prototype.compilation_state(),[])
            prototype.compile_only();prototype.compile_only();self.assertEqual(calls,['compile','compile'])
            d=make_case(s);item=adapter.prepare('after',s,d,Path('.'));self.assertEqual(item.inputs['b'].stride(),(1,128))
            item.compile_only()
            for _ in range(2):
                with self.assertRaises(StudyUnqualified):item.run()
            self.assertEqual(calls.count('launch'),0)
            item.result.copy_(d['expected']);self.assertTrue(adapter.validate('after',item.result,d,s)['passed'])
            item.store[0]=0;self.assertFalse(adapter.validate('after',item.result,d,s)['passed'])

if __name__=='__main__':unittest.main()
