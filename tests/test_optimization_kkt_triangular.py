"""Nonfrozen CPU functional contract/adapter/precision/capture regressions."""
import contextlib,copy,importlib.util,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
BASE=Path(__file__).resolve().parents[1]/'examples/optimization-library/triton/kkt_triangular'
sys.path.insert(0,str(BASE))
import kkt_contract as contract
import kkt_triangular as mod
SPEC=dict(case_id='NONFROZEN-CPU',seed=704,parameters=dict(B=1,H=1,T=5,K=64,BT=32,BC=16,featurewise=True,varlen=False,precision='ieee'))


class Tests(unittest.TestCase):
    def test_reference_and_admission(self):
        f=contract.make_case(SPEC);k=f['inputs']['k'].view(5,64).double();g=f['inputs']['gk'].view(5,64).double();b=f['inputs']['beta'].view(5).double()
        e=f['expected']['ideal'].view(5,32)
        for i in range(5):
            for j in range(32):
                want=float((k[i]*k[j]*(g[i]-g[j]).exp()).sum()*b[i])if j<i else 0.
                self.assertAlmostEqual(float(e[i,j]),want,places=7)
        self.assertTrue(torch.allclose(f['expected']['ideal'],f['expected']['typed'],atol=contract.ATOL,rtol=contract.RTOL))
        for key,value in [('H',2),('varlen',True),('precision','tf32'),('T',0),('BT',48),('featurewise',False)]:
            s=copy.deepcopy(SPEC);s['parameters'][key]=value
            with patch.object(mod,'backend',side_effect=AssertionError('backend reached')),self.assertRaises(ValueError):mod.Adapter().compile_probe('after',s,'.')
        bad=copy.deepcopy(f);bad['inputs']['gk']=bad['inputs']['k']
        with patch.object(mod,'backend',side_effect=AssertionError('backend reached')),self.assertRaises(ValueError):mod.Adapter().prepare('after',SPEC,bad,'.')
    def test_precision_metadata(self):
        dot='%x = tt.dot %a, %b, %c : tensor<16x64xf32> * tensor<64x16xf32> -> tensor<16x16xf32>'
        def item(ir=dot,default='ieee'):return types.SimpleNamespace(asm={'ttir':ir},metadata=types.SimpleNamespace(default_dot_input_precision=default))
        self.assertEqual(mod.ieee_policy(item(),True)['origins'],['actualCompilerMetadataDefaultIEEE'])
        self.assertEqual(mod.ieee_policy(item(dot.replace(' :',' inputPrecision = ieee :'),None),True)['origins'],['explicitIR'])
        for x in [item(default=None),item(default='hf32'),item(dot.replace(' :',' inputPrecision = tf32 :')),item(dot.replace('xf32','xf16')),item('')]:
            with self.assertRaises(ValueError):mod.ieee_policy(x,True)
        self.assertEqual(mod.ieee_policy(item(''),False)['dot_lines'],[])
    def test_full_capture_context_and_errors(self):
        device=[2];seen=[]
        @contextlib.contextmanager
        def guarded(d):
            old=device[0];device[0]=d
            try:yield
            finally:device[0]=old
        item=types.SimpleNamespace(metadata={'prepared_device':'npu:0'},expected_kernels=['zero','inter','intra'])
        def capture(run,out,**kw):
            self.assertEqual(kw,dict(expected_kernel=item.expected_kernels,metric_group='PipeUtilization'))
            for step in ['begin','events','launch','sync','export']:
                self.assertEqual(device[0],'npu:0');seen.append(step)
                if step=='launch':run()
            return 'captured'
        with patch.object(mod,'guard',guarded),patch.dict(sys.modules,{'profile_ten':types.SimpleNamespace(capture_ten=capture)}):
            self.assertEqual(mod.Adapter().capture(lambda:None,item,'.'),'captured');self.assertEqual(device[0],2);self.assertEqual(len(seen),5)
            def fail():raise RuntimeError('launch failed')
            with self.assertRaises(RuntimeError):mod.Adapter().capture(fail,item,'.')
            self.assertEqual(device[0],2)
    def test_adapter_lifecycle_validation_and_compile_publication(self):
        f=contract.make_case(SPEC);fail=[False];seen=[]
        class Kernel:
            def __init__(self,name):self.name=name;self.fn=types.SimpleNamespace(__name__=name)
            def warmup(self,*a,**kw):
                if fail[0]and self.name=='lower':raise RuntimeError('compile failed')
                ir='%x = tt.dot %a, %b, %c : tensor<16x64xf32> * tensor<64x16xf32> -> tensor<16x16xf32>'if self.name in ('lower','full')else'not-a-dot'
                return types.SimpleNamespace(hash='1'*64,asm={'npubin':b'\x7fELFmock','ttir':ir},metadata=types.SimpleNamespace(default_dot_input_precision='ieee'))
            def __getitem__(self,grid):
                def run(*a,**kw):
                    seen.append(self.name)
                    if self.name=='zero':a[0].zero_()
                    elif self.name in ('lower','full'):a[3].copy_(f['expected']['ideal'])
                return run
        kernels=types.SimpleNamespace(kkt_zero=Kernel('zero'),kkt_full_inter=Kernel('full'),kkt_lower_inter=Kernel('lower'),kkt_diagonal=Kernel('diagonal'))
        with patch.object(mod,'backend',return_value=('cpu',kernels,None)):
            a=mod.Adapter();probe=a.compile_probe('after',SPEC,'.');probe.compile_only()
            with self.assertRaises(RuntimeError):probe.run()
            probe.close()
            before=a.prepare('before',SPEC,f,'.');after=a.prepare('after',SPEC,f,'.');self.assertTrue(a.validate('before',before.run(),f,SPEC)['passed']);self.assertTrue(a.validate('after',after.run(),f,SPEC)['passed'])
            after.output[0,0,0,0]=1.;self.assertFalse(a.validate('after',after.output,f,SPEC)['passed'])
            before.close();before.close();self.assertIn('after',a.items)
            with self.assertRaises(RuntimeError):before.run()
            self.assertEqual(seen[:3],['zero','full','diagonal']);after.close();self.assertFalse(a.items)
            item=a.prepare('after',SPEC,f,'.');fail[0]=True
            with self.assertRaises(RuntimeError):item.compile_only()
            self.assertEqual(item.compilation_state(),[]);fail[0]=False;self.assertEqual(len(item.compile_only()),3);item.close()
            sys.path.insert(0,str(BASE.parents[3]/'scripts'))
            from optimization_pair import run_pair
            with tempfile.TemporaryDirectory()as directory:
                a=mod.Adapter();root=Path(directory);report=run_pair(a,SPEC,root/'pass',root/'cache',{},correctness_only=True)
                self.assertEqual(report['status'],'complete');self.assertEqual(report['counts']['correctness_calls'],2);self.assertEqual(report['counts']['profiled_calls'],0);self.assertFalse(a.items)
                self.assertTrue(all(v['cleanup']['status']=='complete'for v in report['arms'].values()))
                a=mod.Adapter()
                with patch.object(a,'validate',return_value={'passed':False}):report=run_pair(a,SPEC,root/'reject',root/'cache',{},correctness_only=True)
                self.assertEqual(report['status'],'failed');self.assertFalse(a.items)



if __name__=='__main__':unittest.main()
