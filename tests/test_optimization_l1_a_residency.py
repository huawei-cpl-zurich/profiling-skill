from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch,sys,unittest,tempfile,json,subprocess,ctypes
from contextlib import contextmanager,ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/catlass'))
import l1_a_adapter as op
import l1_a_contract as c
import l1_a_native as native


class NativeTests(unittest.TestCase):
    def spec(self,**kw):return dict(M=3,N=48,K=32,seed=191201,**kw)

    def test_independent_provider_codec_and_domain(self):
        p=self.spec();f=c.make_case(p);integer=(f['inputs']['a'].double()*64)@(f['inputs']['b'].double()*64)
        self.assertTrue(torch.equal(f['expected'][0]*4096,integer));self.assertTrue(torch.equal(f['expected'][1],(integer/4096).half()))
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            for spec in (p,dict(parameters=p,seed=191202,role='guard')):
                a,x=fixture(op.Adapter(),spec,Path(tmp),torch);b,y=fixture(op.Adapter(),spec,Path(tmp),torch)
                self.assertEqual((x['oracle_calls'],y['oracle_calls']),(1,0));self.assertTrue(c.bits(a['inputs']['a'],b['inputs']['a']))
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for dims in (dict(p,M=129),dict(p,N=47),dict(p,K=1056),dict(p,K=0)):
                with self.assertRaises(ValueError):op.Adapter().compile_probe('after',dims,None)
            for x in (f['inputs']['a'].float(),f['inputs']['a'].T.contiguous().T):
                with self.assertRaises(ValueError):op.Adapter().prepare('after',p,dict(f,inputs=dict(f['inputs'],a=x)),None)
            for value in (float('nan'),float('inf'),.1,.01):
                x=f['inputs']['a'].clone();x[0,0]=value
                with self.assertRaises(ValueError):op.Adapter().prepare('before',p,dict(f,inputs=dict(f['inputs'],a=x)),None)
        alias=c.make_case(dict(M=16,N=16,K=16,seed=191203));alias['inputs']['b']=alias['inputs']['a']
        with self.assertRaises(ValueError):c.check_inputs(alias['inputs'],dict(M=16,N=16,K=16))

    def test_prepared_mock_native_retry_alias_canary_and_dtype(self):
        class Native:
            fail=True;calls=0
            def __init__(self,arm,p,inputs,out,*args):
                if Native.fail:raise RuntimeError('mock compile/ABI failure')
                self.inputs=inputs;self.output=out;self.manifest=dict(status='compiled',arm=arm)
            def run(self):Native.calls+=1;self.output.copy_((self.inputs[0].double()@self.inputs[1].double()).half());return self.output
        default=torch.get_default_dtype()
        try:
            for dtype in (torch.float64,torch.float16):
                torch.set_default_dtype(dtype);spec=self.spec();f=c.make_case(spec);adapter=op.Adapter()
                with patch.object(op,'backend',return_value=('cpu',lambda:0)),patch.object(op,'Native',Native),patch.object(op,'inspect_build',return_value=dict(entrypoints=dict(host_registered_selector='mock_kernel'))):
                    for order in [('before','after'),('after','before')]:
                        for arm in order:
                            item=adapter.prepare(arm,spec,f,None);Native.fail=True;calls=Native.calls
                            with self.assertRaises(RuntimeError):item.run()
                            self.assertEqual(item.compilation_state(),[]);self.assertEqual(Native.calls,calls);Native.fail=False
                            actual=item.run();self.assertEqual(actual.dtype,torch.float16);self.assertTrue(adapter.validate(arm,actual,f,spec)['passed']);self.assertIs(item.run(),actual)
                            actual[0,0]=float('inf');d=adapter.validate(arm,actual,f,spec);self.assertFalse(d['passed']);self.assertEqual(d['mismatch_count'],1);self.assertEqual(d['first_mismatches'][0]['coordinate'],[0,0])
                            item.run();adapter.state[arm]['buffers'][0][0]=0;self.assertFalse(adapter.validate(arm,actual,f,spec)['passed']);self.assertFalse(adapter.validate(arm,(actual,),f,spec)['passed'])
        finally:torch.set_default_dtype(default)

    def test_probe_inspection_resource_errors_and_cli(self):
        class Function:
            def __init__(self,f):self.f=f
            def __call__(self,*args):return self.f(*args)
        def inspect(m,n,k,proof,count):
            for i,v in enumerate([96,8,0,16,24,40,48,64,72,128*k*2,262144,32768,65536,131072,4,0]):proof[i]=v
            return 0
        library=NS(r12_inspect=Function(inspect));manifest=dict(binary='mock.so')
        with patch.object(native,'compile_arm',return_value=manifest),patch.object(native.ctypes,'CDLL',return_value=library),patch.object(native,'entrypoints',return_value=dict(host_registered_selector='mock_kernel')):
            self.assertEqual(native.inspect_build('before',self.spec(),Path('unused'))['case_abi_resources'][9],8192)
            library.r12_inspect=Function(lambda *args:1)
            with self.assertRaises(RuntimeError):native.inspect_build('after',self.spec(),Path('unused'))
        with patch.object(op,'inspect_build',return_value=manifest) as build,patch.object(op,'backend',side_effect=AssertionError('backend/transfer touched')):
            item=op.Adapter().compile_probe('after',self.spec(),None)
            with self.assertRaises(RuntimeError):item.run()
            self.assertEqual(build.call_count,0);item.compile_only();item.compile_only();self.assertEqual(build.call_count,1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'case.json';case.write_text(json.dumps(self.spec())+'\n')
            script="import sys;sys.path[:0]=sys.argv[1:3];import l1_a_adapter;sys.argv=[l1_a_adapter.__file__]+sys.argv[3:];l1_a_adapter.Adapter.compile_probe=lambda self,*args: __import__('types').SimpleNamespace(compile_only=lambda:None);l1_a_adapter.main()"
            args=['--case',str(case),'--output',str(root/'out'),'--inspect-compile'];p=subprocess.run([sys.executable,'-c',script,str(Path(op.__file__).parent),str(Path(c.__file__).parent),*args],cwd=tmp,capture_output=True);self.assertEqual(p.returncode,0,p.stderr)
            case.write_text(json.dumps(dict(self.spec(),N=47)));p=subprocess.run([sys.executable,op.__file__,*args],cwd=tmp,capture_output=True);self.assertNotEqual(p.returncode,0);self.assertFalse((root/'out').exists())


    def library(self):
        class Function:
            def __init__(self,f):self.f=f
            def __call__(self,*args):return self.f(*args)
        state=NS(created=[],destroyed=[],fail=False,observed=[])
        def inspect(m,n,k,proof,count):
            for i,v in enumerate([112,8,0,16,24,48,56,80,88,128*k*2,262144,32768,65536,131072,4,0]):proof[i]=v
            return 0
        def create(*args):
            handle=len(state.created)+1;state.created.append(handle);ctypes.cast(args[-1],ctypes.POINTER(ctypes.c_void_p))[0]=handle;return 0
        def launch(handle,stream):
            state.observed.append(stream)
            if state.fail:raise RuntimeError('mock launch failure')
            item=next(x for x in native._CONTEXTS.values() if x.context.value==handle.value)
            if isinstance(item.output,torch.Tensor):item.output.copy_((item.inputs[0].double()@item.inputs[1].double()).half())
            return 0
        return NS(r12_inspect=Function(inspect),r12_create=Function(create),r12_launch=Function(launch),r12_destroy=Function(lambda h:state.destroyed.append(h.value))),state

    def test_real_pair_cleanup_and_independent_live_items(self):
        from optimization_pair import run_pair
        library,ledger=self.library();baseline=set(native._CONTEXTS);spec=self.spec();fixture=c.make_case(spec);adapter=op.Adapter()
        with tempfile.TemporaryDirectory() as tmp,patch.object(op,'backend',return_value=('cpu',lambda:0)),patch.object(op,'inspect_build',return_value=dict(entrypoints=dict(host_registered_selector='mock'))),patch.object(native,'compile_arm',return_value=dict(binary='mock.so')),patch.object(native.ctypes,'CDLL',return_value=library),patch.object(native,'binary_identity',return_value=[]),patch.object(native,'audit_tensor',return_value={}):
            root=Path(tmp);live=adapter.prepare('before',spec,fixture,None);live.run();live_token=set(native._CONTEXTS)-baseline
            for i,fail in enumerate((False,True,False)):
                ledger.fail=fail;report=run_pair(adapter,spec,root/str(i),root/'cache',{},torch=torch,correctness_only=True)
                self.assertEqual(report['status'],'failed' if fail else 'complete');self.assertTrue(all(x['cleanup']['status']=='complete' for x in report['arms'].values()));self.assertEqual(set(native._CONTEXTS),baseline|live_token);self.assertEqual(adapter.state,{})
                ledger.fail=False;self.assertTrue(c.bits(live.run(),fixture['expected'][1]))
            other=adapter.prepare('before',spec,fixture,None);other.run();live.close();live.close();self.assertIn('before',adapter.state);self.assertTrue(c.bits(other.run(),fixture['expected'][1]));other.close();other.close()
            original_create=library.r12_create.f
            library.r12_create.f=lambda *args:(original_create(*args),1)[1]
            failed=adapter.prepare('after',spec,fixture,None)
            with self.assertRaisesRegex(RuntimeError,'Initialize failed'):failed.run()
            failed.close();failed.close()
            self.assertEqual(set(native._CONTEXTS),baseline);self.assertEqual(sorted(ledger.created),sorted(ledger.destroyed));self.assertEqual(len(ledger.destroyed),len(set(ledger.destroyed)));self.assertEqual(adapter.state,{})
            with self.assertRaisesRegex(RuntimeError,'closed'):other.run()

    def test_explicit_device_stream_context_and_caller_restore(self):
        library,ledger=self.library();current=[1];observed=[]
        @contextmanager
        def guard(device):
            old=current[0];current[0]=int(str(device).split(':')[-1])
            try:yield
            finally:current[0]=old
        def stream(device):observed.append((current[0],device));self.assertEqual((current[0],device),(0,0));return NS(npu_stream=100)
        fake_npu=NS(device=guard,current_stream=stream)
        destroy=library.r12_destroy.f
        def guarded_destroy(handle):self.assertEqual(current[0],0);destroy(handle)
        library.r12_destroy.f=guarded_destroy
        class Buffer:
            device='npu:0'
            def data_ptr(self):return 123
        original_to=torch.Tensor.to
        def transfer(t,device,*args,**kw):
            if device=='npu:0':self.assertEqual(current[0],0);return t
            return original_to(t,device,*args,**kw)
        with patch.dict(sys.modules,torch_npu=NS()),patch.object(torch,'npu',fake_npu,create=True),patch.object(native,'compile_arm',return_value=dict(binary='mock.so')),patch.object(native.ctypes,'CDLL',return_value=library),patch.object(native,'audit_tensor',return_value={}),patch.object(native,'register'),patch.object(torch.ops.catlass_r12,'launch',side_effect=lambda ins,out,token:native._CONTEXTS[token]._launch(),create=True),patch.object(op,'inspect_build',return_value=dict(entrypoints=dict(host_registered_selector='mock'))),patch.object(torch.Tensor,'to',transfer):
            for selected in (1,2):
                current[0]=selected;device,provider=native.backend();adapter=op.Adapter();item=adapter.prepare('before',self.spec(),c.make_case(self.spec()),None);self.assertEqual(current[0],selected);item.close()
                context=native.Native('before',self.spec(),[Buffer(),Buffer()],Buffer(),Path('unused'),provider);self.assertEqual(current[0],selected);context.run();self.assertEqual(current[0],selected);ledger.fail=True
                with self.assertRaisesRegex(RuntimeError,'launch failure'):context.run()
                ledger.fail=False;self.assertEqual(current[0],selected);context.close();context.close();self.assertEqual(current[0],selected)
            self.assertEqual(ledger.observed,[100,100,100,100]);self.assertEqual(ledger.created,ledger.destroyed);self.assertTrue(observed)

    def test_device_elf_identity_through_real_collect_pair(self):
        import copy
        sys.path.insert(0,str(Path(__file__).parent))
        import test_optimization_profile_evidence as fixtures
        import optimization_profile_evidence as evidence
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);host=root/'host.so';host.write_bytes(b'host provenance only')
            entries={}
            for kind in ('aic','aiv'):
                obj=root/(kind+'.o');obj.write_bytes(('NONFROZEN device '+kind).encode());entries[kind]=dict(object_sha256=native.sha(obj),text_sha256=native.sha(obj),selector='mock_'+kind)
            with patch.object(native,'entrypoints',return_value=entries):identity=native.binary_identity(dict(binary=str(host)))
            self.assertIsInstance(identity,list);self.assertEqual(len(identity),2);self.assertEqual(identity[0]['sha256'],native.sha(root/'aic.o'));self.assertNotEqual(identity[0]['hash'],native.sha(host))
            path,report=fixtures.PairTests().fixture(root)
            for row in report['arms'].values():row['compilation_state_before_capture']=copy.deepcopy(identity);row['compilation_state_after_capture']=copy.deepcopy(identity)
            path.write_text(json.dumps(report));result=evidence.collect_pair(path)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['new_device_forwards'],0);self.assertEqual(result['arms']['before']['compilation_before'][0]['format'],'ELF64LE-hiipu-device')
            for row in report['arms'].values():row['compilation_state_before_capture']=row['compilation_state_after_capture']=dict(native=dict(binary_sha256=native.sha(host)))
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(evidence.EvidenceError,'compiled binary identity'):evidence.collect_pair(path)


if __name__=='__main__':unittest.main()
