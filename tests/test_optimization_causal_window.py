from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys,torch,unittest,tempfile,json,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import causal_window as op
import causal_window_contract as c


class WindowTests(unittest.TestCase):
    def spec(self):return dict(B=3,D=32,bias=True,silu=False,indexed=True,slots=5,indices=[-1,4,1],seed=190601)

    def test_scalar_oldx_state_and_guard_cache_codec(self):
        spec=self.spec();f=c.make_case(spec);x,state=f['inputs']['x'],f['inputs']['state'];out,new=f['expected']
        self.assertTrue(c.equal(out[0],x[0],bits=True));self.assertTrue(c.equal(new[[0,2,3]],state[[0,2,3]],bits=True))
        self.assertTrue(c.equal(new[4,:,2],x[1],bits=True));self.assertTrue(c.equal(new[4,:,:2],state[4,:,1:],bits=True))
        self.assertEqual([c.block(d) for d in (2048,2064,4096)],[2048,16,1024])
        for dim in (64,128):
            for taps in (1,3,4):
                flat=torch.arange(dim*taps);sub=dim//16
                permuted=flat.reshape(sub,16*taps).T.reshape(16,taps*sub).T.reshape(-1)
                self.assertTrue(torch.equal(permuted,flat.reshape(dim,taps).T.reshape(-1)))
                inverse=permuted.reshape(taps*16,sub).T.reshape(sub*taps,16).T.reshape(-1)
                self.assertTrue(torch.equal(inverse,flat))
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            for p in (spec,dict(parameters=dict(spec,silu=True),seed=190602,role='guard')):
                provider=NS(case_sources=op.Adapter.case_sources,make_case=c.make_case)
                a,x=fixture(provider,p,Path(tmp),torch);b,y=fixture(provider,p,Path(tmp),torch)
                self.assertEqual((x['oracle_calls'],y['oracle_calls']),(1,0));self.assertTrue(c.equal(a['expected'][1],b['expected'][1],bits=True))

    def test_public_admission_before_backend_and_external_cli(self):
        spec=self.spec();f=c.make_case(spec)
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for indices in ([0,0,1],[-2,4,1],[-1,5,1]):
                with self.assertRaises(ValueError):op.Adapter().compile_probe('after',dict(spec,indices=indices),None)
            for key,value in [('x',f['inputs']['x'].double()),('x',f['inputs']['x'][:,:-1]),('state',f['inputs']['state'].transpose(1,2).contiguous().transpose(1,2)),('indices',f['inputs']['indices'].long())]:
                with self.assertRaises(ValueError):op.Adapter()._prepare('before',spec,dict(f['inputs'],**{key:value}))
            for value in (float('nan'),float('inf'),1.,.01):
                x=f['inputs']['x'].clone();x[0,0]=value
                with self.assertRaises(ValueError):op.Adapter()._prepare('after',spec,dict(f['inputs'],x=x))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'bad.json';case.write_text(json.dumps(dict(spec,seq_len=2)))
            result=subprocess.run([sys.executable,op.__file__,'--case',str(case),'--cache',str(root/'cache'),'--output',str(root/'out'),'--provenance',str(root/'missing')],cwd=tmp,capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'cache').exists())

    def test_unqualified_numeric_rejects_and_compile_inspection_retains_artifacts(self):
        spec=self.spec();adapter=op.Adapter()
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')),patch.object(c,'reference',side_effect=AssertionError('oracle called')):
            with self.assertRaisesRegex(RuntimeError,'numeric API unqualified'):adapter.make_case(spec)
            with self.assertRaisesRegex(RuntimeError,'numeric API unqualified'):op.factory().make_case(spec)
            with self.assertRaisesRegex(RuntimeError,'numeric API unqualified'):adapter.prepare('before',spec,None,None)
        class Kernel:
            fn=NS(__name__='inspection');fail=True;calls=0
            def warmup(self,*a,**kw):
                if self.fail:raise AttributeError('triton.language has no insert_slice')
                return NS(hash='a'*64,name='inspection',asm=dict(npubin=b'ELF-artifact'))
            def __getitem__(self,grid):
                def run(*a,**kw):self.calls+=1
                return run
        kernel=Kernel()
        with tempfile.TemporaryDirectory() as tmp,patch.object(op,'backend',return_value=('cpu',kernel,kernel)),patch.object(c,'reference',side_effect=AssertionError('oracle called')):
            root=Path(tmp);probe=adapter.compile_probe('before',spec,root/'inspection')
            with self.assertRaises(RuntimeError):probe.run()
            with self.assertRaises(AttributeError):probe.compile_only()
            self.assertEqual(probe.compilation_state(),[])
            failure=json.loads((root/'inspection/attempt-1.json').read_text());self.assertEqual(failure['status'],'compile-unsupported');self.assertFalse(failure['numeric_qualification'])
            kernel.fail=False;probe.compile_only()
            self.assertEqual((root/'inspection/attempt-2-kernel-0.npubin').read_bytes(),b'ELF-artifact')
            self.assertEqual(kernel.calls,0);self.assertFalse(probe.metadata['numeric_qualification'])
            case=root/'case.json';case.write_text(json.dumps(spec))
            result=subprocess.run([sys.executable,op.__file__,'--case',str(case),'--cache',str(root/'cache'),'--output',str(root/'out')],cwd=tmp,capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertIn(b'numeric API unqualified',result.stderr);self.assertFalse((root/'cache').exists());self.assertFalse((root/'out').exists())

    def test_stateful_alias_no_reset_compile_retry_probe_and_default_dtype(self):
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.calls=0;self.fail=False
            def warmup(self,*a,**kw):
                if self.fail:raise RuntimeError('mock compile failure')
                return NS(hash='a'*64,name=self.fn.__name__,asm=dict(npubin=b'elf'))
            def __getitem__(self,grid):
                def run(*a,**kw):
                    self.calls+=1;x,state,weight,bias,indices,out=a[:6];inputs=dict(x=x.clone(),state=state.clone(),weight=weight)
                    if bias is not None:inputs['bias']=bias
                    if indices is not None:inputs['indices']=indices
                    expected,updated=c.reference(inputs,dict(B=x.shape[0],D=x.shape[1],slots=state.shape[0],bias=bias is not None,silu=kw['SILU_ACTIVATION'],indexed=indices is not None))
                    state.copy_(updated);out.copy_(expected)
                return run
        kernels=[Kernel('source'),Kernel('direct')];original=torch.get_default_dtype()
        try:
            for dtype in (torch.float64,torch.float16):
                torch.set_default_dtype(dtype)
                for silu in (False,True):
                    spec=dict(self.spec(),silu=silu);f=c.make_case(spec);adapter=op.Adapter()
                    with patch.object(op,'backend',return_value=('cpu',*kernels)):
                        for order in [('before','after'),('after','before')]:
                            for arm in order:
                                kernel=kernels[arm=='after'];probe=adapter.compile_probe(arm,spec,None);calls=kernel.calls
                                with self.assertRaises(RuntimeError):probe.run()
                                probe.compile_only();self.assertEqual(kernel.calls,calls)
                                item=adapter._prepare(arm,spec,f['inputs']);kernel.fail=True
                                with self.assertRaises(RuntimeError):item.run()
                                self.assertEqual(item.compilation_state(),[]);kernel.fail=False
                                actual=item.run();self.assertTrue(adapter.validate(arm,actual,f,spec)['passed']);self.assertIs(actual,adapter.state[arm]['inputs']['x'])
                                state=adapter.state[arm];state['buffers'][0][0]=0
                                self.assertFalse(adapter.validate(arm,actual,f,spec)['passed']);state['buffers'][0][0]=12345
                                self.assertFalse(adapter.validate(arm,(actual,),f,spec)['passed'])
                                cpu={n:v.clone() for n,v in adapter.state[arm]['inputs'].items()};next_x,next_state=c.reference(cpu,c.parameters(spec))
                                self.assertIs(actual,item.run());self.assertTrue(c.equal(actual,next_x));self.assertTrue(c.equal(adapter.state[arm]['inputs']['state'],next_state,bits=True))
                                self.assertFalse(c.equal(actual,f['expected'][0]));self.assertTrue(c.equal(actual[0],f['inputs']['x'][0],bits=True))
        finally:torch.set_default_dtype(original)


if __name__=='__main__':unittest.main()
