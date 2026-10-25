from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch,sys,unittest,tempfile,json,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import mean_inner as op
import mean_inner_contract as c


class MeanTests(unittest.TestCase):
    def spec(self,**kw):return dict(M=3,N=41,K=16,seed=190701,**kw)

    def test_provider_cache_strides_and_admission(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            for spec in (self.spec(),dict(parameters=self.spec(physical_layout='MKN-backing'),seed=190702,role='guard')):
                a,x=fixture(NS(make_case=c.make_case,case_sources=op.Adapter.case_sources),spec,Path(tmp),torch);b,y=fixture(NS(make_case=c.make_case,case_sources=op.Adapter.case_sources),spec,Path(tmp),torch)
                self.assertEqual((x['oracle_calls'],y['oracle_calls']),(1,0));self.assertEqual(a['inputs']['x'].stride(),b['inputs']['x'].stride())
                self.assertTrue(torch.equal(a['expected'],a['inputs']['x'].double().mean(1)));self.assertTrue(torch.equal(a['expected'][0],a['expected'][1]))
        spec=self.spec();f=c.make_case(spec)
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for x in (f['inputs']['x'].double(),f['inputs']['x'].transpose(1,2).contiguous().transpose(1,2),f['inputs']['x'][:1]):
                with self.assertRaises(ValueError):op.Adapter()._prepare('after',spec,dict(x=x))
            for v in (float('nan'),float('inf'),17.):
                x=f['inputs']['x'].clone();x[0,0,0]=v
                with self.assertRaises(ValueError):op.Adapter()._prepare('before',spec,dict(x=x))
            for p in (dict(spec,N=0),dict(spec,K=-1),dict(spec,M=True)):
                with self.assertRaises(ValueError):op.Adapter().compile_probe('after',p,None)

    def test_mock_launch_compile_atomicity_routes_and_diagnostics(self):
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.calls=0;self.fail=False
            def warmup(self,*a,**kw):
                if self.fail:raise RuntimeError('compile failure')
                return NS(hash='a'*64,name=self.fn.__name__,asm=dict(npubin=b'ELF'))
            def __getitem__(self,grid):
                def run(*a,**kw):self.calls+=1;a[1].copy_(a[0].double().mean(1).float())
                return run
        source,direct=Kernel('mean_kernel'),Kernel('_mean_inner');original=torch.get_default_dtype()
        try:
            for dtype in (torch.float64,torch.float16):
                torch.set_default_dtype(dtype)
                with patch.object(op,'backend',return_value=('cpu',source,direct)):
                    for changes in ({},{'K':1},{'physical_layout':'MKN-backing'}):
                        spec=dict(self.spec(),**changes);f=c.make_case(spec);adapter=op.Adapter()
                        for arm in ('before','after'):
                            probe=adapter.compile_probe(arm,spec,None);calls=source.calls+direct.calls
                            with self.assertRaises(RuntimeError):probe.run()
                            probe.compile_only();self.assertEqual(source.calls+direct.calls,calls)
                        for order in [('before','after'),('after','before')]:
                            for arm in order:
                                item=adapter._prepare(arm,spec,f['inputs']);kernel=direct if arm=='after' and not changes else source
                                self.assertEqual(item.metadata['route'],'optimized' if kernel is direct else 'reference')
                                self.assertEqual(item.metadata['input_strides'],list(f['inputs']['x'].stride()))
                                kernel.fail=True;calls=source.calls+direct.calls
                                with self.assertRaises(RuntimeError):item.run()
                                self.assertEqual(item.compilation_state(),[]);self.assertEqual(source.calls+direct.calls,calls);kernel.fail=False
                                actual=item.run();self.assertEqual(actual.dtype,torch.float32);self.assertTrue(adapter.validate(arm,actual,f,spec)['passed'])
                                self.assertIs(actual,item.run());self.assertTrue(adapter.validate(arm,actual,f,spec)['passed'])
                                actual[0,0]=float('nan');report=adapter.validate(arm,actual,f,spec)
                                self.assertFalse(report['passed']);self.assertEqual(report['mismatch_count'],1);self.assertEqual(report['first_mismatches'][0]['coordinate'],[0,0]);self.assertEqual(report['nonfinite_count'],1)
                                item.run();adapter.state[arm]['buffers'][0][0]=0;self.assertFalse(adapter.validate(arm,actual,f,spec)['passed'])
                                self.assertFalse(adapter.validate(arm,(actual,),f,spec)['passed'])
        finally:torch.set_default_dtype(original)

    def test_external_cli_valid_newline_json_and_invalid_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'case.json';case.write_text(json.dumps(self.spec())+'\n');prov=root/'prov.json';prov.write_text('{}\n')
            args=['--case',str(case),'--cache',str(root/'cache'),'--output',str(root/'out'),'--provenance',str(prov),'--correctness-only']
            script="import sys,runpy;sys.path.insert(0,sys.argv[1]);import mean_inner;path=sys.argv[2];sys.argv=[path]+sys.argv[3:]+['--inspect-compile'];mean_inner.Adapter.compile_probe=lambda self,arm,spec,out: __import__('types').SimpleNamespace(compile_only=lambda: None) if spec['N']==41 else None;mean_inner.main()"
            p=subprocess.run([sys.executable,'-c',script,str(Path(op.__file__).parent),op.__file__,*args],cwd=tmp,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
            case.write_text(json.dumps(dict(self.spec(),N=0))+'\n')
            p=subprocess.run([sys.executable,op.__file__,*args],cwd=tmp,capture_output=True);self.assertNotEqual(p.returncode,0);self.assertFalse((root/'cache').exists())

    def test_public_failclosed_before_oracle_or_backend_and_inspection_artifacts(self):
        spec=self.spec()
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')),patch.object(c,'make_case',side_effect=AssertionError('oracle touched')):
            for adapter in (op.Adapter(),op.factory()):
                with self.assertRaises(RuntimeError):adapter.make_case(spec)
                with self.assertRaises(RuntimeError):adapter.prepare('before',spec,{},None)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);calls=[];artifact=NS(hash='a'*64,name='mock',asm=dict(npubin=b'ELF',ttir='IR'))
            def failed():raise RuntimeError('mock backend resource failure')
            item=NS(run=lambda:calls.append('launched'),compile_only=failed,metadata={})
            with patch.object(op.Adapter,'_prepare',return_value=item):
                probe=op.Adapter().compile_probe('before',spec,folder)
                with self.assertRaises(RuntimeError):probe.run()
                with self.assertRaises(RuntimeError):probe.compile_only()
                self.assertEqual(calls,[]);report=json.loads((folder/'attempt-1.json').read_text());self.assertEqual(report['device_forwards'],0);self.assertFalse(report['numeric_qualification'])
            item=NS(run=lambda:calls.append('launched'),compile_only=lambda:[artifact],metadata={})
            with patch.object(op.Adapter,'_prepare',return_value=item):
                op.Adapter().compile_probe('after',spec,folder).compile_only()
                self.assertEqual((folder/'attempt-2-kernel-0.npubin').read_bytes(),b'ELF');self.assertEqual(json.loads((folder/'attempt-2.json').read_text())['status'],'compile-only')


if __name__=='__main__':unittest.main()
