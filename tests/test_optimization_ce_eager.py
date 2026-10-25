"""NONFROZEN functional CE tests; CPU/mocks only, no frozen expected data."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import torch
from ce_eager import Adapter
from ce_eager_contract import dimensions,make_case,validate_inputs,check_output


def spec(need=True,up=-.75,ignored=False):return dict(seed=190910,parameters=dict(rows=4,V=23,needs_grad=need,upstream=up if need else None,smoothing=.1,all_ignored=ignored))


class Kernel:
    def __init__(self,name):self.name=name;self.fn=SimpleNamespace(__name__=name);self.compiles=0;self.launches=0;self.fail=False
    def warmup(self,*a,**kw):
        self.compiles+=1
        if self.fail:raise RuntimeError('late component failure')
        return SimpleNamespace(hash='a'*64,asm={'npubin':b'code'})
    def __getitem__(self,grid):
        def launch(*a,**kw):
            self.launches+=1
            if self.name=='ce_reset_copy':a[1].copy_(a[0]);a[2].zero_()
            elif self.name=='ce_loss_zero':a[0].zero_()
            elif self.name=='element_mul_kernel':a[0].mul_(a[2])
            elif self.name=='ce_delayed_gradient':
                x,y,m,d,c,n=a;x.copy_(self.gradient(x,y,n,kw['label_smoothing']))
            else:
                stats=self.name=='ce_stats_forward'
                x=a[0];y=a[5] if stats else a[2];loss=a[8] if stats else a[5];n=a[13] if stats else a[10]
                # Independent torch CE implementation used only for mocked launches.
                valid=y!=-100;eps=kw['label_smoothing'];logp=x.double().log_softmax(1)
                loss[valid]=(-(1-eps)*logp[valid,y[valid]]-eps*logp[valid].mean(1)).float()/n
                if not stats:x.copy_(self.gradient(x,y,n,eps))
        return launch
    @staticmethod
    def gradient(x,y,n,eps):
        out=torch.zeros_like(x);valid=y!=-100
        if n:
            out[valid]=(x[valid].double().softmax(1)-eps/x.shape[1]).float()/n
            out[valid,y[valid]]-=(1-eps)/n
        return out


class CETests(unittest.TestCase):
    def setUp(self):
        names=('ce_reset_copy','ce_loss_zero','ce_stats_forward','ce_delayed_gradient','cross_entropy_kernel','element_mul_kernel')
        self.k={n:Kernel(n) for n in names}
        m=patch.dict(sys.modules,{'triton':SimpleNamespace(__version__='mock'),'torch_npu':SimpleNamespace(),'ce_eager_kernel':SimpleNamespace(**self.k,__file__=str(ROOT/'examples/optimization-library/triton/ce_eager_kernel.py'))})
        m.start();self.addCleanup(m.stop)

    def test_provider_negative_upstream_ignored_and_none(self):
        for need in (True,False):
            c=spec(need);f=make_case(c);x,y=f['inputs'].values();ref=torch.nn.functional.cross_entropy(x.double(),y,label_smoothing=.1)
            self.assertTrue(torch.allclose(f['expected'][0],ref,atol=1e-14,rtol=1e-14))
            if need:
                xx=x.double().requires_grad_();torch.nn.functional.cross_entropy(xx,y,label_smoothing=.1).backward()
                self.assertTrue(torch.allclose(f['expected'][1],xx.grad*(-.75),atol=1e-14,rtol=1e-14))
            else:self.assertIsNone(f['expected'][1])
            allzero=make_case(spec(need,ignored=True));self.assertEqual(allzero['expected'][0],0)
            if need:self.assertEqual(allzero['expected'][1].count_nonzero(),0)

    def test_domain_dtype_layout_alias_and_output_diagnostics(self):
        for key,value in [('upstream',float('nan')),('upstream',3.),('needs_grad',1),('smoothing',.2),('targets',[0,1,2,23]),('V',1)]:
            c=spec();c['parameters'][key]=value
            with self.assertRaises(ValueError):dimensions(c)
        c=spec();f=make_case(c)
        for name,value in [('X',f['inputs']['X'].double()),('X',f['inputs']['X'].t()),('Y',torch.tensor([-1,0,1,2])),('X',torch.full((4,23),float('inf')))]:
            v=dict(f['inputs']);v[name]=value
            with self.assertRaises(ValueError):validate_inputs(v,c)
        good=tuple(x.float() if x is not None else x for x in f['expected'])
        self.assertTrue(check_output(good,f['expected'])['passed'])
        for bad in ([*good],(good[0],),(good[0].double(),good[1]),(good[0],None)):
            self.assertFalse(check_output(bad,f['expected'])['passed'])
        wrong=good[1].clone();wrong[1,2]=float('nan');r=check_output((good[0],wrong),f['expected'])
        self.assertEqual(r['outputs'][1]['nonfinite_count'],1);self.assertEqual(r['outputs'][1]['first_mismatch']['coordinates'],[1,2]);json.dumps(r,allow_nan=False)

    def test_probe_atomic_retry_and_component_counts(self):
        for need in (True,False):
            c=spec(need);f=make_case(c);n=validate_inputs(f['inputs'],c)
            for arm in ('before','after'):
                item=Adapter()._device_item(arm,c,f['inputs'],n,probe=True)
                with self.assertRaises(RuntimeError):item.run()
                self.assertEqual(sum(x.launches for x in self.k.values()),0)
                self.assertEqual(item.metadata['private_reset_copy'],arm=='after' or need)
                self.assertEqual(item.metadata['saved_stats'],arm=='before' and need)
                self.assertEqual(item.metadata['triton_component_count'],(4 if arm=='before' else 3) if need else 2)
        c=spec();f=make_case(c);item=Adapter()._device_item('before',c,f['inputs'],3,probe=True)
        self.k['element_mul_kernel'].fail=True
        with self.assertRaises(RuntimeError):item.compile_only()
        self.assertEqual(item.compilation_state(),[]);first=self.k['ce_reset_copy'].compiles
        self.k['element_mul_kernel'].fail=False;item.compile_only();self.assertEqual(self.k['ce_reset_copy'].compiles,first+1)
        counts=[k.compiles for k in self.k.values()];item.compile_only();self.assertEqual(counts,[k.compiles for k in self.k.values()])

    def test_reused_adapter_orders_immutability_canaries_defaultdtype(self):
        a=Adapter();old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);self.addCleanup(torch.set_default_dtype,old)
        with patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True):
            for need in (True,False):
                for order in (('before','after'),('after','before'),('before','after')):
                    c=spec(need);f=make_case(c)
                    for arm in order:
                        item=a.prepare(arm,c,f,None);actual=item.run();result=a.validate(arm,actual,f,c)
                        self.assertTrue(result['passed'],result);self.assertEqual(actual[0].dtype,torch.float32)
                        self.assertEqual(result['paired_output_bits_secondary'],None if arm==order[0] else True)
                        self.assertEqual(item.metadata['torch_sum_count'],1)
                        item.buffers[0][0]=0;self.assertFalse(a.validate(arm,actual,f,c)['passed']);item.buffers[0][0]=-701
                        item.inputs['X'][0,0]+=1;self.assertFalse(a.validate(arm,actual,f,c)['passed'])

    def test_externalcwd_cli_success_error_and_strict_json(self):
        cli=ROOT/'examples/optimization-library/triton/ce_eager_cli.py'
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);case=t/'case.json';case.write_text(json.dumps(dict(spec(False),role='guard')));prov=t/'prov.json';prov.write_text('{}')
            args=['--case',str(case),'--output',str(t/'out'),'--cache',str(t/'cache'),'--provenance',str(prov)]
            result=subprocess.run([sys.executable,str(cli),*args,'--validate-only'],cwd=tmp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['role'],'guard')
            wrapper="import runpy,sys,types,json,pathlib; m=types.ModuleType('optimization_pair'); m.run_pair=lambda adapter,spec,out,cache,provenance:(out.mkdir(),(out/'result.json').write_text(json.dumps({'status':'complete'})+'\\n'),{'status':'complete'})[-1]; sys.modules['optimization_pair']=m; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
            result=subprocess.run([sys.executable,'-c',wrapper,str(cli),*args],cwd=tmp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads((t/'out/result.json').read_text())['status'],'complete')
            c=spec();c['parameters']['targets']=[-1,0,1,2];case.write_text(json.dumps(c));(t/'out/result.json').unlink();(t/'out').rmdir()
            result=subprocess.run([sys.executable,str(cli),*args],cwd=tmp,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((t/'out').exists());self.assertFalse((t/'cache').exists())


if __name__=='__main__':unittest.main()
