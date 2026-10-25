"""NONFROZEN functional host tests; mocked launch boundary, no NPU work."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import torch
import request_metadata
from request_metadata import Adapter
from request_metadata_contract import dimensions,make_case,oracle,validate_inputs,check_output

def spec(pre=True):return dict(seed=190905,parameters=dict(tokens=5,topK=256,max_blocks=9,requests=3,HAS_PREFILL=pre,invalid_boundaries=True))

class Kernel:
    def __init__(self):self.fn=SimpleNamespace(__name__='mock_request');self.compiles=0;self.launches=0;self.fail=False
    def warmup(self,*args,**kwargs):
        self.compiles+=1
        if self.fail:self.fail=False;raise RuntimeError('compile failure')
        return SimpleNamespace(hash='a'*64,asm={'npubin':b'code'})
    def __getitem__(self,grid):
        def launch(req,table,idx,out,p,workspace,*args,**kw):
            self.launches+=1;v=dict(R=req,B=table,I=idx)
            if p is not None:v.update(P=p,W=workspace)
            out.copy_(oracle(v))
        return launch

class RequestTests(unittest.TestCase):
    def setUp(self):
        self.kernel=Kernel()
        for p in (patch.dict(sys.modules,{'triton':SimpleNamespace(__version__='mock'),'torch_npu':SimpleNamespace(),
                'request_metadata_kernel':SimpleNamespace(request_tile_kernel=self.kernel,_convert_req_index_to_global_index_kernel=self.kernel)}),
                patch.object(request_metadata,'vector_cores',lambda:48)):
            p.start();self.addCleanup(p.stop)

    def test_integer_oracle_invalid_prefill_and_real_large_indices(self):
        large=spec();large['parameters'].update(tokens=1,requests=3,topK=256,max_blocks=262147,large_integer_indices=True,invalid_boundaries=False)
        for c in (spec(),spec(False),large):
            f=make_case(c);v=f['inputs'];expected=torch.empty_like(v['I'])
            for r in range(v['I'].shape[0]):
                for j in range(v['I'].shape[1]):
                    tok=int(v['I'][r,j]);block=tok//64;pid=int(v['P'][r]) if 'P' in v else -1
                    expected[r,j]=(-1 if tok<0 or block>=v['B'].shape[1] else
                        int(v['W'][pid])+tok if pid>=0 else int(v['B'][int(v['R'][r]),block])*64+tok%64)
            self.assertTrue(torch.equal(f['expected'],expected))
        self.assertGreater(int(make_case(large)['inputs']['I'].max()),2**24)

    def test_domain_layout_alias_overflow_and_output_contract(self):
        for key,value in [('topK',129),('tokens',0),('requests',2**30),('HAS_PREFILL',1)]:
            c=spec();c['parameters'][key]=value
            with self.assertRaises(ValueError):dimensions(c)
        c=spec();f=make_case(c)
        for name,value in [('R',3),('B',100001),('P',3),('W',17),('I',2**31-1)]:
            v={n:x.clone() for n,x in f['inputs'].items()};v[name].view(-1)[0]=value
            with self.assertRaises(ValueError):validate_inputs(v,c)
        for bad in (f['inputs']['I'].long(),f['inputs']['I'].t().contiguous().t()):
            v=dict(f['inputs']);v['I']=bad
            with self.assertRaises(ValueError):validate_inputs(v,c)
        v=dict(f['inputs']);v['P']=v['R']
        with self.assertRaises(ValueError):validate_inputs(v,c)
        for actual in ((f['expected'],),f['expected'].float(),f['expected'][:1]):self.assertFalse(check_output(actual,f['expected'])['passed'])
        with self.assertRaises(ValueError):Adapter()._prepare_historical('invalid',c,f,None)

    def test_compile_failure_retry_probe_guard_and_default_dtype(self):
        c=spec();f=make_case(c);a=Adapter();probe=a._device_item('after',c,f['inputs'],probe=True)
        with self.assertRaises(RuntimeError):probe.run()
        self.assertEqual((self.kernel.compiles,self.kernel.launches),(0,0))
        self.kernel.fail=True
        with self.assertRaises(RuntimeError):probe.compile_only()
        self.assertEqual(probe.compilation_state(),[]);probe.compile_only()
        old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
        try:item=a._device_item('before',c,f['inputs']);self.assertEqual(item.run().dtype,torch.int32);item.run()
        finally:torch.set_default_dtype(old)
        self.assertEqual((self.kernel.compiles,self.kernel.launches),(3,2))

    def test_reused_adapter_orders_private_outputs_canaries_and_inputs(self):
        a=Adapter()
        with patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True):
            for order in (('before','after'),('after','before'),('before','after')):
                c=spec();f=make_case(c)
                for arm in order:
                    item=a._prepare_historical(arm,c,f,None);actual=item.run();r=a.validate(arm,actual,f,c)
                    self.assertTrue(r['passed']);self.assertEqual(r['paired_output_bits'],None if arm==order[0] else True)
                    item.output_storage[0]=0;self.assertFalse(a.validate(arm,actual,f,c)['passed']);item.output_storage[0]=-701
                    item.inputs['R'][0]=(item.inputs['R'][0]+1)%3;self.assertFalse(a.validate(arm,actual,f,c)['passed'])

    def test_public_rejection_inspection_and_singletransfer_diagnostics(self):
        c=spec();f=make_case(c);a=Adapter()
        with self.assertRaises(RuntimeError):a.make_case(c)
        with self.assertRaises(RuntimeError):request_metadata.factory().make_case(c)
        with self.assertRaises(RuntimeError):a.prepare('before',c,f,None)
        with self.assertRaises(RuntimeError):a.probe('before',c)
        self.assertEqual((self.kernel.compiles,self.kernel.launches),(0,0))
        original=torch.empty
        with patch.object(torch,'empty',lambda *args,**kw:original(*args,**dict(kw,device='cpu'))):
            probe=Adapter(compile_inspection=True).probe('after',c)
            with self.assertRaises(RuntimeError):probe.run()
            probe.compile_only()
        self.assertEqual((self.kernel.compiles,self.kernel.launches),(1,0))
        got=f['expected'].clone();got[0,2]=19;got[1,3]=20
        cpu=torch.Tensor.cpu;calls=[]
        with patch.object(torch.Tensor,'cpu',lambda x:(calls.append(1),cpu(x))[1]):
            result=request_metadata.check_output(got,f['expected'])
        self.assertEqual(len(calls),1);self.assertEqual(result['mismatch_count'],2)
        self.assertEqual(result['first_mismatch'],dict(index=[0,2],actual=19,expected=int(f['expected'][0,2])))

    def test_flat_nested_externalcwd_cli(self):
        c=spec();self.assertEqual(dimensions(c),dimensions(dict(c['parameters'],seed=c['seed'])))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'case.json';p.write_text(json.dumps(dict(c,role='guard')))
            cmd=[sys.executable,str(ROOT/'examples/optimization-library/triton/request_metadata_cli.py'),'--case',str(p),'--output',tmp+'/o','--cache',tmp+'/c','--provenance',tmp+'/p','--validate-only']
            r=subprocess.run(cmd,cwd=tmp,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['role'],'guard')
            denied=subprocess.run(cmd[:-1],cwd=tmp,capture_output=True,text=True)
            self.assertEqual(denied.returncode,2);self.assertFalse((Path(tmp)/'c').exists());self.assertFalse((Path(tmp)/'o').exists())

if __name__=='__main__':unittest.main()
