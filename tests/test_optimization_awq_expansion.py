"""Functional host tests use NONFROZEN seeds/shapes and mocked launch boundaries."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import torch
from awq_expansion import Adapter
from awq_expansion_contract import dimensions,make_case,decode,validate_inputs,check_output


def spec(k=35,n=24,g=35):return dict(seed=190901,parameters=dict(K=k,N=n,group=g,all_nibbles_signed_words=True))


class Kernel:
    def __init__(self):self.fn=SimpleNamespace(__name__='mock_awq');self.compiles=0;self.launches=0;self.fail=False
    def warmup(self,*args,**kwargs):
        self.compiles+=1
        if self.fail:self.fail=False;raise RuntimeError('compile failure')
        return SimpleNamespace(hash='a'*64,asm={'npubin':b'code'})
    def __getitem__(self,grid):
        def launch(q,s,z,g,out,c,k,**kwargs):
            self.launches+=1
            out.copy_(((decode(q)-decode(z).repeat_interleave(g,0))*s.repeat_interleave(g,0)).half())
        return launch


class AWQTests(unittest.TestCase):
    def setUp(self):
        self.kernel=Kernel()
        self.mocks=patch.dict(sys.modules,{'triton':SimpleNamespace(__version__='mock'),
            'torch_npu':SimpleNamespace(), 'awq_expansion_kernel':SimpleNamespace(
                awq_lane_kernel=self.kernel,awq_dequantize_kernel=self.kernel)})
        self.mocks.start();self.addCleanup(self.mocks.stop)

    def test_codec_signed_words_and_integer_exact_oracle(self):
        words=torch.tensor([[-1,-2147483648,0x76543210]],dtype=torch.int32)
        self.assertEqual(decode(words).tolist(),[[15]*8+[0,0,0,0,0,0,0,8]+[0,4,1,5,2,6,3,7]])
        for case in (spec(),spec(64,152,32)):
            f=make_case(case);k,n,g=dimensions(case)
            expected=[]
            for row in range(k):
                values=[]
                for col in range(n):
                    shift=(0,4,1,5,2,6,3,7)[col%8]*4
                    q=(int(f['inputs']['Q'][row,col//8])&0xffffffff)>>shift&15
                    z=(int(f['inputs']['Z'][row//g,col//8])&0xffffffff)>>shift&15
                    values.append((q-z)*float(f['inputs']['S'][row//g,col]))
                expected.append(values)
            self.assertTrue(torch.equal(torch.tensor(expected,dtype=torch.float16).view(torch.int16),f['expected'].view(torch.int16)))

    def test_domains_shapes_layout_and_output_arity(self):
        for k,n,g in ((63,24,32),(64,23,32),(96,24,48),(0,8,32),(2**30,128,32)):
            with self.assertRaises(ValueError):dimensions(spec(k,n,g))
        case=spec();f=make_case(case)
        bad=dict(f['inputs']);bad['S']=bad['S'].float()
        with self.assertRaises(ValueError):validate_inputs(bad,case)
        bad=dict(f['inputs']);bad['Q']=bad['Q'].t().contiguous().t()
        with self.assertRaises(ValueError):validate_inputs(bad,case)
        bad=dict(f['inputs']);bad['S']=bad['S'].clone();bad['S'][0,0]=float('nan')
        with self.assertRaises(ValueError):validate_inputs(bad,case)
        for actual in ((f['expected'],),f['expected'].float(),f['expected'][:1]):
            self.assertFalse(check_output(actual,f['expected'])['passed'])
        with self.assertRaises(ValueError):Adapter().prepare('invalid',case,f,None)

    def test_compile_retry_probe_guard_and_firstlaunch_qualification(self):
        a=Adapter();f=make_case(spec());probe=a._device_item('after',spec(),f['inputs'],probe=True)
        with self.assertRaises(RuntimeError):probe.run()
        self.assertEqual((self.kernel.compiles,self.kernel.launches),(0,0))
        self.kernel.fail=True
        with self.assertRaises(RuntimeError):probe.compile_only()
        self.assertEqual(probe.compilation_state(),[])
        probe.compile_only();self.assertEqual(self.kernel.compiles,2)
        item=a._device_item('before',spec(),f['inputs']);item.run();item.run()
        self.assertEqual((self.kernel.compiles,self.kernel.launches),(3,2))
        self.assertTrue(torch.equal(item.output,f['expected']))
        self.assertEqual(len(item.compilation_state()),1)

    def test_reused_adapter_both_orders_contracts_canaries_immutable(self):
        a=Adapter()
        with patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True):
            for order in (('before','after'),('after','before'),('before','after')):
                case=spec();f=make_case(case)
                for arm in order:
                    item=a.prepare(arm,case,f,None);actual=item.run();result=a.validate(arm,actual,f,case)
                    self.assertTrue(result['passed']);self.assertEqual(result['paired_output_bits'],None if arm==order[0] else True)
                    item.output_storage[0]=0
                    self.assertFalse(a.validate(arm,actual,f,case)['passed'])
                    item.output_storage[0]=-701
                    item.inputs['Q'][0,0]=0
                    self.assertFalse(a.validate(arm,actual,f,case)['passed'])

    def test_flat_nested_externalcwd_cli(self):
        case=spec();flat=dict(case['parameters'],seed=case['seed']);self.assertEqual(dimensions(case),dimensions(flat))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'case.json';p.write_text(json.dumps(dict(case,role='guard')))
            cmd=[sys.executable,str(ROOT/'examples/optimization-library/triton/awq_expansion_cli.py'),
                '--case',str(p),'--output',tmp+'/out','--cache',tmp+'/cache','--provenance',tmp+'/provenance','--validate-only']
            done=subprocess.run(cmd,cwd=tmp,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(json.loads(done.stdout)['role'],'guard')


if __name__=='__main__':unittest.main()
