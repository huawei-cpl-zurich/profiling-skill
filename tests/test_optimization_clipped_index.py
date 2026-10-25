import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'examples/optimization-library/triton'
sys.path[:0]=[str(DIRECTORY),str(ROOT/'scripts')]
from clipped_index import Adapter
from clipped_index_contract import make_case,check_output,validate_input,plan
from optimization_pair import fixture


def spec(**p):return dict(seed=93,parameters=dict(M=11,K=23,**p))


class ClippedIndexTests(unittest.TestCase):
    def test_scalar_fp64_oracle_and_guard_patterns(self):
        for case in (spec(),dict(seed=95,M=2,K=19,pattern='clamp_boundaries'),
                     dict(seed=97,M=5,K=9,pattern='zeros_signed_tail')):
            value=make_case(case);gup=value['inputs']['GUP'];m,k=value['expected'].shape
            for i in range(m):
                for j in range(k):
                    g=min(float(gup[i,j]),7);u=max(-7,min(float(gup[i,j+k]),7))
                    self.assertAlmostEqual(float(value['expected'][i,j]),(u+1)*(g/(1+math.exp(-g*1.702))),places=12)
            self.assertTrue(check_output(value['expected'].float(),value['expected'])['passed'])

    def test_padding_integer_plan_without_large_allocation(self):
        p=plan(5,37)
        self.assertEqual((p['block_m'],p['block_k']),(2,512))
        self.assertEqual(p['source_grid'],[3,1]);self.assertEqual(p['flat_grid'],[1])
        self.assertGreater(p['max_padded_input_offset'],5*2*37-1)
        self.assertFalse(p['int64']);self.assertTrue(plan(2**18,4096)['int64'])
        for m,k in ((0,3),(2,2**30),(2**31+1,3),(2**60,2**60)):
            with self.assertRaises(ValueError):plan(m,k)

    def test_admission_before_cache_and_output_errors(self):
        value=make_case(spec());v=value['inputs']['GUP']
        for bad in (v.double(),v[:,::2],v[:,:-1],torch.full_like(v,float('nan')),torch.zeros(12,46)[1:]):
            with self.assertRaises(ValueError):validate_input(bad,spec())
        bad=v.clone();bad[0,0]=.1
        with self.assertRaises(ValueError):validate_input(bad,spec())
        expected=value['expected'];actual=expected.float()
        for bad in ((actual,),actual.double(),actual.flatten(),actual+.01):
            self.assertFalse(check_output(bad,expected)['passed'])
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache'
            a,x=fixture(Adapter(),spec(),cache,torch);b,y=fixture(Adapter(),spec(),cache,torch)
            self.assertEqual((x['oracle_calls'],y['oracle_calls']),(1,0))
            self.assertTrue(torch.equal(a['expected'],b['expected']))
            with self.assertRaises(ValueError):fixture(Adapter(),dict(seed=9,M=0,K=23),Path(tmp)/'bad',torch)
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_public_launch_probe_pair_reuse_and_storage_contracts(self):
        calls=[];compiles=[];fail=[]
        artifact=SimpleNamespace(hash='b'*64,asm={'npubin':b'binary'})
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*a,**kw):
                compiles.append(self.fn.__name__)
                if fail:fail.pop();raise RuntimeError('compile failure')
                return artifact
            def __getitem__(self,grid):
                def run(gup,out,m,k,*a,**kw):
                    calls.append((self.fn.__name__,grid,kw))
                    g=gup[:,:k].clamp(max=7);u=gup[:,k:].clamp(-7,7)
                    out.copy_((u+1)*(g*torch.sigmoid(g*1.702)))
                    return artifact
                return run
        mods=dict(torch_npu=SimpleNamespace(),triton=SimpleNamespace(__version__='mock'),
                  clipped_index_kernel=SimpleNamespace(_clipped_swiglu_kernel=Kernel('source'),_clipped_swiglu_flat_kernel=Kernel('flat')))
        empty=torch.empty
        def cpu_empty(*a,**kw):kw.pop('device',None);return empty(*a,**kw)
        with patch.dict(sys.modules,mods),patch.object(torch.Tensor,'npu',lambda v:v.clone(),create=True),patch.object(torch,'empty',cpu_empty):
            a=Adapter();probe=a.probe('before',spec());probe.compile_only();self.assertEqual(calls,[])
            for order in (('before','after'),('after','before')):
                values=make_case(spec());items={}
                for arm in order:
                    item=a.prepare(arm,spec(),values,Path('.'));items[arm]=item
                    fail.append(True)
                    with self.assertRaises(RuntimeError):item.run()
                    self.assertEqual(item.compilation_state(),[])
                    n=len(compiles);out=item.run();self.assertEqual(len(compiles),n+1)
                    item.run();self.assertEqual(len(compiles),n+1)
                    result=a.validate(arm,out,values,spec());self.assertTrue(result['passed'],result)
                self.assertTrue(result['paired_output_bits'])
                self.assertNotEqual(items['before'].output_storage.data_ptr(),items['after'].output_storage.data_ptr())
                self.assertEqual(items['after'].metadata['grid'],(3,1))
                items[order[-1]].output_storage[0]=0
                self.assertFalse(a.validate(order[-1],out,values,spec())['passed'])
            values=make_case(spec());first=a.prepare('before',spec(),values,Path('.'));v=first.run();a.validate('before',v,values,spec())
            second=a.prepare('after',spec(),values,Path('.'));v=second.run();v[0,0]=torch.nextafter(v[0,0],torch.tensor(float('inf')))
            result=a.validate('after',v,values,spec());self.assertFalse(result['paired_output_bits']);self.assertFalse(result['passed'])
            # Reusing the adapter begins a fresh pair and discards the prior mismatch.
            first=a.prepare('after',spec(),values,Path('.'));v=first.run()
            self.assertIsNone(a.validate('after',v,values,spec())['paired_output_bits'])

    def test_external_cwd_cli_flat_nested_and_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);case=folder/'case.json'
            cmd=[sys.executable,str(DIRECTORY/'clipped_index_cli.py'),'--case',str(case),'--output',str(folder/'out'),
                 '--cache',str(folder/'cache'),'--provenance',str(folder/'provenance.json'),'--validate-only']
            for s in (dict(spec(),role='guard'),dict(seed=93,role='heldout',**spec()['parameters'])):
                case.write_text(json.dumps(s));p=subprocess.run(cmd,cwd=folder,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['role'],s['role'])
            case.write_text(json.dumps(dict(seed=3,M=0,K=17)))
            self.assertNotEqual(subprocess.run(cmd,cwd=folder,capture_output=True).returncode,0)
            self.assertFalse((folder/'out').exists());self.assertFalse((folder/'cache').exists())


if __name__=='__main__':unittest.main()
