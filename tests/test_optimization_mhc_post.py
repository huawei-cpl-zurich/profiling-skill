import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'examples/optimization-library/triton')]
import torch
from mhc_post_contract import dimensions,make_case,reference,validate_inputs,check_output
from mhc_post import Adapter
from optimization_pair import fixture


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(seed=37,case_id='unit',role='guard',parameters=dict(BS=2,D=16))

    def test_flat_nested_and_admission(self):
        self.assertEqual(dimensions(self.spec)[1:],(2,16))
        self.assertEqual(dimensions(dict(seed=37,BS=2,D=16))[1:],(2,16))
        for key,value in [('D',65),('BS',0),('FUSED',True),('enable_fp_fusion',True)]:
            bad=copy.deepcopy(self.spec);bad['parameters'][key]=value
            with self.assertRaises(ValueError):make_case(bad)

    def test_oracle_user_transpose_and_finite_payload(self):
        case=make_case(self.spec);i=case['inputs']
        manual=torch.zeros(2,4,16,dtype=torch.float64)
        for b in range(2):
            for n in range(4):
                manual[b,n]=i['hp'][b,n].double()*i['x'][b].double()
                for j in range(4):manual[b,n]+=i['H'][b,j,n].double()*i['R'][b,j].double()
        self.assertTrue(torch.equal(case['expected'],manual.float()))
        self.assertFalse(torch.equal(reference(i),reference(dict(i,H=i['H'].transpose(1,2).contiguous()))))
        self.assertTrue(check_output(case['expected'],case['expected'])['passed'])
        for bad in [(case['expected'],),case['expected'].double(),case['expected'][:1]]:
            self.assertFalse(check_output(bad,case['expected'])['passed'])

    def test_reject_mutable_views_alias_dtype_and_nonfinite(self):
        for name in ('dtype','nan','view','alias'):
            i=make_case(self.spec)['inputs']
            if name=='dtype':i['x']=i['x'].double()
            elif name=='nan':i['R'][0,0,0]=float('nan')
            elif name=='view':i['x']=torch.zeros(2,17)[:,:16]
            else:i['hp']=i['H'].view(2,16)[:,:4]
            with self.assertRaises(ValueError):validate_inputs(i,self.spec)

    def test_fixture_cache_once_and_invalid_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache'
            a=Adapter()
            _,first=fixture(a,self.spec,cache,torch)
            _,second=fixture(a,self.spec,cache,torch)
            self.assertEqual([first['oracle_calls'],second['oracle_calls']],[1,0])
            bad=copy.deepcopy(self.spec);bad['parameters']['algorithm']='other'
            with self.assertRaises(ValueError):fixture(a,bad,cache,torch)
            self.assertEqual(len(list(cache.iterdir())),1)

    def test_current_pair_bits_canaries_and_input_immutability(self):
        from types import SimpleNamespace
        case=make_case(self.spec);a=Adapter()
        item=SimpleNamespace(inputs={k:v.clone() for k,v in case['inputs'].items()},
                             storages=[torch.tensor([-701.,0.,-701.])])
        a.prepared={'before':item,'after':item}
        expected=case['expected']
        self.assertTrue(a.validate('before',expected,case,self.spec)['passed'])
        self.assertTrue(a.validate('after',expected.clone(),case,self.spec)['paired_exact_bits'])
        changed=expected.clone()
        changed.view(torch.int32).flatten()[0]+=1
        mismatch=a.validate('after',changed,case,self.spec)
        self.assertFalse(mismatch['paired_exact_bits'])
        self.assertFalse(mismatch['passed'])
        a.bits={}
        item.storages[0][0]=0
        self.assertFalse(a.validate('before',expected,case,self.spec)['passed'])
        item.storages[0][0]=-701.
        item.inputs['x'][0,0]=0
        self.assertEqual(a.validate('before',expected,case,self.spec)['mutated_inputs'],['x'])

    def test_standalone_cli_rejects_before_npu_or_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);spec=copy.deepcopy(self.spec);spec['parameters']['D']=65
            (p/'case.json').write_text(json.dumps(spec));(p/'prov.json').write_text('{}')
            # Standard CLI uses module names; import the adapter from an external cwd.
            import os
            env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
                '--adapter','mhc_post:factory','--case',str(p/'case.json'),'--provenance',str(p/'prov.json'),
                '--output',str(p/'out'),'--cache',str(p/'cache'),'--correctness-only'],cwd=tmp,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertIn('unsupported MHC post dimensions',(p/'out/result.json').read_text())
            self.assertFalse((p/'cache').exists())

if __name__=='__main__':unittest.main()
