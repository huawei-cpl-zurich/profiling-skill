import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'examples/optimization-library/triton/pack_selected_contract.py'
spec=importlib.util.spec_from_file_location('pack_selected_contract',MODULE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class PackContractTests(unittest.TestCase):
    def test_flat_nested_and_exact_oracle(self):
        p=dict(lengths=[0,2,1],D=5,pad_value='-inf',signedzero_and_infinity_input=True)
        flat=c.make_case(dict(p,seed=910));nested=c.make_case(dict(parameters=p,seed=910))
        self.assertTrue(torch.equal(flat['x'].view(torch.int32),nested['x'].view(torch.int32)))
        e=flat['expected'];self.assertEqual(tuple(e.shape),(3,2,5))
        self.assertTrue(torch.isneginf(e[0]).all())
        self.assertTrue(torch.equal(e[1].view(torch.int32),flat['x'][:2].view(torch.int32)))
        self.assertEqual(e[1,0,1].view(torch.int32).item(),-2147483648)
        self.assertTrue(torch.isneginf(e[2,1]).all())

    def test_empties_and_compound(self):
        f=c.make_case(dict(lengths=[0,0,0],D=3,also_empty_batch=True,seed=901))
        self.assertEqual([tuple(x.shape) for x in f['expected']],[(3,0,3),(0,0,3)])
        self.assertEqual(tuple(c.make_case(dict(lengths=[],D=4))['expected'].shape),(0,0,4))

    def test_input_rejections(self):
        p=c.dimensions(dict(lengths=[2],D=3));f=c.make_case(dict(p,seed=912))
        c.checked_inputs(f,p)
        for replacement in (f['x'].half(),f['x'].t(),torch.zeros((2,6))[:,::2]):
            with self.assertRaises(ValueError): c.checked_inputs(dict(f,x=replacement),p)
        with self.assertRaises(ValueError):c.checked_inputs(dict(f,lengths=torch.tensor([1],dtype=torch.int32)),p)
        for p in (dict(lengths=[-1],D=3),dict(lengths=[True],D=3),dict(lengths=[2],D=0),dict(lengths=[2**30],D=4),dict(lengths=[1],D=2,pad_value='nan')):
            with self.assertRaises(ValueError):c.make_case(p)

    def test_public_cli_from_external_cwd_validates_flat_nested_before_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder); case=folder/'case.json';cache=folder/'cache'
            script=ROOT/'examples/optimization-library/triton/pack_selected_store.py'
            cmd=[sys.executable,str(script),'--case',str(case),'--cache',str(cache),'--output',str(folder/'out'),'--provenance',str(folder/'missing.json'),'--validate-only']
            for params in (dict(lengths=[3,0],D=2,pad_value=-0.0),dict(parameters=dict(lengths=[3,0],D=2,pad_value=-0.0))):
                case.write_text(json.dumps(params));r=subprocess.run(cmd,cwd=folder,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['parameters']['lengths'],[3,0])
            case.write_text(json.dumps(dict(lengths=[-1],D=2)))
            rejected=subprocess.run(cmd[:-1],cwd=folder,capture_output=True,text=True)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn('nonnegative int32 lengths required',rejected.stderr)
            self.assertFalse(cache.exists());self.assertFalse((folder/'out').exists())
