import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'examples/optimization-library/triton/pack_selected_contract.py'
spec=importlib.util.spec_from_file_location('pack_selected_contract',MODULE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sys.path.insert(0,str(MODULE.parent))
sys.path.insert(0,str(ROOT/'scripts'))
import pack_selected_store as operator
import optimization_pair as pair


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


    def test_validator_rejects_missing_wrong_container_shape_and_dtype(self):
        for compound in (False,True):
            case=dict(lengths=[0,2],D=5,seed=9342,also_empty_batch=compound)
            fixture=c.make_case(case)
            adapter=operator.Adapter()
            adapter.prepared['before']=SimpleNamespace(inputs=(fixture['x'],fixture['lengths']),storage=torch.full((32,),12345.))
            expected=fixture['expected']
            actual=tuple(x.clone() for x in expected) if compound else expected.clone()
            self.assertTrue(adapter.validate('before',actual,fixture,case)['passed'])
            wrong=[(),None,[],{},(None,),torch.empty((0,0,5),dtype=torch.float64),torch.empty((0,0,4))]
            if compound:wrong.extend([(actual[0],),(actual[0],None),list(actual),
                (actual[0].half(),actual[1]),(torch.empty((2,0,4)),actual[1])])
            else:wrong.append((actual,))
            for value in wrong:
                with self.subTest(compound=compound,value=type(value).__name__):
                    self.assertFalse(adapter.validate('before',value,fixture,case)['passed'])

    def test_public_empty_cli_checks_both_arms_without_capture(self):
        calls=[]
        def prepare(adapter,arm,spec,fixture,output):
            result=fixture['expected'].clone()
            item=SimpleNamespace(inputs=(fixture['x'],fixture['lengths']),storage=torch.full((32,),12345.),
                run=lambda:(calls.append(arm),result)[1],expected_kernels=['empty'],sources=[MODULE],
                metadata=dict(kernel_launches_per_api=0),compilation_state=lambda:[])
            adapter.prepared[arm]=item
            return item
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for lengths in ([],[0,0]):
                case=folder/('case'+str(len(lengths))+'.json');case.write_text(json.dumps(dict(lengths=lengths,D=5,seed=9343)))
                provenance=folder/'provenance.json';provenance.write_text('{}')
                out=folder/('out'+str(len(lengths)))
                argv=['pack_selected_store','--case',str(case),'--cache',str(folder/'cache'),'--output',str(out),'--provenance',str(provenance)]
                with mock.patch.object(operator.Adapter,'prepare',prepare),mock.patch.object(pair,'capture_ten',side_effect=AssertionError('empty capture')),mock.patch.object(sys,'argv',argv):
                    self.assertEqual(operator.main(),0)
                report=json.loads((out/'result.json').read_text())
                self.assertEqual(report['mode'],'correctness-only')
                self.assertEqual(report['counts'],dict(correctness_calls=2,profiled_calls=0,capture_calls=0))
            self.assertEqual(calls,['before','after','before','after'])

    def test_public_nonempty_cli_preserves_profile_routing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);case=folder/'case.json';case.write_text(json.dumps(dict(lengths=[1,0],D=5,seed=9344)))
            provenance=folder/'provenance.json';provenance.write_text('{}')
            argv=['pack_selected_store','--case',str(case),'--cache',str(folder/'cache'),'--output',str(folder/'out'),'--provenance',str(provenance)]
            with mock.patch.object(pair,'run_pair',return_value=dict(status='complete')) as run,mock.patch.object(sys,'argv',argv):
                self.assertEqual(operator.main(),0)
            self.assertIs(run.call_args.kwargs['correctness_only'],False)
