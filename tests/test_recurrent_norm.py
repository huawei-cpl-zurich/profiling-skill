import contextlib
import io
from unittest.mock import patch
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'examples/optimization-library/triton'
sys.path.insert(0,str(MODULE))
from recurrent_norm_contract import dimensions,make_case,reference,diagnostics,validate_inputs
import recurrent_norm
from recurrent_norm import Adapter, dimensions as public_dimensions, normalize_spec


class RecurrentNormTests(unittest.TestCase):
    def spec(self,**parameters):
        return dict(seed=991,parameters=dict(B=1,T=2,H=1,HV=2,K=3,V=5,**parameters))

    def test_reference_two_outputs_and_nonzero_initial_state(self):
        f=make_case(self.spec())
        validate_inputs(f['inputs'],self.spec())
        saved=f['inputs']['h0'].clone()
        actual=tuple(x.float() for x in f['expected'])
        self.assertTrue(diagnostics(actual,f)['passed'])
        self.assertTrue(torch.equal(saved,f['inputs']['h0']))
        other={n:x.clone() for n,x in f['inputs'].items()}
        other['h0'].zero_()
        self.assertFalse(torch.equal(reference(other)[1],f['expected'][1]))
        for bad in ((),actual[:1],(*actual,actual[0])):
            self.assertFalse(diagnostics(bad,f)['passed'])
        bad=list(actual);bad[1]=bad[1]+.01
        self.assertFalse(diagnostics(bad,f)['passed'])

    def test_zero_epsilon_and_grouped_mapping(self):
        spec=self.spec(zeros_and_epsilon=True)
        spec['parameters']['T']=3
        f=make_case(spec)
        self.assertTrue(torch.equal(f['expected'][0][:,0],torch.zeros_like(f['expected'][0][:,0])))
        self.assertTrue(torch.isfinite(f['expected'][1]).all())
        wrong={n:x.clone() for n,x in f['inputs'].items()}
        # A different epsilon produces a detectably different near-zero result.
        for name in ('q','k'):
            x=wrong[name]
            wrong[name]=x/torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp_min(1e-12)
        changed=reference(wrong)
        self.assertGreater((changed[0]-f['expected'][0]).abs().max().item(),1e-4)

    def test_cpu_only_exact_signed_epsilon_components(self):
        # Nonfrozen CPU oracle coverage; this establishes no NPU guard coverage.
        import math
        spec=self.spec();spec['parameters'].update(T=1,HV=1,K=2,V=1)
        inputs=make_case(spec)['inputs']
        inputs['q'][0,0,0]=torch.tensor([1e-4,-1e-4])
        inputs['k'][0,0,0]=torch.tensor([-1e-4,2e-4])
        out,ht=reference(inputs)
        q=inputs['q'].flatten().double().tolist()
        k=inputs['k'].flatten().double().tolist()
        q=[x/math.sqrt(sum(y*y for y in q)+1e-6) for x in q]
        k=[x/math.sqrt(sum(y*y for y in k)+1e-6) for x in k]
        state=[x*math.exp(inputs['g'].item()) for x in inputs['h0'].flatten().double().tolist()]
        residual=(inputs['v'].item()-sum(x*y for x,y in zip(state,k)))*inputs['beta'].item()
        state=[x+y*residual for x,y in zip(state,k)]
        self.assertAlmostEqual(out.item(),sum(x*y for x,y in zip(state,q))/math.sqrt(2),places=14)
        self.assertTrue(torch.allclose(ht.flatten(),torch.tensor(state,dtype=torch.float64),atol=1e-14,rtol=0))

    def test_bounds_flat_nested_and_input_rejection(self):
        spec=self.spec()
        self.assertEqual(dimensions(spec),dimensions(dict(seed=991,**spec['parameters'])))
        flat=dict(seed=991,scenario_designation='nonfrozen-unit',hypothesis='unit',**spec['parameters'])
        self.assertEqual(public_dimensions(spec),public_dimensions(flat))
        self.assertTrue(torch.equal(Adapter.make_case(flat)['expected'][0],make_case(spec)['expected'][0]))
        for key,value in (('T',0),('K',65),('V',513),('H',3),('HV',0)):
            bad=self.spec();bad['parameters'][key]=value
            with self.assertRaises(ValueError): dimensions(bad)
        bad=self.spec();bad['parameters']['inplace_final_state']=True
        with self.assertRaises(ValueError):dimensions(bad)
        explicit=self.spec();explicit['inplace_final_state']=False
        explicit['parameters']['is_kda']=False
        self.assertEqual(public_dimensions(explicit),dimensions(self.spec()))
        self.assertTrue(torch.equal(Adapter.make_case(explicit)['expected'][0],make_case(self.spec())['expected'][0]))
        bad=self.spec();bad['inplace_final_state']=True
        with self.assertRaises(ValueError):public_dimensions(bad)
        with self.assertRaises(ValueError):Adapter.make_case(bad)
        bad=self.spec();bad['K']=7
        with self.assertRaises(ValueError):dimensions(bad)
        f=make_case(spec);f['inputs']['h0']=None
        with self.assertRaises(ValueError):validate_inputs(f['inputs'],spec)

    def test_guard_metadata_validate_only_without_execution(self):
        spec=self.spec();spec.update(seed=992,role='guard',expected='two outputs and immutable h0',
                                    hypothesis='nonfrozen',scenario_designation='unit-guard')
        self.assertEqual(normalize_spec(spec)['expected'],spec['expected'])
        self.assertEqual(public_dimensions(spec),dimensions(self.spec()))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);case=p/'case.json';case.write_text(json.dumps(spec))
            with patch.object(recurrent_norm,'contract_make_case',side_effect=AssertionError('oracle called')), \
                 patch.object(recurrent_norm,'kernels',side_effect=AssertionError('kernel requested')), \
                 patch.object(sys,'argv',['recurrent_norm.py','--case',str(case),'--validate-only']), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(recurrent_norm.main(),0)
            self.assertEqual(json.loads(output.getvalue())['status'],'valid')
            result=subprocess.run([sys.executable,str(MODULE/'recurrent_norm.py'),'--case',str(case),
                '--validate-only'],cwd=p,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['case']['role'],'guard')
            self.assertEqual(list(p.iterdir()),[case])
            for nested in (False,True):
                bad=json.loads(json.dumps(spec))
                (bad['parameters'] if nested else bad)['is_kda']=True
                with self.assertRaises(ValueError):normalize_spec(bad)

    def test_external_cwd_cli_rejects_before_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);spec=self.spec();spec['parameters']['T']=0
            (p/'case.json').write_text(json.dumps(spec));(p/'provenance.json').write_text('{}')
            result=subprocess.run([sys.executable,str(MODULE/'recurrent_norm.py'),'--case',str(p/'case.json'),
                '--provenance',str(p/'provenance.json'),'--output',str(p/'out'),'--cache',str(p/'cache')],
                cwd=p,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('unsupported bounded recurrence shape',result.stderr)
            self.assertFalse((p/'cache').exists())
            spec=self.spec();spec['inplace_final_state']=True
            (p/'case.json').write_text(json.dumps(spec))
            result=subprocess.run([sys.executable,str(MODULE/'recurrent_norm.py'),'--case',str(p/'case.json'),
                '--provenance',str(p/'provenance.json'),'--output',str(p/'out'),'--cache',str(p/'cache')],
                cwd=p,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('unsupported optional recurrence mode',result.stderr)
            self.assertFalse((p/'cache').exists())


if __name__=='__main__':unittest.main()
