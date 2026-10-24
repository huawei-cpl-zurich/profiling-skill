"""Observable numerical contract, cache and public CLI behavior."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/triton'), str(ROOT/'scripts')]
from grpo_contract import checked_inputs, diagnostics, make_case, parameters, reference
from grpo_statistics import Adapter
from optimization_pair import fixture


def spec(**extra): return dict(seed=6400, parameters=dict(T=3, V=8, kl_type='unbias', **extra))

class ContractTests(unittest.TestCase):
    def test_global_policy_broadcast_without_kl(self):
        f = dict(logp=torch.zeros(2,3), old=torch.zeros(2,3), ref=torch.ones(2,3), adv=torch.tensor([1., -.5]))
        p = dict(T=2, V=3, kl_type='kl', kl_coef=0)
        self.assertTrue(torch.equal(reference(f,p), torch.full((2,), -.25, dtype=torch.float64)))
        p['kl_coef'] = .1
        f['ref'][1] = -.5
        self.assertTrue(torch.allclose(reference(f,p), torch.tensor([-.2,-.275], dtype=torch.float64)))

    def test_all_kl_forms(self):
        f = dict(logp=torch.zeros(1,2), old=torch.zeros(1,2), ref=torch.full((1,2), .5), adv=torch.zeros(1))
        expected = {'kl':.5,'unbias':__import__('math').exp(.5)-1.5,'mse':.125}
        for kind, value in expected.items():
            actual=reference(f, dict(T=1,V=2,kl_type=kind,kl_coef=1))
            self.assertAlmostEqual(float(actual), value, places=12)

    def test_clip_sign_guard_and_tail_no_kl(self):
        f=make_case(spec(distribution='advantage-sign-clip-boundaries'))
        self.assertEqual(f['adv'].tolist(), [-1.,0.,1.])
        ratio=f['logp'].double().exp()
        self.assertTrue(bool((ratio < .8).any()) and bool((ratio > 1.2).any()))
        tail=make_case(dict(seed=6492,parameters=dict(T=5,V=129,kl_type='mse',kl_coef=0)))
        self.assertTrue(torch.equal(tail['expected'], tail['expected'][:1].expand(5)))
        self.assertNotEqual(float(tail['expected'][0]),0.)

    def test_diagnostics_reject_nonfinite_wrong_shape_dtype_and_values(self):
        f=make_case(spec()); good=f['expected'].float()
        self.assertTrue(diagnostics(good,f)['passed'])
        for bad in (good.double(), good[:1], good+1, torch.full_like(good,float('nan'))):
            self.assertFalse(diagnostics(bad,f)['passed'])

    def test_explicit_fp32_fixture_cache_under_ambient_fp64(self):
        prior=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with tempfile.TemporaryDirectory() as path:
                a=Adapter(); f,m=fixture(a,spec(),Path(path),torch);again,n=fixture(a,spec(),Path(path),torch)
                self.assertEqual((m['oracle_calls'],n['oracle_calls']),(1,0))
                for key in ('logp','old','ref','adv'):
                    self.assertEqual(f[key].dtype,torch.float32);self.assertEqual(f[key].device.type,'cpu')
                    self.assertTrue(torch.equal(f[key],again[key]))
        finally:torch.set_default_dtype(prior)

    def test_reject_invalid_inputs_and_parameters(self):
        f=make_case(spec()); p=parameters(spec())
        for key,value in [('T',0),('kl_type','other'),('kl_coef',float('nan')),('distribution','unknown')]:
            s=spec();s['parameters'][key]=value
            with self.assertRaises(ValueError):parameters(s)
        f['adv'][0]=2
        with self.assertRaises(ValueError):checked_inputs(f,p)

    def test_public_nested_fixture_cli_and_harness_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'case.json').write_text(json.dumps(spec()))
            result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/grpo_contract.py'),
                '--case',str(path/'case.json'),'--output',str(path/'fixture.pt')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            f=torch.load(path/'fixture.pt',weights_only=True);self.assertEqual(f['expected'].shape,(3,))
            bad=spec();bad['parameters']['T']=0;(path/'case.json').write_text(json.dumps(bad));(path/'provenance.json').write_text('{}')
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path[:2]))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','grpo_statistics:Adapter',
                '--case',str(path/'case.json'),'--cache',str(path/'cache'),'--output',str(path/'result'),
                '--provenance',str(path/'provenance.json'),'--correctness-only'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            report=json.loads((path/'result/result.json').read_text());self.assertEqual(report['counts']['correctness_calls'],0)
            self.assertEqual(report['counts']['capture_calls'],0)

if __name__ == '__main__': unittest.main()
