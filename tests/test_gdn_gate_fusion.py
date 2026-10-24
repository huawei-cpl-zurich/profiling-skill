"""Observable gate contract, guards, cached fixtures and public CLI behavior."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/triton'), str(ROOT/'scripts')]
from gdn_gate_contract import INPUTS, checked_inputs, diagnostics, make_case, parameters, reference
from gdn_gate_fusion import Adapter
from optimization_pair import fixture


def spec(r=3, h=7, **extra):
    return dict(case_id='cpu-gdn',role='guard',seed=91,parameters=dict(R=r,H=h,**extra))


class GateTests(unittest.TestCase):
    def test_independent_scalar_reference_and_output_shapes(self):
        a=torch.tensor([[0.,1.]],dtype=torch.float32)
        b=torch.tensor([[0.,-2.]],dtype=torch.float32)
        alog=torch.tensor([0.,1.],dtype=torch.float32)
        bias=torch.tensor([0.,-.5],dtype=torch.float32)
        g,beta=reference(a,b,alog,bias)
        self.assertEqual(g.shape,(1,1,2)); self.assertEqual(beta.shape,(1,1,2))
        self.assertAlmostEqual(g[0,0,0].item(),-math.log(2),14)
        self.assertAlmostEqual(g[0,0,1].item(),-math.e*math.log1p(math.exp(.5)),14)
        self.assertAlmostEqual(beta[0,0,1].item(),1/(1+math.exp(2)),14)

    def test_threshold_guard_has_exact_requested_values(self):
        f=make_case(spec(2,3,distribution='threshold20'))
        self.assertEqual((f['a']+f['dt_bias'])[0].tolist(),[20-1/1024,20,20+1/1024])
        self.assertTrue(diagnostics(tuple(v.float() for v in f['expected']),f)['passed'])
        self.assertAlmostEqual(f['expected'][0][0,0,2].item(),
            -math.exp(f['A_log'][2].item())*(20+1/1024),12)

    def test_odd_tail_symmetric_beta(self):
        f=make_case(spec(3,17,distribution='symmetric-b'))
        self.assertTrue(torch.allclose(f['expected'][1][0,0]+f['expected'][1][0,-1],torch.ones(17,dtype=torch.float64)))
        checked_inputs(spec(3,17,distribution='symmetric-b'),f)

    def test_diagnostics_checks_both_outputs_and_nonfinite_dtype_shape(self):
        f=make_case(spec()); good=tuple(v.float() for v in f['expected'])
        self.assertTrue(diagnostics(good,f)['passed'])
        for bad in ((good[0],good[1]+.01),(good[0].double(),good[1]),(good[0].squeeze(0),good[1])):
            self.assertFalse(diagnostics(bad,f)['passed'])
        bad=good[0].clone();bad[0,0,0]=float('nan')
        self.assertFalse(diagnostics((bad,good[1]),f)['passed'])

    def test_fixture_cache_explicit_cpu_fp32_with_float64_default(self):
        previous=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with tempfile.TemporaryDirectory() as folder:
                f,a=fixture(Adapter(),spec(),Path(folder),torch)
                again,b=fixture(Adapter(),spec(),Path(folder),torch)
                self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0))
                for key in INPUTS:
                    self.assertEqual(f[key].dtype,torch.float32)
                    self.assertEqual(f[key].device.type,'cpu')
                    self.assertTrue(torch.equal(f[key],again[key]))
        finally: torch.set_default_dtype(previous)

    def test_rejects_invalid_shape_distribution_and_input_contract_before_device(self):
        for patch in ({'H':0},{'R':8193},{'distribution':'unknown'}):
            s=spec();s['parameters'].update(patch)
            with self.assertRaises(ValueError):parameters(s)
        f=make_case(spec()); f['a']=f['a'].double()
        with self.assertRaisesRegex(ValueError,'FP32'):Adapter().prepare('before',spec(),f,Path('.'))
        f=make_case(spec()); f['b'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):checked_inputs(spec(),f)

    def test_public_nested_fixture_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); (p/'case.json').write_text(json.dumps(spec()))
            result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/gdn_gate_contract.py'),
                '--case',str(p/'case.json'),'--output',str(p/'fixture.pt')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(torch.load(p/'fixture.pt',weights_only=False)['expected'][0].shape,(1,3,7))

    def test_public_pair_cli_rejection_has_zero_counts_and_no_cache_poison(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); s=spec();s['parameters']['H']=0
            (p/'case.json').write_text(json.dumps(s));(p/'provenance.json').write_text('{}')
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(sys.path[:2]))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
                '--adapter','gdn_gate_fusion:Adapter','--case',str(p/'case.json'),
                '--cache',str(p/'cache'),'--output',str(p/'result'),
                '--provenance',str(p/'provenance.json'),'--correctness-only'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            report=json.loads((p/'result/result.json').read_text())
            self.assertEqual(report['counts']['correctness_calls'],0)
            self.assertEqual(report['counts']['capture_calls'],0)
            self.assertFalse(list((p/'cache').rglob('*.pt')))


if __name__=='__main__':unittest.main()
