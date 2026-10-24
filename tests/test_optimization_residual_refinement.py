import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/triton'))
from refinement_contract import make_case, refinement_cpu, diagnostics, commit


class RefinementContract(unittest.TestCase):
    def specs(self):
        matrix = json.loads((ROOT/'examples/optimization-library/cases/wave004/W13.json').read_text())
        return matrix['cases']+matrix['contract']['guards']

    def test_frozen_equations_both_arms_and_no_input_mutation(self):
        for spec in self.specs():
            fixture = make_case(spec)
            original = {k: v.clone() for k, v in fixture['inputs'].items()}
            for after in (False, True):
                actual = refinement_cpu(fixture['inputs'], after)
                self.assertTrue(diagnostics(actual, fixture)['passed'], (spec, after))
                self.assertNotEqual(actual.data_ptr(), fixture['inputs']['b'].data_ptr())
            for k, value in original.items():
                self.assertTrue(torch.equal(value, fixture['inputs'][k]))

    def test_zero_residual_exact_and_whole_vector_rollback(self):
        exact, overflow = [make_case(s) for s in self.specs()[-2:]]
        self.assertTrue(torch.equal(refinement_cpu(exact['inputs'], False), refinement_cpu(exact['inputs'], True)))
        x, delta = overflow['inputs']['b'], overflow['inputs']['delta']
        proposed = x[:, :8]+delta
        self.assertFalse(torch.isfinite(proposed[0, 0]))
        self.assertTrue(torch.isfinite(proposed[0, 1:]).all())
        self.assertTrue(torch.equal(commit(x, delta, 8), x[:, :8]))
        self.assertTrue(torch.equal(commit(x, torch.ones_like(delta), 8)[:, 1:], proposed[:, 1:]))

    def test_small_residual_does_not_establish_forward_accuracy(self):
        # A nearly singular represented FP32 tridiagonal has an almost-null
        # direction. Adding that direction gives a tiny backward error but
        # changes the solution substantially; the forward oracle rejects it.
        eps = 2**-22
        inputs = dict(dl=torch.tensor([0., -1.]), du=torch.tensor([-1., 0.]),
                      d=torch.tensor([1.+eps, 1.+eps]), b=torch.tensor([[1., -1., 98765.25]]))
        a = torch.tensor([[1.+eps, -1.], [-1., 1.+eps]], dtype=torch.float64)
        expected = inputs['b'].double().clone()
        expected[:, :2] = torch.linalg.solve(a, inputs['b'][:, :2].double().T).T
        wrong = expected.float(); wrong[:, :2] += 1
        result = diagnostics(wrong, dict(inputs=inputs, expected=expected))
        self.assertLess(result['componentwise_backward_error'], 1e-6)
        self.assertGreater(result['max_forward_error'], .9)
        self.assertFalse(result['passed'])

    def test_committed_nested_case_through_cli(self):
        # A CPU execution adapter exercises the public CLI and actual committed
        # guard JSON without substituting NPU evidence or loading Triton.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path/'cpu_refine.py').write_text('''from types import SimpleNamespace
from refinement_contract import *
class Adapter:
 case_sources=CASE_SOURCES
 make_case=staticmethod(make_case)
 def prepare(self,arm,spec,fixture,output):
  return SimpleNamespace(run=lambda:refinement_cpu(fixture['inputs'],arm=='after'),sources=CASE_SOURCES,expected_kernels=['cpu_component'],metadata={'host_fallback':True},compilation_state=lambda:[])
 def validate(self,arm,actual,fixture,spec):return diagnostics(actual,fixture)
def factory():return Adapter()
''')
            (path/'provenance.json').write_text('{}')
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([d,str(ROOT/'examples/optimization-library/triton')]))
            command = [sys.executable, str(ROOT/'scripts/optimization_pair.py'), '--adapter','cpu_refine:factory',
                       '--case',str(ROOT/'examples/optimization-library/cases/wave004/W004-W13-GB.json'),
                       '--output',d+'/out','--cache',d+'/cache','--provenance',d+'/provenance.json']
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((path/'out/result.json').read_text())
            self.assertEqual(report['status'],'complete')
            self.assertTrue(all(a['capture_calls']==a['profiled_calls']==0 for a in report['arms'].values()))

if __name__ == '__main__':unittest.main()
