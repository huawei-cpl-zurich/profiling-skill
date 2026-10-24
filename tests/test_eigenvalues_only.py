"""Functional CPU contracts and executable launch-plan checks; device gate separate."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
from eigenvalues_contract import diagnostics, make_case, validate_input
from eigenvalues_only import Solver
from jacobi_contract import round_pairs


class ContractTests(unittest.TestCase):
    def setUp(self):
        matrix = json.loads((ROOT/'experiments/optimization-wave004/W08.json').read_text())
        self.specs = matrix['cases']+matrix['contract']['guards']

    def test_actual_nested_cases_under_ambient_fp64(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for spec in self.specs:
                fixture = make_case(spec)
                validate_input(spec,fixture)
                self.assertEqual(fixture['a'].dtype,torch.float32)
                values,v = torch.linalg.eigh(fixture['a'].double())
                actual = (values.float(),v.float()) if spec['parameters']['want_vectors'] else values.float()
                self.assertTrue(diagnostics(actual,fixture,spec['parameters']['want_vectors'])['passed'])
        finally:
            torch.set_default_dtype(previous)

    def test_bad_outputs_and_domain_rejected(self):
        spec = self.specs[0]; fixture = make_case(spec)
        self.assertFalse(diagnostics(torch.full_like(fixture['expected'],float('nan')).float(),fixture,False)['passed'])
        self.assertFalse(diagnostics(fixture['expected'].flip(-1).float(),fixture,False)['passed'])
        fixture['a'][0,0,0] += 1
        with self.assertRaises(ValueError): validate_input(spec,fixture)
        spec = self.specs[-1]; fixture = make_case(spec)
        self.assertFalse(diagnostics((fixture['expected'].float(),torch.zeros_like(fixture['a'])),fixture,True)['passed'])

    def test_five_complete_sweeps_and_vector_fallback_plan(self):
        for track in (False,True):
            s = Solver.__new__(Solver)
            s.n,s.batch,s.track,s.vectors = 8,3,track,track
            s.a,s.work,s.scratch,s.v,s.vtmp,s.values,s.sorted_v,s.c,s.s = range(9)
            s.maps = [(r,-r) for r in range(7)]
            s.k = ('copy','angles','stage','identity','sort')
            plan = list(s.calls())
            self.assertEqual(sum(k=='angles' for k,_,_ in plan),35)
            self.assertEqual(sum(k=='stage' for k,_,_ in plan),35*(3 if track else 2))
            self.assertEqual(sum(k=='identity' for k,_,_ in plan),int(track))
            self.assertEqual(plan,list(s.calls()))  # Reset/init included on every API call.
            self.assertTrue(all(grid[1]==3 for _,grid,_ in plan))
        for n in (4,8,64,96):
            pairs = [tuple(p) for r in range(n-1) for p in round_pairs(n,r)]
            self.assertEqual(len(set(pairs)),n*(n-1)//2)

    def test_actual_json_cli_fixture_boundary(self):
        # Invoke the real harness CLI and cache boundary with a CPU-only adapter.
        # The numerical device Solver is independently exercised by the A3 gate.
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p/'cpu_adapter.py').write_text('''from types import SimpleNamespace
from pathlib import Path
from eigenvalues_contract import make_case, CASE_SOURCES, diagnostics
class Adapter:
 case_sources=CASE_SOURCES
 make_case=staticmethod(make_case)
 def prepare(self,arm,spec,fixture,output):
  import torch
  def run():
   values,v=torch.linalg.eigh(fixture['a'].double())
   return (values.float(),v.float()) if spec['parameters']['want_vectors'] else values.float()
  return SimpleNamespace(run=run,sources=[Path(__file__),*CASE_SOURCES],metadata={},expected_kernels=['cpu'])
 def validate(self,arm,actual,fixture,spec):
  return diagnostics(actual,fixture,spec['parameters']['want_vectors'])
''')
            (p/'provenance.json').write_text('{}')
            import os
            env = dict(os.environ,PYTHONPATH=str(p)+':'+str(ROOT/'examples/optimization-library/triton'))
            for index,spec in enumerate((self.specs[1],self.specs[-1])):
                (p/'case.json').write_text(json.dumps(spec))
                output=p/f'out-{index}'
                result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
                    '--adapter','cpu_adapter:Adapter','--case',str(p/'case.json'),'--cache',str(p/'cache'),
                    '--provenance',str(p/'provenance.json'),'--output',str(output),'--correctness-only'],
                    env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                report=json.loads((output/'result.json').read_text())
                self.assertEqual(report['counts']['correctness_calls'],2)
                self.assertEqual(report['counts']['profiled_calls'],0)


if __name__=='__main__': unittest.main()
