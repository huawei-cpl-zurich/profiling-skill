"""Functional frozen fixtures and actual device-flag dispatch control flow."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
from diagonal_contract import make_case,validate_input,diagnostics
from diagonal_eigensolve import Solver


class Tests(unittest.TestCase):
    def setUp(self):
        matrix=json.loads((ROOT/'experiments/optimization-wave004/W10.json').read_text())
        self.specs=matrix['cases']+matrix['contract']['guards']

    def test_all_fixtures_ambient_double(self):
        old=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for spec in self.specs:
                fixture=make_case(spec);validate_input(spec,fixture)
                self.assertEqual(fixture['a'].device.type,'cpu')
                self.assertEqual(fixture['a'].dtype,torch.float32)
                self.assertTrue(diagnostics(fixture['expected'].float(),fixture,False)['passed'])
            tiny=make_case(self.specs[-2])['a']
            self.assertEqual(tiny[0,0,1].item(),2**-20)
        finally:torch.set_default_dtype(old)

    def repeated_nonmultiple_specs(self):
        return [dict(case_id=f'W10-regression-N{n}',role='guard',seed=5092,
                     parameters=dict(batch=1,N=n,sweeps=5,want_vectors=False,
                                     structure='unsorted-repeated-diagonal')) for n in (6,10)]

    def test_repeated_diagonal_nonmultiple_of_four(self):
        for spec in self.repeated_nonmultiple_specs():
            n=spec['parameters']['N']
            fixture=make_case(spec)
            validate_input(spec,fixture)
            self.assertEqual(fixture['a'].shape,(1,n,n))
            pattern=torch.tensor([2.,-1.,1.,0.],dtype=torch.float32)
            expected=pattern[torch.arange(n)%4]
            self.assertTrue(torch.equal(fixture['a'][0].diagonal(),expected))
            self.assertTrue(torch.equal(fixture['expected'].float(),expected.sort().values.unsqueeze(0)))

    def test_actual_flag_controls_each_call(self):
        calls=[]
        class Kernel:
            def __getitem__(self,grid):
                def launch(*args,**kwargs):calls.append(args);return None
                return launch
        class Flag:
            def __init__(self):self.bits=iter([1,0,1])
            def item(self):return next(self.bits)
        s=Solver.__new__(Solver)
        s.arm='after';s.flag=Flag();s.certificate=Kernel();s.k=[Kernel()]
        s.a='current-device-input';s.n=16;s.batch=1;s.values='values';s.work='work';s.options={}
        s.keep=lambda _:None
        with patch('eigenvalues_only.Solver.run',return_value='fallback') as parent:
            self.assertEqual(s.run(),'values');self.assertEqual(s.last_dispatch,'diagonal-sort')
            self.assertEqual(s.run(),'fallback');self.assertEqual(s.last_dispatch,'five-sweep-fallback')
            self.assertEqual(s.run(),'values')
            self.assertEqual(parent.call_count,1)
        self.assertEqual(len(calls),5) # Three real certificates and two sorts.
        self.assertTrue(all(c[0]=='current-device-input' for c in calls))

    def test_bad_output_and_invalid_domain(self):
        fixture=make_case(self.specs[0])
        self.assertFalse(diagnostics(fixture['expected'].flip(-1).float(),fixture,False)['passed'])
        fixture['a'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):validate_input(self.specs[0],fixture)

    def test_actual_nested_matrix_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'cpu_diagonal.py').write_text('''from pathlib import Path
from types import SimpleNamespace
from diagonal_contract import CASE_SOURCES,make_case,diagnostics
class Adapter:
 case_sources=CASE_SOURCES
 make_case=staticmethod(make_case)
 def prepare(self,arm,spec,fixture,output):
  import torch
  return SimpleNamespace(run=lambda:torch.linalg.eigvalsh(fixture['a'].double()).float(),sources=[Path(__file__),*CASE_SOURCES],metadata={},expected_kernels=['cpu'])
 def validate(self,arm,actual,fixture,spec):return diagnostics(actual,fixture,False)
''')
            import os
            env=dict(os.environ,PYTHONPATH=str(p)+':'+str(ROOT/'examples/optimization-library/triton'))
            (p/'provenance.json').write_text('{}')
            for spec in self.specs+self.repeated_nonmultiple_specs():
                (p/'case.json').write_text(json.dumps(spec))
                out=p/spec['case_id']
                run=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
                    '--adapter','cpu_diagonal:Adapter','--case',str(p/'case.json'),'--cache',str(p/'cache'),
                    '--provenance',str(p/'provenance.json'),'--output',str(out),'--correctness-only'],
                    env=env,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                report=json.loads((out/'result.json').read_text())
                self.assertEqual(report['counts']['correctness_calls'],2)
                self.assertEqual(report['counts']['profiled_calls'],0)


if __name__=='__main__':unittest.main()
