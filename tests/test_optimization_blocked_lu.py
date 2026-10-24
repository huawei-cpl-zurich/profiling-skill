"""CPU functional coverage for nested frozen cases, pivot semantics and gates."""
import importlib.util
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/triton'))
import blocked_lu_contract as contract
import blocked_lu


class ContractTests(unittest.TestCase):
    def test_committed_matrix(self):
        family = json.loads((ROOT/'examples/optimization-library/cases/wave004/W07.json').read_text())
        for spec in family['cases']+family['contract']['guards']:
            with self.subTest(case=spec['case_id']):
                fixture = contract.make_case(spec)
                self.assertEqual(contract.dimensions(spec), (spec['parameters']['N'], 16))
                lu, p = fixture['expected']
                result = contract.diagnostics((lu.float(), p), fixture, spec)
                self.assertTrue(result['passed'], result)
                self.assertTrue(contract.validate_inputs(fixture, spec) or spec['case_id'].endswith('GA'))

    def test_cli_nested_committed_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            provenance = tmp/'provenance.json'
            provenance.write_text('{}')
            env = dict(os.environ, PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
            result = subprocess.run([sys.executable, str(ROOT/'scripts/optimization_pair.py'),
                '--adapter', 'blocked_lu:factory', '--case',
                str(ROOT/'examples/optimization-library/cases/wave004/W004-W07-GA.json'),
                '--output', str(tmp/'out'), '--cache', str(tmp/'cache'),
                '--provenance', str(provenance)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((tmp/'out/result.json').read_text())
            self.assertEqual(report['counts']['correctness_calls'], 2)
            self.assertEqual(report['counts']['profiled_calls'], 0)
            self.assertEqual(report['case']['parameters']['N'], 4)
            for row in report['arms'].values():
                self.assertEqual(row['metadata']['npu_forwards_per_call'], 0)
                self.assertTrue(row['correctness']['diagnostics']['passed'])

    def test_pivot_fallback_computes_live_input(self):
        spec = dict(seed=4791, role='guard', parameters=dict(N=4, panel=16, distribution='nonsingular-zero-first-pivot'))
        fixture = contract.make_case(spec)
        original = fixture['inputs']['a'].clone()
        adapter = blocked_lu.Adapter()
        for arm in ('before', 'after'):
            item = adapter.prepare(arm, spec, fixture, Path('.'))
            actual = item.run()
            self.assertTrue(adapter.validate(arm, actual, fixture, spec)['passed'])
            self.assertEqual(item.metadata['npu_forwards_per_call'], 0)
            self.assertFalse(torch.equal(actual[1], torch.arange(4)))
            fixture['expected'] = (torch.full_like(actual[0], float('nan')), actual[1])
            self.assertTrue(torch.isfinite(item.run()[0]).all())
            fixture['expected'] = contract.dense_lu(original, pivot=True)
        self.assertTrue(torch.equal(original, fixture['inputs']['a']))

    def test_reject_profile_pivot_and_bad_shape(self):
        spec = dict(seed=1, role='guard', parameters=dict(N=4, panel=16, distribution='nonsingular-zero-first-pivot'))
        fixture = contract.make_case(spec)
        spec['role'] = 'gallery'
        with self.assertRaises(ValueError):
            contract.validate_inputs(fixture, spec)
        for params in (dict(N=0, panel=16), dict(N=32, panel=8), dict(N=257, panel=16)):
            with self.assertRaises(ValueError):
                contract.dimensions(dict(seed=1, parameters=params))

    def test_gate_rejects_nan_wrong_permutation_and_inaccurate_factor(self):
        spec = dict(seed=4702, parameters=dict(N=16, panel=16))
        fixture = contract.make_case(spec)
        lu, p = fixture['expected']
        bad = lu.float().clone()
        bad[0, 0] += .1
        self.assertFalse(contract.diagnostics((bad, p), fixture, spec)['passed'])
        self.assertFalse(contract.diagnostics((torch.full_like(bad, float('nan')), p), fixture, spec)['passed'])
        self.assertFalse(contract.diagnostics((lu.float(), p.flip(0)), fixture, spec)['passed'])


if __name__ == '__main__':
    unittest.main()
