import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/triton'))
from tridiagonal_contract import make_case, thomas_cpu, diagnostics, validate_inputs
from shared_tridiagonal import Adapter


class ThomasContract(unittest.TestCase):
    def spec(self, m=17, r=3, ldb=24):
        return dict(seed=5192, parameters=dict(m=m, R=r, ldb=ldb))

    def test_both_schedules_padding_and_immutable(self):
        fixture = make_case(self.spec())
        original = {k: v.clone() for k, v in fixture['inputs'].items()}
        outputs = [thomas_cpu(fixture['inputs'], shared) for shared in (False, True)]
        for x in outputs:
            self.assertTrue(diagnostics(x, fixture)['passed'])
            self.assertNotEqual(x.data_ptr(), fixture['inputs']['b'].data_ptr())
        self.assertTrue(torch.equal(*outputs))
        for k, v in original.items():
            self.assertTrue(torch.equal(v, fixture['inputs'][k]))

    def test_pivot_fallback_uses_current_inputs_not_expected(self):
        spec = self.spec(4, 2, 8)
        spec['parameters']['distribution'] = 'nonsingular-zero-first-pivot'
        fixture = make_case(spec)
        adapter = Adapter()
        with tempfile.TemporaryDirectory() as d:
            for arm in ('before', 'after'):
                item = adapter.prepare(arm, spec, fixture, Path(d))
                self.assertEqual(item.metadata['npu_calls_per_forward'], 0)
                x = item.run()
                self.assertTrue(diagnostics(x, fixture)['passed'])
                fixture['inputs']['b'][:, :4] *= 2
                y = item.run()
                self.assertTrue(torch.allclose(y[:, :4], x[:, :4]*2))
                fixture['inputs']['b'][:, :4] /= 2

    def test_empty_rhs_and_bad_inputs(self):
        spec = self.spec(17, 0, 24)
        fixture = make_case(spec)
        for shared in (False, True):
            self.assertEqual(tuple(thomas_cpu(fixture['inputs'], shared).shape), (0, 24))
        fixture['inputs']['d'][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_inputs(fixture['inputs'], spec)

    def test_real_nested_case_cli(self):
        # Exercise the committed nested spec via the public pair CLI. Pivot guard
        # executes on CPU without loading Triton/NPU or making any capture call.
        with tempfile.TemporaryDirectory() as d:
            cmd = [sys.executable, str(ROOT/'scripts/optimization_pair.py'), '--help']
            help_result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            import os
            env = dict(os.environ, PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
            Path(d+'/provenance.json').write_text('{}')
            cmd = [sys.executable, str(ROOT/'scripts/optimization_pair.py'), '--adapter', 'shared_tridiagonal:factory',
                   '--case', str(ROOT/'examples/optimization-library/cases/wave004/W004-W11-GA.json'),
                   '--output', d+'/out', '--cache', d+'/cache', '--provenance', d+'/provenance.json']
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(Path(d+'/out/result.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertTrue(all(a['capture_calls'] == 0 and a['profiled_calls'] == 0 for a in report['arms'].values()))


if __name__ == '__main__':
    unittest.main()
