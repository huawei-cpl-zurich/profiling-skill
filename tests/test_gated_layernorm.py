"""Behavioral checks of gate order, statistics, fixture and CLI rejection."""
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
from gated_layernorm_contract import diagnostics, make_case, parameters, reference
from optimization_pair import fixture
from gated_layernorm import Adapter


def spec(n=7, first=False, seed=6300, **extra):
    return dict(seed=seed, parameters=dict(R=2, N=n, gate_first=first, eps=1e-5, **extra))


class ContractTests(unittest.TestCase):
    def test_gate_order_and_statistics_are_observable(self):
        x = torch.ones((1, 2), dtype=torch.float32)
        z = torch.tensor([[-1., 2.]], dtype=torch.float32)
        gamma, beta = torch.ones(2), torch.tensor([.1, -.1])
        first = reference(x, z, gamma, beta, True, 1e-5)
        last = reference(x, z, gamma, beta, False, 1e-5)
        u = z.double() * z.double().sigmoid()
        self.assertTrue(torch.allclose(first[1], u.mean(-1)))
        self.assertEqual(float(last[1]), 1.)
        self.assertGreater(float((first[0]-last[0]).abs().max()), .1)
        self.assertTrue(torch.allclose(last[0], beta.double()*u))

    def test_diagnostics_accepts_all_three_and_rejects_bad_stat_or_dtype(self):
        f = make_case(spec(n=13))
        actual = tuple(v.float() for v in f['expected'])
        self.assertTrue(diagnostics(actual, f)['passed'])
        self.assertFalse(diagnostics((actual[0], actual[1]+.1, actual[2]), f)['passed'])
        self.assertFalse(diagnostics((actual[0].double(), *actual[1:]), f)['passed'])
        bad = actual[0].clone(); bad[0,0] = float('nan')
        self.assertFalse(diagnostics((bad, *actual[1:]), f)['passed'])

    def test_odd_width_fixture_cache_is_explicit_cpu_fp32_under_float64_default(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with tempfile.TemporaryDirectory() as folder:
                adapter = Adapter()
                f, first = fixture(adapter, spec(n=17), Path(folder), torch)
                again, second = fixture(adapter, spec(n=17), Path(folder), torch)
                self.assertEqual((first['oracle_calls'], second['oracle_calls']), (1, 0))
                for key in ('x','z','gamma','beta'):
                    self.assertEqual(f[key].dtype, torch.float32)
                    self.assertEqual(f[key].device.type, 'cpu')
                    self.assertTrue(torch.equal(f[key], again[key]))
                self.assertEqual(f['expected'][0].shape, (2,17))
        finally:
            torch.set_default_dtype(previous)

    def test_invalid_parameters_and_public_nested_case_cli(self):
        for field, value in (('N', 0), ('gate_first', 1), ('eps', 0)):
            s = spec(); s['parameters'][field] = value
            with self.assertRaises(ValueError): parameters(s)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            s = spec(); s['parameters']['N'] = 0
            (folder/'case.json').write_text(json.dumps(s))
            (folder/'provenance.json').write_text('{}')
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path[:2]))
            result = subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
                '--adapter','gated_layernorm:Adapter','--case',str(folder/'case.json'),
                '--cache',str(folder/'cache'),'--output',str(folder/'result'),
                '--provenance',str(folder/'provenance.json'),'--correctness-only'],
                env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads((folder/'result/result.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['counts']['correctness_calls'], 0)
            self.assertEqual(report['counts']['capture_calls'], 0)


if __name__ == '__main__': unittest.main()
