import importlib.util
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples/optimization-library/triton'))
import attention
from attention_contract import diagnostics, dimensions, make_case, reference


class AttentionContractTests(unittest.TestCase):
    def spec(self, **updates):
        return dict(b=1, h=4, q=23, k=79, d=64, seed=7301, **updates)

    def test_zero_queries_produce_uniform_value_mean(self):
        q = torch.zeros(1, 4, 3, 64, dtype=torch.float16)
        k = torch.randn(1, 4, 5, 64).half()
        v = torch.arange(5, dtype=torch.float16)[None, None, :, None].expand(1, 4, 5, 64)
        actual = reference(q, k, v)
        torch.testing.assert_close(actual, torch.full_like(actual, 2.0), rtol=0, atol=0)

    def test_single_key_returns_value_for_every_query(self):
        q = torch.randn(1, 4, 7, 64).half()
        k = torch.randn(1, 4, 1, 64).half()
        v = torch.randn(1, 4, 1, 64).half()
        torch.testing.assert_close(reference(q, k, v), v.float().expand_as(q), rtol=0, atol=0)

    def test_nonuniform_probabilities_have_independent_closed_form(self):
        q = torch.zeros(1, 1, 1, 64, dtype=torch.float16)
        q[..., 0] = 8
        k = torch.zeros(1, 1, 2, 64, dtype=torch.float16)
        k[..., 1, 0] = 1
        v = torch.zeros_like(k)
        v[..., 1, :] = 1
        expected = torch.full((1, 1, 1, 64), torch.sigmoid(torch.tensor(1.)).item())
        torch.testing.assert_close(reference(q, k, v), expected)

    def test_tail_fixture_is_deterministic_fp32_and_not_an_alias(self):
        first, second = make_case(self.spec()), make_case(self.spec())
        for key in ('q', 'k', 'v', 'expected'):
            self.assertTrue(torch.equal(first[key], second[key]))
            self.assertNotEqual(first[key].data_ptr(), second[key].data_ptr())
        self.assertEqual(first['expected'].shape, (1, 4, 23, 64))
        self.assertEqual(first['expected'].dtype, torch.float32)
        self.assertTrue(diagnostics(first['expected'], first)['passed'])

    def test_shared_tolerance_rejects_corrupted_value(self):
        fixture = make_case(self.spec())
        actual = fixture['expected'].clone()
        actual[..., -1, -1] += .1
        result = diagnostics(actual, fixture)
        self.assertFalse(result['passed'])
        self.assertGreater(result['max_tolerance_fraction'], 1)

    def test_small_error_is_reported_without_relaxing_tolerance(self):
        fixture = make_case(self.spec())
        result = diagnostics(fixture['expected'] + .0001, fixture)
        self.assertTrue(result['passed'])
        self.assertEqual(result['rtol'], .001)
        self.assertEqual(result['atol'], .001)
        self.assertGreater(result['max_absolute_error'], 0)

    def test_dtype_shape_and_nonfinite_rejected(self):
        fixture = make_case(self.spec())
        for actual in (fixture['expected'].half(), fixture['expected'][..., :-1, :],
                       torch.full_like(fixture['expected'], float('nan'))):
            with self.subTest(shape=actual.shape, dtype=actual.dtype):
                self.assertFalse(diagnostics(actual, fixture)['passed'])

    def test_explicit_envelope_rejects_unsupported_inputs(self):
        for key, value in [('b', 2), ('h', 1), ('q', 0), ('k', 1537), ('d', 32), ('seed', -1), ('q', True)]:
            spec = self.spec()
            spec[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                dimensions(spec)

    def test_cpu_import_and_frontend_hashes_need_no_device(self):
        self.assertIs(attention.Adapter.make_case, make_case)
        if importlib.util.find_spec('triton') is None:
            self.skipTest('Triton frontend unavailable')
        for kernel in attention.kernels():
            self.assertEqual(len(kernel.cache_key), 64)
            self.assertFalse(kernel.fn.__closure__)


if __name__ == '__main__':
    unittest.main()
