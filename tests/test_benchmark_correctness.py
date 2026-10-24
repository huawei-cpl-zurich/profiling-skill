"""Functional upstream comparison checks; BENCHMARK_TEST_DEVICE=npu:0 exercises A3."""
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from benchmark_correctness import verify_outputs, snapshot, failure_details


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get('BENCHMARK_TEST_DEVICE', 'cpu')
        if cls.device.startswith('npu'):
            import torch_npu  # noqa: F401

    def tensor(self, values, dtype=torch.float32):
        return torch.tensor(values, device=self.device, dtype=dtype)

    def test_tuple_optional_output_and_nan_inf_pass(self):
        value = self.tensor([1., float('nan'), float('inf'), -float('inf')])
        verify_outputs((value.cpu(), None), (value, None), [value])

    def test_numerical_failure_preserves_upstream_metrics_and_chinese(self):
        with self.assertRaises(AssertionError) as caught:
            verify_outputs(self.tensor([1., 2.]), self.tensor([9., 2.]))
        details = failure_details(caught.exception)
        self.assertIn('验证失败', details['error_msg'])
        self.assertIsInstance(details['metrics'], dict)

    def test_shape_nan_and_inf_failures(self):
        original = self.tensor([[1., 2.]])
        for actual, text in ((original.reshape(2), '形状不一致'),
                             (self.tensor([[float('nan'), 2.]]), 'NaN'),
                             (self.tensor([[float('inf'), 2.]]), 'Inf')):
            with self.subTest(text=text), self.assertRaisesRegex(AssertionError, text):
                verify_outputs(original, actual)
        with self.assertRaisesRegex(AssertionError, 'Inf 符号'):
            verify_outputs(self.tensor([float('inf')]), self.tensor([-float('inf')]))

    def test_optional_and_output_count_mismatches(self):
        for expected, actual in (((None,), (self.tensor([1.]),)),
                                 ((self.tensor([1.]),), ()),
                                 ({'state': None}, {'wrong': None})):
            with self.subTest(expected=expected), self.assertRaises(AssertionError):
                verify_outputs(expected, actual)

    def test_dtype_aware_integer_quantization_and_non_compute(self):
        golden = self.tensor([10], torch.int32)
        actual = self.tensor([11], torch.int32)
        verify_outputs(golden, actual, ((self.tensor([1.]), 7), {'scale': 1.}))  # upstream fp->int allows one
        with self.assertRaises(AssertionError):
            verify_outputs(golden, actual, [self.tensor([1], torch.int32)])
        value = self.tensor([1., 2.])
        with self.assertRaises(AssertionError):
            verify_outputs(value, value + .0001, [value], non_compute=True)

    def test_snapshot_independence(self):
        value = self.tensor([1., 2.])
        frozen = snapshot({'state': (value,)})
        value.add_(10)
        self.assertTrue(torch.equal(frozen['state'][0], torch.tensor([1., 2.])))

    def test_campaign_rejects_wrong_output_dtype(self):
        from private_campaign_cases import validate
        value = self.tensor([1., 2.])
        case = {'expected': value, 'args': (value.to(torch.float16),)}
        validate(case, 'candidate', value.clone())
        for dtype in (torch.float16, torch.int32):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(AssertionError, 'dtype 不一致'):
                validate(case, 'candidate', value.to(dtype))


if __name__ == '__main__':
    unittest.main()
