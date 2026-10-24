from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

GALLERY = Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton'
sys.path.insert(0, str(GALLERY))
import paged_kv_contract as contract


def spec(**updates):
    return dict(dict(B=1, Hq=4, Hkv=2, Q=3, KV=19, D=32, Dv=32, page=16, seed=1191), **updates)


class PagedKVTests(unittest.TestCase):
    def test_page_permutation_partial_page_and_gqa_match_dense(self):
        case = spec()
        f = contract.make_case(case)
        contract.check_inputs(case, f)
        k = f['k'][f['table'].long()].reshape(1, -1, 2, 32)[:, :19]
        v = f['v'][f['table'].long()].reshape(1, -1, 2, 32)[:, :19]
        scores = f['q'].double() @ k.repeat_interleave(2, 2).permute(0, 2, 3, 1).double() / 32**0.5
        mask = torch.arange(19)[None, :] <= 16 + torch.arange(3)[:, None]
        dense = torch.softmax(scores.masked_fill(~mask, -float('inf')), -1) @ v.repeat_interleave(2, 2).permute(0, 2, 1, 3).double()
        torch.testing.assert_close(f['expected'], dense, atol=1e-14, rtol=1e-14)
        changed = f['k'].clone()
        changed[f['table'][0, -1], 3:] = 30000
        result = contract.reference(f['q'], changed, f['v'], f['table'], 19)
        self.assertTrue(torch.equal(result, f['expected']))

    def test_empty_sequence_defined_zero(self):
        f = contract.make_case(spec(KV=0, Q=1))
        self.assertTrue(torch.equal(f['expected'], torch.zeros_like(f['expected'])))
        self.assertTrue(contract.diagnostics(f['expected'].float(), f)['passed'])

    def test_output_checks_detect_sentinel_shape_and_numeric_failure(self):
        f = contract.make_case(spec())
        self.assertTrue(contract.diagnostics(f['expected'].float(), f)['passed'])
        for actual in (torch.full_like(f['expected'].float(), float('nan')),
                       f['expected'].float() + 1, f['expected'].float()[..., :1], f['expected']):
            self.assertFalse(contract.diagnostics(actual, f)['passed'])

    def test_invalid_dimensions_and_page_table_rejected(self):
        for changes in (dict(Hq=3), dict(Q=20), dict(KV=-1), dict(page=8), dict(seed=True)):
            with self.assertRaises(ValueError):
                contract.dimensions(spec(**changes))
        f = contract.make_case(spec())
        f['table'].zero_()
        with self.assertRaises(ValueError):
            contract.check_inputs(spec(), f)

    def test_forward_contains_both_gathers_and_attention_without_extra_launches(self):
        import paged_kv_pairs
        calls = []
        class Kernel:
            def __init__(self, name):
                self.fn = SimpleNamespace(__name__=name)
            def __getitem__(self, grid):
                def launch(*args):
                    calls.append((self.fn.__name__, grid, args))
                    return SimpleNamespace(hash=self.fn.__name__, asm={'binary': b'test'})
                return launch
        original_empty, original_full = torch.empty, torch.full
        def empty(*args, **kwargs):
            kwargs.pop('device', None)
            return original_empty(*args, **kwargs)
        def full(*args, **kwargs):
            kwargs.pop('device', None)
            return original_full(*args, **kwargs)
        frontend = SimpleNamespace(__version__='test', cdiv=lambda a, b: (a+b-1)//b)
        f = contract.make_case(spec())
        with patch.dict(sys.modules, {'torch_npu': SimpleNamespace(), 'triton': frontend}), \
                patch.object(torch.Tensor, 'npu', lambda x: x.clone(), create=True), \
                patch.object(torch, 'empty', empty), patch.object(torch, 'full', full), \
                patch.object(paged_kv_pairs, 'kernels', lambda: (Kernel('gather'), Kernel('attention'))):
            adapter = paged_kv_pairs.factory()
            before = adapter.prepare('before', spec(), f, None)
            after = adapter.prepare('after', spec(), f, None)
            self.assertEqual(calls, [])
            before.run()
            self.assertEqual([x[0] for x in calls], ['gather', 'attention'])
            gather, attention = calls
            self.assertIs(gather[2][3], attention[2][1])
            self.assertIs(gather[2][4], attention[2][2])
            self.assertFalse(attention[2][-3])
            calls.clear()
            after.run()
            self.assertEqual([x[0] for x in calls], ['attention'])
            self.assertTrue(calls[0][2][-3])
            self.assertEqual(before.metadata['query_tile'], after.metadata['query_tile'])
            self.assertEqual(before.metadata['key_tile'], after.metadata['key_tile'])
            self.assertEqual(before.expected_kernels, ['gather', 'attention'])
            self.assertEqual(after.expected_kernels, ['attention'])

    def test_frontend_resolves_without_launch(self):
        try:
            import triton
        except ImportError:
            self.skipTest('Triton frontend unavailable')
        import paged_kv_pairs
        for kernel in paged_kv_pairs.kernels():
            self.assertTrue(kernel.cache_key)


if __name__ == '__main__':
    unittest.main()
