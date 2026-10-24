import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton'))
import ragged_pairs as pairs
from ragged_contract import check_spec, make_case, reference, validate_actual


class RaggedTests(unittest.TestCase):
    def spec(self, **updates):
        return dict(dict(family='prefix-offsets', batch=4, length=7, width=3,
                         seed=8301, pad=-3.25), **updates)

    def test_sequential_oracle_tails_and_empty_segments(self):
        x = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        lengths = torch.tensor([0, 2, 1, 3], dtype=torch.int32)
        result = reference(x, lengths, 4, -3.25)
        self.assertTrue(torch.equal(result[0], torch.full((4, 3), -3.25)))
        self.assertTrue(torch.equal(result[1, :2], x[:2]))
        self.assertTrue(torch.equal(result[2, :1], x[2:3]))
        self.assertTrue(torch.equal(result[3, :3], x[3:]))
        self.assertEqual(result[3, 3].tolist(), [-3.25] * 3)

    def test_generated_fixtures_partition_input_and_reproduce(self):
        for dtype in ('float16', 'float32'):
            for pattern in ('empty', 'mixed', 'full'):
                spec = self.spec(dtype=dtype, pattern=pattern, width=19, length=5)
                first, second = make_case(spec), make_case(spec)
                lengths = first['inputs']['lengths'].tolist()
                recovered = torch.cat([first['expected'][i, :n] for i, n in enumerate(lengths)])
                self.assertTrue(torch.equal(recovered, first['inputs']['x']))
                self.assertTrue(torch.equal(first['expected'], second['expected']))
                self.assertTrue(validate_actual(first['expected'], first)['passed'])
                self.assertEqual(first['expected'].shape, (4, 5, 19))

    def test_explicit_lengths_zero_input_and_quantized_padding(self):
        spec = self.spec(lengths=[0, 0, 0, 0], dtype='float16', pad=1.2345)
        case = make_case(spec)
        self.assertEqual(case['inputs']['x'].shape, (0, 3))
        self.assertEqual(case['padding'], float(torch.tensor(1.2345).half()))
        self.assertTrue(torch.equal(case['expected'], torch.full((4, 7, 3), case['padding']).half()))

    def test_json_null_lengths_matches_omitted_override(self):
        for family in ('prefix-offsets', 'padding-store'):
            for pattern in ('empty', 'mixed', 'full'):
                for dtype in ('float16', 'float32'):
                    with self.subTest(family=family, pattern=pattern, dtype=dtype):
                        spec = self.spec(family=family, pattern=pattern, dtype=dtype)
                        omitted = make_case(spec)
                        nullable = make_case(json.loads(json.dumps(dict(spec, lengths=None))))
                        self.assertEqual(nullable['padding'], omitted['padding'])
                        self.assertTrue(torch.equal(nullable['expected'], omitted['expected']))
                        for name in ('x', 'lengths'):
                            self.assertTrue(torch.equal(nullable['inputs'][name], omitted['inputs'][name]))
            explicit = make_case(self.spec(family=family, lengths=[0, 1, 3, 2]))
            self.assertEqual(explicit['inputs']['lengths'].tolist(), [0, 1, 3, 2])

    def test_invalid_domains_rejected_before_device(self):
        for update in [dict(batch=0), dict(batch=True), dict(length=2049), dict(width=0),
                dict(batch=1024, length=2048, width=1024), dict(dtype='bfloat16'),
                dict(seed=-1), dict(seed=2**64), dict(pad=float('inf')), dict(pad=True),
                dict(dtype='float16', pad=1e20), dict(pattern='unknown'),
                dict(lengths=[1]), dict(lengths=[0, 1, 2, 8]), dict(lengths=[False, 0, 0, 0]),
                dict(lengths=''), dict(lengths={}), dict(lengths=(0, 1, 2, 3)), dict(lengths=0)]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                check_spec(self.spec(**update))
        with self.assertRaises(ValueError):
            reference(torch.ones(3, 2), torch.tensor([2], dtype=torch.int32), 2, 0)

    def test_validation_rejects_output_error_shape_dtype_and_mutation(self):
        case = make_case(self.spec())
        changed = case['expected'].clone()
        changed[-1, -1, -1] += 1
        self.assertFalse(validate_actual(changed, case)['passed'])
        for output in (case['expected'].half(), case['expected'][:, :-1]):
            with self.assertRaises(ValueError):
                validate_actual(output, case)
        inputs = {k: v.clone() for k, v in case['inputs'].items()}
        inputs['lengths'][0] += 1
        with self.assertRaisesRegex(ValueError, 'input mutated'):
            validate_actual(case['expected'], case, inputs)

    def test_pair_controls_change_only_the_named_mechanism(self):
        for family in ('prefix-offsets', 'padding-store'):
            spec = self.spec(family=family, batch=257, length=193, width=129)
            before, after = (pairs.configuration(arm, spec) for arm in ('before', 'after'))
            self.assertEqual(before['grid'], (257, 7, 3))
            self.assertEqual(after['grid'], before['grid'])
            self.assertEqual(after['scan_width'], 512)
            changed = {key for key in before if before[key] != after[key]}
            self.assertEqual(changed, {'cached_offsets'} if family == 'prefix-offsets' else {'single_store'})

    def test_preparation_is_launch_free_and_complete_forward_includes_scan(self):
        calls = []

        class FakeKernel:
            def __init__(self, name):
                self.fn = SimpleNamespace(__name__=name)
                self.name = name

            def __getitem__(self, grid):
                def launch(*args):
                    calls.append((self.name, grid))
                    if self.name == 'ragged_prefix':
                        lengths, starts, _, _ = args
                        starts.copy_(lengths.cumsum(0) - lengths)
                    else:
                        x, lengths, starts, output, maximum, _, pad, *_ = args
                        if args[-2]:
                            self_check = lengths.cumsum(0) - lengths
                            assert torch.equal(starts, self_check)
                        output.copy_(reference(x, lengths, maximum, pad))
                    return SimpleNamespace(name=self.name, hash=self.name, asm={'binary': self.name.encode()})
                return launch

        empty, empty_like = torch.empty, torch.empty_like
        def cpu_empty(*args, **kwargs):
            kwargs.pop('device', None)
            return empty(*args, **kwargs)
        def cpu_empty_like(*args, **kwargs):
            kwargs.pop('device', None)
            return empty_like(*args, **kwargs)

        with patch.dict(sys.modules, {'torch_npu': SimpleNamespace()}), \
                patch.object(torch.Tensor, 'to', lambda t, *a, **kw: t.clone()), \
                patch.object(torch, 'empty', cpu_empty), patch.object(torch, 'empty_like', cpu_empty_like), \
                patch.object(pairs, 'kernels', return_value=(FakeKernel('ragged_prefix'), FakeKernel('ragged_pack'))):
            for family in ('prefix-offsets', 'padding-store'):
                for arm in ('before', 'after'):
                    calls.clear()
                    spec = self.spec(family=family)
                    fixture = make_case(spec)
                    adapter = pairs.Adapter()
                    prepared = adapter.prepare(arm, spec, fixture, Path('/unused'))
                    self.assertEqual(calls, [])
                    self.assertEqual(prepared.compilation_state(), [])
                    actual = prepared.run()
                    self.assertTrue(adapter.validate(arm, actual, fixture, spec)['passed'])
                    expected = ['ragged_prefix', 'ragged_pack'] if family == 'prefix-offsets' and arm == 'after' else ['ragged_pack']
                    self.assertEqual([name for name, _ in calls], expected)
                    self.assertEqual(prepared.expected_kernels, expected)
                    self.assertEqual(len(prepared.compilation_state()), len(expected))
                    self.assertEqual(prepared.metadata['workspace_bytes'], 16 if len(expected) == 2 else 0)

    def test_lazy_frontend_definitions_resolve_without_device(self):
        if importlib.util.find_spec('triton') is None:
            self.skipTest('Triton frontend unavailable')
        for kernel in pairs.kernels():
            self.assertIsInstance(kernel.cache_key, str)
            self.assertTrue(kernel.cache_key)
            self.assertFalse(kernel.fn.__closure__)


if __name__ == '__main__':
    unittest.main()
