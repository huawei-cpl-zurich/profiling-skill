from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

GALLERY = Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton'
sys.path.insert(0, str(GALLERY))
import gemm
import gemm_contract as contract


class GemmTests(unittest.TestCase):
    def test_oracle_preserves_fp32_add_contract(self):
        spec = dict(family='02', m=13, n=19, k=33, seed=44)
        fixture = contract.make_case(spec)
        a, b, z = (fixture['inputs'][key] for key in ('a', 'b', 'z'))
        oracle = torch.stack([torch.stack([(a[i].double() * b[:, j].double()).sum()
                              for j in range(spec['n'])]) for i in range(spec['m'])]).float() + z
        torch.testing.assert_close(fixture['expected'], oracle, rtol=1e-5, atol=1e-5)
        self.assertEqual(fixture['expected'].dtype, torch.float32)
        self.assertFalse(torch.equal(fixture['expected'], (a.float() @ b.float()).half().float() + z))
        self.assertTrue(contract.validate_actual(fixture['expected'].clone(), fixture)['passed'])

    def test_layouts_have_same_logical_values_and_different_strides(self):
        for family in ('02', '05', '12'):
            row = contract.make_case(dict(family=family, m=5, n=7, k=9, seed=3))
            column = contract.make_case(dict(family=family, m=5, n=7, k=9, seed=3, b_layout='column'))
            self.assertNotEqual(row['inputs']['b'].stride(), column['inputs']['b'].stride())
            torch.testing.assert_close(row['inputs']['b'], column['inputs']['b'], rtol=0, atol=0)
            torch.testing.assert_close(row['expected'], column['expected'], rtol=0, atol=0)

    def test_mapping_is_bijective_including_rectangular_and_tail_grids(self):
        for rows in (1, 2, 3, 7, 32):
            for columns in (1, 2, 5, 8, 37):
                expected = {(r, c) for r in range(rows) for c in range(columns)}
                for diagonal in (False, True):
                    mapped = [gemm.program_tile(pid, rows, columns, diagonal)
                              for pid in range(rows * columns)]
                    self.assertEqual(len(mapped), len(set(mapped)))
                    self.assertEqual(set(mapped), expected)
        # Cover physical elements for a non-tile-aligned matrix, not just tiles.
        spec = dict(family='12', m=193, n=385, k=17)
        cfg = gemm.configuration('after', spec)
        covered = set()
        for pid in range(cfg['rows'] * cfg['columns']):
            row, column = gemm.program_tile(pid, cfg['rows'], cfg['columns'], cfg['diagonal'])
            for r in range(row * cfg['bm'], min((row + 1) * cfg['bm'], spec['m'])):
                for c in range(column * cfg['bn'], min((column + 1) * cfg['bn'], spec['n'])):
                    self.assertNotIn((r, c), covered)
                    covered.add((r, c))
        self.assertEqual(len(covered), spec['m'] * spec['n'])

    def test_tile_balance_crosses_core_wave_boundary(self):
        spec = dict(family='05', m=2048, n=384, k=6144)
        before, after = (gemm.configuration(arm, spec) for arm in ('before', 'after'))
        self.assertEqual(before['rows'] * before['columns'], 32)
        self.assertEqual(after['rows'] * after['columns'], 24)
        self.assertEqual(before['bk'], after['bk'])
        self.assertEqual(before['bm'] * before['bn'], after['bm'] * after['bn'])

    def test_validation_rejects_wrong_dtype_nonfinite_and_mutation(self):
        fixture = contract.make_case(dict(family='05', m=3, n=5, k=7))
        actual = fixture['expected'].clone()
        with self.assertRaisesRegex(ValueError, 'dtype'):
            contract.validate_actual(actual.float(), fixture)
        actual[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            contract.validate_actual(actual, fixture)
        actual = fixture['expected'].clone() + 1
        self.assertFalse(contract.validate_actual(actual, fixture)['passed'])
        inputs = {name: tensor.clone() for name, tensor in fixture['inputs'].items()}
        inputs['a'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'input mutated'):
            contract.validate_actual(fixture['expected'], fixture, inputs)

    def test_dispatch_rejects_unsupported_cases(self):
        for changes in (dict(family='03'), dict(k=0), dict(m=True), dict(b_layout='sliced')):
            spec = dict(family='12', m=2, n=3, k=4)
            spec.update(changes)
            with self.assertRaises(ValueError):
                contract.make_case(spec)
        with self.assertRaises(ValueError):
            gemm.configuration('unrelated', dict(family='02', m=2, n=3, k=4))
        self.assertFalse(gemm.configuration('after', dict(family='12', m=1, n=385, k=32))['diagonal'])

    def test_binary_snapshot_records_bytes_not_compiler_text(self):
        compiled = dict(gemm=SimpleNamespace(name='actual_kernel', hash='cache-key',
                                            asm=dict(binary=b'abc', mlir='unverified text')))
        snapshot = gemm.compiled_identity(compiled)
        self.assertEqual(snapshot[0]['name'], 'actual_kernel')
        self.assertEqual(set(snapshot[0]['binaries']), {'binary'})
        compiled['gemm'].asm['binary'] = b'changed'
        self.assertNotEqual(snapshot, gemm.compiled_identity(compiled))

    def test_preparation_does_not_launch_and_run_preserves_whole_call(self):
        calls = []
        class Launch:
            def __init__(self, name):
                self.fn = SimpleNamespace(__name__=name)
            def __getitem__(self, grid):
                def invoke(*args):
                    calls.append((self.fn.__name__, grid))
                    if self.fn.__name__ == 'gallery_gemm':
                        a, b, z, output = args[:4]
                        value = a.float() @ b.float()
                        if args[-2]:
                            value += z
                        output.copy_(value)
                    else:
                        x, z, output = args[:3]
                        output.copy_(x + z)
                    return SimpleNamespace(name=self.fn.__name__, hash='stable', asm={'binary': b'cpu-fixture'})
                return invoke
        original_empty = torch.empty_like
        def allocate(value, **kwargs):
            kwargs.pop('device', None)
            return original_empty(value, **kwargs)
        def transfer(value, **kwargs):
            self.assertEqual(kwargs['device'], 'npu:0')
            return value.clone(memory_format=torch.preserve_format)
        modules = dict(torch_npu=SimpleNamespace(), triton=SimpleNamespace(__version__='fixture'))
        with patch.dict(sys.modules, modules), patch.object(gemm, 'kernels', return_value=(
                Launch('gallery_gemm'), Launch('gallery_add'))), patch.object(torch.Tensor, 'to', transfer), \
                patch.object(torch, 'empty_like', side_effect=allocate):
            for family in ('02', '05', '12'):
                spec = dict(family=family, m=3, n=5, k=7, b_layout='column')
                fixture = contract.make_case(spec)
                adapter = gemm.Adapter()
                for arm in ('before', 'after'):
                    previous = len(calls)
                    prepared = adapter.prepare(arm, spec, fixture, None)
                    self.assertEqual(len(calls), previous)
                    self.assertEqual(prepared.compilation_state(), [])
                    actual = prepared.run()
                    expected_calls = 2 if family == '02' and arm == 'before' else 1
                    self.assertEqual(len(calls) - previous, expected_calls)
                    self.assertTrue(adapter.validate(arm, actual, fixture, spec)['passed'])
                    self.assertEqual(len(prepared.expected_kernels), expected_calls)
                    self.assertEqual(len(prepared.compilation_state()), expected_calls)


if __name__ == '__main__':
    unittest.main()
