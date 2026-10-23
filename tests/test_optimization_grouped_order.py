from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

GALLERY = Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton'
sys.path.insert(0, str(GALLERY))
import gemm
import grouped_order as grouped


class GroupedOrderTests(unittest.TestCase):
    def test_bijection_covers_partial_groups_and_rectangles(self):
        for rows in (1, 2, 3, 4, 5, 7, 8, 31, 32):
            for columns in (1, 2, 3, 5, 37):
                expected = {(m, n) for m in range(rows) for n in range(columns)}
                for size in (1, 2, 4, 8):
                    actual = [grouped.program_tile(p, rows, columns, size)
                              for p in range(rows * columns)]
                    self.assertEqual(len(actual), len(set(actual)))
                    self.assertEqual(set(actual), expected)

    def test_mapping_groups_m_rows_for_shared_b_column(self):
        self.assertEqual([grouped.program_tile(p, 7, 5) for p in range(4)],
                         [(0, 0), (1, 0), (2, 0), (3, 0)])
        self.assertEqual([grouped.program_tile(p, 7, 5) for p in range(20, 23)],
                         [(4, 0), (5, 0), (6, 0)])
        self.assertEqual([grouped.program_tile(p, 7, 5) for p in range(23, 26)],
                         [(4, 1), (5, 1), (6, 1)])
        self.assertEqual([grouped.program_tile(p, 7, 5, 1) for p in range(35)],
                         [gemm.program_tile(p, 7, 5, False) for p in range(35)])

    def test_rejects_invalid_program_domains(self):
        for args in ((-1, 3, 5), (15, 3, 5), (0, 0, 3), (0, 2, 0),
                     (True, 2, 3), (0, 2.5, 3), (0, 2, 3, 0)):
            with self.assertRaises(ValueError):
                grouped.program_tile(*args)

    def test_tiled_output_matches_oracle_with_physical_and_tile_tails(self):
        spec = dict(family='12', m=257, n=141, k=33, seed=84, b_layout='column')
        fixture = grouped.Adapter.make_case(spec)
        a, b = fixture['inputs']['a'], fixture['inputs']['b']
        output = torch.empty_like(fixture['expected'])
        visits = torch.zeros_like(output, dtype=torch.int32)
        config = gemm.configuration('before', spec)
        for pid in range(config['rows'] * config['columns']):
            m, n = grouped.program_tile(pid, config['rows'], config['columns'])
            ms = slice(m * config['bm'], min((m + 1) * config['bm'], spec['m']))
            ns = slice(n * config['bn'], min((n + 1) * config['bn'], spec['n']))
            output[ms, ns] = a[ms].float() @ b[:, ns].float()
            visits[ms, ns] += 1
        self.assertTrue(torch.equal(visits, torch.ones_like(visits)))
        self.assertTrue(gemm.validate_actual(output, fixture)['passed'])

    def test_preserves_shared_provider_and_baseline_boundary(self):
        self.assertIs(grouped.Adapter.make_case, gemm.Adapter.make_case)
        self.assertIs(grouped.Adapter.case_sources, gemm.Adapter.case_sources)
        adapter = grouped.Adapter()
        marker = SimpleNamespace(metadata={})
        spec = dict(family='12', m=320, n=384, k=32)
        with patch.object(gemm.Adapter, 'prepare', return_value=marker) as original:
            self.assertIs(adapter.prepare('before', spec, {}, None), marker)
            original.assert_called_once_with('before', spec, {}, None)
        self.assertEqual(marker.metadata['program_order'], 'row-major, unchanged baseline')
        for bad in (dict(spec, family='02'), dict(spec, k=0)):
            with self.assertRaises(ValueError):
                adapter.prepare('after', bad, {}, None)

    def test_prepare_launch_count_identity_and_no_capture_inspection(self):
        calls = []
        class Launch:
            fn = SimpleNamespace(__name__='gallery_grouped_gemm')
            def __getitem__(self, grid):
                def invoke(a, b, output, *parameters):
                    calls.append((grid, parameters))
                    output.copy_((a.float() @ b.float()).half())
                    return SimpleNamespace(name=self.fn.__name__, hash='cache', asm={'npubin': b'proof'})
                return invoke
        original_empty = torch.empty_like
        def allocate(value, **kwargs):
            kwargs.pop('device', None)
            return original_empty(value, **kwargs)
        def transfer(value, **kwargs):
            return value.clone(memory_format=torch.preserve_format)
        fixture = grouped.Adapter.make_case(dict(family='12', m=3, n=5, k=7, seed=8))
        modules = dict(torch_npu=SimpleNamespace(), triton=SimpleNamespace(__version__='fixture'))
        with patch.dict(sys.modules, modules), patch.object(grouped, 'kernel', return_value=Launch()), \
                patch.object(torch.Tensor, 'to', transfer), patch.object(torch, 'empty_like', side_effect=allocate):
            adapter = grouped.Adapter()
            prepared = adapter.prepare('after', dict(family='12', m=3, n=5, k=7), fixture, None)
            self.assertEqual(calls, [])
            self.assertEqual(prepared.compilation_state(), [])
            actual = prepared.run()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], (1,))
            self.assertEqual(calls[0][1][-4:], (64, 128, 32, 4))
            self.assertEqual(prepared.expected_kernels, ['gallery_grouped_gemm'])
            self.assertTrue(adapter.validate('after', actual, fixture, {})['passed'])
            self.assertEqual(prepared.compilation_state()[0]['name'], 'gallery_grouped_gemm')
            self.assertEqual(prepared.metadata['workspace_scope'], 'adapter-explicit allocation only')

    def test_lazy_frontend_resolves_all_dependencies_without_launch(self):
        try:
            import triton
        except ImportError:
            self.skipTest('Triton frontend unavailable')
        compiled = grouped.kernel()
        self.assertTrue(compiled.cache_key)
        self.assertEqual(compiled.fn.__name__, 'gallery_grouped_gemm')


if __name__ == '__main__':
    unittest.main()
