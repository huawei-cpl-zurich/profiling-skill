import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton'
sys.path.insert(0, str(EXAMPLES))
import int4_contract as contract
import int4_pairs as pair


class Int4Tests(unittest.TestCase):
    def spec(self, **updates):
        return dict(dict(family='int4-consumer', m=3, n=16, k=64, group_size=32, seed=244101), **updates)

    def test_natural_nibble_order_and_signed_storage(self):
        values = torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]])
        words = contract.pack8(values)
        self.assertEqual(words.tolist(), [[-19088744, 1985229328]])
        self.assertTrue(torch.equal(contract.unpack8(words), values))
        for bad in (values[:, :-1], values.float(), values - 1, values + 1):
            with self.assertRaises(ValueError):
                contract.pack8(bad)

    def test_group_decode_has_explicit_half_rounding(self):
        words = torch.tensor([[-19088744, 1985229328]], dtype=torch.int32).repeat(64, 1)
        zeros = torch.tensor([[-2004318072, -2004318072], [0, 0]], dtype=torch.int32)
        scales = torch.linspace(.013, .087, 32).reshape(2, 16).half()
        decoded = contract.dequantize(words, zeros, scales, 32)
        values = torch.tensor([*range(8, 16), *range(8)], dtype=torch.float32)
        expected = torch.cat([((values - 8) * scales[0].float()).half().repeat(32, 1),
                              (values * scales[1].float()).half().repeat(32, 1)])
        self.assertTrue(torch.equal(decoded, expected))
        with self.assertRaises(ValueError):
            contract.dequantize(words, zeros, scales.float(), 32)

    def test_fixture_reference_and_error_reporting(self):
        spec = self.spec(m=17, n=72, k=96)
        first, second = contract.make_case(spec), contract.make_case(spec)
        self.assertTrue(torch.equal(first['expected'], second['expected']))
        self.assertEqual(first['expected'].dtype, torch.float16)
        weight = contract.dequantize(first['inputs']['packed'], first['inputs']['zeros'],
                                     first['inputs']['scales'], spec['group_size'])
        expected = (first['inputs']['a'].float() @ weight.float()).half()
        self.assertTrue(torch.equal(first['expected'], expected))
        self.assertTrue(contract.validate_actual(expected, first)['passed'])
        diagnostic = contract.validate_actual(expected + 1, first)
        self.assertFalse(diagnostic['passed'])
        self.assertGreater(diagnostic['max_tolerance_fraction'], 1)
        changed = {name: value.clone() for name, value in first['inputs'].items()}
        changed['packed'][0, 0] ^= 16
        with self.assertRaisesRegex(ValueError, 'input mutated'):
            contract.validate_actual(expected, first, changed)

    def test_dispatch_rejects_unsupported_contract(self):
        for changes in [dict(family='02'), dict(m=True), dict(n=17), dict(k=65),
                        dict(group_size=16), dict(seed=-1), dict(seed=2**64)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                contract.check_spec(self.spec(**changes))
        before, after = (pair.configuration(arm, self.spec()) for arm in ('before', 'after'))
        self.assertEqual({k: v for k, v in before.items() if k != 'fused'},
                         {k: v for k, v in after.items() if k != 'fused'})
        self.assertFalse(before['fused'])
        self.assertTrue(after['fused'])

    def test_prepare_has_no_launch_and_run_counts_complete_operation(self):
        calls = []
        class Kernel:
            def __init__(self, name): self.fn = SimpleNamespace(__name__=name)
            def __getitem__(self, grid):
                def launch(*args):
                    calls.append((self.fn.__name__, grid))
                    return SimpleNamespace(name=self.fn.__name__, hash='compiled', asm={'binary': b'ELF'})
                return launch
        empty, empty_like = torch.empty, torch.empty_like
        def allocate(*args, **kwargs):
            return empty(*args, **dict(kwargs, device='cpu'))
        def allocate_like(*args, **kwargs):
            return empty_like(*args, **dict(kwargs, device='cpu'))
        fixture = contract.make_case(self.spec())
        for arm, count in [('before', 2), ('after', 1)]:
            with patch.dict(sys.modules, {'torch_npu': SimpleNamespace(get_npu_format=lambda value: 2)}), \
                    patch.object(torch.Tensor, 'to', lambda value, *a, **k: value), \
                    patch.object(torch, 'empty', allocate), patch.object(torch, 'empty_like', allocate_like), \
                    patch.object(pair, 'kernels', return_value=(Kernel('dequant'), Kernel('gemm'))):
                start = len(calls)
                prepared = pair.Adapter().prepare(arm, self.spec(), fixture, Path('.'))
                self.assertEqual(len(calls), start)
                self.assertEqual(prepared.compilation_state(), [])
                prepared.run()
                self.assertEqual(len(calls) - start, count)
                self.assertEqual(len(prepared.compilation_state()), count)
                self.assertEqual(prepared.metadata['workspace_bytes'], 64 * 16 * 2 if arm == 'before' else 0)
                self.assertEqual(set(prepared.metadata['input_formats'].values()), {2})
                sys.modules['torch_npu'].get_npu_format = lambda value: 29
                with self.assertRaisesRegex(ValueError, 'ND storage'):
                    pair.Adapter().prepare(arm, self.spec(), fixture, Path('.'))
                self.assertEqual(len(calls) - start, count)

    @unittest.skipUnless(importlib.util.find_spec('triton'), 'Triton frontend required')
    def test_frontend_resolution_has_no_closure_or_device(self):
        for kernel in pair.kernels():
            self.assertIsInstance(kernel.cache_key, str)
            self.assertTrue(kernel.cache_key)
            self.assertFalse(kernel.fn.__closure__)

    @unittest.skipUnless(importlib.util.find_spec('triton'), 'Triton interpreter required')
    def test_actual_kernel_arithmetic_in_cpu_interpreter(self):
        code = textwrap.dedent('''
            import sys,torch
            sys.path.insert(0,sys.argv[1])
            import int4_contract as c,int4_pairs as p
            torch.set_num_threads(2)
            dequant,gemm=p.kernels()
            for m,n,k,g in [(3,16,64,32),(33,72,96,32),(1,8,128,128)]:
                spec=dict(family='int4-consumer',m=m,n=n,k=k,group_size=g,seed=123)
                f=c.make_case(spec);a,q,z,s=(f['inputs'][x] for x in ('a','packed','zeros','scales'))
                d=torch.empty(k,n,dtype=torch.float16);before=torch.empty(m,n,dtype=torch.float16)
                after=torch.empty_like(before)
                dequant[((k*n+1023)//1024,)](q,z,s,d,k,n,g,1024)
                assert torch.equal(d,c.dequantize(q,z,s,g))
                grid=((m+31)//32*((n+63)//64),)
                gemm[grid](a,q,z,s,d,before,m,n,k,g,32,64,32,False)
                gemm[grid](a,q,z,s,after,after,m,n,k,g,32,64,32,True)
                assert c.validate_actual(before,f)['passed']
                assert c.validate_actual(after,f)['passed']
                assert torch.equal(before,after)
            print('9 actual interpreted kernel calls passed')
        ''')
        result = subprocess.run([sys.executable, '-c', code, str(EXAMPLES)],
                                env=dict(os.environ, TRITON_INTERPRET='1'), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
