import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/triton'))
sys.path.insert(0, str(ROOT/'scripts'))
from slot_mapping_contract import make_case, parameters, reference, checked_inputs
from optimization_pair import fixture
from slot_mapping import Adapter


def spec(**updates):
    p = dict(T=3, K=9, requests=3, block_size=32, mode='mixed')
    p.update(updates)
    return dict(case_id='test', role='guard', seed=13, parameters=p)


class SlotContractTests(unittest.TestCase):
    def test_nested_parameters_and_large_exact_boundary(self):
        s = spec(T=2, K=8, requests=2, block_size=64, mode='decode', distribution='boundaries-and-large-exact-base')
        f = make_case(s)
        self.assertEqual(f['expected'][0, :5].tolist(), [-1, -1, 33554496, 33554559, int(f['block_table'][0, 1])*64])
        self.assertEqual(int(f['expected'][0, 6]), -1)
        self.assertEqual(int(f['expected'][0, 7]), 33554497)

    def test_mixed_prefill_keeps_invalid_block_rule(self):
        s = spec(K=5)
        f = make_case(s)
        f['tok'][1] = torch.tensor([-2, -1, 0, 4095, 4096], dtype=torch.int32)
        expected = reference(f, parameters(s))
        start = int(f['starts'][f['prefill_id'][1]])
        self.assertEqual(expected[1].tolist(), [-1, -1, start, start+4095, -1])
        for row in (0, 2):
            for col in range(5):
                tok = int(f['tok'][row, col])
                result = -1 if tok < 0 or tok >= 4096 else int(f['block_table'][f['request_id'][row], tok//32])*32 + tok%32
                self.assertEqual(int(expected[row, col]), result)

    def test_invalid_layout_metadata_and_overflow(self):
        s = spec(); p = parameters(s); f = make_case(s)
        f['tok'] = f['tok'].t().contiguous().t()
        with self.assertRaises(ValueError): checked_inputs(f, p)
        f = make_case(s); f['request_id'][0] = 3
        with self.assertRaises(ValueError): checked_inputs(f, p)
        f = make_case(s); f['block_table'][0, 0] = 2147483647
        with self.assertRaises(ValueError): checked_inputs(f, p)
        with self.assertRaises(ValueError): parameters(spec(block_size=3))

    def test_fixture_cache_reuses_oracle_with_nested_cli_case(self):
        s = json.loads(json.dumps(spec(K=33)))
        with tempfile.TemporaryDirectory() as temp:
            adapter = Adapter()
            first, a = fixture(adapter, s, Path(temp), torch)
            adapter.make_case = lambda _: self.fail('cache regenerated oracle')
            second, b = fixture(adapter, s, Path(temp), torch)
            self.assertEqual((a['oracle_calls'], b['oracle_calls']), (1, 0))
            self.assertTrue(torch.equal(first['expected'], second['expected']))

    def test_public_cli_reads_nested_case(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)/'case.json'
            output = Path(temp)/'payload.pt'
            case.write_text(json.dumps(spec(T=2, K=33)))
            subprocess.run([sys.executable, str(ROOT/'examples/optimization-library/triton/slot_mapping_contract.py'),
                            '--case', str(case), '--output', str(output)], check=True)
            result = torch.load(output, weights_only=True)
            self.assertEqual(tuple(result['expected'].shape), (2, 33))
            self.assertTrue(torch.equal(result['expected'], reference(result, parameters(spec(T=2, K=33)))))

    def test_fixture_is_explicit_int32_under_default_float64(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            f = make_case(spec())
            self.assertTrue(all(x.dtype == torch.int32 and x.device.type == 'cpu' for x in f.values()))
        finally:
            torch.set_default_dtype(previous)

if __name__ == '__main__':
    unittest.main()
