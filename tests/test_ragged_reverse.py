import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/triton'), str(ROOT/'scripts')]
from ragged_reverse_contract import checked_inputs, make_case, parameters, reference
from ragged_reverse import Adapter
from optimization_pair import fixture


def spec(**updates):
    p = dict(B=3, T=19, D=5, length_pattern='mixed-short')
    p.update(updates)
    return dict(case_id='test', role='guard', seed=18, parameters=p)


class ReverseTests(unittest.TestCase):
    def test_clamp_and_involution_against_explicit_prefixes(self):
        s = spec(B=4, T=17, D=3, length_pattern='clamp-extremes')
        f = make_case(s)
        self.assertEqual(f['lengths'].tolist(), [-3, 0, 17, 22])
        for b, length in enumerate((0, 0, 17, 17)):
            want = f['x'][b].clone()
            want[:length] = f['x'][b, :length].flip(0)
            self.assertTrue(torch.equal(f['one_reverse'][b], want))
        self.assertTrue(torch.equal(reference(f['one_reverse'], f['lengths']), f['x']))
        self.assertTrue(torch.equal(f['expected'], f['x']))

    def test_suffix_and_channel_order(self):
        f = make_case(spec())
        for b, length in enumerate((1, 7, 0)):
            for i in range(19):
                source = length-1-i if i < length else i
                self.assertTrue(torch.equal(f['expected'][b, i], f['x'][b, source]))
            self.assertTrue(torch.equal(f['expected'][b, length:], f['x'][b, length:]))
        self.assertNotEqual(f['x'].data_ptr(), f['expected'].data_ptr())

    def test_signed_zero_bits_preserved(self):
        x = torch.tensor([0., -0., 1., -1.], dtype=torch.float32).view(1, 4, 1)
        out = reference(x, torch.tensor([3], dtype=torch.int32))
        self.assertEqual(out.view(torch.int32).flatten().tolist(), [1065353216, -2147483648, 0, -1082130432])

    def test_input_rejection(self):
        s = spec(); p = parameters(s); f = make_case(s)
        f['x'] = f['x'].transpose(1, 2).contiguous().transpose(1, 2)
        with self.assertRaises(ValueError): checked_inputs(f, p)
        f = make_case(s); f['lengths'] = f['lengths'].long()
        with self.assertRaises(ValueError): checked_inputs(f, p)
        f = make_case(s); f['x'][0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): checked_inputs(f, p)
        for update in (dict(B=0), dict(T=True), dict(length_pattern='unknown')):
            with self.assertRaises(ValueError): parameters(spec(**update))

    def test_validation_observes_corrupt_output_and_input(self):
        s = spec(); f = make_case(s); adapter = Adapter()
        adapter.prepared['before'] = SimpleNamespace(inputs=(f['x'].clone(), f['lengths'].clone()), intermediate=None)
        self.assertTrue(adapter.validate('before', f['expected'].clone(), f, s)['passed'])
        corrupt = f['expected'].clone(); corrupt[2, -1, -1] += 1
        self.assertFalse(adapter.validate('before', corrupt, f, s)['passed'])
        adapter.prepared['before'].inputs[0][0, 0, 0] += 1
        self.assertFalse(adapter.validate('before', f['expected'].clone(), f, s)['passed'])

    def test_validation_requires_nontrivial_guard_intermediate(self):
        s = spec(B=4, T=17, D=3, length_pattern='clamp-extremes'); f = make_case(s)
        adapter = Adapter()
        item = SimpleNamespace(inputs=(f['x'].clone(), f['lengths'].clone()), intermediate=f['one_reverse'].clone())
        adapter.prepared['after'] = item
        self.assertTrue(adapter.validate('after', f['expected'].clone(), f, s)['passed'])
        item.intermediate = f['x'].clone()
        self.assertFalse(adapter.validate('after', f['expected'].clone(), f, s)['passed'])

    def test_fixture_cache_under_ambient_float64(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with tempfile.TemporaryDirectory() as temp:
                adapter = Adapter(); first, a = fixture(adapter, spec(), Path(temp), torch)
                adapter.make_case = lambda _: self.fail('oracle regenerated')
                second, b = fixture(adapter, spec(), Path(temp), torch)
                self.assertEqual((a['oracle_calls'], b['oracle_calls']), (1, 0))
                self.assertEqual(first['x'].dtype, torch.float32)
                self.assertEqual(first['lengths'].dtype, torch.int32)
                self.assertTrue(torch.equal(first['expected'], second['expected']))
        finally: torch.set_default_dtype(previous)

    def test_public_nested_case_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            case, output = Path(temp)/'case.json', Path(temp)/'fixture.pt'
            case.write_text(json.dumps(spec()))
            subprocess.run([sys.executable, str(ROOT/'examples/optimization-library/triton/ragged_reverse_contract.py'), '--case', str(case), '--output', str(output)], check=True)
            f = torch.load(output, weights_only=True)
            self.assertEqual(tuple(f['expected'].shape), (3, 19, 5))
            self.assertTrue(torch.equal(f['expected'], reference(f['x'], f['lengths'])))

if __name__ == '__main__': unittest.main()
