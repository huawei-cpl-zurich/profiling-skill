import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton'))
from logsoftmax_contract import dimensions, make_case, diagnostics
from logsoftmax_residency import Adapter

class ContractTests(unittest.TestCase):
    def test_flat_nested_equivalence(self):
        flat = dict(rows=3, width=47, seed=981)
        nested = dict(parameters=dict(rows=3, width=47), seed=981)
        self.assertEqual(dimensions(flat), dimensions(nested))
        a, b = make_case(flat), make_case(nested)
        self.assertTrue(torch.equal(a['inputs'][0], b['inputs'][0]))
        self.assertTrue(diagnostics(a['expected'], a)['passed'])
        bad = tuple(x + .1 for x in a['expected'])
        self.assertFalse(diagnostics(bad, a)['passed'])

    def test_compound_same_bits_and_independent_neighbors(self):
        case = make_case(dict(rows=1, width=53, seed=982, batch_contexts=[1,7,32]))
        self.assertEqual([x.shape[0] for x in case['inputs']], [1,7,32])
        self.assertTrue(diagnostics(case['expected'], case)['passed'])
        self.assertFalse(torch.equal(case['inputs'][1][1], case['inputs'][2][1]))
        wrong = list(case['expected']); wrong[1] = wrong[1].clone()
        wrong[1][0,0] = torch.nextafter(wrong[1][0,0], torch.tensor(float('inf')))
        self.assertFalse(diagnostics(tuple(wrong), case)['passed'])

    def test_reject_unsupported_before_cache_or_device(self):
        for spec in [dict(rows=0,width=10,seed=12),dict(rows=1,width=10,seed=-1),
                     dict(rows=1,width=10,seed=12,batch_contexts=[1,2])]:
            with self.assertRaises(ValueError):make_case(spec)

if __name__ == '__main__': unittest.main()
