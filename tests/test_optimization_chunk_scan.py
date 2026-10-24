"""Functional host checks for the W04 input domain and prefix-reset contract."""
import json
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/triton'))
import chunk_scan_contract as contract
from chunk_scan import selection


class ChunkScanTests(unittest.TestCase):
    def specs(self):
        return [json.loads(p.read_text()) for p in sorted((ROOT/'examples/optimization-library/cases/wave004').glob('W004-W04-*.json'))]

    def test_seeded_fixtures_and_sequential_prefixes(self):
        for spec in self.specs():
            with self.subTest(case=spec['case_id']):
                fixture = contract.make_case(spec)
                contract.validate_inputs(fixture['inputs'], spec)
                self.assertTrue(torch.equal(fixture['inputs'],contract.make_case(spec)['inputs']))
                b,t,h,chunk = contract.dimensions(spec)
                serial = torch.empty_like(fixture['inputs'])
                for start in range(0,t,chunk):
                    state = torch.zeros(b,h)
                    for token in range(start,min(start+chunk,t)):
                        state += fixture['inputs'][:,token]
                        serial[:,token] = state
                self.assertTrue(contract.diagnostics(serial,fixture['expected'])['passed'])

    def test_reset_and_partial_chunk(self):
        guards = {s['case_id']:s for s in self.specs() if s['role']=='guard'}
        reset = contract.make_case(guards['W004-W04-GA'])['expected']
        self.assertTrue(torch.equal(reset[:,63],torch.full((1,2),64/1024)))
        self.assertTrue(torch.equal(reset[:,64],torch.full((1,2),-1/1024)))
        partial = contract.make_case(guards['W004-W04-GB'])
        self.assertTrue(torch.equal(partial['expected'][:,64],partial['inputs'][:,64]))
        self.assertTrue(torch.equal(partial['expected'][:,69],partial['inputs'][:,64:].sum(1)))
        for arm in ('before','after'):
            self.assertEqual(selection(arm,guards['W004-W04-GB']),0)

    def test_domain_and_shape_rejection(self):
        spec = self.specs()[0]
        x = contract.make_case(spec)['inputs']
        for invalid in (x.double(), x.transpose(1,2), torch.full_like(x,float('nan')), x+1/2048):
            with self.assertRaises(ValueError):contract.validate_inputs(invalid,spec)
        bad = dict(spec,parameters=dict(spec['parameters'],chunk=63))
        with self.assertRaises(ValueError):contract.dimensions(bad)
        with self.assertRaises(ValueError):selection('unknown',spec)

    def test_output_diagnostics_reject_corruption(self):
        fixture = contract.make_case(self.specs()[0]);expected=fixture['expected']
        self.assertFalse(contract.diagnostics(expected.double(),expected)['passed'])
        actual=expected.clone();actual.flatten()[-1]+=1/1024
        self.assertFalse(contract.diagnostics(actual,expected)['passed'])


if __name__=='__main__':unittest.main()
