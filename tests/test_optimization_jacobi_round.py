from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
from jacobi_contract import diagnostics, make_case, pair_map, reference, validate_input
from jacobi_round import Stage


class JacobiRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads((ROOT/'experiments/optimization-wave004/W09.json').read_text())
        cls.cases = cls.matrix['cases'] + cls.matrix['contract']['guards']
        cls.fixtures = [make_case(spec) for spec in cls.cases]

    def test_all_six_frozen_oracles_and_fp32_serial_staged_equivalence(self):
        for spec, fixture in zip(self.cases,self.fixtures):
            clean = json.loads(json.dumps(spec))
            clean['parameters'].pop('invalid_duplicate_pair_map',None)
            pairs, mate, sign = validate_input(clean,fixture)
            a = fixture['a']
            c,s = torch.ones(a.shape[0]),torch.zeros(a.shape[0])
            for p,q in pairs:
                tau = (a[q,q]-a[p,p])/(2*a[p,q])
                t = torch.where(tau>=0,1.,-1.)/(tau.abs()+torch.sqrt(1+tau*tau)) if a[p,q] != 0 else torch.tensor(0.)
                c[p] = c[q] = 1/torch.sqrt(1+t*t)
                s[p],s[q] = -t*c[p],t*c[p]
            serial = a.clone()
            for p,q in pairs:
                ap,aq = serial[p].clone(),serial[q].clone()
                serial[p],serial[q] = c[p]*ap+s[p]*aq,c[q]*aq+s[q]*ap
                ap,aq = serial[:,p].clone(),serial[:,q].clone()
                serial[:,p],serial[:,q] = c[p]*ap+s[p]*aq,c[q]*aq+s[q]*ap
            scratch = c[:,None]*a+s[:,None]*a[mate,:]
            staged = c[None,:]*scratch+s[None,:]*scratch[:,mate]
            self.assertTrue(diagnostics(serial,fixture)['passed'],spec['case_id'])
            self.assertTrue(diagnostics(staged,fixture)['passed'],spec['case_id'])
            torch.testing.assert_close(serial,staged,atol=2e-6,rtol=2e-6)
            if spec['parameters']['N'] % 2:
                self.assertEqual(sum(i==v for i,v in enumerate(mate)),1)

    def test_round_generator_covers_each_pair_once_per_full_schedule(self):
        for n in range(2,12):
            seen = []
            for r in range(n+n%2-1):
                pairs,mate,_ = pair_map(dict(N=n,round_index=r))
                self.assertEqual(len({x for pair in pairs for x in pair}),len(pairs)*2)
                self.assertTrue(all(mate[mate[i]]==i for i in range(n)))
                seen.extend(tuple(p) for p in pairs)
            self.assertEqual(len(seen),n*(n-1)//2)
            self.assertEqual(len(set(seen)),len(seen))

    def test_duplicate_rejected_both_arms_before_device_import(self):
        spec,fixture = self.cases[-1],self.fixtures[-1]
        for arm in ('before','after'):
            with self.assertRaisesRegex(ValueError,'prescribed disjoint round'):
                Stage(arm,spec,fixture)

    def test_diagnostics_reject_nan_incomplete_and_wrong_transform(self):
        f = self.fixtures[0]
        self.assertTrue(diagnostics(f['expected'].float(),f)['passed'])
        self.assertFalse(diagnostics(torch.full_like(f['a'],float('nan')),f)['passed'])
        self.assertFalse(diagnostics(f['a'],f)['passed'])
        self.assertFalse(diagnostics(f['a'][:-1],f)['passed'])
        diagonal = torch.diag(torch.arange(1,5,dtype=torch.float32))
        torch.testing.assert_close(reference(diagonal,[[0,3],[1,2]]),diagonal.double())

    def test_committed_nested_case_through_actual_cli_rejects_before_npu(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d/'case.json').write_text(json.dumps(self.cases[-1]))
            (d/'provenance.json').write_text('{}')
            env = dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
            run = subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
                '--adapter','jacobi_round:factory','--case',str(d/'case.json'),
                '--output',str(d/'out'),'--cache',str(d/'cache'),
                '--provenance',str(d/'provenance.json'),'--correctness-only'],env=env,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            report = d/'out/pair.json'
            # CLI may propagate or record adapter rejection; either exposes the host error.
            evidence = run.stderr + ''.join(p.read_text() for p in (d/'out').glob('*.json'))
            self.assertIn('prescribed disjoint round',evidence)
            self.assertNotIn('torch_npu',evidence)


if __name__ == '__main__':
    unittest.main()
