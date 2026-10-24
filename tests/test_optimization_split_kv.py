import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/triton'))
from split_kv_contract import dimensions, diagnostics, reference


class DecodeContractTests(unittest.TestCase):
    def test_grouped_heads_and_dense_normalization(self):
        q = torch.zeros((1,4,1,32), dtype=torch.float16)
        k = torch.zeros((1,2,3,32), dtype=torch.float16)
        v = torch.tensor([1.,3.]).view(1,2,1,1).expand(1,2,3,32).half()
        actual = reference(q,k,v)
        torch.testing.assert_close(actual.flatten(2).mean(-1), torch.tensor([[1.,1.,3.,3.]],dtype=torch.float64))

    def test_empty_output_and_numeric_rejection(self):
        q = torch.ones((1,2,1,32), dtype=torch.float16)
        k = torch.empty((1,1,0,32), dtype=torch.float16)
        v = torch.empty((1,1,0,64), dtype=torch.float16)
        expected = reference(q,k,v)
        self.assertEqual(expected.shape, (1,2,1,64))
        self.assertEqual(expected.abs().sum().item(), 0)
        self.assertTrue(diagnostics(expected.half(), dict(expected=expected))['passed'])
        self.assertFalse(diagnostics(expected.half()+.01, dict(expected=expected))['passed'])
        self.assertFalse(diagnostics(expected.half()*float('nan'), dict(expected=expected))['passed'])

    def test_nonlinear_merge_and_tail(self):
        # Independent partition-stat math checks why normalized partial sums fail.
        scores = torch.tensor([0.,1.,2.,3.,4.],dtype=torch.float64)
        values = torch.tensor([1.,-2.,4.,-3.,2.],dtype=torch.float64)
        stats=[]
        for s,v in zip(scores.tensor_split(4), values.tensor_split(4)):
            m=s.max();p=(s-m).exp();stats.append((m,p.sum(),(p*v).sum()))
        maximum=max(s[0] for s in stats)
        denominator=sum((s[0]-maximum).exp()*s[1] for s in stats)
        numerator=sum((s[0]-maximum).exp()*s[2] for s in stats)
        torch.testing.assert_close(numerator/denominator, (scores.softmax(0)*values).sum())
        self.assertNotAlmostEqual(sum((s[2]/s[1]).item() for s in stats), (numerator/denominator).item())

    def test_reject_non_decode(self):
        spec=dict(parameters=dict(B=1,Hq=8,Hkv=2,Q=2,KV=65,D=64,Dv=32,split=4,blockN=64))
        with self.assertRaises(ValueError): dimensions(spec)


if __name__ == '__main__': unittest.main()
