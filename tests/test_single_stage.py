import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/optimization-library/catlass'))
from single_stage_adapter import route, make_case, check_spec

class SingleStageTests(unittest.TestCase):
    def spec(self, **kw):
        return dict(dict(m=128,n=256,k=64,seed=7101), **kw)
    def test_actual_cube_gate(self):
        spec=self.spec(m=512,n=1024)
        self.assertEqual(route(spec,'after',16),'direct')
        self.assertEqual(route(spec,'after',15),'general')
        self.assertEqual(route(spec,'before',20),'general')
    def test_guards_share_general(self):
        for spec in (self.spec(m=1024,n=1024), self.spec(k=272)):
            self.assertEqual([route(spec,a,20) for a in ('before','after')],['general','general'])
    def test_contract_rejections(self):
        for kw in (dict(k=0),dict(k=17),dict(m=True),dict(seed=1.5)):
            with self.assertRaises(ValueError):check_spec(self.spec(**kw))
        for cores in (0,True,1.5):
            with self.assertRaises(ValueError):route(self.spec(),'after',cores)
    def test_represented_oracle_and_storage(self):
        spec=self.spec(m=16,n=16,k=16)
        fixture=make_case(spec);a,b=fixture['inputs'].values()
        self.assertEqual(a.dtype,torch.float16)
        self.assertEqual(a.device.type,'cpu')
        self.assertEqual(a.stride(),(16,1));self.assertEqual(b.stride(),(16,1))
        self.assertLessEqual(float(a.abs().max()),.25)
        self.assertLessEqual(float(b.abs().max()),.25)
        expected=torch.tensor([[sum(float(a[i,t])*float(b[t,j]) for t in range(16))
                                for j in range(16)] for i in range(16)],dtype=torch.float16,device='cpu')
        self.assertTrue(torch.equal(expected,fixture['expected']))
        again=make_case(spec)
        self.assertTrue(torch.equal(a,again['inputs']['a']))

if __name__=='__main__':unittest.main()
