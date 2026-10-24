import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
from moe_rows_contract import make_case,reference,validate_inputs,dimensions
import torch


class RowMovement(unittest.TestCase):
    def spec(self, **params):
        return dict(seed=9193,parameters=dict(N=13,D=16,forward=True,with_probs=True,
                    chunks=5,map_kind='shuffled_chunks',**params))

    def test_bitwise_roundtrip(self):
        spec=self.spec(); data=make_case(spec)
        moved=data['expected']
        back=reference(dict(x=moved[0],probs=moved[1],row_map=data['inputs']['row_map']),False)
        for original,result in zip((data['inputs']['x'],data['inputs']['probs']),back):
            self.assertTrue(torch.equal(original.view(torch.int32),result.view(torch.int32)))

    def test_inverse_without_probs(self):
        spec=self.spec();spec['parameters'].update(forward=False,with_probs=False,N=9,D=32)
        fixture=make_case(spec)
        self.assertIsNone(fixture['expected'][1])
        self.assertTrue(torch.equal(fixture['expected'][0].view(torch.int32),
            fixture['inputs']['x'][fixture['inputs']['row_map'].long()].view(torch.int32)))

    def test_admission(self):
        spec=self.spec();data=make_case(spec)
        for bad in (-1,13,0):
            broken={k:v.clone() for k,v in data['inputs'].items()}
            broken['row_map'][1]=bad
            if bad==0 and data['inputs']['row_map'][1]==0: broken['row_map'][0]=0
            with self.assertRaises(ValueError): validate_inputs(broken,spec)
        bad={**data['inputs'],'x':data['inputs']['x'].double()}
        with self.assertRaises(ValueError): validate_inputs(bad,spec)
        for n,d in ((2**24,8),(5,3),(2**23,4096)):
            changed=self.spec();changed['parameters'].update(N=n,D=d)
            with self.assertRaises(ValueError): dimensions(changed)

    def test_empty_and_flat_metadata(self):
        spec=dict(seed=9921,N=0,D=16,with_probs=False,forward=False,map_kind='empty',role='ordinary')
        data=make_case(spec)
        self.assertEqual(tuple(data['expected'][0].shape),(0,16))
        path=ROOT/'examples/optimization-library/triton/moe_rows.py'
        module_spec=importlib.util.spec_from_file_location('outside_adapter',path)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        self.assertTrue(module.Adapter.correctness_only(spec))
        self.assertIsNone(data['expected'][1])


if __name__=='__main__': unittest.main()
