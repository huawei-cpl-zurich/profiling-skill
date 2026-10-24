"""Functional CPU contract and the same configuration ABI used by the device library."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import torch

EXAMPLES=Path(__file__).resolve().parents[1]/'examples/optimization-library/catlass'
sys.path.insert(0,str(EXAMPLES))
from interval_adapter import configure
from interval_contract import check_spec,copy_plan,make_case,gemm_contract
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from optimization_native_bridge import build_shared,digest


class IntervalTests(unittest.TestCase):
    def spec(self,**changes):
        return dict(dict(family='E12',m=1,n=48,k=64,seed=1201),**changes)

    def test_small_and_tail_rows_have_independent_complete_oracle(self):
        for m in (1,4,17,129):
            spec=self.spec(m=m)
            fixture=make_case(spec)
            a,b=(fixture['inputs'][name] for name in ('a','b'))
            expected=(a.double()@b.double()).half()
            self.assertTrue(gemm_contract.validate_actual(expected,fixture)['passed'])
            self.assertEqual(fixture['expected'].shape,(m,48))

    def test_alignment_and_layout_rejected_before_preparation(self):
        for changes in (dict(family='05'),dict(m=0),dict(m=True),dict(n=17),dict(k=31),
                        dict(k=64.),dict(b_layout='column'),dict(m=32769)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                check_spec(self.spec(**changes))

    def test_copy_request_model_accounts_for_all_row_and_k_tails(self):
        one=copy_plan(self.spec(n=512,k=272))
        self.assertEqual((one['nd2nz_requests'],one['interval_row_requests']),(4,4))
        tail=copy_plan(self.spec(m=129,n=512,k=272))
        self.assertEqual((tail['nd2nz_requests'],tail['interval_row_requests']),(8,516))
        self.assertTrue(tail['source_model_only'])
        self.assertEqual(tail['workspace_bytes'],0)

    def test_output_and_mutation_rejected_without_tolerance_change(self):
        fixture=make_case(self.spec(m=4))
        self.assertFalse(gemm_contract.validate_actual(fixture['expected']+1,fixture)['passed'])
        inputs={key:value.clone() for key,value in fixture['inputs'].items()}
        inputs['a'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'input mutated'):
            gemm_contract.validate_actual(fixture['expected'],fixture,inputs)

    @unittest.skipUnless(shutil.which('c++'),'host C++ compiler required')
    def test_actual_configuration_header_accepts_tail_rows_and_binds_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            header=root/'interval_config.h'
            shutil.copyfile(EXAMPLES/'interval_config.h',header)
            source=root/'host.cpp'
            source.write_text('#include "interval_config.h"\n')
            for arm in ('before','after'):
                binary=root/f'{arm}.so'
                manifest=build_shared(['c++','-shared','-fPIC',f'-DAFTER={int(arm=="after")}',
                    str(source),'-o',str(binary)],source_paths=[source,header],binary=binary,
                    output=root/f'build-{arm}',source_revision='configuration-only')
                self.assertIn(digest(header),[entry['sha256'] for entry in manifest['sources']])
                library=ctypes.CDLL(str(binary))
                for m in (1,4,17,129):
                    self.assertEqual(configure(library,arm,self.spec(m=m),20)['workspace_bytes'],0)
                for wrong_arm,cores in [('after' if arm=='before' else 'before',20),
                                        (arm,0),(arm,True)]:
                    with self.assertRaises(ValueError):
                        configure(library,wrong_arm,self.spec(),cores)
                for shape in ((0,16,16),(1,15,16),(1,16,15),(32769,16,16)):
                    self.assertNotEqual(library.catlass_bridge_configure(*shape,20),0)


if __name__=='__main__':
    unittest.main()
