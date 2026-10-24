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
from preload_contract import check_spec,make_case,gemm_contract
from preload_adapter import configure


class PreloadTests(unittest.TestCase):
    def spec(self,family='08',**changes):
        return dict(dict(family=family,m=32,n=48,k=64,seed=8501),**changes)

    def test_shared_oracle_matches_independent_fp32_accumulation(self):
        for family in ('08','09','11'):
            fixture=make_case(self.spec(family))
            a,b=fixture['inputs']['a'],fixture['inputs']['b']
            expected=(a.float()@b.float()).half()
            self.assertTrue(torch.equal(expected,fixture['expected']))
            self.assertTrue(gemm_contract.validate_actual(expected,fixture)['passed'])

    def test_capacity_gate_accepts_boundary_and_rejects_over_capacity(self):
        check_spec(self.spec('09',m=128,k=1024))
        for spec in (self.spec('09',m=144),self.spec('09',k=1040),
                     self.spec('09',k=384,b_layout='column'),self.spec('10'),
                     self.spec(m=17),self.spec(k=64.0),self.spec(k=0)):
            with self.assertRaises(ValueError):
                check_spec(spec)

    def test_output_error_is_reported_without_tolerance_changes(self):
        fixture=make_case(self.spec('11'))
        result=gemm_contract.validate_actual(fixture['expected']+1,fixture)
        self.assertFalse(result['passed'])
        self.assertGreater(result['max_absolute_error'],.9)
        self.assertEqual(result['rtol'],.001)

    def test_input_mutation_is_rejected(self):
        fixture=make_case(self.spec('09'))
        inputs={name:value.clone() for name,value in fixture['inputs'].items()}
        inputs['b'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'input mutated'):
            gemm_contract.validate_actual(fixture['expected'],fixture,inputs)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required for ABI integration')
    def test_resource_abi_and_regime_information(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'configuration.cpp'
            source.write_text('''#include <stdint.h>
static uint32_t n_;
extern "C" uint32_t catlass_preload_family(){return FAMILY;}
extern "C" uint32_t catlass_preload_after(){return AFTER;}
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t c){
 n_=n; return !c||m%16||n%16||k%16;}
extern "C" uint64_t catlass_preload_l1_bytes(){return FAMILY==9&&AFTER?524288:393216;}
extern "C" uint64_t catlass_preload_workspace(){return n_==16?4:0;}
''')
            for family in ('08','09','11'):
                binary=root/f'{family}.so'
                subprocess.run(['c++','-shared','-fPIC',f'-DFAMILY={int(family)}','-DAFTER=1',
                                str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
                library=ctypes.CDLL(str(binary))
                shape=[128,32768,256] if family=='09' else [2048,2048,512]
                result=configure(library,family,'after',shape,20)
                self.assertEqual(result['workspace_bytes'],0)
                self.assertEqual(result['output_tasks'],128)
                self.assertEqual(result['max_tasks_per_core'],7)
                self.assertEqual(result['source_reuse_opportunity'],family=='09')
                self.assertEqual(result['shuffle_k'],family=='11')
                small=configure(library,family,'after',[128,256,256],20)
                self.assertEqual(small['k_tiles'],1)
                self.assertFalse(small['source_reuse_opportunity'])
                for wrong_family,arm,shape,cores in [(family,'before',[128,256,256],20),
                        ('09' if family!='09' else '08','after',[128,256,256],20),
                        (family,'after',[128,16,256],20),
                        (family,'after',[128,256,256],0),
                        (family,'after',[128,256,256],True)]:
                    with self.assertRaises(ValueError):
                        configure(library,wrong_family,arm,shape,cores)


if __name__=='__main__':
    unittest.main()
