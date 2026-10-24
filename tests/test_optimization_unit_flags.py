import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import torch
EXAMPLES = Path(__file__).resolve().parents[1]/'examples/optimization-library/catlass'
sys.path.insert(0,str(EXAMPLES))
from unit_flags_contract import check_spec, make_case, validate_actual
from unit_flags_adapter import configure, Composite, isolate_binary


class UnitFlagsTests(unittest.TestCase):
    def spec(self,**values):
        return dict(dict(m=64,n=128,k=192,seed=7291),**values)

    def test_fp64_represented_input_oracle_and_numerical_rejection(self):
        fixture = make_case(self.spec())
        self.assertEqual(fixture['expected'].dtype,torch.float64)
        for value in fixture['inputs'].values():
            self.assertEqual(value.dtype,torch.float16)
            self.assertEqual(value.device.type,'cpu')
            self.assertTrue(bool((value.abs()<=.25).all()))
        self.assertTrue(validate_actual(fixture['expected'].half(),fixture,fixture['inputs'])['passed'])
        self.assertFalse(validate_actual(fixture['expected'].half()+1,fixture,fixture['inputs'])['passed'])
        with self.assertRaises(ValueError):
            validate_actual(torch.full_like(fixture['expected'].half(),float('nan')),fixture,fixture['inputs'])

    def test_grid_is_bound_to_cube_count_and_unsupported_contracts_reject(self):
        self.assertEqual(check_spec(self.spec(m=512,n=1024),20),16)
        for spec,cores in [(self.spec(m=512,n=1024),15),(self.spec(k=272),20),
                           (self.spec(m=65),20),(self.spec(b_layout='column'),20),
                           (self.spec(consecutive_distinct_calls=2),20)]:
            with self.assertRaises(ValueError):check_spec(spec,cores)

    def test_composite_guard_caches_two_distinct_gemms_and_executes_both(self):
        fixture = make_case(self.spec(role='guard',consecutive_distinct_calls=2))
        self.assertEqual(len(fixture['expected']),2)
        self.assertFalse(torch.equal(fixture['components'][0]['inputs']['a'],fixture['components'][1]['inputs']['a']))
        calls=[]
        class Part:
            sources=['source'];expected_kernels=['kernel'];metadata={}
            def __init__(self,index):self.index=index
            def run(self):calls.append(self.index);return self.index
            def close(self):calls.append('closed')
            def compilation_state(self):return {'identity':self.index}
        item=Composite([Part(0),Part(1)])
        self.assertEqual(item.run(),(0,1));self.assertEqual(calls,[0,1])
        self.assertEqual(item.metadata['component_gemms_per_api'],2)
        item.close();self.assertEqual(calls[-2:],['closed','closed'])

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
    def test_real_cabi_configuration_checks_binary_arm_and_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'abi.cpp').write_text('''#include <stdint.h>
static uint32_t m_;
extern "C" uint32_t case_m(){return m_;}
extern "C" uint32_t catlass_unit_flag_enabled(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\"wave006_unit_flags_after\\"]";}
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t c){m_=m;return k==64;}
''')
            subprocess.run(['c++','-shared','-fPIC',str(path/'abi.cpp'),'-o',str(path/'abi.so')],check=True,capture_output=True)
            library=ctypes.CDLL(str(path/'abi.so'))
            self.assertEqual(configure(library,'after',self.spec(),20)['tasks'],1)
            first=ctypes.CDLL(str(isolate_binary(path/'abi.so',path/'case-a')))
            second=ctypes.CDLL(str(isolate_binary(path/'abi.so',path/'case-b')))
            configure(first,'after',self.spec(m=64),20)
            configure(second,'after',self.spec(m=128),20)
            self.assertEqual(first.case_m(),64)
            self.assertEqual(second.case_m(),128)
            configure(first,'after',self.spec(m=256),20)
            self.assertEqual(second.case_m(),128)

            for arm,spec,cores in [('before',self.spec(),20),('after',self.spec(k=64),20),
                                   ('after',self.spec(m=512,n=1024),15)]:
                with self.assertRaises(ValueError):configure(library,arm,spec,cores)
