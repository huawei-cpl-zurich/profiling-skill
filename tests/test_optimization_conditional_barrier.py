import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import torch
EXAMPLES=Path(__file__).resolve().parents[1]/'examples/optimization-library/catlass'
sys.path.insert(0,str(EXAMPLES))
from conditional_barrier_contract import check_spec,make_case,validate_actual,TILES
from conditional_barrier_adapter import configure,isolate_binary,Composite
from conditional_barrier_compile import prepare_dependency,HEADER,PREDICATE,sha,PIN


class ConditionalBarrierTests(unittest.TestCase):
    def spec(self,**values):
        return dict(dict(m=64,n=64,k=192,tile_m=64,tile_n=64,seed=7592),**values)

    def test_all_regimes_and_exact_boundary_are_real_tiles(self):
        for m,n in TILES:
            self.assertEqual(check_spec(self.spec(m=m,n=n,tile_m=m,tile_n=n),1),1)
        for bad in [dict(tile_m=128),dict(n=80),dict(k=272),dict(b_layout='column'),dict(consecutive_distinct_calls=2)]:
            with self.assertRaises(ValueError):make_case(self.spec(**bad))
        with self.assertRaises(ValueError):check_spec(self.spec(),0)

    def test_oracle_composite_inputs_and_numerical_rejection(self):
        f=make_case(self.spec(role='guard',consecutive_distinct_calls=2))
        self.assertEqual(len(f['expected']),2)
        self.assertFalse(torch.equal(f['components'][0]['inputs']['a'],f['components'][1]['inputs']['a']))
        for part in f['components']:
            self.assertEqual(part['expected'].dtype,torch.float64)
            self.assertTrue(validate_actual(part['expected'].half(),part,part['inputs'])['passed'])
            self.assertFalse(validate_actual(part['expected'].half()+1,part,part['inputs'])['passed'])
            changed={k:v.clone() for k,v in part['inputs'].items()};changed['a'][0,0]+=1
            with self.assertRaises(ValueError):validate_actual(part['expected'].half(),part,changed)

    def test_dependency_patcher_preserves_input_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stock=root/'stock';(stock/HEADER).parent.mkdir(parents=True)
            text='template <\n    CATLASS_DEVICE\n    void operator()(\nAscendC::Mmad(l0CTensor\n'+PREDICATE+'\nbody\n    CATLASS_DEVICE\n    void operator()(\n'+PREDICATE
            (stock/HEADER).write_text(text)
            (stock/'manifest.json').write_text(json.dumps(dict(revision=PIN,files=[dict(file=HEADER,sha256=sha(text.encode()))])))
            before=prepare_dependency(stock,root/'before','before');after=prepare_dependency(stock,root/'after','after')
            self.assertEqual((stock/HEADER).read_text(),text)
            self.assertEqual((root/'after'/HEADER).read_text(),text)
            self.assertEqual((root/'before'/HEADER).read_text(),text.replace(PREDICATE,'if (true) {',1))
            self.assertEqual(before['stock_header_sha256'],after['compiled_header_sha256'])
            (stock/HEADER).write_text(text+'corrupt')
            with self.assertRaises(ValueError):prepare_dependency(stock,root/'corrupt','before')

    def test_public_compile_cli_rejects_unpinned_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dependency=root/'dependency';dependency.mkdir()
            (dependency/'manifest.json').write_text(json.dumps(dict(revision='wrong',files=[])))
            result=subprocess.run([sys.executable,str(EXAMPLES/'conditional_barrier_compile.py'),
                '--catlass',str(dependency),'--output',str(root/'output')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(list((root/'output').rglob('*.so')))
            self.assertFalse((root/'output'/'dependency-before').exists())

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
    def test_private_cabi_binary_arm_tile_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'abi.cpp').write_text('''#include <stdint.h>
static uint32_t m_;
extern "C" uint32_t case_m(){return m_;}
extern "C" uint32_t catlass_stock_barrier_enabled(){return 1;}
extern "C" uint32_t catlass_tile_m(){return 64;}
extern "C" uint32_t catlass_tile_n(){return 64;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\"wave006_conditional_barrier_after\\"]";}
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t c){m_=m;return k==64;}
''')
            subprocess.run(['c++','-shared','-fPIC',str(root/'abi.cpp'),'-o',str(root/'abi.so')],check=True,capture_output=True)
            first=ctypes.CDLL(str(isolate_binary(root/'abi.so',root/'a')))
            second=ctypes.CDLL(str(isolate_binary(root/'abi.so',root/'b')))
            self.assertFalse(configure(first,'after',self.spec(),1)['enable_unit_flag'])
            self.assertEqual(second.case_m(),0)
            for arm,spec in [('before',self.spec()),('after',self.spec(m=32,tile_m=32)),('after',self.spec(k=64))]:
                with self.assertRaises(ValueError):configure(first,arm,spec,1)
