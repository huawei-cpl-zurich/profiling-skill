import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/catlass'),str(ROOT/'scripts')]
from mixed_adapter import Adapter,configure
from mixed_contract import check_spec,make_padding_case,gemm_contract
from optimization_native_bridge import NativeArm
from optimization_pair import fixture


class MixedTests(unittest.TestCase):
    def test_epilogue_reuses_exact_cross_language_oracle(self):
        with tempfile.TemporaryDirectory() as folder:
            spec=dict(family='02',m=16,n=16,k=16,seed=9201,b_layout='row')
            first=Adapter('02',folder,20)
            value,record=fixture(first,spec,Path(folder)/'cache',torch)
            second=Adapter('02',folder,20)
            def never(spec): raise AssertionError('duplicate oracle generation')
            second.make_case=never
            cached,again=fixture(second,spec,Path(folder)/'cache',torch)
            self.assertEqual((record['oracle_calls'],again['oracle_calls']),(1,0))
            self.assertTrue(torch.equal(value['expected'],cached['expected']))
            self.assertEqual(cached['expected'].dtype,torch.float32)

    def test_padding_oracle_tails_and_numeric_diagnostics(self):
        spec=dict(family='10',m=17,n=19,k=21,seed=9202)
        value=make_padding_case(spec)
        expected=(value['inputs']['a'].float()@value['inputs']['b'].float()).half()
        self.assertTrue(torch.equal(value['expected'],expected))
        self.assertTrue(gemm_contract.validate_actual(expected,value)['passed'])
        self.assertFalse(gemm_contract.validate_actual(expected+1,value)['passed'])
        inputs={k:v.clone() for k,v in value['inputs'].items()}
        inputs['a'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'input mutated'):
            gemm_contract.validate_actual(expected,value,inputs)

    def test_dispatch_guards_run_before_device_preparation(self):
        for spec,family in [(dict(family='10',m=16,n=16,k=16),'02'),
                (dict(family='02',m=17,n=16,k=16),'02'),
                (dict(family='10',m=16,n=16,k=16,b_layout='column'),'10')]:
            with self.assertRaises(ValueError): check_spec(spec,family)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required for actual ABI boundary')
    def test_real_abi_workspace_pointer_launch_and_immutable_addend(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            source=folder/'model.cpp'
            source.write_text(r'''#include <stdint.h>
static uint32_t m_,n_,family_=2,calls_=0;static bool bad_;
extern "C" uint32_t catlass_mixed_family(){return family_;}
extern "C" void select_family(uint32_t f){family_=f;}
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t c){
 m_=m;n_=n;bad_=k==32;return !c||!m||!n||!k;}
extern "C" uint64_t catlass_mixed_workspace(){return bad_?1:uint64_t(m_)*n_*4;}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\"test_add\"]";}
extern "C" uint32_t call_count(){return calls_;}
extern "C" int catlass_bridge_launch(void **p,uint64_t n,void*){
 if(n!=5)return 2;++calls_;auto z=(const float*)p[2];auto out=(float*)p[4];
 for(uint64_t i=0;i<uint64_t(m_)*n_;++i)out[i]=1+z[i];return 0;}
''')
            binary=folder/'model.so'
            subprocess.run(['c++','-shared','-fPIC',str(source),'-o',str(binary)],
                           capture_output=True,text=True,check=True)
            lib=ctypes.CDLL(str(binary))
            record=configure(lib,'02','after',[16,16,16],20)
            self.assertEqual(record['workspace_bytes'],1024)
            self.assertEqual(record['launch_count'],1)
            a=torch.ones(16,16,dtype=torch.float16)
            b=torch.full((16,16),1/16,dtype=torch.float16)
            z=torch.arange(256,dtype=torch.float32).reshape(16,16)
            original=z.clone();workspace=torch.empty(1024,dtype=torch.uint8);out=torch.empty_like(z)
            arm=NativeArm(binary,[a,b,z,workspace],out,sources=[source],source_revision='unit-boundary',stream_provider=lambda:0)
            try:
                for _ in range(11): result=arm.run()
                self.assertTrue(torch.equal(result,a.float()@b.float()+z))
                self.assertTrue(torch.equal(z,original))
                self.assertEqual(lib.call_count(),11)
            finally: arm.close()
            for family,shape,cores in [('10',[16,16,16],20),('02',[16,16,32],20),('02',[16,16,16],0)]:
                with self.assertRaises(ValueError): configure(lib,family,'after',shape,cores)


if __name__=='__main__': unittest.main()
