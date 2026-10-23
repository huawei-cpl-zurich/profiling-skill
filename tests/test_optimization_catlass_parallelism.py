import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

EXAMPLES = Path(__file__).resolve().parents[1]/'examples/optimization-library/catlass'
sys.path.insert(0, str(EXAMPLES))
from parallelism_contract import check_spec, make_case, gemm_contract
from parallelism_adapter import configure


class ParallelismTests(unittest.TestCase):
    def spec(self, family='06', **extra):
        return dict(family=family, m=32, n=48, k=64, seed=8201, **extra)

    def test_oracle_preserves_shared_fp16_contract(self):
        for family in ('06','07'):
            fixture = make_case(self.spec(family))
            self.assertEqual(fixture['expected'].dtype, torch.float16)
            self.assertTrue(gemm_contract.validate_actual(fixture['expected'], fixture)['passed'])
            self.assertTrue(torch.equal(fixture['expected'], make_case(self.spec(family))['expected']))

    def test_unsupported_layout_and_alignment_fail_before_device_use(self):
        for spec in (self.spec('02'), self.spec(b_layout='column'),
                     dict(self.spec(), m=17), dict(self.spec(), k=0)):
            with self.assertRaises(ValueError):
                check_spec(spec)

    def test_numeric_rejection_retains_actual_error(self):
        fixture = make_case(self.spec())
        diagnostic = gemm_contract.validate_actual(fixture['expected']+1, fixture)
        self.assertFalse(diagnostic['passed'])
        self.assertGreater(diagnostic['max_absolute_error'], .9)

    def test_input_mutation_is_not_confused_with_workspace_mutation(self):
        fixture = make_case(self.spec())
        inputs = {key:value.clone() for key,value in fixture['inputs'].items()}
        inputs['a'][0,0] += 1
        with self.assertRaisesRegex(ValueError, 'input mutated'):
            gemm_contract.validate_actual(fixture['expected'], fixture, inputs)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required for actual ABI boundary')
    def test_actual_configuration_abi_workspace_and_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder/'config.cpp'
            source.write_text("""#include <stdint.h>
static uint32_t m_,n_; static bool bad_;
extern "C" uint32_t catlass_parallelism_family(){return 6;}
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t c){
 m_=m;n_=n;bad_=k==128;return (m%16||n%16||!c)?1:0;}
extern "C" uint32_t catlass_parallelism_factor(){return 4;}
extern "C" uint64_t catlass_parallelism_workspace(){return bad_?1:uint64_t(m_)*n_*4*4;}
""")
            binary = folder/'config.so'
            subprocess.run(['c++','-shared','-fPIC',str(source),'-o',str(binary)],check=True,
                           capture_output=True,text=True)
            library = ctypes.CDLL(str(binary))
            record = configure(library,'06','after',[64,64,16384],20)
            self.assertEqual(record['split_factor'],4)
            self.assertEqual(record['workspace_bytes'],65536)
            self.assertEqual(record['output_tiles'],1)
            for family,arm,shape,cores in [('07','after',[64,64,64],20),
                    ('06','before',[64,64,64],20),('06','after',[64,64,128],20),
                    ('06','after',[17,64,64],20),('06','after',[64,64,64],0)]:
                with self.assertRaises(ValueError):
                    configure(library,family,arm,shape,cores)


if __name__ == '__main__':
    unittest.main()
