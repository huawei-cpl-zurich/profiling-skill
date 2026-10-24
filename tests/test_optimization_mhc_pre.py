import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'examples/optimization-library/triton'
sys.path[:0] = [str(DIRECTORY), str(ROOT / 'scripts')]
from mhc_pre_contract import make_case, validate_inputs, check_outputs
from mhc_pre import Adapter
from optimization_pair import fixture


def spec(**changes):
    return dict(seed=47, parameters=dict(BS=7, D=128, tile_D=128, **changes))


class MHCPreTests(unittest.TestCase):
    def test_independent_scalar_oracle(self):
        value = make_case(spec())
        h, x, dy = (value['inputs'][k] for k in ('H', 'X', 'dY'))
        dh, dx = value['expected']
        for b in range(7):
            for n in range(4):
                dot = sum(float(x[b, n, j]) * float(dy[b, j]) for j in range(128))
                self.assertEqual(float(dh[b,n]), float(torch.tensor(dot, dtype=torch.float32)))
                self.assertTrue(torch.equal(dx[b,n], dy[b] * h[b,n]))

    def test_zero_and_cancellation(self):
        case = dict(seed=83, BS=3, D=16, tile_D=16, zero_cancelled_and_signedzero=True)
        value = make_case(case)
        self.assertTrue(bool((value['expected'][0] == 0).all()))
        result = check_outputs(value['expected'], value['expected'])
        self.assertTrue(result['passed'])
        self.assertFalse(result['signed_zero_bitwise_guarantee'])

    def test_output_contract_negative_checks(self):
        expected = make_case(spec())['expected']
        for bad in (expected[:1], expected + expected, list(expected),
                    (expected[0].double(), expected[1]),
                    (expected[0].flatten(), expected[1]),
                    (expected[0] + .1, expected[1])):
            self.assertFalse(check_outputs(bad, expected)['passed'])

    def test_bad_inputs_reject_before_backend(self):
        for key, replacement in [('H', torch.zeros(7,4,dtype=torch.float64)),
                                 ('dY', torch.full((7,128), float('nan'))),
                                 ('X', torch.zeros(7,4,129))]:
            value = make_case(spec()); value['inputs'][key] = replacement
            with self.assertRaises(ValueError):
                Adapter().prepare('after', spec(), value, Path('.'))
        value = make_case(spec()); value['inputs']['dY'] = torch.zeros(8,128)[1:]
        with self.assertRaises(ValueError):
            validate_inputs(value['inputs'], spec())

    def test_cpu_cache_and_empty_before_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/'cache'
            a, first = fixture(Adapter(), spec(), cache, torch)
            b, second = fixture(Adapter(), spec(), cache, torch)
            self.assertEqual((first['oracle_calls'],second['oracle_calls']), (1,0))
            self.assertTrue(torch.equal(a['expected'][0],b['expected'][0]))
            with self.assertRaises(ValueError):
                fixture(Adapter(), dict(seed=9,BS=0,D=16,tile_D=16), Path(directory)/'empty', torch)
            self.assertFalse((Path(directory)/'empty').exists())

    def test_prepare_reuse_compile_and_exact_fallback(self):
        launches = []
        artifact = SimpleNamespace(hash='a'*64, asm={'npubin':b'code'})
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*args,**kwargs):return artifact
            def __getitem__(self,grid):
                def launch(*args,**flags):
                    launches.append((self.fn.__name__, grid, flags))
                    return artifact
                return launch
        original_empty = torch.empty
        def cpu_empty(*args,**kwargs):
            kwargs.pop('device',None)
            return original_empty(*args,**kwargs)
        modules = dict(torch_npu=SimpleNamespace(), triton=SimpleNamespace(__version__='fake'),
            mhc_pre_kernel=SimpleNamespace(hc_pre_bmm_bwd_fused_kernel=Kernel('bwd'),zero_gradient_kernel=Kernel('zero')))
        with patch.dict(sys.modules,modules), patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True), patch.object(torch,'empty',cpu_empty):
            a=Adapter()
            item=a.prepare('before',spec(),make_case(spec()),Path('.'))
            self.assertEqual(len(item.compile_only()),2); self.assertEqual(launches,[])
            item.run(); self.assertEqual([x[0] for x in launches],['zero','bwd'])
            newer=a.prepare('after',spec(),make_case(spec()),Path('.'))
            newer.run(); self.assertEqual([x[0] for x in launches],['zero','bwd','bwd'])
            self.assertTrue(newer.metadata['exclusive']); self.assertIs(a.prepared['after'],newer)
            fallback=dict(seed=71,BS=9,D=256,tile_D=64)
            before=a.prepare('before',fallback,make_case(fallback),Path('.'))
            after=a.prepare('after',fallback,make_case(fallback),Path('.'))
            self.assertEqual(before.metadata['flags'],after.metadata['flags'])
            after.run(); self.assertEqual([x[0] for x in launches[-2:]],['zero','bwd'])
            self.assertFalse(after.metadata['exclusive'])
            self.assertNotEqual(item.output_storages[0].data_ptr(),newer.output_storages[0].data_ptr())
            self.assertEqual(newer.output_storages[0][[0,-1]].tolist(),[-701.,-701.])

    def test_external_cwd_cli_flat_nested_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); case=folder/'case.json'
            command=[sys.executable,str(DIRECTORY/'mhc_pre_cli.py'),'--case',str(case),
                     '--output',str(folder/'out'),'--cache',str(folder/'cache'),
                     '--provenance',str(folder/'provenance.json'),'--validate-only']
            for value in (spec(),dict(seed=47,**spec()['parameters'])):
                case.write_text(json.dumps(value))
                result=subprocess.run(command,cwd=folder,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            case.write_text(json.dumps(dict(seed=47,BS=2,D=24,tile_D=8)))
            result=subprocess.run(command,cwd=folder,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((folder/'out').exists()); self.assertFalse((folder/'cache').exists())

    def test_standalone_import(self):
        s=importlib.util.spec_from_file_location('standalone_mhc',DIRECTORY/'mhc_pre.py')
        m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
        with self.assertRaises(ValueError):
            m.factory().prepare('invalid',spec(),make_case(spec()),Path('.'))


if __name__ == '__main__':unittest.main()
