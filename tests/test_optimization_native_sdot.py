import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/optimization-library/ascendc/sdot'))
import sdot
import sdot_contract as contract
import sdot_compile as compiler


class Tests(unittest.TestCase):
    def test_tiling_ownership_and_exact_oracle(self):
        for n in (1,7,31,79,11779):
            t=struct.unpack('<83I',contract.tiling(n));self.assertEqual(t[:3],(n,min(n,8),0));self.assertEqual(len(contract.tiling(n)),332)
            ownership=[]
            for o,c in zip(t[3:11],t[43:51]):ownership.extend(range(o,o+c))
            self.assertEqual(ownership,list(range(n)));self.assertEqual(t[11:43],(0,)*32);self.assertEqual(t[51:],(0,)*32)
            f=contract.make_case({'n':n,'seed':33219});contract.check_fixture({'n':n},f)
            qx=(f['x']*256).to(torch.int64);qy=(f['y']*256).to(torch.int64)
            self.assertEqual(float(f['expected'][0]),int((qx*qy).sum())/65536)
        f=contract.make_case({'n':1,'signed_zero':True});self.assertTrue(torch.signbit(f['x'][0]));self.assertEqual(float(f['expected'][0]),0)

    def test_strict_domain_before_backend(self):
        for p in ({'n':0},{'n':True},{'n':1048577},{'n':4,'incx':-1},{'n':4,'isConj':1}):
            with self.assertRaises(ValueError):sdot.Adapter(compile_inspection=True).compile_probe('before',p,None)
        spec={'n':5};f=contract.make_case(spec)
        for x in (f['x'].double(),f['x'].clone(),torch.full((5,),float('nan')),torch.ones((5,))):
            with self.assertRaises(ValueError):contract.check_fixture(spec,dict(f,x=x))
        with self.assertRaises(ValueError):contract.check_fixture(spec,dict(f,y=f['x']))
        with self.assertRaises(ValueError):sdot.Adapter().prepare('x',spec,f,None)

    def test_atomic_cache_probe_and_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);obj=d/'device.o';obj.write_bytes(b'selected-native-device');binary=d/'host.so';binary.write_bytes(b'host')
            b=dict(object=obj,binary=binary,resources={});adapter=sdot.Adapter(d,compile_inspection=True)
            previous=torch.get_default_dtype()
            try:
                for dtype in (torch.float64,torch.float16):
                    torch.set_default_dtype(dtype)
                    f=contract.make_case({'n':11,'seed':33220});self.assertEqual(f['x'].dtype,torch.float32);self.assertEqual(f['expected'].dtype,torch.float32)
                    for arm in ('after','before'):
                        probe=adapter.compile_probe(arm,{'n':11},d)
                        with patch.object(sdot,'build',side_effect=ValueError('failed')) as build:
                            with self.assertRaises(sdot.StudyUnqualified):probe.run()
                            self.assertEqual(build.call_count,0)
                            if arm not in adapter.builds:
                                with self.assertRaises(ValueError):probe.compile_only()
                                self.assertEqual(probe.compilation_state(),[])
                        with patch.object(sdot,'build',return_value=b) as build:
                            probe.compile_only();calls=build.call_count;probe.compile_only();self.assertEqual(build.call_count,calls)
                            self.assertEqual(len(probe.compilation_state()),1)
                        with self.assertRaises(sdot.StudyUnqualified):probe.run()
            finally:torch.set_default_dtype(previous)
        with self.assertRaises(ValueError):compiler.derivative('missing source body')
        with self.assertRaises(ValueError):compiler.sections(b'not an ELF')
        self.assertEqual(compiler.registered_selectors({'.ascend.meta._Z4testPh':{},'.ascend.meta.other':{}},['test']),['_Z4testPh'])
        with self.assertRaises(RuntimeError):compiler.registered_selectors({'.ascend.meta.test':{},'.ascend.meta._Z4testPh':{}},['test'])

    def test_default_workflow_rejects_before_provider_or_build(self):
        with patch.object(contract,'make_case',side_effect=AssertionError('provider called')),patch.object(sdot,'build',side_effect=AssertionError('compiler called')):
            adapter=sdot.Adapter()
            calls=[lambda:sdot.factory(),lambda:adapter.make_case({'n':31}),lambda:adapter.prepare('before',{},None,None),
                lambda:adapter.validate('before',None,None,{}),lambda:adapter.capture(None,None,None),lambda:adapter.compile_probe('before',{'n':31},None)]
            for call in calls:
                with self.assertRaises(sdot.StudyUnqualified):call()

    def test_cli_external_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);case=d/'case.json';case.write_text('{"n":0}\n')
            args=[sys.executable,str(Path(sdot.__file__)),'--case',str(case),'--output',str(d/'out'),'--cache',str(d/'cache'),'--provenance',str(d/'absent')]
            p=subprocess.run(args,cwd=d,capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('native SDOT runtime study failed',p.stderr);self.assertFalse((d/'cache').exists())
            case.write_text('{"n":11}\\n');p=subprocess.run(args,cwd=d,capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('native SDOT runtime study failed',p.stderr)


if __name__=='__main__':unittest.main()
