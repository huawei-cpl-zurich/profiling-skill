import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'examples/optimization-library/ascendc/sasum'
sys.path[:0]=[str(HERE),str(ROOT/'scripts'),str(ROOT/'tests')]
import sasum
import sasum_contract as provider
import sasum_compile as compiler
import test_optimization_profile_evidence as evidence_tests
import optimization_profile_evidence as evidence


class SasumTests(unittest.TestCase):
    def test_source_partition_and_independent_nonfrozen_expected(self):
        for n in (1,7,8,23,67,1059,164003):
            raw=struct.unpack('<Ii81I',provider.tiling(n,2));offset=raw[3:43];count=raw[43:83]
            owner=[j for a,c in zip(offset[:8],count[:8]) for j in range(a,a+c)]
            self.assertEqual(owner,list(range(n)))
            self.assertTrue(all(a%8==0 for a,c in zip(offset,count) if c))
        old=torch.get_default_dtype()
        try:
            for dt in (torch.float64,torch.float16):
                torch.set_default_dtype(dt);f=provider.make_case(dict(n=37,incx=2,seed=214))
                self.assertEqual(f['x'].dtype,torch.float32);self.assertEqual(f['x'].stride(),(2,))
                self.assertEqual(float(f['expected'][0]),sum(abs(float(v)) for v in f['x']))
                provider.check_fixture(dict(n=37,incx=2),f)
        finally:torch.set_default_dtype(old)

    def test_domain_alias_holes_and_default_strictness(self):
        for s in ({'n':True},{'n':0},{'n':9,'incx':True},{'n':9,'incx':3}):
            with self.assertRaises(ValueError):sasum.Adapter(compile_inspection=True).compile_probe('before',s,'.')
        s=dict(n=19,incx=2);f=provider.make_case(s)
        for key,value in [('x',f['x'].double()),('expected',f['x'][:1])]:
            with self.assertRaises(ValueError):provider.check_fixture(s,dict(f,**{key:value}))
        provider.full(f['x'])[33]=0
        with self.assertRaisesRegex(ValueError,'hole'):provider.check_fixture(s,f)

    def test_probe_zero_launch_atomic_failure_retry_and_repeated_adapters(self):
        with tempfile.TemporaryDirectory() as t:
            obj=Path(t)/'device-vec.o';obj.write_bytes(b'device-ELF');(Path(t)/'device-cube.o').write_bytes(b'cube-ELF');b=dict(object=obj,resources={},binary=obj)
            a=sasum.Adapter(compile_inspection=True);p=a.compile_probe('after',dict(n=11),t)
            with patch.object(sasum,'build',side_effect=[RuntimeError('compile failed'),b]) as build,patch.object(sasum,'sections',return_value={'.text':{'bytes':10},'.ascend.meta.actual':{}}):
                with self.assertRaisesRegex(RuntimeError,'cannot execute'):p.run()
                self.assertEqual(build.call_count,0)
                with self.assertRaisesRegex(RuntimeError,'compile failed'):p.compile_only()
                self.assertEqual(a.builds,{})
                self.assertEqual(len(p.compile_only()),2);p.compile_only();self.assertEqual(build.call_count,2)
                with self.assertRaisesRegex(RuntimeError,'cannot execute'):p.run()
                self.assertEqual(p.compilation_state(),sasum.binary_state(b));self.assertEqual(sasum.Adapter().builds,{})

    def test_actual_collector_normalization_accepts_device_identity_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);obj=root/'selected-aiv.o';obj.write_bytes(b'selected device ELF');(root/'selected-cube.o').write_bytes(b'actual cube component')
            path,report=evidence_tests.PairTests().fixture(root)
            for row in report['arms'].values():
                row['compilation_state_before_capture']=sasum.binary_state(dict(object=obj))
                row['compilation_state_after_capture']=sasum.binary_state(dict(object=obj))
            path.write_text(json.dumps(report)+'\n');out=evidence.collect_pair(path)
            self.assertEqual(out['status'],'complete');self.assertEqual(out['arms']['after']['event_latency_ms']['count'],10)
            report['arms']['after']['compilation_state_after_capture'][0]['binaries']['device_aic_elf']='f'*64
            path.write_text(json.dumps(report)+'\n')
            with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path)

    def test_default_numeric_workflows_reject_before_provider_or_build(self):
        with patch.object(sasum,'build',side_effect=AssertionError('compiler forbidden')),patch.object(provider,'make_case',side_effect=AssertionError('oracle forbidden')):
            a=sasum.Adapter()
            for fn,args in ((sasum.factory,()),(a.make_case,({'n':13},)),(a.prepare,('before',{},None,'.')),(a.compile_probe,('before',{'n':13},'.')),(a.validate,()),(a.capture,())):
                with self.assertRaises(sasum.StudyUnqualified):fn(*args)
            self.assertEqual(a.builds,{})

    def test_registration_suffix_and_external_cwd_cli_strict_json(self):
        name='_Z12sasum_kernelPhS_S_S__mix_aiv'
        self.assertEqual(compiler.registered_selectors({'.ascend.meta.'+name:{}},['sasum_kernel']),[name])
        with self.assertRaises(RuntimeError):compiler.registered_selectors({'.ascend.meta.sasum_kernel':{},'.ascend.meta.sasum_kernel_mix_aiv':{}},['sasum_kernel'])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);case=p/'case.json';case.write_text('{"n":false}\n');prov=p/'provenance.json';prov.write_text('{}\n')
            out=subprocess.run([sys.executable,str(HERE/'sasum.py'),'--case',str(case),'--output',str(p/'out'),'--cache',str(p/'cache'),'--provenance',str(prov)],cwd=t,capture_output=True,text=True)
            self.assertNotEqual(out.returncode,0);self.assertIn('ACL507015',out.stderr);self.assertFalse((p/'cache').exists())


if __name__=='__main__':unittest.main()
