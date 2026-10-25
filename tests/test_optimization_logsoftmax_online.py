import contextlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import logsoftmax_online as op
import logsoftmax_online_contract as contract


class Kernel:
    def __init__(self,name):self.fn=SimpleNamespace(__name__=name);self.calls=0;self.launches=0;self.fail=False
    def warmup(self,*args,**kwargs):
        self.calls+=1
        if self.fail:raise ValueError('compile failed')
        return SimpleNamespace(hash='a'*64,asm={'npubin':b'\x7fELFmock'})
    def __getitem__(self,grid):
        def run(*a,**kw):self.launches+=1;return self.warmup(*a,**kw)
        return run


class Tests(unittest.TestCase):
    def test_oracle_strides_and_diagnostics(self):
        for params in [dict(rows=4,cols=19),dict(rows=2,cols=33,row_padding=7,extreme_finite=True),dict(rows=3,cols=13,uniform_row=True)]:
            spec={'parameters':params,'seed':49001};f=contract.make_case(spec);contract.check_fixture(spec,f)
            x=f['inputs']['x'];self.assertEqual(x.stride(),(params['cols']+params.get('row_padding',0),1))
            self.assertTrue(contract.diagnostics(torch.log_softmax(x,1),f['expected'])['passed'])
            bad=f['expected'].float();bad[0,0]+=1
            d=contract.diagnostics(bad,f['expected']);self.assertFalse(d['passed']);self.assertEqual(d['first_mismatches'][0]['coordinates'],[0,0])

    def test_reject_before_backend(self):
        for p in [dict(rows=0,cols=3),dict(rows=3,cols=True),dict(rows=1,cols=4,row_padding=-1)]:
            with self.assertRaises(ValueError):op.Adapter().compile_probe('before',p,None)
        spec={'rows':2,'cols':9};f=contract.make_case(spec)
        for x in [f['inputs']['x'].double(),f['inputs']['x'].t(),torch.full((2,9),float('nan')),torch.full((2,9),81.)]:
            with self.assertRaises(ValueError):op.Adapter().prepare('before',spec,dict(inputs={'x':x},expected=f['expected']),None)
        with self.assertRaises(ValueError):op.Adapter().prepare('before',spec,dict(inputs={'x':f['inputs']['x'],'extra':None},expected=f['expected']),None)
        with self.assertRaises(ValueError):op.Adapter().prepare('bad',spec,f,None)

    def test_mock_probe_atomic_and_repeated_adapter(self):
        driver=SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda _:dict(num_vectorcore=48))))))
        before,after=Kernel('source'),Kernel('online')
        modules={'triton':driver,'torch_npu':SimpleNamespace(),'logsoftmax_online_kernels':SimpleNamespace(logsoftmax_source=before,logsoftmax_online=after)}
        with patch.dict(sys.modules,modules):
            adapter=op.Adapter();spec={'rows':2,'cols':17,'row_padding':7};f=contract.make_case(spec);x=f['inputs']['x']
            for dtype in (torch.float64,torch.float16):
                previous=torch.get_default_dtype()
                try:
                    torch.set_default_dtype(dtype)
                    for order in [('before','after'),('after','before')]:
                        for arm in order:
                            k=before if arm=='before' else after
                            probe=adapter._prepare(arm,spec,x,True)
                            n=k.calls
                            with self.assertRaises(RuntimeError):probe.run()
                            self.assertEqual(k.calls,n);self.assertEqual(k.launches,0)
                            k.fail=True
                            with self.assertRaises(ValueError):probe.compile_only()
                            self.assertEqual(probe.compilation_state(),[])
                            k.fail=False;probe.compile_only();n=k.calls;probe.compile_only();self.assertEqual(k.calls,n)
                            self.assertEqual(probe.out.dtype,torch.float32);self.assertEqual(probe.metadata['input_stride'],[24,1])
                            item=adapter._prepare(arm,spec,x,False);item.out.copy_(f['expected'])
                            self.assertTrue(adapter.validate(arm,item.out,f,spec)['passed'])
                        self.assertEqual(set(adapter.items),{'before','after'})
                        self.assertEqual(set(adapter.numeric),{'before','after'})
                    fresh=adapter._prepare('before',spec,x,False);self.assertEqual(adapter.numeric,{})
                    fresh.out.copy_(f['expected']);fresh.store[0]=0
                    self.assertFalse(adapter.validate('before',fresh.out,f,spec)['passed'])
                finally:torch.set_default_dtype(previous)

    def test_external_cli_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);case=d/'case.json';case.write_text('{"rows":0,"cols":17}\n')
            args=[sys.executable,str(Path(op.__file__)), '--case',str(case),'--output',str(d/'out'),'--cache',str(d/'cache'),'--provenance',str(d/'absent')]
            result=subprocess.run(args,cwd=d,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('invalid rows',result.stderr);self.assertFalse((d/'cache').exists())
            case.write_text('{"rows":2,"cols":17}\\n')
            result=subprocess.run(args,cwd=d,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('JSONDecodeError',result.stderr)


if __name__=='__main__':unittest.main()
