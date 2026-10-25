import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import compressor_projection as op
import compressor_projection_contract as contract


class ProjectionTests(unittest.TestCase):
    def test_integer_certificate_and_both_outputs(self):
        spec = dict(parameters=dict(M=3, signed_cancelled_distinct_weights=True), seed=112)
        f = contract.make_case(spec)
        for index, name in enumerate(('w0','w1')):
            x = (f['inputs']['x'].float()*128).to(torch.int64)
            w = (f['inputs'][name].float()*128).to(torch.int64)
            for row, col in ((0,1),(1,7),(2,21)):
                exact = (x[row]*w[col]).sum().item()/16384
                self.assertEqual(f['expected'][index][row,col].item(), exact)
        self.assertTrue(contract.validate_actual(f['expected'],f)['passed'])
        for wrong in (f['expected'][:1], [f['expected'][1],f['expected'][0]],
                      [f['expected'][0].double(), f['expected'][1]]):
            with self.assertRaises(ValueError):contract.validate_actual(wrong,f)

    def test_guards_before_backend_import(self):
        for change in (dict(M=-1),dict(M=True),dict(M=257),dict(M=1,BK=1024),dict(M=1,IN_DIM=5121),dict(M=1,OUT_DIM=513),dict(M=1,BN=32)):
            with self.assertRaises(ValueError):op.Adapter().prepare('after',change,{},None)
        f=contract.make_case(dict(M=0,seed=113))
        self.assertEqual([tuple(x.shape) for x in f['expected']],[(0,512),(0,512)])
        self.assertTrue(contract.validate_actual(f['expected'],f)['passed'])

    def test_layout_alias_mutation_canaries(self):
        f=contract.make_case(dict(M=2,seed=114))
        contract.check_inputs(f['inputs'],2,'cpu')
        for inputs in (dict(f['inputs'],w1=f['inputs']['w0']),
                       dict(f['inputs'],x=f['inputs']['x'].float()),
                       dict(f['inputs'],w0=f['inputs']['w0'].T.contiguous().T)):
            with self.assertRaises(ValueError):contract.check_inputs(inputs,2,'cpu')
        b=torch.full((80,),12345.)
        contract.validate_actual(f['expected'],f,f['inputs'],[b])
        b[0]=0
        with self.assertRaises(ValueError):contract.validate_actual(f['expected'],f,f['inputs'],[b])
        altered={k:v.clone() for k,v in f['inputs'].items()};altered['x'][0,0]+=1
        with self.assertRaises(ValueError):contract.validate_actual(f['expected'],f,altered)
        with self.assertRaises(ValueError):contract.validate_actual([f['expected'][0]]*2,f)

    def test_caller_numeric_domain_before_backend_import(self):
        import builtins
        fixture = contract.make_case(dict(M=2, seed=118))
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in ('torch_npu', 'compressor_projection_kernels', 'compressor_projection_utils'):
                raise AssertionError('backend imported before invalid fixture rejection')
            return original_import(name, *args, **kwargs)
        for name in ('x', 'w0', 'w1'):
            for bad in (float('nan'), float('inf'), -float('inf'), .25, -.25, 1/256, -1/256):
                with self.subTest(name=name, bad=bad):
                    inputs = dict(fixture['inputs']); inputs[name] = inputs[name].clone()
                    inputs[name][0, 0] = bad
                    with patch.object(builtins, '__import__', guarded_import), self.assertRaises(ValueError):
                        op.Adapter().prepare('after', dict(M=2), dict(fixture, inputs=inputs), None)
        for name in ('x', 'w0', 'w1'):
            fixture['inputs'][name][0, :3] = torch.tensor([-.125, 0., .125], dtype=torch.bfloat16)
        op.check_numeric_domain(fixture['inputs'])
        op.check_numeric_domain(contract.make_case(dict(M=0, seed=119))['inputs'])

    def test_repeated_adapter_both_orders_compile_once_and_reuse_outputs(self):
        from types import SimpleNamespace as NS
        import struct
        data=bytearray(300);data[:6]=b'\x7fELF\x02\x01'
        struct.pack_into('<Q',data,40,64);struct.pack_into('<HHH',data,58,64,2,0)
        struct.pack_into('<IIQQQQIIQQ',data,64,0,3,0,0,192,7,0,0,1,0)
        struct.pack_into('<IIQQQQIIQQ',data,128,1,1,6,0,200,1,0,0,1,0)
        data[192:199]=b'\0.text\0';data[200]=1
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.compiles=0;self.launches=0
            def warmup(self,*args,**kwargs):
                self.compiles+=1
                return NS(asm=dict(npubin=bytes(data)),hash='a'*64,name=self.fn.__name__)
            def __getitem__(self,grid):
                def launch(*args,**kwargs):self.launches+=1
                return launch
        k0,k1=Kernel('single'),Kernel('dual')
        modules=dict(torch_npu=NS(),compressor_projection_kernels=NS(_proj_kernel=k0,_proj2_kernel=k1),
                     compressor_projection_utils=NS(get_npu_aicore_num=lambda:20))
        full=torch.full
        def cpu_full(*args,**kwargs):kwargs.pop('device',None);return full(*args,**kwargs)
        f=contract.make_case(dict(M=1,seed=117));adapter=op.Adapter()
        with patch.dict(sys.modules,modules),patch.object(op,'check_inputs'),patch.object(torch.Tensor,'to',lambda self,*a,**k:self.clone()),patch.object(torch,'full',cpu_full),patch.object(op,'check_numeric_domain',wraps=op.check_numeric_domain) as numeric_guard:
            for order in [('before','after'),('after','before')]:
                for arm in order:
                    guard_calls=numeric_guard.call_count
                    item=adapter.prepare(arm,dict(M=1),f,None)
                    self.assertEqual(numeric_guard.call_count,guard_calls+1)
                    kernel=k0 if arm=='before' else k1
                    before=(kernel.compiles,kernel.launches)
                    a=item.run();b=item.run()
                    self.assertEqual(numeric_guard.call_count,guard_calls+1)
                    self.assertIs(a,b)
                    self.assertEqual(kernel.compiles-before[0],2 if arm=='before' else 1)
                    self.assertEqual(kernel.launches-before[1],4 if arm=='before' else 2)
                    self.assertEqual(len(item.compilation_state()),2 if arm=='before' else 1)

    def test_external_cwd_cli_rejects_before_cache(self):
        import subprocess, json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'case.json';case.write_text(json.dumps(dict(M=2,BK=1024)))
            result=subprocess.run([sys.executable,op.__file__,'--case',str(case),'--output',str(root/'out'),
                                   '--cache',str(root/'cache'),'--provenance',str(root/'unused.json')],
                                  cwd=tmp,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'cache').exists());self.assertFalse((root/'out').exists())

    def test_generic_cache_flat_nested_role_and_reuse(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            for spec in (dict(M=1,seed=115),dict(parameters=dict(M=2),seed=116,role='guard')):
                first,meta=fixture(op.Adapter(),spec,Path(tmp),torch)
                second,again=fixture(op.Adapter(),spec,Path(tmp),torch)
                self.assertEqual(meta['oracle_calls'],1);self.assertEqual(again['oracle_calls'],0)
                self.assertTrue(contract.validate_actual(second['expected'],first)['passed'])


if __name__=='__main__':unittest.main()
