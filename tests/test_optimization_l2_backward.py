from pathlib import Path
import sys,tempfile,unittest,math
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import l2_backward as op
import l2_backward_contract as c


class L2Tests(unittest.TestCase):
    def test_given_state_oracle_zero_parallel_cancellation(self):
        f=c.make_case(dict(rows=3,D=517,zero_parallel_cancel=True,seed=911))
        y,r,dy=(f['inputs'][n] for n in ('y','rstd','dy'))
        self.assertTrue(torch.equal(y[0],torch.zeros_like(y[0])))
        self.assertTrue(torch.equal(dy[1],y[1]))
        for row in range(3):
            dot=math.fsum(float(a)*float(b) for a,b in zip(y[row],dy[row]))
            for col in (0,71,516):
                expected=float(r[row])*(float(dy[row,col])-float(y[row,col])*dot)
                self.assertAlmostEqual(float(f['expected'][row,col]),expected,delta=2e-6+2e-5*abs(expected))
        self.assertTrue(c.validate_actual(f['expected'],f)['passed'])

    def test_host_rejection_dtype_shape_values_alias_and_canary(self):
        for p in (dict(rows=17,D=64),dict(rows=0,D=517),dict(rows=2,D=4097)):
            with self.assertRaises(ValueError):op.Adapter().prepare('after',p,{},None)
        f=c.make_case(dict(rows=2,D=519,seed=912));p=dict(rows=2,D=519)
        invalid=[dict(f['inputs'],rstd=f['inputs']['rstd'].double()),dict(f['inputs'],dy=f['inputs']['y']),
                 dict(f['inputs'],y=f['inputs']['y'][:,:-1]),dict(f['inputs'],y=f['inputs']['y'].T.contiguous().T),
                 dict(f['inputs'],rstd=torch.zeros(2)),dict(f['inputs'],dy=torch.full((2,519),1.5)),
                 dict(f['inputs'],y=torch.full((2,519),float('nan')))]
        for inputs in invalid:
            with self.assertRaises(ValueError):op.Adapter().prepare('after',p,dict(f,inputs=inputs),None)
        for actual in (f['expected'].double(),f['expected'][:,:-1],[f['expected']]):
            with self.assertRaises(ValueError):c.validate_actual(actual,f)
        b=torch.full((80,),12345.);b[0]=0
        with self.assertRaises(ValueError):c.validate_actual(f['expected'],f,buffer=b)
        inputs={n:t.clone() for n,t in f['inputs'].items()};inputs['dy'][0,0]+=1
        with self.assertRaises(ValueError):c.validate_actual(f['expected'],f,inputs)

    def test_cache_roles_and_cli_host_guard_no_cache(self):
        import json,subprocess
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for spec in (dict(rows=2,D=521,seed=913),dict(parameters=dict(rows=1,D=523),seed=914,role='guard')):
                f,a=fixture(op.Adapter(),spec,root/'cache',torch);g,b=fixture(op.Adapter(),spec,root/'cache',torch)
                self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0));self.assertTrue(c.validate_actual(g['expected'],f)['passed'])
            case=root/'case.json';case.write_text(json.dumps(dict(rows=19,D=64)))
            result=subprocess.run([sys.executable,op.__file__,'--case',str(case),'--output',str(root/'out'),'--cache',str(root/'invalid-cache'),
                                   '--provenance',str(root/'missing.json')],cwd=tmp,capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'invalid-cache').exists())

    def test_compile_probe_no_launch_cached_runs_both_orders_fail_closed(self):
        from types import SimpleNamespace as NS
        import struct
        f=c.make_case(dict(rows=2,D=525,seed=915));data=bytearray(300);data[:6]=b'\x7fELF\x02\x01'
        struct.pack_into('<Q',data,40,64);struct.pack_into('<HHH',data,58,64,2,0)
        struct.pack_into('<IIQQQQIIQQ',data,64,0,3,0,0,192,7,0,0,1,0)
        struct.pack_into('<IIQQQQIIQQ',data,128,1,1,6,0,200,1,0,0,1,0);data[192:199]=b'\0.text\0';data[200]=1
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.compiles=0;self.launches=0;self.binary=bytes(data)
            def warmup(self,*a,**kw):self.compiles+=1;return NS(asm=dict(npubin=self.binary),hash='a'*64,name=self.fn.__name__)
            def __getitem__(self,grid):
                def run(*a,**kw):self.launches+=1;a[3].copy_(f['expected'])
                return run
        before,after=Kernel('before'),Kernel('after');full=torch.full;check=c.check_inputs
        def host_full(*a,**kw):kw.pop('device',None);return full(*a,**kw)
        modules=dict(torch_npu=NS(),l2_backward_kernels=NS(l2norm_bwd_kernel1=before,l2norm_bwd_factored=after))
        with patch.dict(sys.modules,modules),patch.object(torch.Tensor,'to',lambda self,*a,**kw:self.clone()),patch.object(torch,'full',host_full),patch.object(op,'check_inputs',lambda inputs,p,device:check(inputs,p,'cpu')):
            for arm,kernel in [('before',before),('after',after)]:
                probe=op.Adapter().compile_probe(arm,dict(rows=2,D=525),None);count=kernel.compiles;calls=kernel.launches
                with self.assertRaisesRegex(RuntimeError,'compile-only probe'):probe.run()
                self.assertEqual((kernel.compiles,kernel.launches),(count,calls))
                probe.compile_only();self.assertEqual(kernel.launches,calls)
                self.assertEqual(len(probe.compilation_state()),1);self.assertEqual(probe.metadata['device_kernel_calls'],0)
                with self.assertRaisesRegex(RuntimeError,'compile-only probe'):probe.run()
                self.assertEqual((kernel.compiles,kernel.launches),(count+1,calls))
            adapter=op.Adapter()
            for order in [('before','after'),('after','before')]:
                for i,arm in enumerate(order):
                    k=before if arm=='before' else after;count=k.compiles;calls=k.launches
                    item=adapter.prepare(arm,dict(rows=2,D=525),f,None);a=item.run();b=item.run();self.assertIs(a,b)
                    self.assertEqual((k.compiles-count,k.launches-calls),(1,2));self.assertEqual(item.metadata['device_kernel_calls'],1);result=adapter.validate(arm,a,f,dict(rows=2,D=525))
                    self.assertTrue(result['passed']);self.assertEqual('paired_exact_bits' in result,i==1)
            after.binary=b'invalid';calls=after.launches;item=adapter.prepare('after',dict(rows=2,D=525),f,None)
            with self.assertRaises(ValueError):item.run()
            self.assertEqual(after.launches,calls)


if __name__=='__main__':unittest.main()
