from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import contextmanager
import torch,sys,unittest,tempfile,hashlib,weakref
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton/rope_backward'),str(Path(__file__).resolve().parents[1]/'scripts')]
import rope_backward as op
import rope_backward_contract as contract


class Kernel:
    fail=None;checks=staticmethod(lambda:None);launches=0
    def __init__(self,kind):self.kind=kind;self.fn=NS(__name__='mock_'+kind)
    def warmup(self,*args,**kw):
        Kernel.checks()
        if self.kind==Kernel.fail:raise RuntimeError('mock compile failure')
        return NS(hash=hashlib.sha256(self.kind.encode()).hexdigest(),name=self.fn.__name__,asm=dict(npubin=self.kind.encode(),ttir='mocktypes'))
    def __getitem__(self,grid):
        def launch(*args,**kw):
            Kernel.checks();Kernel.launches+=1
            if self.kind=='zero':dc,ds,*_=args;dc.zero_();ds.zero_();return
            dy,x,c,s,dx,dc,ds,total,S,H,nope,rd,half,br,block=args
            de,do=dy[...,nope::2].float(),dy[...,nope+1::2].float();e,o=x[...,::2].float(),x[...,1::2].float();c=c.unsqueeze(2);s=s.unsqueeze(2)
            dx[...,:nope].copy_(dy[...,:nope]);dx[...,nope::2].copy_(de*c+do*s);dx[...,nope+1::2].copy_(-de*s+do*c)
            pc,ps=de*e+do*o,-de*o+do*e
            if self.kind=='owned':dc.copy_(pc.sum(2));ds.copy_(ps.sum(2))
            else:
                for j in range(0,H,br):dc.add_(pc[:,:,j:j+br].sum(2));ds.add_(ps[:,:,j:j+br].sum(2))
        return launch


def kernels():return NS(rope_backward_zero=Kernel('zero'),rope_backward_atomic=Kernel('atomic'),rope_backward_owned=Kernel('owned'))


class RopeTests(unittest.TestCase):
    def spec(self):return dict(case_id='nonfrozenCPU',role='guard',seed=7912,parameters=dict(B=1,S=3,H=16,half=8,nope_dim=64))
    def tearDown(self):Kernel.fail=None;Kernel.checks=staticmethod(lambda:None)
    def test_public_quarantine_precedes_provider_backend_and_moves(self):
        spec=self.spec()
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')),patch.object(op,'_make_case',side_effect=AssertionError('oracle touched')):
            public=op.factory()
            for call in [lambda:op.make_case(spec),lambda:public.make_case(spec),lambda:public.prepare('after',spec,{},None),lambda:public.validate('after',None,{},spec),lambda:public.capture(lambda:None,None,None),lambda:public.compile_probe('before',spec,None)]:
                with self.assertRaisesRegex(RuntimeError,'quarantined'):call()
        with patch.object(op,'backend',return_value=('cpu',kernels())):
            probe=op.Adapter().compile_probe('before',spec,None,inspect_compile=True);self.assertEqual(len(probe.compile_only()),2)
            with self.assertRaises(RuntimeError):probe.run()
            probe.close()
    def test_integer_oracle_and_reject_before_backend(self):
        spec=self.spec();p=spec['parameters'];f=contract.make_case(spec);i={k:t.double()*16 for k,t in f['inputs'].items()};de,do=i['dy'][...,64::2],i['dy'][...,65::2];e,o=i['x_rope'][...,::2],i['x_rope'][...,1::2]
        self.assertTrue(torch.equal(f['expected'][1].double()*256,(de*e+do*o).sum(2)));self.assertTrue(torch.equal(f['expected'][2].double()*256,(-de*o+do*e).sum(2)));self.assertTrue(torch.equal(f['expected'][0][...,:64],f['inputs']['dy'][...,:64]))
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for key,value in [('H',24),('half',12),('nope_dim',65),('B',True),('S',0)]:
                with self.assertRaises(ValueError):op._MeasuredAdapter().compile_probe('after',dict(spec,parameters=dict(p,**{key:value})),None)
            for name,value in [('dy',f['inputs']['dy'].float()),('x_rope',f['inputs']['x_rope'].transpose(1,2)),('cos',f['inputs']['sin']),('sin',f['inputs']['sin'][:,:,:1].expand_as(f['inputs']['sin']))]:
                with self.assertRaises(ValueError):op._MeasuredAdapter().prepare('after',spec,dict(f,inputs=dict(f['inputs'],**{name:value})),None)
            for value in (.01,float('nan'),float('inf'),.5):
                bad=f['inputs']['cos'].clone();bad[0,0,0]=value
                with self.assertRaises(ValueError):op._MeasuredAdapter().prepare('before',spec,dict(f,inputs=dict(f['inputs'],cos=bad)),None)
    def test_probe_atomic_publication_reuse_diagnostics_and_close(self):
        spec=self.spec();f=contract.make_case(spec);a=op._MeasuredAdapter()
        with patch.object(op,'backend',return_value=('cpu',kernels())):
            probe=a.compile_probe('before',spec,None);before=Kernel.launches
            with self.assertRaises(RuntimeError):probe.run()
            self.assertEqual(Kernel.launches,before);Kernel.fail='atomic'
            with self.assertRaises(RuntimeError):probe.compile_only()
            self.assertEqual(probe.compilation_state(),[]);Kernel.fail=None;self.assertEqual(len(probe.compile_only()),2);probe.close();probe.close()
            for order in (('before','after'),('after','before')):
                items={arm:a.prepare(arm,spec,f,None) for arm in order};refs=[weakref.ref(t) for item in items.values() for t in item.outputs]
                for arm,item in items.items():
                    actual=item.run();self.assertTrue(a.validate(arm,actual,f,spec)['passed']);self.assertIs(item.run(),actual);self.assertFalse(a.validate(arm,actual[:2],f,spec)['passed'])
                    actual[1][0,0,0]=float('inf');diagnostic=a.validate(arm,actual,f,spec);self.assertFalse(diagnostic['passed']);self.assertEqual(diagnostic['outputs'][1]['mismatch_count'],1);item.run();a.validate(arm,actual,f,spec)
                del actual
                for item in items.values():item.close();item.close()
                self.assertEqual(a.state,{});self.assertTrue(all(t() is None for t in refs))
                with self.assertRaises(RuntimeError):items['after'].run()
    def test_real_generic_pair_cleanup_and_signed_zero(self):
        from optimization_pair import run_pair,fixture
        with tempfile.TemporaryDirectory() as tmp,patch.object(op,'backend',return_value=('cpu',kernels())):
            a=op._MeasuredAdapter();root=Path(tmp)
            for i in range(2):
                report=run_pair(a,self.spec(),root/str(i),root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(report['status'],'complete');self.assertEqual(a.state,{})
            _,receipt=fixture(a,self.spec(),root/'cache',torch);self.assertEqual(receipt['oracle_calls'],0)
            Kernel.fail='atomic';report=run_pair(a,self.spec(),root/'failed',root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(report['status'],'failed');self.assertEqual(a.state,{})
    def test_entire_capture_device_and_metric_group_restore(self):
        current=['caller2'];events=[]
        @contextmanager
        def device(name):
            old=current[0];current[0]=name
            try:yield
            finally:current[0]=old
        def capture(run,output,**kw):
            self.assertEqual(current[0],'npu:0');self.assertEqual(kw['metric_group'],'PipeUtilization');events.append('begin')
            for _ in range(10):run()
            self.assertEqual(current[0],'npu:0');events.append('end');return dict(status='complete',kernels=kw['expected_kernel'])
        item=NS(metadata={'prepared_device':'npu:0'},expected_kernels=['owned'])
        with patch.object(torch,'npu',NS(device=device),create=True),patch.dict(sys.modules,{'profile_ten':NS(capture_ten=capture)}):
            result=op._MeasuredAdapter().capture(lambda:self.assertEqual(current[0],'npu:0'),item,None);self.assertEqual(result['status'],'complete');self.assertEqual(events,['begin','end']);self.assertEqual(current[0],'caller2')
            def fail(*a,**kw):self.assertEqual(current[0],'npu:0');raise RuntimeError('captureerror')
            with patch.dict(sys.modules,{'profile_ten':NS(capture_ten=fail)}),self.assertRaises(RuntimeError):op._MeasuredAdapter().capture(lambda:None,item,None)
            self.assertEqual(current[0],'caller2')
    def test_real_run_pair_prepared_device_restore(self):
        from optimization_pair import run_pair
        current=[2]
        @contextmanager
        def device(d):
            old=current[0];current[0]=int(str(d).split(':')[-1])
            try:yield
            finally:current[0]=old
        original=torch.Tensor.to
        def transfer(t,d,*a,**kw):
            if d=='npu:0':self.assertEqual(current[0],0);return t
            return original(t,d,*a,**kw)
        Kernel.checks=staticmethod(lambda:self.assertEqual(current[0],0))
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch,'npu',NS(device=device),create=True),patch.object(torch.Tensor,'to',transfer),patch.object(op,'backend',return_value=('npu:0',kernels())):
            a=op._MeasuredAdapter();report=run_pair(a,self.spec(),Path(tmp)/'run',Path(tmp)/'cache',{},torch=torch,correctness_only=True);self.assertEqual(report['status'],'complete');self.assertEqual(a.state,{});self.assertEqual(current[0],2)


if __name__=='__main__':unittest.main()
