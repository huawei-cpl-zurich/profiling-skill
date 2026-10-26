from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import contextmanager
import math,sys,unittest,tempfile,weakref,json,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'examples/optimization-library/triton/gdn_local_dv'),str(ROOT/'scripts'),str(ROOT/'tests')]
import gdn_local_dv as op
import gdn_local_dv_contract as provider
SPEC=dict(B=2,T=2,H=4,K=64,V=64,BT=16,dense=True,USE_G=True,USE_G_GAMMA=False,g_mode='zigzag',scale=.5,seed=17601)
IR='%d = tt.dot %a, %b, %c : tensor<64x128xf16> * tensor<128x128xf16> -> tensor<64x128xf32>\n'


class Kernel:
    failure=None;seen=[];check=staticmethod(lambda:None)
    def __init__(self,name,expected,dots):self.name=name;self.expected=expected;self.dots=dots;self.fn=NS(__name__='mock_'+name)
    def warmup(self,*a,**kw):
        Kernel.check()
        if self.failure==self.name:raise RuntimeError('mock compile failure')
        return NS(hash=hashlib.sha256(self.name.encode()).hexdigest(),asm=dict(npubin=b'\x7fELF'+self.name.encode(),ttir=IR*self.dots))
    def __getitem__(self,grid):
        def run(*args,**kw):
            Kernel.check();Kernel.seen.append(self.name)
            if self.name=='scores':args[5].zero_()
            elif self.name=='fused':args[5].copy_(self.expected)
            else:args[2].copy_(self.expected)
        return run


def kernels(expected):return NS(**{'gdn_local_dv_'+name:Kernel(name,expected,n) for name,n in [('scores',1),('fused',2),('consume',1)]})


class LocalDVTests(unittest.TestCase):
    def tearDown(self):Kernel.failure=None;Kernel.check=staticmethod(lambda:None);Kernel.seen=[]
    def test_typed_oracle_upper_diagonal_positive_gate_and_domain(self):
        old=torch.get_default_dtype()
        try:
            for dtype in (torch.float64,torch.float16):
                torch.set_default_dtype(dtype);f=provider.make_case(SPEC);z=f['inputs'];head=3;row=0;col=2
                fp32=lambda v:float(torch.tensor(v,dtype=torch.float32,device='cpu'))
                total=0.
                for j in range(row,SPEC['T']):
                    dot=fp32(float(z['k'][1,row,head].double()@z['q'][1,j,head].double()));diff=fp32(float(z['g'][1,head,j])-float(z['g'][1,head,row]));self.assertGreaterEqual(diff,0)
                    gate=fp32(math.exp(diff));a=float(torch.tensor(fp32(fp32(dot*gate)*.5),dtype=torch.float16,device='cpu'));total+=a*float(z['do'][1,j,head,col])
                expected=torch.tensor(fp32(total),dtype=torch.float16,device='cpu');self.assertEqual(f['expected'][1,row,head,col],expected)
        finally:torch.set_default_dtype(old)
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for change in ({'B':3},{'H':2},{'K':32},{'V':32},{'dense':False},{'T':0},{'T':True},{'BT':64},{'scale':1.},{'USE_G':False},{'USE_G_GAMMA':True}):
                with self.assertRaises(ValueError):op.Adapter().compile_probe('after',dict(SPEC,**change),None)
            for key in ('g','q','layout','alias','halo'):
                f=provider.make_case(SPEC)
                if key=='g':f['inputs']['g'][0,0,0]=20
                elif key=='q':f['inputs']['q'][0,0,0,0]=.5
                elif key=='layout':f['inputs']['q']=f['inputs']['q'].transpose(0,1)
                elif key=='halo':provider.backing(f['inputs']['q'])[0]=0
                else:f['inputs']['k']=f['inputs']['q']
                with self.assertRaises(ValueError):op.Adapter().prepare('before',SPEC,f,None)

    def test_probe_atomic_compile_reuse_validation_and_cleanup(self):
        f=provider.make_case(SPEC);adapter=op.Adapter()
        with patch.object(op,'backend',return_value=('cpu',kernels(f['expected']))):
            probe=adapter.compile_probe('before',SPEC,None)
            with self.assertRaises(RuntimeError):probe.run()
            Kernel.failure='consume'
            with self.assertRaises(RuntimeError):probe.compile_only()
            self.assertEqual(probe.compilation_state(),[]);self.assertEqual(Kernel.seen,[]);Kernel.failure=None;self.assertEqual(len(probe.compile_only()),2);probe.close();probe.close()
            for order in [('before','after'),('after','before')]:
                items={arm:adapter.prepare(arm,SPEC,f,None) for arm in order}
                for arm,item in items.items():
                    value=item.run();self.assertTrue(adapter.validate(arm,value,f,SPEC)['passed']);self.assertIs(value,item.run())
                self.assertTrue(adapter.validate(order[-1],value,f,SPEC)['paired_all_bits_secondary']);value.flatten()[0]=float('inf');diag=adapter.validate(order[-1],value,f,SPEC);self.assertFalse(diag['passed']);self.assertEqual(diag['nonfinite_count'],1)
                items[order[-1]].run();items[order[-1]].state['buffers'][0][0]=0;self.assertFalse(adapter.validate(order[-1],value,f,SPEC)['passed']);del value
                refs=[weakref.ref(item.state['out'])for item in items.values()]
                for item in items.values():item.close();item.close()
                self.assertEqual(adapter.items,{});self.assertTrue(all(ref() is None for ref in refs))
                with self.assertRaises(RuntimeError):items['after'].run()
            self.assertIn('consume',Kernel.seen)
        for text,n in [(IR.replace('xf16','xf32'),1),('',1),(IR,2)]:
            with self.assertRaises(ValueError):op.typed_policy(NS(asm={'ttir':text}),n)
        padded=IR+'%x = tt.load %p {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}\n'
        with self.assertRaisesRegex(ValueError,'zero-padded'):op.typed_policy(NS(asm={'ttir':padded,'ttadapter':''}),1)
        self.assertEqual(op.typed_policy(NS(asm={'ttir':padded,'ttadapter':'%z = arith.constant 0.000000e+00 : f16; linalg.fill'}),1)['dot_count'],1)

    def test_actual_pair_cleanup_and_foreign_device_restore(self):
        from optimization_pair import run_pair
        f=provider.make_case(SPEC);current=[2]
        @contextmanager
        def device(d):
            old=current[0];current[0]=int(str(d).split(':')[-1])
            try:yield
            finally:current[0]=old
        real_to=torch.Tensor.to
        def transfer(t,d,*a,**kw):
            if d=='npu:0':self.assertEqual(current[0],0);return t
            return real_to(t,d,*a,**kw)
        Kernel.check=staticmethod(lambda:self.assertEqual(current[0],0))
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch,'npu',NS(device=device),create=True),patch.object(torch.Tensor,'to',transfer),patch.object(op,'backend',return_value=('npu:0',kernels(f['expected']))):
            a=op.Adapter();root=Path(tmp)
            for i,c in enumerate((1,2)):
                current[0]=c;report=run_pair(a,SPEC,root/str(i),root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(report['status'],'complete');self.assertEqual(a.items,{});self.assertEqual(current[0],c);self.assertTrue(all(x['cleanup']['status']=='complete'for x in report['arms'].values()))
            Kernel.failure='consume';report=run_pair(a,SPEC,root/'error',root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(report['status'],'failed');self.assertEqual(a.items,{});self.assertEqual(current[0],2)

    def test_entire_capture_device_metric_and_error_restoration(self):
        import profile_ten
        current=[2];events=[]
        def check(n):self.assertEqual(current[0],0);events.append(n)
        @contextmanager
        def device(d):
            old=current[0];current[0]=int(str(d).split(':')[-1])
            try:yield
            finally:current[0]=old
        @contextmanager
        def profile(**kw):
            check('begin');self.assertEqual(kw['experimental_config']['aic_metrics'],'pipes')
            try:yield NS(step=lambda:check('step'))
            finally:check('end')
        class Event:
            def __init__(self,**kw):check('event')
            def record(self):check('record')
            def elapsed_time(self,end):check('elapsed');return 1.
        profiler=NS(ProfilerLevel=NS(Level1=1),AiCMetrics=NS(PipeUtilization='pipes'),ProfilerActivity=NS(CPU=1,NPU=2),profile=profile,schedule=lambda **kw:kw,_ExperimentalConfig=lambda **kw:kw,tensorboard_trace_handler=lambda *a,**kw:None)
        def analyse(path,**kw):check('analyse');Path(path).mkdir();(Path(path)/'trace_view.json').write_text('[]')
        cap=NS(CaptureError=RuntimeError,normalize_trace=lambda *a,**kw:dict(iterations=[{'iteration_id':0}],tasks=[{'kind':'device','name':'mock_fused','iteration_id':0}]))
        accounting=NS(account_profile=lambda iterations,tasks,**kw:dict(iterations=iterations,tasks=tasks))
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch,'npu',NS(device=device,Event=Event,synchronize=lambda:check('sync')),create=True),patch.dict(sys.modules,{'torch_npu':NS(profiler=profiler),'torch_npu.profiler.profiler':NS(analyse=analyse),'profile_capture':cap,'profile_accounting':accounting}):
            item=NS(metadata={'prepared_device':'npu:0'},expected_kernels=['mock_fused'])
            for selected,fail in ((1,False),(2,True)):
                current[0]=selected;events.clear()
                def run():
                    check('launch')
                    if fail:raise RuntimeError('mock launch failure')
                result=op.Adapter().capture(run,item,Path(tmp)/str(selected));self.assertEqual(current[0],selected);self.assertIn('end',events);self.assertEqual(result['profile']['status'],'failed'if fail else'complete')
                if not fail:self.assertEqual(events.count('launch'),10);self.assertIn('sync',events);self.assertIn('analyse',events)

    def test_actual_evidence_consumer_binary_binding(self):
        import test_optimization_profile_evidence as fixtures
        import optimization_profile_evidence as evidence
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixtures.PairTests().fixture(Path(tmp));compiled=op.identity([NS(hash='a'*64,asm={'npubin':b'\x7fELF','ttir':IR})]*3)
            for row in report['arms'].values():row['compilation_state_before_capture']=compiled;row['compilation_state_after_capture']=json.loads(json.dumps(compiled))
            path.write_text(json.dumps(report));self.assertEqual(evidence.collect_pair(path)['status'],'complete');report['arms']['after']['compilation_state_after_capture'][1]['binaries']['npubin']='f'*64;path.write_text(json.dumps(report))
            with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path)


if __name__=='__main__':unittest.main()
