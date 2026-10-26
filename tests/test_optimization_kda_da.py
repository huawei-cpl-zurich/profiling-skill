from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import contextmanager
import hashlib,json,sys,tempfile,unittest,weakref
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton/kda_da'),str(ROOT/'scripts'),str(ROOT/'tests')]
import kda_da as op
import kda_da_contract as provider
SPEC=dict(B=2,T=5,H=2,BT=64,dense=True,seed=90127)
torch.set_num_threads(1)


def ir(tile):
    return f'tt.func public @stage(%a: !tt.ptr<f16>, %p: !tt.ptr<f32>, %o: !tt.ptr<f32>) {{\n%c = arith.truncf %p : tensor<{tile}x{tile}xf32> to tensor<{tile}x{tile}xf16>\n%d = tt.dot %a, %b, %c : tensor<{tile}x{tile}xf16> * tensor<{tile}x{tile}xf16> -> tensor<{tile}x{tile}xf32>\n}}'


class Kernel:
    fail=None;seen=[];check=staticmethod(lambda:None)
    def __init__(self,name):self.name=name;self.fn=NS(__name__='mock_'+name)
    def warmup(self,*args,**flags):
        Kernel.check()
        if self.fail==self.name:raise RuntimeError('mock compile failure')
        name=self.name+str(flags['BM']);return NS(hash=hashlib.sha256(name.encode()).hexdigest(),asm=dict(npubin=b'\x7fELF'+name.encode(),ttir=ir(flags['BM'])))
    def __getitem__(self,grid):
        def run(a,p,out,*args,**flags):
            Kernel.check();Kernel.seen.append(self.name);b,t,h,bt=a.shape
            for batch in range(b):
                for start in range(0,t,bt):
                    n=min(bt,t-start)
                    for head in range(h):
                        aa=torch.zeros((bt,bt),dtype=torch.float64);pp=aa.clone();aa[:n]=a[batch,start:start+n,head].double();pp[:n]=p[batch,start:start+n,head].half().double()
                        value=pp@aa.T if self.name=='right'else torch.tril(-(aa.T@pp),diagonal=-1)
                        out[batch,start:start+n,head].copy_(value[:n].float())
        return run


def kernels():return NS(chunk_kda_bwd_kernel_dA_post_right=Kernel('right'),chunk_kda_bwd_kernel_dA_post_left=Kernel('left'))


class DATests(unittest.TestCase):
    def tearDown(self):Kernel.fail=None;Kernel.seen=[];Kernel.check=staticmethod(lambda:None)
    def test_independent_exact_oracle_defaults_tails_and_admission(self):
        old=torch.get_default_dtype()
        try:
            for dtype in (torch.float16,torch.float64):
                torch.set_default_dtype(dtype);f=provider.make_case(SPEC);a=f['inputs']['A'][1,:,1].double();p=f['inputs']['P'][1,:,1].double();row,col=2,1;want=0.
                for k in range(5):
                    inner=sum(float(p[k,j])*float(a[col,j])for j in range(64))
                    cast=float(torch.tensor(inner,dtype=torch.float16,device='cpu'));want-=float(a[k,row])*cast
                self.assertEqual(float(f['expected'][1,row,1,col]),want);self.assertTrue(torch.equal(f['expected'].double(),f['expected_fp64']))
        finally:torch.set_default_dtype(old)
        f=provider.make_case(dict(SPEC,T=67))
        self.assertEqual(int(torch.count_nonzero(f['expected'][:,64])),0)
        self.assertEqual(int(torch.count_nonzero(f['expected'][:,65,:,1:])),0)
        self.assertEqual(int(torch.count_nonzero(f['expected'][:,66,:,2:])),0)
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for change in ({'B':3},{'T':True},{'H':0},{'BT':32},{'dense':False},{'dtype_A':'float32'}):
                with self.assertRaises(ValueError):op.Adapter().compile_probe('after',dict(SPEC,**change),None)
            for change in ('A','P','halo','layout','alias'):
                f=provider.make_case(SPEC)
                if change=='A':f['inputs']['A'][0,0,0,1]=.25
                elif change=='P':f['inputs']['P'][0,0,0,0]=.25
                elif change=='halo':provider.backing(f['inputs']['A'])[0]=0
                elif change=='layout':f['inputs']['P']=f['inputs']['P'].transpose(1,2)
                else:f['expected']=f['inputs']['P']
                with self.assertRaises(ValueError):op.Adapter().prepare('before',SPEC,f,None)

    def test_atomic_compile_reuse_two_stages_validation_and_close(self):
        f=provider.make_case(SPEC);adapter=op.Adapter()
        with patch.object(op,'backend',return_value=('cpu',kernels(),None)):
            probe=adapter.compile_probe('before',SPEC,None)
            with self.assertRaises(RuntimeError):probe.run()
            Kernel.fail='left'
            with self.assertRaises(RuntimeError):probe.compile_only()
            self.assertEqual(probe.compilation_state(),[]);self.assertEqual(Kernel.seen,[]);Kernel.fail=None;self.assertEqual(len(probe.compile_only()),2);probe.close()
            for order in [('before','after'),('after','before')]:
                items={arm:adapter.prepare(arm,SPEC,f,None)for arm in order}
                for arm,item in items.items():
                    value=item.run();self.assertTrue(adapter.validate(arm,value,f,SPEC)['passed']);self.assertIs(value,item.run());self.assertEqual(len(item.compilation_state()),2)
                value[0,1,0,0]=float('nan');self.assertFalse(adapter.validate(order[-1],value,f,SPEC)['passed']);items[order[-1]].run();items[order[-1]].stores[0][0]=0;self.assertFalse(adapter.validate(order[-1],value,f,SPEC)['passed']);del value
                refs=[weakref.ref(item.output)for item in items.values()]
                for item in items.values():item.close();item.close()
                self.assertEqual(adapter.items,{});self.assertTrue(all(ref()is None for ref in refs))
                with self.assertRaises(RuntimeError):items['after'].run()
            first=adapter.prepare('before',SPEC,f,None);other=adapter.prepare('after',SPEC,f,None);new=adapter.prepare('before',SPEC,f,None);first.close();self.assertIs(adapter.items['before'],new);self.assertIs(adapter.items['after'],other);new.close();other.close()
        self.assertIn('right',Kernel.seen);self.assertIn('left',Kernel.seen)
        for text in ('',ir(32).replace('xf16','xf32'),ir(32).replace('arith.truncf','arith.bitcast'),ir(64)):
            with self.assertRaises(ValueError):op.typed_policy(NS(asm={'ttir':text}),32)

    def test_real_pair_cleanup_and_foreign_device_restoration(self):
        from optimization_pair import run_pair
        current=[2]
        @contextmanager
        def device(d):
            old=current[0];current[0]=int(str(d).split(':')[-1])
            try:yield
            finally:current[0]=old
        real_to=torch.Tensor.to
        def transfer(x,d,*a,**kw):
            if d=='npu:0':self.assertEqual(current[0],0);return x
            return real_to(x,d,*a,**kw)
        Kernel.check=staticmethod(lambda:self.assertEqual(current[0],0))
        with tempfile.TemporaryDirectory()as tmp,patch.object(torch,'npu',NS(device=device),create=True),patch.object(torch.Tensor,'to',transfer),patch.object(op,'backend',return_value=('npu:0',kernels(),None)):
            root=Path(tmp);a=op.Adapter()
            for caller in (1,2):
                current[0]=caller;report=run_pair(a,SPEC,root/str(caller),root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(report['status'],'complete');self.assertEqual(current[0],caller);self.assertEqual(a.items,{})
            Kernel.fail='left';report=run_pair(a,SPEC,root/'error',root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(report['status'],'failed');self.assertEqual(current[0],2);self.assertEqual(a.items,{})

    def test_entire_capture_device_metric_events_and_failure(self):
        import profile_ten
        current=[2];events=[]
        def check(name):self.assertEqual(current[0],0);events.append(name)
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
        cap=NS(CaptureError=RuntimeError,normalize_trace=lambda *a,**kw:dict(iterations=[{'iteration_id':0}],tasks=[{'kind':'device','name':name,'iteration_id':0}for name in('mock_right','mock_left')]))
        accounting=NS(account_profile=lambda iterations,tasks,**kw:dict(iterations=iterations,tasks=tasks))
        with tempfile.TemporaryDirectory()as tmp,patch.object(torch,'npu',NS(device=device,Event=Event,synchronize=lambda:check('sync')),create=True),patch.dict(sys.modules,{'torch_npu':NS(profiler=profiler),'torch_npu.profiler.profiler':NS(analyse=analyse),'profile_capture':cap,'profile_accounting':accounting}):
            item=NS(metadata={'prepared_device':'npu:0'},expected_kernels=['mock_right','mock_left'])
            for caller,fail in((1,False),(2,True)):
                current[0]=caller;events.clear()
                def run():
                    check('launch')
                    if fail:raise RuntimeError('mock launch failed')
                result=op.Adapter().capture(run,item,Path(tmp)/str(caller));self.assertEqual(current[0],caller);self.assertIn('end',events);self.assertEqual(result['profile']['status'],'failed'if fail else'complete')
                if not fail:self.assertEqual(events.count('launch'),10);self.assertIn('sync',events);self.assertIn('analyse',events)

    def test_actual_evidence_binary_binding(self):
        import test_optimization_profile_evidence as fixtures
        import optimization_profile_evidence as evidence
        with tempfile.TemporaryDirectory()as tmp:
            path,report=fixtures.PairTests().fixture(Path(tmp));compiled=op.identity([NS(hash='a'*64,asm={'npubin':b'\x7fELF','ttir':ir(32)})]*2)
            for row in report['arms'].values():row['compilation_state_before_capture']=compiled;row['compilation_state_after_capture']=json.loads(json.dumps(compiled))
            path.write_text(json.dumps(report));self.assertEqual(evidence.collect_pair(path)['status'],'complete');report['arms']['after']['compilation_state_after_capture'][1]['binaries']['npubin']='f'*64;path.write_text(json.dumps(report))
            with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path)


if __name__=='__main__':unittest.main()
