from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import contextmanager
import torch,sys,unittest,tempfile,json,subprocess,weakref,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import dsa_indexer as op
import dsa_indexer_contract as contract


class Kernel:
    fail=None;launches=0;checks=staticmethod(lambda:None)
    def __init__(self,kind):self.kind=kind;self.fn=NS(__name__='mock_'+kind);self.warmups=0
    def warmup(self,*args,**kw):
        Kernel.checks();self.warmups+=1
        if Kernel.fail==self.kind:raise RuntimeError('mock compiler failure')
        return NS(hash=hashlib.sha256(self.kind.encode()).hexdigest(),name=self.fn.__name__,asm=dict(npubin=self.kind.encode(),ttir='mock FP32'))
    def __getitem__(self,grid):
        def launch(*args,**flags):
            Kernel.checks();Kernel.launches+=1
            if self.kind in ('qk','fused'):
                if self.kind=='qk':q,k,out,b,h,m,n,d,_=args
                else:q,k,out,w,b,h,m,n,d,_=args
                score=(q.float().reshape(b,m*h,d)@k.float().transpose(1,2)).reshape(b,m,h,n)
                out.copy_(score if self.kind=='qk' else (score.clamp_min(0)*w.unsqueeze(-1)).sum(2))
            elif self.kind=='activate':s,w,a,*_=args;a.copy_(s.clamp_min(0)*w.unsqueeze(-1))
            else:a,out,*_=args;out.copy_(a.sum(2))
        return launch


def kernels():return NS(dsa_indexer_qk=Kernel('qk'),dsa_indexer_activate=Kernel('activate'),dsa_indexer_headsum=Kernel('headsum'))


class DSATests(unittest.TestCase):
    def spec(self):return dict(parameters=dict(B=2,M=3,H=16,D=32,N=131,dtype='float16'),seed=291201,role='guard')
    def tearDown(self):Kernel.fail=None;Kernel.checks=staticmethod(lambda:None)

    def test_independent_exact_oracle_domain_and_cache(self):
        from optimization_pair import fixture
        spec=self.spec();x=contract.make_case(spec);q=x['inputs']['q'].double()*64;k=x['inputs']['k'].double()*64;w=x['inputs']['weights'].double()*16;p=spec['parameters']
        integer=(q.reshape(p['B'],p['M']*p['H'],p['D'])@k.transpose(1,2)).reshape(p['B'],p['M'],p['H'],p['N']).clamp_min(0)
        self.assertTrue(torch.equal(x['expected'].double()*65536,(integer*w.unsqueeze(-1)).sum(2)))
        with tempfile.TemporaryDirectory() as tmp:
            for case in (spec,dict(p,seed=291203,role='guard')):
                _,a=fixture(op.Adapter(),case,Path(tmp),torch);_,b=fixture(op.Adapter(),case,Path(tmp),torch);self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0))
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for key,wrong in [('q',x['inputs']['q'].float()),('k',x['inputs']['k'].transpose(1,2)),('weights',x['inputs']['weights'][:,:,:1].expand_as(x['inputs']['weights']))]:
                with self.assertRaises(ValueError):op.Adapter().prepare('after',spec,dict(x,inputs=dict(x['inputs'],**{key:wrong})),None)
            alias=x['inputs']['k'].flatten()[:x['inputs']['q'].numel()].view_as(x['inputs']['q'])
            with self.assertRaises(ValueError):op.Adapter().prepare('after',spec,dict(x,inputs=dict(x['inputs'],q=alias)),None)
            for value in (float('nan'),float('inf'),.01,.5):
                weights=x['inputs']['weights'].clone();weights[0,0,0]=value
                with self.assertRaises(ValueError):op.Adapter().prepare('before',spec,dict(x,inputs=dict(x['inputs'],weights=weights)),None)
            for name,value in [('H',17),('D',48),('N',0),('B',True)]:
                with self.assertRaises(ValueError):op.Adapter().compile_probe('after',dict(spec,parameters=dict(p,**{name:value})),None)

    def test_atomic_probe_and_prepared_reuse_dtype_diagnostics_cleanup(self):
        spec=self.spec();x=contract.make_case(spec);k=kernels();fused=Kernel('fused');adapter=op.Adapter();default=torch.get_default_dtype()
        try:
            with patch.object(op,'backend',return_value=('cpu',k,fused)):
                probe=adapter.compile_probe('before',spec,None);launches=Kernel.launches
                with self.assertRaises(RuntimeError):probe.run()
                self.assertEqual(Kernel.launches,launches);Kernel.fail='activate'
                with self.assertRaises(RuntimeError):probe.compile_only()
                self.assertEqual(probe.compilation_state(),[]);Kernel.fail=None;self.assertEqual(len(probe.compile_only()),3);probe.close();probe.close()
                for dtype,order in [(torch.float64,('before','after')),(torch.float16,('after','before'))]:
                    torch.set_default_dtype(dtype);items={arm:adapter.prepare(arm,spec,x,None) for arm in order}
                    for arm,item in items.items():
                        out=item.run();self.assertEqual(out.dtype,torch.float32);self.assertTrue(adapter.validate(arm,out,x,spec)['passed']);self.assertIs(item.run(),out);self.assertFalse(adapter.validate(arm,(out,),x,spec)['passed']);self.assertFalse(adapter.validate(arm,out.double(),x,spec)['passed'])
                        out[0,0,0]=float('inf');d=adapter.validate(arm,out,x,spec);self.assertFalse(d['passed']);self.assertEqual(d['mismatch_count'],1);self.assertEqual(d['first_mismatches'][0]['coordinate'],[0,0,0]);item.run();adapter.validate(arm,out,x,spec)
                    refs=[weakref.ref(adapter.state[a]['out']) for a in items];del out
                    items[order[0]].close();self.assertTrue(adapter.validate(order[1],items[order[1]].run(),x,spec)['passed']);items[order[1]].close();self.assertTrue(all(r() is None for r in refs));self.assertEqual(adapter.state,{})
                    with self.assertRaises(RuntimeError):items[order[1]].run()
        finally:torch.set_default_dtype(default)

    def test_real_pair_lifecycle_and_device_restore(self):
        from optimization_pair import run_pair
        current=[1]
        @contextmanager
        def device(d):
            old=current[0];current[0]=int(str(d).split(':')[-1])
            try:yield
            finally:current[0]=old
        original_to=torch.Tensor.to
        def transfer(t,d,*a,**kw):
            if d=='npu:0':self.assertEqual(current[0],0);return t
            return original_to(t,d,*a,**kw)
        Kernel.checks=staticmethod(lambda:self.assertEqual(current[0],0))
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch,'npu',NS(device=device),create=True),patch.object(torch.Tensor,'to',transfer),patch.object(op,'backend',return_value=('npu:0',kernels(),Kernel('fused'))):
            adapter=op.Adapter();root=Path(tmp)
            for i,selected in enumerate((1,2)):
                current[0]=selected;r=run_pair(adapter,self.spec(),root/str(i),root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(r['status'],'complete');self.assertTrue(all(a['cleanup']['status']=='complete' for a in r['arms'].values()));self.assertEqual(adapter.state,{});self.assertEqual(current[0],selected)
            Kernel.fail='activate';r=run_pair(adapter,self.spec(),root/'error',root/'cache',{},torch=torch,correctness_only=True);self.assertEqual(r['status'],'failed');self.assertEqual(adapter.state,{});self.assertEqual(current[0],2)

    def test_capture_entire_lifecycle_uses_prepared_device_and_restores(self):
        import optimization_pair,profile_ten
        current=[2];events=[]
        def check(name):
            self.assertEqual(current[0],0);events.append(name)
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
        fake=NS(profiler=profiler)
        item=NS(metadata={'prepared_device':'npu:0'},expected_kernels=['mock_fused'])
        def analyse(path,**kw):
            check('analyse');vendor=Path(path);vendor.mkdir();(vendor/'trace_view.json').write_text('[]')
        capture=NS(CaptureError=RuntimeError,normalize_trace=lambda *a,**kw:dict(iterations=[{'iteration_id':0}],tasks=[{'kind':'device','name':'mock_fused','iteration_id':0}]))
        accounting=NS(account_profile=lambda iterations,tasks,**kw:dict(iterations=iterations,tasks=tasks))
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch,'npu',NS(device=device,Event=Event,synchronize=lambda:check('sync')),create=True),patch.dict(sys.modules,{'torch_npu':fake,'torch_npu.profiler.profiler':NS(analyse=analyse),'profile_capture':capture,'profile_accounting':accounting}):
            for selected,fail in ((1,False),(2,True)):
                current[0]=selected;events.clear()
                def run():
                    check('kernel')
                    if fail:raise RuntimeError('mock launch failure')
                result=op.Adapter().capture(run,item,Path(tmp)/str(selected))
                self.assertEqual(current[0],selected);self.assertIn('begin',events);self.assertIn('end',events)
                self.assertEqual(result['profile']['status'],'failed' if fail else 'complete');self.assertEqual(result['kernel_identity']['expected'],item.expected_kernels)
                if not fail:
                    self.assertEqual(events.count('kernel'),10);self.assertEqual(events.count('event'),20);self.assertEqual(events.count('record'),20);self.assertIn('sync',events);self.assertIn('analyse',events)
            with patch.object(profile_ten,'capture_ten',side_effect=RuntimeError('setup failure')):
                with self.assertRaises(RuntimeError):op.Adapter().capture(run,item,Path(tmp)/'setup')
                self.assertEqual(current[0],2)

    def test_signed_zeros_secondary_and_real_collect_pair_identity(self):
        import copy
        sys.path.insert(0,str(Path(__file__).parent));import test_optimization_profile_evidence as fixtures
        from optimization_profile_evidence import collect_pair,EvidenceError
        spec=dict(parameters=dict(B=1,M=2,H=16,D=32,N=17),seed=291202,weight_pattern='materialized_broadcast_heads_rows_0_and_minus4_over16');x=contract.make_case(spec);adapter=op.Adapter()
        with patch.object(op,'backend',return_value=('cpu',kernels(),Kernel('fused'))):
            a=adapter.prepare('before',spec,x,None);b=adapter.prepare('after',spec,x,None);out=a.run();adapter.validate('before',out,x,spec);other=b.run();other[0,0].fill_(-0.);d=adapter.validate('after',other,x,spec);self.assertTrue(d['passed']);self.assertFalse(d['paired_all_bits_secondary']);identity=b.compilation_state();a.close();b.close()
        with tempfile.TemporaryDirectory() as tmp:
            path,report=fixtures.PairTests().fixture(Path(tmp))
            for row in report['arms'].values():row['compilation_state_before_capture']=copy.deepcopy(identity);row['compilation_state_after_capture']=copy.deepcopy(identity)
            path.write_text(json.dumps(report));self.assertEqual(collect_pair(path)['status'],'complete')
            report['arms']['before']['compilation_state_before_capture']={'bad':'dict'};path.write_text(json.dumps(report))
            with self.assertRaises(EvidenceError):collect_pair(path)

    def test_external_cwd_cli_valid_json_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'case.json';case.write_text(json.dumps(self.spec())+'\n');provenance=root/'provenance.json';provenance.write_text('{}\n')
            script="import sys;sys.path[:0]=sys.argv[1:4];import dsa_indexer as op;import test_optimization_dsa_indexer_fusion as t;op.backend=lambda:('cpu',t.kernels(),t.Kernel('fused'));sys.argv=[op.__file__]+sys.argv[4:];raise SystemExit(op.main())"
            base=Path(__file__).resolve().parents[1];args=[sys.executable,'-c',script,str(Path(op.__file__).parent),str(base/'scripts'),str(base/'tests'),'--case',str(case),'--output',str(root/'out'),'--cache',str(root/'cache'),'--provenance',str(provenance),'--correctness-only'];p=subprocess.run(args,cwd=tmp,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads((root/'out/result.json').read_text())['status'],'complete')
            case.write_text('{bad JSON}\n');args[args.index('--output')+1]=str(root/'invalid');p=subprocess.run(args,cwd=tmp,capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertFalse((root/'invalid').exists())


if __name__=='__main__':unittest.main()
