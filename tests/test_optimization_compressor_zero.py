from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,os,subprocess,sys,tempfile,unittest
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
import torch
from compressor_zero_contract import make_case,dimensions,check_fixture,complete_write,output_shape
from compressor_zero import Adapter
from optimization_pair import fixture


def spec(b=2,s=18,r=4,**extra):
    return dict(case_id='host-only-zero',role='guard',seed=92913,parameters=dict(B=b,S=s,ratio=r,**extra))


class HostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.data=make_case(spec())

    def test_cpu_projection_certificate_padding_and_cache(self):
        d=self.data;check_fixture(spec(),d)
        for weight,key in [('wkv','kv'),('wgate','score')]:
            exact=sum(int(float(d['x'][1,3,k])*128)*int(float(d[weight][7,k])*128) for k in range(5120))/16384
            self.assertEqual(float(d['expected'][key][21,7]),exact)
        self.assertTrue((d['expected']['output'][:,-1]==0).all())
        for batch in range(2):
            for token in range(2,18):
                self.assertTrue(torch.equal(d['expected']['cache'][batch,token%16,:512],d['expected']['kv'][batch*18+token]))
        packed=spec(1,19,4,packed=True);p=make_case(packed)
        self.assertEqual(output_shape(packed),(5,512));self.assertTrue((p['expected']['output'][-1]==0).all())
        partial=spec(1,21,4,used_len=17,start_pos=2);p=make_case(partial)
        self.assertTrue((p['expected']['output'][0,4:]==0).all())

    def test_complete_write_partition_and_fallback(self):
        for b,s,r in [(2,24,4),(3,40,8),(1,10,2)]:
            case=spec(b,s,r);self.assertTrue(complete_write(case))
            gt=16//r;tasks=1+((s+r-1)//r+gt-1)//gt
            writes=[g for task in range(tasks) for g in range(task*gt,min(task*gt+gt,s//r))]
            self.assertEqual(sorted(writes),list(range(s//r)))
            self.assertEqual(len(writes),len(set(writes)))
        for case in [spec(),spec(1,24,4,packed=True),spec(1,24,4,used_len=19),spec(1,24,4,start_pos=1)]:self.assertFalse(complete_write(case))

    def test_domain_layout_alias_and_cache(self):
        for params in [dict(B=True),dict(S=2),dict(ratio=3),dict(packed=1),dict(B=2,packed=True),dict(block_table=[0,0]),dict(used_len=4),dict(start_pos=-1)]:
            bad=spec();bad['parameters'].update(params)
            with self.assertRaises(ValueError):dimensions(bad)
            with patch.dict(sys.modules,{'torch_npu':None,'triton':None}):
                with self.assertRaises(ValueError):Adapter().prepare('before',bad,self.data,Path('.'))
                with self.assertRaises(ValueError):Adapter().compile_probe('after',bad,Path('.'))
        for field,value in [('wgate',self.data['wkv']),('cache',self.data['cache'].double()),('x',self.data['x'].transpose(0,1))]:
            with self.assertRaises(ValueError):check_fixture(spec(),{**self.data,field:value})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);_,a=fixture(Adapter(),spec(),p/'cache',torch);_,b=fixture(Adapter(),spec(),p/'cache',torch)
            self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0))
            invalid=spec();invalid['parameters']['block_table']=[0,0];(p/'case.json').write_text(json.dumps(invalid))
            run=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/compressor_zero.py'),'--case',str(p/'case.json'),'--cache',str(p/'absent-cache'),'--output',str(p/'out'),'--provenance',str(p/'absent-provenance')],cwd=p,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0);self.assertIn('distinct private cache',run.stderr)
            self.assertFalse((p/'absent-cache').exists());self.assertFalse((p/'out').exists())
            subprocess.run([sys.executable,'-c','from compressor_zero import factory; assert not factory().prepared'],cwd=p,env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')),check=True)

    def test_non_kernel_zero_tasks_are_full_operation_work(self):
        from profile_capture import normalize_trace,TASK_KINDS,CaptureError
        from profile_accounting import account_profile
        from optimization_profile_evidence import operation_windows
        events=[dict(name='model_iteration_0',pid=1,tid=1,ts=0,dur=100,ph='X',cat='cpu_op')]
        for i,(kind,start,duration) in enumerate([('MEMSET',20,7),('KERNEL_AIVEC',40,3)]):
            events.extend([dict(name='zero' if i==0 else 'math',pid=1,tid=1,ts=10+i*10,dur=2,ph='X',cat='cpu_op'),dict(name=kind,pid=2,tid=3,ts=start,dur=duration,ph='X',args={'Task Type':kind,'Physic Stream Id':3,'Task Id':i,'Batch Id':0,'Subtask Id':0}),dict(ph='s',cat='async_npu',name='torch_to_npu',id=i,pid=1,tid=1,ts=10+i*10),dict(ph='f',cat='async_npu',name='torch_to_npu',id=i,pid=2,tid=3,ts=start)])
        with self.assertRaises(CaptureError):normalize_trace(events,['model_iteration_0'],export_completed=True)
        with patch.dict(TASK_KINDS,{'MEMSET':'device','MEMSET_ASYNC':'device'}):
            normalized=normalize_trace(events,['model_iteration_0'],export_completed=True)
            result=account_profile(normalized['iterations'],normalized['tasks'],capture_complete=True)
            self.assertEqual(result['iterations'][0]['task_count'],2)
            self.assertEqual(operation_windows(result)[0]['device_operation_span_us'],23)
            self.assertEqual(operation_windows(result)[0]['operation_active_union_us'],10)
            self.assertEqual(normalized['acquisition_evidence']['links'][0]['task_type'],'MEMSET')

    def test_mocked_full_api_atomic_compile_and_pair_state(self):
        calls=[];fail=[True]
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def artifact(self):return SimpleNamespace(hash=hashlib.sha256(self.fn.__name__.encode()).hexdigest(),asm={'npubin':self.fn.__name__.encode()})
            def warmup(self,*args,**kw):
                calls.append('compile:'+self.fn.__name__)
                if fail[0] and self.fn.__name__=='_cache_update_kernel':raise RuntimeError('unit compile failure')
                return self.artifact()
            def __getitem__(self,grid):
                def run(*args,**kw):calls.append(self.fn.__name__);return self.artifact()
                return run
        vendor=SimpleNamespace(**{n:Kernel(n) for n in ('_proj2_kernel','_proj_kernel','_pool_kernel','_cache_update_kernel')})
        driver=SimpleNamespace(active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda _:dict(num_aicore=4))))
        triton=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda x:1<<(x-1).bit_length(),runtime=SimpleNamespace(driver=driver))
        zero=torch.Tensor.zero_
        def counted_zero(x):calls.append('zero');return zero(x)
        with patch.dict(sys.modules,{'torch_npu':SimpleNamespace(),'triton':triton,'compressor_zero_vendor':vendor}),patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True),patch.object(torch,'npu',SimpleNamespace(current_device=lambda:0),create=True),patch.object(torch.Tensor,'zero_',counted_zero):
            adapter=Adapter();item=adapter.prepare('before',spec(),self.data,Path('.'))
            self.assertFalse(calls)
            with self.assertRaises(RuntimeError):item.run()
            self.assertEqual(item.compilation_state(),[]);self.assertNotIn('zero',calls)
            fail[0]=False;calls.clear();item.run();item.run()
            self.assertEqual(calls.count('compile:_cache_update_kernel'),1);self.assertEqual(calls.count('zero'),2)
            self.assertEqual(calls.count('_cache_update_kernel'),2)
            probe=adapter._build('after',spec(),{k:v.clone() for k,v in self.data.items() if k!='expected'},probe=True)
            calls.clear()
            with self.assertRaises(ValueError):probe.run()
            self.assertFalse(calls)
            # Changed process defaults must not alter the FP32 launch ABI or CPU diagnostics.
            default_dtype=torch.get_default_dtype()
            try:
                for dtype in (torch.float64,torch.float16):
                    torch.set_default_dtype(dtype)
                    for arm in ('before','after'):
                        changed=adapter.prepare(arm,spec(),self.data,Path('.'))
                        self.assertEqual(changed.kv.dtype,torch.float32);self.assertEqual(changed.score.dtype,torch.float32)
                        self.assertEqual(changed.kv.device.type,'cpu');self.assertEqual(changed.score.device.type,'cpu')
                        changed.compile_only()
                        for key in ('kv','score'):getattr(changed,key).copy_(self.data['expected'][key])
                        changed.output.copy_(self.data['expected']['output']);changed.cache.copy_(self.data['expected']['cache'])
                        diagnostics=adapter.validate(arm,dict(output=changed.output,cache=changed.cache),self.data,spec())
                        self.assertTrue(diagnostics['passed'])
                        self.assertEqual(diagnostics['absolute_error_quantiles'],[0.]*5)
                        self.assertEqual(diagnostics['relative_error_quantiles'],[0.]*5)
            finally:
                torch.set_default_dtype(default_dtype)
            for case in [spec(1,12,4),spec(1,13,4),spec(1,19,4,packed=True),spec(1,19,4,start_pos=2)]:
                data=make_case(case)
                for order in [('before','after'),('after','before')]:
                    items={a:adapter.prepare(a,case,data,Path('.')) for a in order}
                    for a in order:
                        calls.clear();actual=items[a].run()
                        self.assertEqual(calls.count('zero'),int(a=='before' or not complete_write(case)))
                        for key in ('kv','score'):getattr(items[a],key).copy_(data['expected'][key])
                        items[a].output.copy_(data['expected']['output']);items[a].cache.copy_(data['expected']['cache'])
                        self.assertTrue(adapter.validate(a,actual,data,case)['passed'])
                    bad=items[order[-1]]
                    self.assertFalse(adapter.validate(order[-1],{'output':bad.output},data,case)['passed'])
                    bad.cache_store[0]=0;self.assertFalse(adapter.validate(order[-1],dict(output=bad.output,cache=bad.cache),data,case)['passed'])
            with self.assertRaises(ValueError):adapter.validate('before',{},self.data,spec())


if __name__=='__main__':unittest.main()
