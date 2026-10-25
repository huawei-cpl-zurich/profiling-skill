from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,os,subprocess,sys,tempfile,unittest
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
import torch
from compressor_cache_contract import make_case,dimensions,check_fixture,cache_writers
from compressor_cache import Adapter,StudyUnsupported
from optimization_pair import fixture


def spec(start=0):
    return dict(case_id='host-unit-cache',role='guard' if start else 'gallery',seed=9287,
                parameters=dict(M=5,cache_size=11,ratio=4,start_pos=start))


class HostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.data=make_case(spec())

    def test_exact_certificate_cache_and_zero_padding(self):
        d=self.data;check_fixture(spec(),d)
        for weight,key in [('wkv','kv'),('wgate','score')]:
            expected=sum(int(float(d['x'][0,2,k])*128)*int(float(d[weight][7,k])*128) for k in range(5120))/16384
            self.assertEqual(float(d['expected'][key][2,7]),expected)
        self.assertTrue(torch.equal(d['expected']['cache'][0,5:],d['cache'][0,5:]))
        self.assertTrue((d['expected']['output'][0,1]==0).all())
        for m,c,start in [(3,9,0),(19,7,3),(34,13,1)]:
            writes=cache_writers(m,c,start)
            self.assertEqual(len(writes),min(m,c));self.assertEqual(len({s for _,s in writes}),len(writes))
            self.assertTrue(all(max(0,m-c)<=t<m for t,_ in writes))

    def test_rejections_cache_and_cli(self):
        for key,value in [('batch',2),('used_len',4),('block_table',[0,0]),('start_pos',4)]:
            bad=spec();bad['parameters'][key]=value
            with self.assertRaises(ValueError):dimensions(bad)
        alias=self.data['x'].view(torch.float32).flatten()[:11*1024].view(1,11,1024)
        for key,value in [('wgate',self.data['wkv']),('cache',alias),('cache',self.data['cache'].double())]:
            with self.assertRaises(ValueError):check_fixture(spec(),{**self.data,key:value})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);_,a=fixture(Adapter(),spec(),p/'unit-cache',torch);_,b=fixture(Adapter(),spec(),p/'unit-cache',torch)
            self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0))
            bad=spec();bad['parameters']['batch']=2;(p/'case.json').write_text(json.dumps(bad))
            result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/compressor_cache.py'),
                '--case',str(p/'case.json'),'--cache',str(p/'absent'),'--output',str(p/'out'),
                '--provenance',str(p/'absent-provenance')],cwd=p,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('T07 failed study',result.stderr)
            self.assertFalse((p/'absent').exists());self.assertFalse((p/'out').exists())
            subprocess.run([sys.executable,'-c','from compressor_cache import Adapter,StudyUnsupported; assert not Adapter().prepared'],
                cwd=p,env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')),check=True)

    def test_default_public_interfaces_fail_closed(self):
        with patch.dict(sys.modules,{'torch_npu':None,'triton':None}):
            with self.assertRaises(StudyUnsupported):Adapter().prepare('after',spec(),self.data,Path('.'))
            with self.assertRaises(StudyUnsupported):Adapter().compile_probe('after',spec(),Path('.'))
        calls=[]
        with self.assertRaises(StudyUnsupported):Adapter().capture(lambda:calls.append('forward'),None,None)
        self.assertFalse(calls)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','compressor_cache:factory',
                '--case',str(p/'missing'),'--provenance',str(p/'missing-provenance'),
                '--cache',str(p/'cache'),'--output',str(p/'out')],cwd=p,capture_output=True,text=True,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')))
            self.assertNotEqual(result.returncode,0);self.assertIn('T07 failed study',result.stderr)
            self.assertFalse((p/'cache').exists());self.assertFalse((p/'out').exists())

    def test_mocked_full_api_and_pair_state(self):
        calls=[]
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*args,**kw):
                calls.append('compile:'+self.fn.__name__)
                return SimpleNamespace(hash=hashlib.sha256(self.fn.__name__.encode()).hexdigest(),asm={'npubin':b'unit-only'})
            def __getitem__(self,grid):
                def run(*a,**kw):
                    calls.append(self.fn.__name__)
                    return SimpleNamespace(hash=hashlib.sha256(self.fn.__name__.encode()).hexdigest(),asm={'npubin':b'unit-only'})
                return run
        vendor=SimpleNamespace(**{n:Kernel(n) for n in ('_proj2_kernel','_proj_kernel','_pool_kernel','_cache_update_kernel')})
        projection=SimpleNamespace(**{n:Kernel(n) for n in ('cache_proj2_kernel','cache_proj_kernel')})
        driver=SimpleNamespace(active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda _:dict(num_aicore=4))))
        triton=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda x:1<<(x-1).bit_length(),runtime=SimpleNamespace(driver=driver))
        with patch.dict(sys.modules,{'torch_npu':SimpleNamespace(),'triton':triton,'compressor_cache_vendor':vendor,'compressor_cache_projection':projection}),patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True),patch.object(torch,'npu',SimpleNamespace(current_device=lambda:0),create=True):
            adapter=Adapter(compile_inspection=True);item=adapter.prepare('after',spec(),self.data,Path('.'));self.assertFalse(calls)
            item.compile_only()
            for _ in range(2):
                with self.assertRaises(StudyUnsupported):item.run()
            self.assertEqual(calls.count('compile:cache_proj2_kernel'),1)
            self.assertFalse([x for x in calls if not x.startswith('compile:')])
            residual=spec(2);data=make_case(residual)
            for order in [('before','after'),('after','before')]:
                items={a:adapter.prepare(a,residual,data,Path('.')) for a in order}
                for a in order:
                    calls.clear();items[a].compile_only()
                    with self.assertRaises(StudyUnsupported):items[a].run()
                    self.assertFalse([x for x in calls if not x.startswith('compile:')])
                    for key in ('kv','score'):getattr(items[a],key).copy_(data['expected'][key])
                    items[a].output.copy_(data['expected']['output']);items[a].cache.copy_(data['expected']['cache'])
                    self.assertTrue(adapter.validate(a,dict(output=items[a].output,cache=items[a].cache),data,residual)['passed'])
                items[order[-1]].cache_store[0]=0
                self.assertFalse(adapter.validate(order[-1],dict(output=items[order[-1]].output,cache=items[order[-1]].cache),data,residual)['passed'])

if __name__=='__main__':unittest.main()
