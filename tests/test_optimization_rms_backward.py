import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'examples/optimization-library/triton'
sys.path[:0] = [str(DIRECTORY),str(ROOT/'scripts')]
from rms_backward_contract import make_case, validate_inputs, check_outputs
from rms_backward import Adapter
from optimization_pair import fixture


def spec(**changes):
    return dict(seed=49,parameters=dict(rows=7,D=17,offset=1,eps=1e-6,**changes))


class RMSBackwardTests(unittest.TestCase):
    def test_scalar_saved_state_oracle(self):
        value=make_case(spec()); x,dy,w,r=(value['inputs'][k] for k in ('X','dY','W','RSTD'))
        dx,dw=value['expected']
        for i in range(7):
            dot=sum(float(dy[i,j])*(float(w[j])+1)*float(x[i,j]) for j in range(17))/17
            for j in range(17):
                expected=float(r[i])*float(dy[i,j])*(float(w[j])+1)-float(r[i])**3*float(x[i,j])*dot
                self.assertAlmostEqual(float(dx[i,j]),expected,places=12)
        for j in range(17):
            self.assertAlmostEqual(float(dw[j]),sum(float(dy[i,j])*float(x[i,j])*float(r[i]) for i in range(7)),places=12)

    def test_zero_and_cancellation(self):
        case=dict(seed=79,rows=5,D=19,offset=0,eps=1e-6,zero_cancellation_poison_tail=True)
        value=make_case(case)
        self.assertEqual(float(value['inputs']['RSTD'][0]),1000.)
        expected=value['inputs']['dY'][0].double()*value['inputs']['W'].double()*1000
        torch.testing.assert_close(value['expected'][0][0],expected)

    def test_output_contract(self):
        expected=make_case(spec())['expected']; good=tuple(v.float() for v in expected)
        self.assertTrue(check_outputs(good,expected)['passed'])
        for bad in (good[:1],good+good,list(good),(good[0].double(),good[1]),
                    (good[0].flatten(),good[1]),(good[0]+.01,good[1]),
                    (good[0].fill_(float('nan')),good[1])):
            self.assertFalse(check_outputs(bad,expected)['passed'])

    def test_reject_before_backend_and_cache(self):
        for key,v in [('X',torch.zeros(7,17,dtype=torch.float64)),('RSTD',torch.full((7,),float('nan'))),
                      ('dY',torch.zeros(8,17)[1:])]:
            value=make_case(spec());value['inputs'][key]=v
            with self.assertRaises(ValueError):Adapter().prepare('after',spec(),value,Path('.'))
        value=make_case(spec());value['inputs']['dY']=value['inputs']['X']
        with self.assertRaises(ValueError):validate_inputs(value['inputs'],spec())
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache'
            a,one=fixture(Adapter(),spec(),cache,torch);b,two=fixture(Adapter(),spec(),cache,torch)
            self.assertEqual((one['oracle_calls'],two['oracle_calls']),(1,0))
            self.assertTrue(torch.equal(a['expected'][0],b['expected'][0]))
            with self.assertRaises(ValueError):fixture(Adapter(),dict(seed=3,rows=1,D=2049,offset=0,eps=1e-6),Path(tmp)/'bad',torch)
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_public_compile_reuse_routes_and_private_outputs(self):
        calls=[];compiles=[];fail_next=[]
        artifact=SimpleNamespace(hash='a'*64,asm={'npubin':b'code'})
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*args,**kwargs):
                compiles.append(self.fn.__name__)
                if fail_next and fail_next[0] == self.fn.__name__:
                    fail_next.pop(0)
                    raise RuntimeError('mock late compile failure')
                return artifact
            def __getitem__(self,grid):
                def launch(*args,**kw):calls.append((self.fn.__name__,grid,kw));return artifact
                return launch
        driver=SimpleNamespace(active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda n:dict(num_vectorcore=40))))
        triton=SimpleNamespace(__version__='fake',language=SimpleNamespace(float32='f32'),next_power_of_2=lambda n:1<<(n-1).bit_length(),cdiv=lambda a,b:(a+b-1)//b)
        mods={'torch_npu':SimpleNamespace(),'triton':triton,'triton.runtime':SimpleNamespace(driver=driver),'triton.runtime.driver':driver,
              'rms_backward_kernel':SimpleNamespace(_block_rms_norm_backward_kernel=Kernel('bwd'),zero_dx=Kernel('zero'),sum_dw=Kernel('sum'))}
        triton.runtime=mods['triton.runtime']
        empty,full=torch.empty,torch.full
        def cpu(fn):
            def call(*a,**kw):kw.pop('device',None);return fn(*a,**kw)
            return call
        with patch.dict(sys.modules,mods),patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True),patch.object(torch,'npu',SimpleNamespace(current_device=lambda:0),create=True),patch.object(torch,'empty',cpu(empty)),patch.object(torch,'full',cpu(full)):
            adapter=Adapter()
            for order in (('before','after'),('after','before')):
                for arm in order:
                    item=adapter.prepare(arm,spec(),make_case(spec()),Path('.'))
                    previous=len(compiles);out=item.run();self.assertEqual(len(compiles)-previous,3)
                    item.run();self.assertEqual(len(compiles)-previous,3)
                    self.assertEqual([x[0] for x in calls[-3:]],['zero','bwd','sum'])
                    self.assertEqual(calls[-2][2]['BLOCK_ROW'],4 if arm=='after' else 1)
                    self.assertEqual(item.metadata['grid'],[40]);self.assertEqual(len(item.compilation_state()),3)
                    self.assertEqual(len(out),2);self.assertEqual(item.output_storages[0][[0,-1]].tolist(),[-701.,-701.])
                self.assertNotEqual(adapter.prepared['before'].output_storages[0].data_ptr(),adapter.prepared['after'].output_storages[0].data_ptr())
            wide=dict(seed=99,rows=2,D=2048,offset=0,eps=1e-6)
            item=adapter.prepare('after',wide,make_case(wide),Path('.'));item.run()
            self.assertEqual(calls[-2][2]['BLOCK_ROW'],1)
            for failing in ('bwd','sum'):
                with self.subTest(failing_component=failing):
                    retry=adapter.prepare('after',spec(),make_case(spec()),Path('.'))
                    previous_calls=len(calls)
                    for qualify in (retry.compile_only,retry.run):
                        fail_next.append(failing)
                        with self.assertRaisesRegex(RuntimeError,'mock late compile failure'):
                            qualify()
                        self.assertEqual(retry.compilation_state(),[])
                        self.assertEqual(len(calls),previous_calls)
                    previous_compiles=len(compiles)
                    retry.run()
                    self.assertEqual(compiles[previous_compiles:],['zero','bwd','sum'])
                    self.assertEqual([c[0] for c in calls[previous_calls:]],['zero','bwd','sum'])
                    self.assertEqual(len(retry.compilation_state()),3)
                    previous_compiles=len(compiles)
                    retry.run()
                    self.assertEqual(len(compiles),previous_compiles)


    def test_external_cwd_cli_metadata_and_standalone(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);case=folder/'case.json'
            cmd=[sys.executable,str(DIRECTORY/'rms_backward_cli.py'),'--case',str(case),'--output',str(folder/'out'),
                 '--cache',str(folder/'cache'),'--provenance',str(folder/'provenance.json'),'--validate-only']
            for value in (dict(spec(),role='guard'),dict(seed=49,role='heldout',**spec()['parameters'])):
                case.write_text(json.dumps(value));p=subprocess.run(cmd,cwd=folder,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['role'],value['role'])
            case.write_text(json.dumps(dict(seed=4,rows=0,D=17,offset=1,eps=1e-6)))
            self.assertNotEqual(subprocess.run(cmd,cwd=folder,capture_output=True).returncode,0)
            self.assertFalse((folder/'cache').exists());self.assertFalse((folder/'out').exists())
        s=importlib.util.spec_from_file_location('standalone',DIRECTORY/'rms_backward.py')
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        with self.assertRaises(ValueError):m.factory().prepare('bad',spec(),make_case(spec()),Path('.'))


if __name__=='__main__':unittest.main()
