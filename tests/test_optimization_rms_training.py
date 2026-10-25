import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/'examples/optimization-library/triton'
sys.path[:0]=[str(DIRECTORY),str(ROOT/'scripts')]
from rms_training import Adapter
from rms_training_contract import make_case,validate_inputs,check_outputs
from optimization_pair import fixture


def spec():return dict(seed=109,parameters=dict(rows=5,D=11,offset=1))


class RMSTrainingTests(unittest.TestCase):
    def test_scalar_full_fp64_oracle(self):
        value=make_case(spec());x,dy,w=(value['inputs'][k] for k in ('X','DY','W'));y,dx,dw=value['expected']
        for i in range(5):
            r=1/math.sqrt(sum(float(v)**2 for v in x[i])/11+1e-6)
            dot=sum(float(dy[i,j])*(float(w[j])+1)*float(x[i,j]) for j in range(11))/11
            for j in range(11):
                self.assertAlmostEqual(float(y[i,j]),float(x[i,j])*r*(float(w[j])+1),places=12)
                self.assertAlmostEqual(float(dx[i,j]),r*float(dy[i,j])*(float(w[j])+1)-r**3*float(x[i,j])*dot,places=12)
        for j in range(11):
            expected=sum(float(dy[i,j])*float(x[i,j])/math.sqrt(sum(float(v)**2 for v in x[i])/11+1e-6) for i in range(5))
            self.assertAlmostEqual(float(dw[j]),expected,places=12)
        zero=make_case(dict(seed=111,rows=3,D=19,offset=0,zero_cancel=True))
        torch.testing.assert_close(zero['expected'][0][0],torch.zeros(19,dtype=torch.float64))
        torch.testing.assert_close(zero['expected'][1][0],zero['inputs']['DY'][0].double()*zero['inputs']['W'].double()*1000)

    def test_admission_cache_and_outputs(self):
        value=make_case(spec());expected=value['expected'];good=tuple(v.float() for v in expected)
        self.assertTrue(check_outputs(good,expected)['passed'])
        for bad in (good[:2],good+good,list(good),(good[0].double(),*good[1:]),
                    (good[0],good[0],good[2]),(good[0]+.1,*good[1:])):
            self.assertFalse(check_outputs(bad,expected)['passed'])
        for key,v in [('X',torch.zeros(5,11,dtype=torch.float64)),('DY',torch.full((5,11),float('inf'))),('W',torch.full((11,),.1))]:
            bad=make_case(spec());bad['inputs'][key]=v
            with self.assertRaises(ValueError):Adapter().prepare('after',spec(),bad,Path('.'))
        bad=make_case(spec());bad['inputs']['DY']=bad['inputs']['X']
        with self.assertRaises(ValueError):validate_inputs(bad['inputs'],spec())
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache';_,a=fixture(Adapter(),spec(),cache,torch);_,b=fixture(Adapter(),spec(),cache,torch)
            self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0))
            for eps in (0,-1,1e-5):
                with self.assertRaises(ValueError):fixture(Adapter(),dict(seed=117,rows=2,D=13,offset=0,eps=eps),Path(tmp)/'bad',torch)
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_full_api_compile_retry_probe_routes_and_pair_reuse(self):
        calls=[];compiles=[];fail=[];artifact=SimpleNamespace(hash='c'*64,asm={'npubin':b'code'})
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*a,**kw):
                compiles.append(self.fn.__name__)
                if fail and fail[0]==self.fn.__name__:fail.pop(0);raise RuntimeError('late qualification failure')
                return artifact
            def __getitem__(self,grid):
                def run(*a,**kw):
                    name=self.fn.__name__;calls.append((name,grid,kw))
                    if name=='forward':
                        y,_,x,_,w,_,stats,_,d,eps,offset,*_=a
                        r=(x.square().sum(-1)/d+eps).rsqrt();y.copy_(x*r[:,None]*(w+offset))
                        if kw['SAVE_STATS']:stats.copy_(r)
                    elif name=='zero':a[0].zero_()
                    elif name=='backward':
                        dy,_,dx,_,x,_,_,w,_,stats,_,partial,_,rows,d,offset,eps,*_=a
                        r=(x.square().sum(-1)/d+eps).rsqrt() if kw['RECOMPUTE'] else stats
                        m=dy*(w+offset);dx.copy_(r[:,None]*m-r[:,None]**3*x*(m*x).mean(-1,keepdim=True))
                        partial.zero_();partial[0].copy_((dy*x*r[:,None]).sum(0))
                    else:a[1].copy_(a[0].sum(0))
                    return artifact
                return run
        driver=SimpleNamespace(active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda n:dict(num_vectorcore=40))))
        triton=SimpleNamespace(language=SimpleNamespace(float32='f32'),next_power_of_2=lambda n:1<<(n-1).bit_length(),cdiv=lambda a,b:(a+b-1)//b,runtime=SimpleNamespace(driver=driver))
        mods={'torch_npu':SimpleNamespace(),'triton':triton,'triton.runtime':triton.runtime,'triton.runtime.driver':driver,
              'rms_training_kernel':SimpleNamespace(_rms_norm_forward_kernel=Kernel('forward'),_block_rms_norm_backward_kernel=Kernel('backward'),zero_dx=Kernel('zero'),sum_dw=Kernel('sum'))}
        empty=torch.empty
        def cpu_empty(*a,**kw):kw.pop('device',None);return empty(*a,**kw)
        with patch.dict(sys.modules,mods),patch.object(torch.Tensor,'npu',lambda v:v.clone(),create=True),patch.object(torch,'npu',SimpleNamespace(current_device=lambda:0),create=True),patch.object(torch,'empty',cpu_empty):
            a=Adapter();probe=a.probe('after',spec())
            for compile_first in (False,True):
                if compile_first:probe.compile_only()
                previous=len(compiles)
                with self.assertRaisesRegex(RuntimeError,'compile-only probe cannot execute'):probe.run()
                self.assertEqual(len(compiles),previous);self.assertEqual(calls,[])
            for order in (('before','after'),('after','before')):
                value=make_case(spec())
                for arm in order:
                    item=a.prepare(arm,spec(),value,Path('.'));previous=len(calls)
                    fail.append('sum')
                    with self.assertRaises(RuntimeError):item.run()
                    self.assertEqual(item.compilation_state(),[]);self.assertEqual(len(calls),previous)
                    out=item.run();self.assertEqual([c[0] for c in calls[-4:]],['forward','zero','backward','sum'])
                    self.assertEqual(calls[-2][2]['BLOCK_ROW'],1)
                    self.assertEqual(calls[-4][2]['SAVE_STATS'],arm=='before')
                    self.assertEqual(calls[-2][2]['RECOMPUTE'],arm=='after')
                    count=len(compiles);item.run();self.assertEqual(len(compiles),count)
                    result=a.validate(arm,out,value,spec());self.assertTrue(result['passed'],result)
                self.assertTrue(result['paired_y_bits']);self.assertFalse(result['internal_rstd_observed'])
                item.output_storages[1][0]=0;self.assertFalse(a.validate(arm,out,value,spec())['passed'])
            wide=dict(seed=113,rows=4,D=2048,offset=1)
            for arm in ('before','after'):
                item=a.prepare(arm,wide,make_case(wide),Path('.'));item.run()
                self.assertTrue(calls[-4][2]['SAVE_STATS']);self.assertFalse(calls[-2][2]['RECOMPUTE'])
            value=make_case(spec());first=a.prepare('before',spec(),value,Path('.'));a.validate('before',first.run(),value,spec())
            second=a.prepare('after',spec(),value,Path('.'));out=second.run();out[0][0,0]=torch.nextafter(out[0][0,0],torch.tensor(float('inf')))
            self.assertFalse(a.validate('after',out,value,spec())['paired_y_bits'])
            fresh=a.prepare('after',spec(),value,Path('.'));self.assertIsNone(a.validate('after',fresh.run(),value,spec())['paired_y_bits'])
            fresh.inputs['X'][0,0]=0;self.assertFalse(a.validate('after',out,value,spec())['immutable'])

    def test_external_cwd_exploratory_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);case=folder/'case.json';cmd=[sys.executable,str(DIRECTORY/'rms_training_cli.py'),'--case',str(case),'--output',str(folder/'out'),'--cache',str(folder/'cache'),'--provenance',str(folder/'p.json'),'--validate-only']
            for value in (dict(spec(),role='guard'),dict(seed=109,role='heldout',**spec()['parameters'])):
                case.write_text(json.dumps(value));p=subprocess.run(cmd,cwd=folder,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['role'],value['role'])
            case.write_text(json.dumps(dict(seed=109,rows=3,D=17,offset=1,eps=0)))
            self.assertNotEqual(subprocess.run(cmd,cwd=folder,capture_output=True).returncode,0)
            self.assertFalse((folder/'out').exists());self.assertFalse((folder/'cache').exists())


if __name__=='__main__':unittest.main()
