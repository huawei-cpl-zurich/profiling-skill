import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from scaled_mm_contract import make_case, check_fixture, reference
from scaled_mm import Adapter, StudyUnsupported
from optimization_pair import fixture


def spec(mode='scalar',bias=True):
    return dict(case_id='unit',role='gallery',seed=1927,parameters=dict(M=3,N=7,K=128,
                scale_mode=mode,exp_a=-3,exp_b=2,scale_a=.3,scale_b=.7,bias=bias))


class ScaledContractTests(unittest.TestCase):
    def test_independent_integer_dot_and_order(self):
        for mode in ('scalar','row_col','nonpower_scalar'):
            data=make_case(spec(mode)); check_fixture(spec(mode),data)
            expected=torch.empty_like(data['expected'])
            for i in range(3):
                for j in range(7):
                    acc=sum(int(data['a'][i,q])*int(data['b'][q,j]) for q in range(128))
                    v=data['sa'][i if mode=='row_col' else 0]*torch.tensor(acc,dtype=torch.float32)
                    v=data['sb'][j if mode=='row_col' else 0]*v
                    expected[i,j]=v+data['bias'][j]
            self.assertTrue(torch.equal(expected.view(torch.int32),data['expected'].view(torch.int32)))

    def test_rejection_before_backend_import(self):
        good=make_case(spec())
        for key,value in [('K',129),('M',129),('scale_mode','other'),('bias',1),('exp_a',9)]:
            bad=spec(); bad['parameters'][key]=value
            with self.assertRaises(ValueError): check_fixture(bad,good)
        for key,value in [('a',good['a'].float()),('sa',good['sa'].expand(3)),('bias',None)]:
            data=dict(good);data[key]=value
            with self.assertRaises(ValueError):check_fixture(spec(),data)
        data=dict(good); data['extra']=1
        with self.assertRaises(ValueError):check_fixture(spec(),data)
        data=copy.deepcopy(good);data['a'][0,0]=9
        with self.assertRaises(ValueError):check_fixture(spec(),data)

    def test_cache_once_and_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Adapter(); x,first=fixture(a,spec(),Path(tmp),torch)
            y,second=fixture(a,spec(),Path(tmp),torch)
            self.assertEqual((first['oracle_calls'],second['oracle_calls']),(1,0))
            self.assertTrue(torch.equal(x['expected'],y['expected']))
            other=spec();other['role']='heldout'
            _,third=fixture(a,other,Path(tmp),torch)
            self.assertEqual(third['oracle_calls'],1)

    def test_external_working_directory_import(self):
        import os
        import subprocess
        env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([sys.executable,'-c',
                "from scaled_mm import Adapter, StudyUnsupported; from scaled_mm_contract import dimensions; "
                "assert Adapter().prepared == {}; "
                "assert dimensions(dict(M=1,N=2,K=128,seed=17,scale_mode='scalar',exp_a=0,exp_b=0,bias=False))[:3] == (1,2,128)"],
                cwd=cwd,env=env,check=True)

    def test_flat_parameters_and_no_bias(self):
        s=spec(bias=False);flat={**s['parameters'],'seed':s['seed']}
        data=make_case(flat);check_fixture(flat,data)
        self.assertIsNone(data['bias'])
        self.assertTrue(torch.equal(data['expected'],reference(data['a'],data['b'],data['sa'],data['sb'],None)))

class PreparedInterfaceTests(unittest.TestCase):
    def test_explicit_compile_inspection_cannot_launch(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        calls=[]
        artifact=SimpleNamespace(hash='a'*64,asm={'npubin':b'ELF'})
        class Kernel:
            def warmup(self,*args,**kwargs):
                calls.append('compile'); return artifact
            def __getitem__(self,grid):
                def run(*args,**kwargs):
                    calls.append('launch'); return artifact
                return run
        modules={'torch_npu':SimpleNamespace(), 'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),
                 'triton.language':SimpleNamespace(int32='i32'),
                 'scaled_mm_kernel':SimpleNamespace(scaled_mm_kernel=Kernel())}
        with patch.dict(sys.modules,modules), patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True):
            adapter=Adapter(compile_inspection=True); item=adapter.prepare('after',spec(),make_case(spec()),Path('.'))
            self.assertEqual(calls,[])
            item.compile_only()
            for _ in range(2):
                with self.assertRaises(StudyUnsupported):item.run()
            self.assertEqual(calls,['compile'])
            self.assertEqual(item.compilation_state()[0]['hash'],'a'*64)
            self.assertIn('npubin',item.compilation_state()[0]['binaries'])

    def test_default_cli_and_capture_fail_before_work(self):
        import os
        import subprocess
        from unittest.mock import patch
        with patch.dict(sys.modules,{'torch_npu':None,'triton':None}):
            with self.assertRaises(StudyUnsupported):
                Adapter().prepare('after',spec(),make_case(spec()),Path('.'))
        calls=[]
        with self.assertRaises(StudyUnsupported):Adapter().capture(lambda:calls.append('forward'),None,None)
        self.assertEqual(calls,[])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
                '--adapter','scaled_mm:factory','--case',str(folder/'absent-case'),
                '--provenance',str(folder/'absent-provenance'),'--cache',str(folder/'cache'),
                '--output',str(folder/'out')],cwd=folder,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('T03 failed study',result.stderr)
            self.assertFalse((folder/'cache').exists())
            self.assertFalse((folder/'out').exists())

    def test_numeric_mode_canary_and_immutable_rejections(self):
        from types import SimpleNamespace
        data=make_case(spec());adapter=Adapter()
        item=SimpleNamespace(spec=spec(),mode=torch.ones(1,dtype=torch.int32),
            out_storage=torch.tensor([-76543.,-76543.]),mode_storage=torch.tensor([-9876,-9876]),
            inputs={k:v.clone() for k,v in data.items() if k!='expected' and v is not None})
        adapter.prepared['after']=item
        actual=data['expected'].clone()
        self.assertTrue(adapter.validate('after',actual,data,spec())['passed'])
        item.mode[0]=2
        self.assertFalse(adapter.validate('after',actual,data,spec())['passed'])
        item.mode[0]=1; item.out_storage[0]=0
        self.assertFalse(adapter.validate('after',actual,data,spec())['passed'])
        item.out_storage[0]=-76543.;item.inputs['a'][0,0]+=1
        self.assertFalse(adapter.validate('after',actual,data,spec())['passed'])
        item.inputs['a']=data['a'].clone();actual[0,0]=torch.nextafter(actual[0,0],torch.tensor(float('inf')))
        self.assertFalse(adapter.validate('after',actual,data,spec())['passed'])
        other=spec();other['seed']+=1
        with self.assertRaises(ValueError):adapter.validate('after',actual,data,other)

if __name__=='__main__':unittest.main()
