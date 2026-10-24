import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'examples/optimization-library/triton'
sys.path[:0]=[str(DIRECTORY),str(ROOT/'scripts')]
from sinkhorn_partial_contract import make_case, validate_inputs, check_outputs
from sinkhorn_partial import Adapter
from optimization_pair import fixture


def spec(n=13):
    return dict(seed=113,parameters=dict(num_blocks=n))


class PartialTests(unittest.TestCase):
    def test_scalar_oracle_bound_and_sentinels(self):
        f=make_case(spec())
        actual=[]
        for name,c in (('scale',2),('base',8)):
            value=f['inputs'][name].clone()
            for j in range(c):
                value[j]=sum(float(x) for x in f['inputs']['partials_'+name][:,j])
            actual.append(value)
        result=check_outputs(tuple(actual),f)
        self.assertTrue(result['passed'])
        self.assertEqual([len(x['bounds']) for x in result['outputs']],[2,8])
        actual[0][2]=0.  # numeric equality does not excuse changed negative-zero sentinel.
        self.assertFalse(check_outputs(tuple(actual),f)['passed'])

    def test_cancellation(self):
        s=dict(seed=151,num_blocks=19,cancellation_and_zeros=True)
        f=make_case(s)
        self.assertTrue(check_outputs(tuple(v.float() for v in f['expected']),f)['passed'])

    def test_output_rejections(self):
        f=make_case(spec()); output=tuple(v.float() for v in f['expected'])
        for bad in (output[:1],output+output,list(output),(output[0].double(),output[1]),
                    (output[0].flatten()[:2],output[1]),(output[0]+.1,output[1])):
            self.assertFalse(check_outputs(bad,f)['passed'])

    def test_domain_layout_alias_rejections(self):
        for replacement in (torch.full((13,2),1.01),torch.full((13,2),2**-21),
                            torch.full((13,2),float('nan')),torch.zeros(13,2,dtype=torch.float64),
                            torch.zeros(14,2)[1:],torch.zeros(2,13).T):
            f=make_case(spec());f['inputs']['partials_scale']=replacement
            with self.assertRaises(ValueError):
                Adapter().prepare('after',spec(),f,Path('.'))
        f=make_case(spec());f['inputs']['scale']=f['inputs']['base'][:3]
        with self.assertRaises(ValueError):validate_inputs(f['inputs'],spec())

    def test_cache_once_and_precache_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'cache'
            _,a=fixture(Adapter(),spec(),cache,torch)
            _,b=fixture(Adapter(),spec(),cache,torch)
            self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0))
            for n in (0,4097,True):
                with self.assertRaises(ValueError):fixture(Adapter(),spec(n),Path(directory)/'invalid',torch)
            self.assertFalse((Path(directory)/'invalid').exists())

    def test_prepared_reuse_compile_first_and_fallback(self):
        calls=[]
        artifact=SimpleNamespace(hash='a'*64,asm={'npubin':b'code'})
        class Kernel:
            fn=SimpleNamespace(__name__='reduce')
            def warmup(self,*args,**kwargs):
                calls.append(('compile',kwargs));return artifact
            def __getitem__(self,grid):
                def launch(*args,**kwargs):calls.append(('launch',kwargs));return artifact
                return launch
        original=torch.empty
        def empty(*args,**kw):kw.pop('device',None);return original(*args,**kw)
        modules=dict(torch_npu=SimpleNamespace(),triton=SimpleNamespace(__version__='fake'),
                     sinkhorn_partial_kernel=SimpleNamespace(deterministic_reduce_kernel=Kernel()))
        with patch.dict(sys.modules,modules),patch.object(torch.Tensor,'npu',lambda s:s.clone(),create=True),patch.object(torch,'empty',empty):
            a=Adapter();item=a.prepare('after',spec(),make_case(spec()),Path('.'))
            self.assertEqual(calls,[])
            first=item.run();second=item.run()
            self.assertEqual([r[0] for r in calls],['compile','launch','launch'])
            self.assertIs(first[0],second[0]);self.assertTrue(item.metadata['vector'])
            self.assertEqual(first[0][2:].view(torch.int32).tolist(),[-2147483648])
            self.assertEqual(item.output_storages[0][[0,-1]].tolist(),[-701.,-701.])
            self.assertEqual(len(item.compilation_state()),1)
            f=spec(1031)
            before=a.prepare('before',f,make_case(f),Path('.'))
            after=a.prepare('after',f,make_case(f),Path('.'))
            self.assertEqual(before.metadata['flags'],after.metadata['flags'])
            self.assertFalse(after.metadata['vector'])
            with self.assertRaises(ValueError):a.prepare('wrong',spec(),make_case(spec()),Path('.'))

    def test_external_cli_flat_nested(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);case=p/'case.json'
            command=[sys.executable,str(DIRECTORY/'sinkhorn_partial_cli.py'),'--case',str(case),
                     '--output',str(p/'out'),'--cache',str(p/'cache'),'--provenance',str(p/'prov'),'--validate-only']
            for s in (spec(),dict(seed=113,num_blocks=13,expected='ok',role='guard')):
                case.write_text(json.dumps(s))
                result=subprocess.run(command,cwd=p,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            case.write_text(json.dumps(spec(0)))
            self.assertNotEqual(subprocess.run(command,cwd=p,capture_output=True).returncode,0)
            self.assertFalse((p/'cache').exists());self.assertFalse((p/'out').exists())

    def test_standalone_import(self):
        s=importlib.util.spec_from_file_location('external_partial',DIRECTORY/'sinkhorn_partial.py')
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        self.assertIsInstance(m.factory().make_case(spec()),dict)


if __name__=='__main__':unittest.main()
