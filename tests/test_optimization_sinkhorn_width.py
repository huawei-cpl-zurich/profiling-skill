import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'examples/optimization-library/triton'
sys.path[:0]=[str(DIRECTORY),str(ROOT/'scripts')]
from sinkhorn_contract import make_case, check_outputs, validate_inputs
from sinkhorn_width import Adapter
from optimization_pair import fixture


def spec(**changes):
    return dict(seed=97,parameters=dict(BS=5,iters=3,eps=1e-5,**changes))


class SinkhornTests(unittest.TestCase):
    def test_source_initial_smoothing_equal_logits(self):
        case=dict(seed=62,BS=2,iters=1,eps=1e-3,zero_scale_equal_logits=True)
        value=make_case(case); pre,post,comb=value['expected']
        self.assertTrue(torch.equal(pre,torch.full_like(pre,.501)))
        self.assertTrue(torch.equal(post,torch.ones_like(post)))
        wanted=torch.full_like(comb,.251/(1.004+.001))
        self.assertTrue(torch.equal(comb,wanted))
        self.assertFalse(torch.equal(comb,torch.full_like(comb,.25)))

    def test_independent_scalar_oracle(self):
        import math
        value=make_case(spec()); x,s,b=(value['inputs'][k] for k in ('mixes','scale','base'))
        pre,post,comb=value['expected']
        for batch in range(5):
            rows=[]
            for r in range(4):
                z=[float(x[batch,8+r*4+c])*float(s[2])+float(b[8+r*4+c]) for c in range(4)]
                ex=[math.exp(v-max(z)) for v in z]
                rows.append([v/sum(ex)+1e-5 for v in ex])
            def colnorm(a):
                return [[a[r][c]/(sum(a[k][c] for k in range(4))+1e-5) for c in range(4)] for r in range(4)]
            rows=colnorm(rows)
            for _ in range(2):
                rows=colnorm([[v/(sum(row)+1e-5) for v in row] for row in rows])
            self.assertTrue(torch.equal(comb[batch],torch.tensor(rows,dtype=torch.float32)))
            for j in range(4):
                expected=1/(1+math.exp(-(float(x[batch,j])*float(s[0])+float(b[j]))))+1e-5
                self.assertEqual(pre[batch,j],torch.tensor(expected,dtype=torch.float32))

    def test_contract_and_cache(self):
        expected=make_case(spec())['expected']
        for bad in (expected[:2],expected+expected,list(expected),(expected[0].double(),*expected[1:]),(expected[0]+.01,*expected[1:])):
            self.assertFalse(check_outputs(bad,expected)['passed'])
        with tempfile.TemporaryDirectory() as folder:
            a,first=fixture(Adapter(),spec(),Path(folder)/'cache',torch)
            b,second=fixture(Adapter(),spec(),Path(folder)/'cache',torch)
            self.assertEqual((first['oracle_calls'],second['oracle_calls']),(1,0))
            self.assertTrue(torch.equal(a['expected'][2],b['expected'][2]))
            with self.assertRaises(ValueError):
                fixture(Adapter(),dict(seed=5,BS=0,iters=1,eps=1e-6),Path(folder)/'bad',torch)
            self.assertFalse((Path(folder)/'bad').exists())
        for key,replacement in [('scale',torch.ones(3,dtype=torch.float64)),('base',torch.full((24,),float('nan'))),('mixes',torch.zeros(5,25))]:
            value=make_case(spec());value['inputs'][key]=replacement
            with self.assertRaises(ValueError):Adapter().prepare('after',spec(),value,Path('.'))

    def test_prepared_interface_reuses_outputs_and_two_launches(self):
        launches=[]
        artifact=SimpleNamespace(hash='b'*64,asm={'npubin':b'code'})
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*args,**kwargs):return artifact
            def __getitem__(self,grid):
                def run(*args,**kwargs):launches.append((self.fn.__name__,grid,kwargs));return artifact
                return run
        original=torch.empty
        def allocate(*args,**kwargs):kwargs.pop('device',None);return original(*args,**kwargs)
        modules=dict(torch_npu=SimpleNamespace(),triton=SimpleNamespace(__version__='fake'),
            sinkhorn_kernel=SimpleNamespace(hc_split_sinkhorn_forward_prepost_kernel=Kernel('prepost'),hc_split_sinkhorn_forward_comb_kernel=Kernel('comb')))
        with patch.dict(sys.modules,modules),patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True),patch.object(torch,'empty',allocate):
            adapter=Adapter(); before=adapter.prepare('before',spec(),make_case(spec()),Path('.'));after=adapter.prepare('after',spec(),make_case(spec()),Path('.'))
            self.assertEqual(len(before.compile_only()),2);self.assertEqual(launches,[])
            a=before.run();self.assertIs(a,before.run());b=after.run()
            self.assertEqual(len(launches),6);self.assertEqual(launches[1][2]['BLOCK_ALIGN'],8);self.assertEqual(launches[-1][2]['BLOCK_ALIGN'],4)
            self.assertEqual(a[2].stride(),(32,8,1));self.assertEqual(b[2].stride(),(32,8,1))
            self.assertNotEqual(a[0].data_ptr(),b[0].data_ptr())
            self.assertEqual(before.output_storages[0][[0,-1]].tolist(),[-701.,-701.])
            self.assertIsInstance(before.compilation_state(),list)
            for output,wanted in zip(b,make_case(spec())['expected']):output.copy_(wanted)
            after.backing[:,:,4:]=0
            self.assertTrue(adapter.validate('after',b,make_case(spec()),spec())['passed'])
            after.backing[0,0,4]=-0.
            self.assertFalse(adapter.validate('after',b,make_case(spec()),spec())['passed'])

    def test_external_cli_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);case=folder/'case.json'
            command=[sys.executable,str(DIRECTORY/'sinkhorn_width_cli.py'),'--case',str(case),'--output',str(folder/'out'),'--cache',str(folder/'cache'),'--provenance',str(folder/'prov'),'--validate-only']
            for value in (spec(),dict(seed=97,role='gallery',expected='metadata',**spec()['parameters'])):
                case.write_text(json.dumps(value));result=subprocess.run(command,cwd=folder,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            for change in (dict(BS=0),dict(iters=0),dict(eps=.01)):
                value=spec();value['parameters'].update(change);case.write_text(json.dumps(value))
                self.assertNotEqual(subprocess.run(command,cwd=folder,capture_output=True).returncode,0)
            self.assertFalse((folder/'out').exists());self.assertFalse((folder/'cache').exists())

if __name__=='__main__':unittest.main()
