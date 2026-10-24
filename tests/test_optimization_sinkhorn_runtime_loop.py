import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/'examples/optimization-library/triton'
sys.path[:0] = [str(DIRECTORY),str(ROOT/'scripts')]
from sinkhorn_runtime_contract import dimensions, make_case
from sinkhorn_contract import reference
from sinkhorn_runtime_loop import Adapter
from optimization_pair import fixture, run_pair


def case(compound=False):
    return dict(seed=151,role='guard',parameters=dict(BS=3,iters=2,eps=1e-5,
                **(dict(counts_sequence=[2,5]) if compound else {})))


class RuntimeTests(unittest.TestCase):
    def test_compound_trajectory_cached_once(self):
        import sinkhorn_runtime_contract as contract
        with tempfile.TemporaryDirectory() as folder,patch.object(contract.torch,'softmax',wraps=torch.softmax) as softmax:
            a,first = fixture(Adapter(),case(True),Path(folder),torch)
            b,second = fixture(Adapter(),case(True),Path(folder),torch)
            self.assertEqual((first['oracle_calls'],second['oracle_calls']),(1,0))
            self.assertEqual(softmax.call_count,1)
            self.assertEqual(len(a['expected']),2)
            self.assertEqual(a['expected'][0][0].data_ptr(),a['expected'][1][0].data_ptr())
            for index,count in enumerate((2,5)):
                wanted = reference(a['inputs'],count,1e-5)
                for got,expected,cached in zip(a['expected'][index],wanted,b['expected'][index]):
                    self.assertTrue(torch.equal(got,expected));self.assertTrue(torch.equal(got,cached))

    def test_public_prepared_and_pair_lifecycle(self):
        launches = []
        class Kernel:
            def __init__(self,name,runtime=False):self.fn=SimpleNamespace(__name__=name);self.runtime=runtime
            def artifact(self,count=None):
                return SimpleNamespace(hash=('a' if self.runtime or count is None else str(count))*64,asm={'npubin':b'code'})
            def warmup(self,*args,**flags):return self.artifact(flags.get('SINKHORN_ITERS'))
            def __getitem__(self,grid):
                def run(*args,**flags):
                    launches.append((self.fn.__name__,flags))
                    mixes,scale,base = args[:3]
                    count=flags.get('SINKHORN_ITERS',1);eps=args[-2]
                    outputs = reference(dict(mixes=mixes,scale=scale,base=base),count,eps)
                    if 'prepost' in self.fn.__name__:
                        args[3].copy_(outputs[0]);args[4].copy_(outputs[1])
                    else:
                        args[3][:,:,:4].copy_(outputs[2]);args[3][:,:,4:].zero_()
                    return self.artifact(flags.get('SINKHORN_ITERS'))
                return run
        original = torch.empty
        def allocate(*args,**kwargs):kwargs.pop('device',None);return original(*args,**kwargs)
        modules = dict(torch_npu=SimpleNamespace(),triton=SimpleNamespace(__version__='cpu-double'),
            sinkhorn_kernel=SimpleNamespace(hc_split_sinkhorn_forward_prepost_kernel=Kernel('prepost'),
                hc_split_sinkhorn_forward_comb_kernel=Kernel('static')),
            sinkhorn_runtime_kernel=SimpleNamespace(hc_runtime_sinkhorn_forward_comb_kernel=Kernel('runtime',True)))
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,modules), \
                patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True),patch.object(torch,'empty',allocate):
            root=Path(folder);adapter=Adapter()
            for index,order in enumerate((('before','after'),('after','before'))):
                for compound in (False,True):
                    spec=case(compound)
                    report=run_pair(adapter,spec,root/f'{index}-{compound}',root/'cache',{},order=order,torch=torch)
                    self.assertEqual(report['status'],'complete',report)
                    self.assertEqual(report['counts']['correctness_calls'],2)
                    self.assertEqual(report['counts']['profiled_calls'],0)
                    diag=report['arms']['after']['correctness']['diagnostics']
                    self.assertEqual(len(diag['components']),2 if compound else 1)
                    self.assertEqual(diag['runtime_variant_unified'],True if compound else None)
                    self.assertEqual(adapter.pair_bits,{})
            value=make_case(case(True));prepared=adapter.prepare('after',case(True),value,root/'separate'/'after')
            count=len(launches);prepared.compile_only();self.assertEqual(len(launches),count)
            first=prepared.run();self.assertIs(first,prepared.run());self.assertEqual(len(first),2)
            self.assertNotEqual(first[0][0].data_ptr(),first[1][0].data_ptr())
            self.assertTrue(all(c.backing.stride()==(32,8,1) for c in prepared.components))
            self.assertTrue(adapter.validate('after',first,value,case(True))['passed'])
            prepared.components[1].backing[0,0,4]=-0.
            self.assertFalse(adapter.validate('after',first,value,case(True))['passed'])
            self.assertFalse(adapter.validate('after',first[:1],value,case(True))['passed'])
            self.assertTrue(all(flags.get('BLOCK_ALIGN',8)==8 for _,flags in launches))

    def test_equal_logits_zero_loop_and_bad_cases(self):
        value=make_case(dict(seed=152,BS=2,iters=1,eps=.001,zero_scale_equal_logits=True))
        self.assertTrue(torch.equal(value['expected'][2],torch.full((2,4,4),.251/1.005)))
        for changes in (dict(iters=0),dict(iters=-1),dict(counts_sequence=[2,21]),dict(counts_sequence=[5,2])):
            spec=case();spec['parameters'].update(changes)
            with tempfile.TemporaryDirectory() as folder:
                cache=Path(folder)/'absent'
                with self.assertRaises(ValueError):fixture(Adapter(),spec,cache,torch)
                self.assertFalse(cache.exists())

    def test_external_cli_flat_and_nested(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'case.json'
            cmd=[sys.executable,str(DIRECTORY/'sinkhorn_runtime_loop_cli.py'),'--case',str(source),
                '--output',str(root/'out'),'--cache',str(root/'cache'),'--provenance',str(root/'prov'),'--validate-only']
            for spec in (case(),case(True),dict(seed=153,BS=7,iters=3,eps=1e-6,expected='metadata',role='heldout')):
                source.write_text(json.dumps(spec));result=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            spec=case();spec['parameters']['iters']=0;source.write_text(json.dumps(spec))
            self.assertNotEqual(subprocess.run(cmd,cwd=root,capture_output=True).returncode,0)
            self.assertFalse((root/'cache').exists());self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
