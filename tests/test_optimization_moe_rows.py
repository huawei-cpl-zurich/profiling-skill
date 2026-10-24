import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
from moe_rows_contract import make_case,reference,validate_inputs,dimensions
import torch


class RowMovement(unittest.TestCase):
    def spec(self, **params):
        return dict(seed=9193,parameters=dict(N=13,D=16,forward=True,with_probs=True,
                    chunks=5,map_kind='shuffled_chunks',**params))

    def test_bitwise_roundtrip(self):
        spec=self.spec(); data=make_case(spec)
        moved=data['expected']
        back=reference(dict(x=moved[0],probs=moved[1],row_map=data['inputs']['row_map']),False)
        for original,result in zip((data['inputs']['x'],data['inputs']['probs']),back):
            self.assertTrue(torch.equal(original.view(torch.int32),result.view(torch.int32)))

    def test_inverse_without_probs(self):
        spec=self.spec();spec['parameters'].update(forward=False,with_probs=False,N=9,D=32)
        fixture=make_case(spec)
        self.assertIsNone(fixture['expected'][1])
        self.assertTrue(torch.equal(fixture['expected'][0].view(torch.int32),
            fixture['inputs']['x'][fixture['inputs']['row_map'].long()].view(torch.int32)))

    def test_admission(self):
        spec=self.spec();data=make_case(spec)
        for bad in (-1,13,0):
            broken={k:v.clone() for k,v in data['inputs'].items()}
            broken['row_map'][1]=bad
            if bad==0 and data['inputs']['row_map'][1]==0: broken['row_map'][0]=0
            with self.assertRaises(ValueError): validate_inputs(broken,spec)
        bad={**data['inputs'],'x':data['inputs']['x'].double()}
        with self.assertRaises(ValueError): validate_inputs(bad,spec)
        for n,d in ((2**24,8),(5,3),(2**23,4096)):
            changed=self.spec();changed['parameters'].update(N=n,D=d)
            with self.assertRaises(ValueError): dimensions(changed)

    def test_empty_and_flat_metadata(self):
        spec=dict(seed=9921,N=0,D=16,with_probs=False,forward=False,map_kind='empty',role='ordinary')
        data=make_case(spec)
        self.assertEqual(tuple(data['expected'][0].shape),(0,16))
        path=ROOT/'examples/optimization-library/triton/moe_rows.py'
        module_spec=importlib.util.spec_from_file_location('outside_adapter',path)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        self.assertTrue(module.Adapter.correctness_only(spec))
        self.assertIsNone(data['expected'][1])



class PublicCompilationBoundary(unittest.TestCase):
    """Exercise the documented runner using CPU doubles, never a Triton compiler."""
    def exercise(self, lowering_fails):
        import tempfile
        import types
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'scripts'))
        from optimization_pair import run_pair
        from moe_rows import Adapter
        events=[]
        class Kernel:
            def __init__(self,name):
                self.fn=types.SimpleNamespace(__name__=name)
                self.artifact=types.SimpleNamespace(hash=('a' if name=='before' else 'b')*64,
                    asm={'npubin':name.encode()})
                self.compiles=[]
            def warmup(self,*args,**flags):
                self.compiles.append(flags['USE_ELEMENT'])
                events.append(('compile',self.fn.__name__,flags['USE_ELEMENT']))
                if lowering_fails and self.fn.__name__=='before' and flags['USE_ELEMENT']:
                    raise RuntimeError('extension lowering unsupported')
                return self.artifact
            def __getitem__(self,grid):
                def launch(x,y,m,p,q,*args,**flags):
                    events.append(('launch',self.fn.__name__,flags['USE_ELEMENT']))
                    expected=reference(dict(x=x,row_map=m,probs=p),flags['FORWARD'])
                    y.copy_(expected[0]);q.copy_(expected[1])
                    return self.artifact
                return launch
        before,after=Kernel('before'),Kernel('after')
        modules=dict(torch_npu=types.ModuleType('torch_npu'),
            triton=types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda x:1<<(x-1).bit_length()),
            moe_rows_kernel=types.SimpleNamespace(moe_rows_before=before,moe_rows_after=after,
                get_element=object(),GET_ELEMENT_SOURCE='importable extension'))
        full=torch.full
        def cpu_full(*args,**kwargs):
            kwargs.pop('device',None);return full(*args,**kwargs)
        adapter=Adapter()
        # Only the physical backend/validator is doubled; public prepare/run and
        # generic fixture, source binding, correctness accounting remain real.
        def validate(arm,actual,fixture,spec):
            return dict(passed=all(torch.equal(a.view(torch.int32),b.view(torch.int32))
                                  for a,b in zip(actual,fixture['expected'])))
        adapter.validate=validate
        spec=dict(seed=8631,role='ordinary',parameters=dict(N=11,D=8,forward=True,
            with_probs=True,chunks=4,map_kind='reverse_chunks'))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,modules), \
                patch.object(torch.Tensor,'npu',lambda v:v.clone(),create=True), \
                patch.object(torch,'npu',types.SimpleNamespace(get_device_properties=lambda _:types.SimpleNamespace(vector_core_num=3)),create=True), \
                patch.object(torch,'full',cpu_full):
            report=run_pair(adapter,spec,Path(directory)/'out',Path(directory)/'cache',{},correctness_only=True)
            self.assertEqual(report['status'],'complete',report.get('error'))
            item=adapter.prepared['before'];state=item.compilation_state()
            item.run();item.compile_only();item.run()
            self.assertEqual(item.compilation_state(),state)
            self.assertEqual(before.compiles,[True,False] if lowering_fails else [True])
            self.assertEqual(after.compiles,[True])
            self.assertEqual(item.metadata['flags']['USE_ELEMENT'],not lowering_fails)
            self.assertEqual(item.expected_kernels,['before'])
            self.assertEqual(report['counts'],dict(correctness_calls=2,profiled_calls=0,capture_calls=0))
            if lowering_fails:
                self.assertIn('portable-source-style',item.metadata['baseline'])
                self.assertIn('lowering unsupported',item.metadata['get_element_compile_failure']['message'])
            self.assertEqual(events[:3] if lowering_fails else events[:2],
                [('compile','before',True),('compile','before',False),('launch','before',False)]
                if lowering_fails else [('compile','before',True),('launch','before',True)])

    def test_importable_extension_compile_failure_uses_portable_public_runner(self):
        self.exercise(True)

    def test_supported_extension_qualified_once_before_public_launch(self):
        self.exercise(False)


if __name__=='__main__': unittest.main()
