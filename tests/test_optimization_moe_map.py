import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from moe_map_contract import MAX_N,dimensions,make_case,validate_inputs


class MapContract(unittest.TestCase):
    def spec(self,**params):
        return dict(seed=99232,role='guard',parameters=dict(N=137,num_splits=7,**params))

    def test_chunk_reference_and_bijection(self):
        for kind in ('identity','reverse','uneven_zero_shuffle'):
            data=make_case(self.spec(map_kind=kind));i=data['inputs'];mapping=data['expected']
            self.assertTrue(torch.equal(torch.sort(mapping).values,torch.arange(137,dtype=torch.int32)))
            # Independent scalar formulation for these small nonfrozen tests.
            expected=[]
            for token in range(137):
                chunk=next(c for c in range(7) if int(i['input_cumsum'][c+1])>token)
                expected.append(int(i['output_cumsum'][i['inverse_sorted_indices'][chunk]])+token-int(i['input_cumsum'][chunk]))
            self.assertEqual(mapping.tolist(),expected)

    def test_explicit_permutation_with_generated_sizes(self):
        spec=dict(seed=314159,N=9,num_splits=3,sorted_indices=[2,1,0])
        data=make_case(spec)
        self.assertEqual(data['inputs']['split_sizes'].tolist(),[3,3,3])
        self.assertEqual(data['inputs']['inverse_sorted_indices'].tolist(),[2,1,0])
        self.assertEqual(data['expected'].tolist(),[6,7,8,3,4,5,0,1,2])
        from moe_map import Adapter
        from optimization_pair import fixture
        for order in ([2,2,0],[3,1,0],[2,1],[2,True,0]):
            with tempfile.TemporaryDirectory() as d:
                cache=Path(d)/'cache'
                with self.assertRaises(ValueError):fixture(Adapter(),dict(spec,sorted_indices=order),cache,torch)
                self.assertFalse(cache.exists())

    def test_invalid_metadata(self):
        spec=self.spec();data=make_case(spec)
        for key,index,value in [('split_sizes',0,-1),('input_cumsum',0,1),('output_cumsum',1,-2),('inverse_sorted_indices',0,99)]:
            broken={k:v.clone() for k,v in data['inputs'].items()};broken[key][index]=value
            with self.assertRaises(ValueError):validate_inputs(broken,spec)
        broken={k:v.clone() for k,v in data['inputs'].items()};broken['split_sizes']=broken['split_sizes'].long()
        with self.assertRaises(ValueError):validate_inputs(broken,spec)

    def test_invalid_before_cache(self):
        from optimization_pair import fixture
        from moe_map import Adapter
        for n in (0,-1,MAX_N+1):
            spec=self.spec();spec['parameters']['N']=n
            with tempfile.TemporaryDirectory() as d:
                cache=Path(d)/'cache'
                with self.assertRaises(ValueError):fixture(Adapter(),spec,cache,torch)
                self.assertFalse(cache.exists())

    def test_flat_metadata_cache_and_external_import(self):
        from optimization_pair import fixture
        from moe_map import Adapter
        spec=dict(seed=98443,N=23,num_splits=3,role='guard',expected='integer exact')
        with tempfile.TemporaryDirectory() as d:
            _,first=fixture(Adapter(),spec,Path(d),torch)
            _,second=fixture(Adapter(),spec,Path(d),torch)
            self.assertEqual((first['oracle_calls'],second['oracle_calls']),(1,0))
        module_spec=importlib.util.spec_from_file_location('external_map',ROOT/'examples/optimization-library/triton/moe_map.py')
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        self.assertEqual(module.factory().make_case(spec)['expected'].tolist(),list(range(23)))

    def test_float_intermediate_counterexample(self):
        token=torch.tensor(9000001,dtype=torch.float32)
        self.assertEqual(int(torch.tensor(9000000,dtype=torch.int32)+token-9000000),9000000)
        self.assertNotEqual(int(token),9000000)


class PublicBoundary(unittest.TestCase):
    def test_first_launch_qualifies_and_compilation_is_stable(self):
        import types
        from moe_map import Adapter
        events=[]
        artifact=types.SimpleNamespace(hash='a'*64,asm={'npubin':b'compiled'})
        class Kernel:
            fn=types.SimpleNamespace(__name__='map_test')
            def warmup(self,*args,**kwargs):events.append('compile');return artifact
            def __getitem__(self,grid):
                def launch(*args,**kwargs):
                    events.append('launch');return artifact
                return launch
        modules=dict(torch_npu=types.ModuleType('torch_npu'),
            triton=types.SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b,next_power_of_2=lambda s:1<<(s-1).bit_length()),
            moe_map_kernel=types.SimpleNamespace(moe_map_before=Kernel(),moe_map_after=Kernel()))
        full=torch.full
        def cpu_full(*args,**kwargs):kwargs.pop('device');return full(*args,**kwargs)
        spec=dict(seed=1942,parameters=dict(N=17,num_splits=5));data=make_case(spec)
        with patch.dict(sys.modules,modules), patch.object(torch.Tensor,'npu',lambda v:v.clone(),create=True), \
             patch.object(torch,'full',cpu_full), patch.object(torch,'npu',types.SimpleNamespace(
                 get_device_properties=lambda _:types.SimpleNamespace(vector_core_num=8)),create=True):
            item=Adapter().prepare('before',spec,data,Path('.'))
            item.run();state=item.compilation_state();item.run();item.compile_only()
            self.assertEqual(events,['compile','launch','launch'])
            self.assertEqual(item.compilation_state(),state)
            self.assertEqual(item.metadata['block_size'],3)
            self.assertEqual(item.expected_kernels,['map_test'])
        with patch.dict(sys.modules,modules), patch.object(torch,'npu',types.SimpleNamespace(
                 get_device_properties=lambda _:types.SimpleNamespace(vector_core_num=65)),create=True):
            with self.assertRaises(ValueError):Adapter().prepare('before',spec,data,Path('.'))

    def test_cli_explicit_permutation_uses_generated_sizes(self):
        import subprocess,json,os
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            # Replace only the execution backend; retain the real provider,
            # fixture cache, CLI, and correctness accounting in the subprocess.
            (root/'cpu_map.py').write_text("""
from types import SimpleNamespace
from pathlib import Path
from moe_map import Adapter
import moe_map_contract
class CPUAdapter(Adapter):
    def prepare(self,arm,spec,fixture,output):
        return SimpleNamespace(run=lambda:fixture['expected'],
            expected_kernels=['cpu_map'],sources=[Path(moe_map_contract.__file__)],metadata={})
    def validate(self,arm,actual,fixture,spec):
        return dict(passed=actual.tolist()==[6,7,8,3,4,5,0,1,2],mapping=actual.tolist())
def factory():return CPUAdapter()
""")
            env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(root),str(ROOT/'examples/optimization-library/triton'))))
            (root/'provenance.json').write_text('{}')
            for index,order in enumerate(([2,1,0],[2,2,0])):
                (root/'case.json').write_text(json.dumps(dict(seed=314159,N=9,num_splits=3,role='guard',sorted_indices=order)))
                result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','cpu_map:factory',
                    '--case',str(root/'case.json'),'--provenance',str(root/'provenance.json'),
                    '--output',str(root/f'out{index}'),'--cache',str(root/f'cache{index}')],
                    cwd=root,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,index,result.stderr)
                report=json.loads((root/f'out{index}/result.json').read_text())
                self.assertEqual(report['counts']['correctness_calls'],0 if index else 2)
                if index:self.assertFalse((root/f'cache{index}').exists())
                else:
                    for arm in ('before','after'):
                        self.assertEqual(report['arms'][arm]['correctness']['diagnostics']['mapping'],[6,7,8,3,4,5,0,1,2])

    def test_standalone_cli_rejects_before_cache(self):
        import subprocess,json,os
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'case.json').write_text(json.dumps(dict(seed=19842,N=MAX_N+1,num_splits=2,role='guard')))
            (root/'provenance.json').write_text('{}')
            env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','moe_map:factory',
                '--case',str(root/'case.json'),'--provenance',str(root/'provenance.json'),
                '--output',str(root/'out'),'--cache',str(root/'cache')],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertFalse((root/'cache').exists())
            self.assertEqual(json.loads((root/'out/result.json').read_text())['counts']['correctness_calls'],0)


if __name__=='__main__':unittest.main()
