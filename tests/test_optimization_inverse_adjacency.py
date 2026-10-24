import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/triton'))
sys.path.insert(0, str(ROOT/'scripts'))
from inverse_contract import dimensions, make_case, reference, validate_inputs, diagnostics
from inverse_adjacency import InverseReduction


class InverseContractTests(unittest.TestCase):
    def test_actual_cli_accepts_committed_nested_case(self):
        import optimization_pair
        case=ROOT/'examples/optimization-library/cases/wave004/W004-W06-GB.json'
        observed=[]
        def execute(adapter,spec,*args,**kwargs):
            observed.append(dimensions(spec))
            return dict(status='complete')
        with tempfile.TemporaryDirectory() as folder:
            provenance=Path(folder)/'provenance.json';provenance.write_text('{}')
            argv=['optimization_pair.py','--adapter','inverse_adjacency:Adapter','--case',str(case),
                  '--output',folder+'/result','--cache',folder+'/cache','--provenance',str(provenance)]
            with patch.object(sys,'argv',argv),patch.object(optimization_pair,'run_pair',side_effect=execute):
                self.assertEqual(optimization_pair.main(),0)
        self.assertEqual(observed,[(257,17,8,'seededpermuted-edge-order')])

    def test_committed_cases_through_cli_parser_and_fixture_cache(self):
        # The real CLI consumes nested parameters. Exercise its load and oracle
        # cache boundary, without device work or regenerating scientific cases.
        from optimization_pair import fixture
        from inverse_adjacency import Adapter
        matrix = json.loads((ROOT/'examples/optimization-library/cases/wave004/W06.json').read_text())
        for spec in matrix['cases'] + matrix['contract']['guards']:
            e, n, d, _ = dimensions(spec)
            self.assertEqual((e,n,d), tuple(spec['parameters'][k] for k in ('E','N','D')))
        # Use a distinct testing seed, retaining each frozen nested case shape.
        with tempfile.TemporaryDirectory() as folder:
            spec = dict(matrix['contract']['guards'][0], seed=99)
            first, a = fixture(Adapter(), spec, Path(folder), torch)
            second, b = fixture(Adapter(), spec, Path(folder), torch)
            self.assertEqual((a['oracle_calls'], b['oracle_calls']), (1,0))
            self.assertTrue(torch.equal(first['expected'], second['expected']))
            validate_inputs(first['inputs'], spec)
            self.assertTrue(diagnostics(first['expected'], first)['passed'])

    def test_duplicate_multiplicity_and_empty_rows(self):
        inputs = dict(dest=torch.tensor([2,0,2],dtype=torch.int32),
                      values=torch.tensor([[.1,-.1],[.05,0],[.025,.1]],dtype=torch.float32))
        expected = reference(inputs,4)
        self.assertTrue(torch.equal(expected[2],torch.tensor([.125,0.])))
        self.assertTrue(torch.equal(expected[[1,3]],torch.zeros(2,2)))
        fixture = dict(inputs=inputs,expected=expected)
        self.assertTrue(diagnostics(expected,fixture)['passed'])
        bad=expected.clone();bad[1,0]=1e-8
        self.assertFalse(diagnostics(bad,fixture)['passed'])
        bad=expected.clone();bad[0,0]=float('nan')
        self.assertFalse(diagnostics(bad,fixture)['passed'])

    def test_invalid_input_rejected_before_device_transfer(self):
        spec=dict(E=3,N=4,D=4,seed=99,distribution='seededpermuted-edge-order')
        fixture=make_case(spec)
        for change in ('id','range','dtype'):
            inputs={k:v.clone() for k,v in fixture['inputs'].items()}
            if change=='id':inputs['dest'][0]=4
            if change=='range':inputs['values'][0,0]=.126
            if change=='dtype':inputs['dest']=inputs['dest'].long()
            with self.assertRaises(ValueError):validate_inputs(inputs,spec)

    def test_compile_only_and_execution_order(self):
        class Kernel:
            def __init__(self):self.calls=[]
            def warmup(self,*args,**kw):self.calls.append(('compile',args,kw));return self
            def __getitem__(self,grid):
                def launch(*args,**kw):self.calls.append(('launch',args,kw))
                return launch
            hash='a'*64
            asm={'npubin':b'binary'}
        first,second=Kernel(),Kernel()
        op=InverseReduction.__new__(InverseReduction)
        op.calls=[(first,(1,),(1,)),(second,(2,),(2,))]
        op.options=dict(num_stages=1,multibuffer=False)
        op.output='result';op.compile()
        self.assertEqual([x[0] for x in first.calls],['compile'])
        self.assertIn('binaries',op.compilation_state()[0])
        self.assertEqual(op.run(),'result')
        self.assertEqual([x[0] for x in second.calls],['compile','launch'])


if __name__=='__main__':unittest.main()
