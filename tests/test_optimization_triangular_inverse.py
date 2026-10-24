from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import triangular_inverse as inverse
import triangular_inverse_contract as contract


class TriangularInverseTests(unittest.TestCase):
    def test_oracle_solves_packed_heads_and_tail_without_crossing(self):
        spec=dict(B=2,T=39,H=2,BT=32,seed=17,segments=[0,3,38,39])
        fixture=contract.make_case(spec)
        contract.validate_inputs(fixture['inputs'],spec)
        result=contract.diagnostics(fixture['expected'],fixture,spec)
        self.assertTrue(result['passed'],result)
        self.assertLess(result['residual_infinity'],1e-6)
        self.assertEqual(contract.chunks(spec),[(b,s,l) for b in range(2) for s,l in [(0,3),(3,32),(35,3),(38,1)]])

    def test_hierarchical_equation_matches_independent_solve(self):
        # Exercise each supported hierarchy, including partially valid blocks.
        for bt in (16,32,64,128):
            spec=dict(B=1,T=bt+3,H=1,BT=bt,seed=41)
            fixture=contract.make_case(spec)
            out=torch.zeros_like(fixture['expected'])
            for batch,start,length in contract.chunks(spec):
                a=torch.zeros(bt,bt)
                a[:length,:length]=fixture['inputs']['a'][batch,start:start+length,0,:length]
                x=torch.zeros_like(a)
                for base in range(0,bt,16):
                    for col in range(base,base+16):
                        for row in range(base,base+16):
                            x[row,col]=float(row==col)-(a[row]*x[:,col]).sum()
                size=16
                while size<bt:
                    y=torch.zeros_like(x)
                    for base in range(0,bt,2*size):
                        top=slice(base,base+size)
                        bottom=slice(base+size,base+2*size)
                        y[top,top]=x[top,top]
                        y[bottom,bottom]=x[bottom,bottom]
                        y[bottom,top]=-x[bottom,bottom]@a[bottom,top]@x[top,top]
                    x=y
                    size*=2
                out[batch,start:start+length,0,:length]=x[:length,:length]
            self.assertTrue(contract.diagnostics(out,fixture,spec)['passed'])

    def test_rejects_input_envelope_and_unsupported_before_allocation(self):
        spec=dict(B=1,T=35,H=1,BT=32,seed=12)
        fixture=contract.make_case(spec)
        for index,value in [((0,0,0,0),.1),((0,1,0,0),.3),((0,34,0,4),.1)]:
            bad={'a':fixture['inputs']['a'].clone()}
            bad['a'][index]=value
            with self.assertRaises(ValueError):
                contract.validate_inputs(bad,spec)
        reject=dict(spec,BT=24,expect_rejection=True)
        payload=contract.make_case(reject)
        with patch.dict(sys.modules,dict(torch_npu=object())),patch.object(torch,'empty',side_effect=AssertionError('allocated')):
            for arm in ('before','after'):
                with self.assertRaisesRegex(ValueError,'unsupported BT'):
                    inverse.Adapter().prepare(arm,reject,payload,None)

    def test_adapter_preparation_and_whole_forward_launch_accounting(self):
        calls=[]
        class Launch:
            def __init__(self,name):
                self.name=name
                self.fn=SimpleNamespace(__name__=name)
            def __getitem__(self,grid):
                def run(*args):
                    calls.append((self.name,grid))
                    if self.name=='solve':
                        a,x,starts,lengths,batches,t,h,bt,leaf=args
                        x.zero_()
                        for chunk,(start,length,batch) in enumerate(zip(starts,lengths,batches)):
                            for head in range(h):
                                task=chunk*h+head
                                for col in range(bt):
                                    for row in range(col//leaf*leaf,(col//leaf+1)*leaf):
                                        total=(a[batch,start+row,head,:length].float()*x[task,:length,col]).sum() if row<length else 0
                                        x[task,row,col]=float(row==col)-total
                    elif self.name=='left':
                        a,x,temp,starts,lengths,batches,t,h,bt,size=args
                        for chunk,(start,length,batch) in enumerate(zip(starts,lengths,batches)):
                            for head in range(h):
                                matrix=torch.zeros(bt,bt)
                                matrix[:length,:length]=a[batch,start:start+length,head,:length]
                                task=chunk*h+head
                                for base in range(0,bt,2*size):
                                    top=slice(base,base+size);bottom=slice(base+size,base+2*size)
                                    temp[task,bottom,top]=x[task,bottom,bottom]@matrix[bottom,top]
                    elif self.name=='right':
                        x,xright,temp,y,bt,size=args
                        for base in range(0,bt,2*size):
                            top=slice(base,base+size);bottom=slice(base+size,base+2*size)
                            y[:,top,top]=xright[:,top,top]
                            y[:,bottom,bottom]=x[:,bottom,bottom]
                            y[:,bottom,top]=-temp[:,bottom,top]@xright[:,top,top]
                            y[:,top,bottom]=0
                    else:
                        x,out,starts,lengths,batches,t,h,bt=args
                        out.zero_()
                        for chunk,(start,length,batch) in enumerate(zip(starts,lengths,batches)):
                            for head in range(h):
                                out[batch,start:start+length,head,:length]=x[chunk*h+head,:length,:length]
                    return SimpleNamespace(name=self.name,hash='stable',asm={'npubin':b'cpu-boundary-fixture'})
                return run
        original_empty,original_tensor=torch.empty,torch.tensor
        def empty(*args,**kwargs):
            kwargs.pop('device',None)
            return original_empty(*args,**kwargs)
        def tensor(*args,**kwargs):
            kwargs.pop('device',None)
            return original_tensor(*args,**kwargs)
        with patch.dict(sys.modules,dict(torch_npu=object())),patch.object(torch.Tensor,'to',lambda value,*args,**kwargs:value.clone()),patch.object(torch,'empty',side_effect=empty),patch.object(torch,'tensor',side_effect=tensor),patch.object(inverse,'kernels',return_value=tuple(Launch(n) for n in ('solve','left','right','unpack'))):
            inverse.triton=SimpleNamespace(__version__='cpu-fixture')
            for bt in (16,32,64,128):
                spec=dict(B=1,T=bt+3,H=2,BT=bt,seed=87)
                fixture=contract.make_case(spec)
                adapter=inverse.Adapter()
                for arm in ('before','after'):
                    previous=len(calls)
                    item=adapter.prepare(arm,spec,fixture,None)
                    self.assertEqual(len(calls),previous)
                    self.assertEqual(item.compilation_state(),[])
                    self.assertTrue(adapter.validate(arm,item.run(),fixture,spec)['passed'])
                    self.assertEqual(len(calls)-previous,item.metadata['launches_per_forward'])

    def test_diagnostics_enforce_residual_structure_and_dtype(self):
        spec=dict(B=1,T=35,H=1,BT=32,seed=4)
        fixture=contract.make_case(spec)
        self.assertFalse(contract.diagnostics(fixture['expected'].half(),fixture,spec)['passed'])
        bad=fixture['expected'].clone()
        bad[0,0,0,1]=1e-7
        self.assertFalse(contract.diagnostics(bad,fixture,spec)['passed'])
        bad=fixture['expected'].clone()
        bad[0,32,0,3]=1e-7
        self.assertFalse(contract.diagnostics(bad,fixture,spec)['passed'])
        # Every element is within 1e-3 but the infinity residual gate rejects.
        bad=fixture['expected'].clone()
        for row in range(1,32):
            bad[0,row,0,:row]+=0.0002
        result=contract.diagnostics(bad,fixture,spec)
        self.assertLess(result['max_absolute_error'],.001)
        self.assertGreater(result['residual_infinity'],.002)
        self.assertFalse(result['passed'])


if __name__=='__main__':
    unittest.main()
