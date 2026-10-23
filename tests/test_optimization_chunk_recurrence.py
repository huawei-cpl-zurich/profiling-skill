from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'examples/optimization-library/triton'))
from recurrence_contract import diagnostics,dimensions,make_case,reference,validate_inputs


def chunk_algebra(fixture,bounds):
    """CPU FP64 mathematical check of blocked equations, not candidate execution."""
    inp={key:value.double() for key,value in fixture['inputs'].items()}
    q,k,v,beta,g,initial=(inp[key] for key in ('q','k','v','beta','g','initial'))
    t,hv,dv=v.shape
    hk=k.shape[1]
    output=torch.empty(t,hv,dv,dtype=torch.float64)
    final=torch.empty_like(initial)
    for segment,(start,end) in enumerate(zip(bounds,bounds[1:])):
        for h in range(hv):
            state=initial[segment,h].clone()
            kh=h//(hv//hk)
            for first in range(start,end,32):
                last=min(first+32,end)
                key,query,value=k[first:last,kh],q[first:last,kh],v[first:last,h]
                b=beta[first:last,h]
                c=g[first:last,h].cumsum(0)
                decay=(c[:,None]-c[None,:]).exp()
                lower=torch.tril((key@key.T)*b[:,None]*decay,diagonal=-1)
                rhs=b[:,None]*(value-c.exp()[:,None]*(key@state))
                delta=torch.linalg.solve_triangular(torch.eye(last-first,dtype=torch.float64)+lower,rhs,upper=False)
                output[first:last,h]=c.exp()[:,None]*(query@state)+torch.tril((query@key.T)*decay)@delta
                state=c[-1].exp()*state+(key*(c[-1]-c).exp()[:,None]).T@delta
            final[segment,h]=state
    return output.half(),final.float()


class RecurrenceTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(t=35,hk=1,hv=2,dk=64,dv=64,segments=[0,2,35],seed=71)
        self.fixture=make_case(self.spec)

    def test_chunk_equations_match_independent_sequential_oracle(self):
        output,final=chunk_algebra(self.fixture,self.spec['segments'])
        expected_output,expected_final=self.fixture['expected']
        torch.testing.assert_close(output,expected_output,rtol=.001,atol=.0001)
        torch.testing.assert_close(final,expected_final,rtol=1e-5,atol=1e-6)

    def test_packed_segments_reset_nonzero_initial_state(self):
        inp=self.fixture['inputs']
        one={key:(value[1:2] if key=='initial' else value[2:]) for key,value in inp.items()}
        out,final=reference(one,[0,33])
        torch.testing.assert_close(out,self.fixture['expected'][0][2:])
        torch.testing.assert_close(final[0],self.fixture['expected'][1][1])
        self.assertGreater(float(inp['initial'].abs().sum()),0)

    def test_gqa_mapping_and_independent_value_heads(self):
        inp=self.fixture['inputs']
        repeated={key:value.clone() for key,value in inp.items()}
        repeated['q']=inp['q'].repeat_interleave(2,dim=1)
        repeated['k']=inp['k'].repeat_interleave(2,dim=1)
        actual=reference(repeated,self.spec['segments'])
        for a,b in zip(actual,self.fixture['expected']):
            torch.testing.assert_close(a,b)
        self.assertFalse(torch.equal(actual[0][:,0],actual[0][:,1]))

    def test_single_token_gate_and_final_state_semantics(self):
        spec=dict(self.spec,t=1,segments=[0,1])
        f=make_case(spec)
        inp=f['inputs']; initial=inp['initial'][0]
        k=inp['k'][0,0].float(); q=inp['q'][0,0].float()
        state=initial*inp['g'][0].exp()[:,None,None]
        delta=inp['beta'][0,:,None]*(inp['v'][0].float()-(state*k[None,:,None]).sum(1))
        state=state+k[None,:,None]*delta[:,None,:]
        torch.testing.assert_close(state,f['expected'][1][0])
        torch.testing.assert_close((state*q[None,:,None]).sum(1).half(),f['expected'][0][0])

    def test_nilpotent_inverse_doubling_factorization(self):
        rng=torch.Generator().manual_seed(9)
        a=torch.tril(torch.randn(32,32,generator=rng,dtype=torch.float64)*.01,diagonal=-1)
        inverse=torch.eye(32,dtype=torch.float64)-a
        power=a@a
        for _ in range(4):
            inverse=inverse+inverse@power
            power=power@power
        torch.testing.assert_close(inverse@(torch.eye(32,dtype=torch.float64)+a),torch.eye(32,dtype=torch.float64),rtol=1e-12,atol=1e-12)

    def test_diagnostics_require_both_outputs_and_fixed_tolerance(self):
        self.assertTrue(diagnostics(self.fixture['expected'],self.fixture)['passed'])
        output,final=(value.clone() for value in self.fixture['expected'])
        final[0,0,0,0]+=1
        result=diagnostics((output,final),self.fixture)
        self.assertFalse(result['passed'])
        self.assertFalse(result['final_state']['passed'])
        self.assertEqual(result['atol'],.02)
        self.assertFalse(diagnostics((output,),self.fixture)['passed'])
        output[0,0,0]=float('nan')
        self.assertFalse(diagnostics((output,final),self.fixture)['passed'])

    def test_dispatch_and_fixture_reject_unsupported_inputs(self):
        validate_inputs(self.fixture['inputs'],self.spec)
        for change in (dict(hk=3),dict(segments=[0,35,35]),dict(dk=32),dict(t=True)):
            with self.assertRaises(ValueError): dimensions(dict(self.spec,**change))
        bad={k:v.clone() for k,v in self.fixture['inputs'].items()}
        bad['beta'][0,0]=.5
        with self.assertRaises(ValueError): validate_inputs(bad,self.spec)
        bad['beta'][0,0]=0
        bad['k'][0,0]*=2
        with self.assertRaises(ValueError): validate_inputs(bad,self.spec)

    def test_lazy_jit_frontend_names_and_dependencies(self):
        try:
            import triton
        except ImportError:
            self.skipTest('Triton frontend unavailable')
        import recurrence
        for kernel in recurrence.kernels():
            self.assertIsInstance(kernel.cache_key,str)
            self.assertTrue(kernel.cache_key)
            self.assertIn('tl',kernel.fn.__globals__)


if __name__=='__main__':
    unittest.main()
