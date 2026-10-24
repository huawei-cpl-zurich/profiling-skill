import json
import hashlib
from pathlib import Path
import sys
import unittest
import subprocess
import tempfile
import os
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
from guarded_jacobi_contract import make_case,model,certificate,diagnostics,validate_inputs

class JacobiContract(unittest.TestCase):
    def spec(self,regime='strong-contraction',m=129):
        return dict(seed=5292,parameters=dict(m=m,R=1,ldb=m+8,regime=regime))
    def test_forty_updates_boundary_and_padding(self):
        for regime in ('strong-contraction','boundary-impulses','exact-dominance-boundary'):
            f=make_case(self.spec(regime)); originals={k:v.clone() for k,v in f['inputs'].items()}
            self.assertEqual(certificate(f['inputs']),regime!='exact-dominance-boundary')
            for after in (False,True):self.assertTrue(diagnostics(model(f['inputs'],after),f)['passed'])
            self.assertTrue(all(torch.equal(v,originals[k]) for k,v in f['inputs'].items()))
    def test_current_inputs_ranges_and_finiteness(self):
        f=make_case(self.spec());x=f['inputs'];self.assertTrue(certificate(x))
        x['b'][0,0]=100.;self.assertFalse(certificate(x))
        x['b'][0,0]=float('nan');self.assertFalse(certificate(x))
        x['b'][0,0]=0.;x['d'][0]=2**25;self.assertFalse(certificate(x))
        self.assertFalse(certificate(make_case(self.spec(m=127))['inputs']))
    def test_nested_matrix_fixture_interface(self):
        p=ROOT/'examples/optimization-library/cases/wave004/W004-W12-GB.json'
        spec=json.loads(p.read_text());f=make_case(spec);validate_inputs(f['inputs'],spec)
        self.assertTrue(diagnostics(model(f['inputs'],True),f)['passed'])
        bad=model(f['inputs'],True);bad[0,0]+=0.1
        self.assertFalse(diagnostics(bad,f)['passed'])

    def test_nested_public_cli_with_cpu_operation_double(self):
        # The hardware boundary is replaced; the actual adapter, case loader,
        # oracle cache, CLI argument parser and result writer execute normally.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'provenance.json').write_text('{}')
            (p/'driver.py').write_text("""
import sys, types
from pathlib import Path
import torch
sys.modules['torch_npu']=types.ModuleType('torch_npu')
import guarded_jacobi as module
from guarded_jacobi_contract import model
from optimization_pair import main
torch.Tensor.npu=lambda self:self
class Operation:
    def __init__(self, inputs, spec, arm):
        self.inputs=inputs; self.after=arm=='after'
        self.compiled=[]; self.calls=[(types.SimpleNamespace(fn=types.SimpleNamespace(__name__='cpu_operation_double')),(),())]; self.workspace=[]; self.options={}
    def run(self):return model(self.inputs,self.after)
    def compilation_state(self):return []
module.Operation=Operation
raise SystemExit(main())
""")
            env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(ROOT/'scripts'),str(ROOT/'examples/optimization-library/triton'))))
            cmd=[sys.executable,str(p/'driver.py'),'--adapter','guarded_jacobi:factory',
                '--case',str(ROOT/'examples/optimization-library/cases/wave004/W004-W12-GB.json'),
                '--output',str(p/'output'),'--cache',str(p/'cache'),'--provenance',str(p/'provenance.json')]
            r=subprocess.run(cmd,env=env,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr+((p/'output/result.json').read_text() if (p/'output/result.json').exists() else ''))
            result=json.loads((p/'output/result.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['counts'],dict(correctness_calls=2,capture_calls=0,profiled_calls=0))


@unittest.skipUnless(os.environ.get('W12_NPU_COMPONENT_TEST') == '1', 'requires selected healthy A3')
class CertificateDeviceBoundary(unittest.TestCase):
    def test_current_device_inputs_strict_boundary_and_range(self):
        import torch_npu
        import guarded_jacobi
        import triton
        from optimization_pair import fixture
        from guarded_jacobi import Adapter
        cache=Path(os.environ['W12_FIXTURE_CACHE'])
        prepared=[]
        for name in ('GA','GB'):
            spec=json.loads((ROOT/f'examples/optimization-library/cases/wave004/W004-W12-{name}.json').read_text())
            value,identity=fixture(Adapter(),spec,cache,torch)
            self.assertEqual(identity['oracle_calls'],0)
            original={k:v.npu() for k,v in value['inputs'].items()}
            variants=[('frozen-'+name,original,name=='GB')]
            if name=='GB':
                for field,number in [('b',100.),('d',2**25),('du',float('nan'))]:
                    changed={k:v.clone() for k,v in original.items()}
                    changed[field].flatten()[0]=number
                    variants.append((field+'-changed',changed,False))
            for label,inputs,expected in variants:
                fn=guarded_jacobi.kernels()[0];m=inputs['d'].numel()
                cert=torch.empty((),device='npu',dtype=torch.int32)
                args=tuple(inputs[k] for k in ('dl','d','du','b'))+(cert,m,triton.next_power_of_2(m))
                options=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
                compiled=fn.warmup(*args,grid=(1,),**options)
                identity=dict(hash=compiled.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in compiled.asm.items() if isinstance(v,(str,bytes))})
                prepared.append((label,fn,args,options,cert,expected,identity))
        rows=[]
        for label,fn,args,options,cert,expected,identity in prepared:
            fn[(1,)](*args,**options)
            actual=bool(cert.cpu().item())
            rows.append(dict(case=label,expected=expected,actual=actual,compilation=[identity]))
            self.assertEqual(actual,expected,label)
        Path(os.environ['W12_COMPONENT_REPORT']).write_text(json.dumps(dict(
            component_npu_calls=len(rows),full_api_calls=0,captures=0,rows=rows),indent=2)+'\n')

if __name__=='__main__':unittest.main()
