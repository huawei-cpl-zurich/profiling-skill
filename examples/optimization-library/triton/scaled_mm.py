"""Failed-study scaled matmul component; public execution is disabled.

Public outputs remain FP32 matrices. The private mode store is common to both
arms and is inspected only during correctness, never in the ten forwards.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from scaled_mm_contract import CASE_SOURCES, dimensions, make_case, check_fixture


def binary_identity(artifact):
    binaries = {}
    for key, value in artifact.asm.items():
        if isinstance(value,str): value=value.encode()
        if isinstance(value,bytes): binaries[key]=hashlib.sha256(value).hexdigest()
    return dict(hash=artifact.hash, binaries=binaries)


FAILURE = ('T03 failed study: candidate backend compilation failed before correctness; '
           'public execution and capture are disabled. See experiments/optimization-wave008/T03-result.json.')


class StudyUnsupported(ValueError):
    """The recorded candidate has no passing device correctness gate."""


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self, *, compile_inspection=False):
        self.prepared = {}
        self.compile_inspection = compile_inspection

    def capture(self, *args, **kwargs):
        raise StudyUnsupported(FAILURE)

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before','after'): raise ValueError('unknown arm')
        if not self.compile_inspection: raise StudyUnsupported(FAILURE)
        # Reject unsupported host domain before backend imports or compile cache.
        m,n,k,p = check_fixture(spec,fixture)
        import torch
        import torch_npu
        import triton
        from scaled_mm_kernel import scaled_mm_kernel
        device = {name:x.npu() for name,x in fixture.items() if name != 'expected' and x is not None}
        count = triton.cdiv(m,16)*triton.cdiv(n,64)
        out_storage = torch.full((m*n+2,), -76543., dtype=torch.float32, device=device['a'].device)
        result = out_storage[1:-1].view(m,n)
        mode_storage = torch.full((count+2,), -9876, dtype=torch.int32, device=result.device)
        mode = mode_storage[1:-1]
        args = (device['a'],device['b'],device['sa'],device['sb'],result,device.get('bias'),mode,
                arm=='after',m,n,k,k,1,n,1,n,1)
        kw = dict(ACCUMULATOR_DTYPE=__import__('triton.language',fromlist=['int32']).int32,
                  BLOCK_SIZE_M=16,BLOCK_SIZE_N=64,BLOCK_SIZE_K=128,
                  BLOCK_SIZE_SCALE_A=1 if device['sa'].numel()==1 else 16,
                  BLOCK_SIZE_SCALE_B=1 if device['sb'].numel()==1 else 64,
                  num_stages=1,multibuffer=False,enable_fp_fusion=False)
        compiled = {}
        def compile_only():
            artifact = scaled_mm_kernel.warmup(*args,grid=(count,),**kw)
            if not any(key in artifact.asm for key in ('npubin','cubin')):
                raise RuntimeError('no compiled device binary')
            compiled[artifact.hash] = artifact
            return [artifact]
        def run():
            # Even explicit compilation inspection never authorizes execution.
            raise StudyUnsupported(FAILURE)
        item = SimpleNamespace(run=run,compile_only=compile_only,
            compilation_state=lambda:[binary_identity(x) for x in compiled.values()],
            expected_kernels=['scaled_mm_kernel'],sources=[Path(__file__),Path(__file__).with_name('scaled_mm_kernel.py'),*CASE_SOURCES],
            metadata=dict(arm=arm,M=m,N=n,K=k,tile=[16,64,128],prepared_component=True,
                execution_enabled=False,failure_cause='unresolved backend malloc diagnostic',
                role=spec.get('role'),parameters=p,mode_proof_bytes_per_api=4*count,
                allocator_excluded=True,prepared_output_reuse=True,physical_offsets='source int64 unchanged',
                machine_multiply_count='unverified; source logical two to one only'),
            inputs=device,out_storage=out_storage,mode_storage=mode_storage,mode=mode,result=result,
            spec=spec)
        self.prepared[arm]=item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item=self.prepared[arm]
        if item.spec != spec: raise ValueError('prepared pair does not match current case')
        expected=fixture['expected']
        host=actual.detach().cpu()
        valid=(tuple(host.shape)==tuple(expected.shape) and host.dtype==torch.float32 and
               torch.equal(host.view(torch.int32),expected.view(torch.int32)))
        flags=item.mode.cpu()
        # Expected route by frozen case designation; observed flags establish it.
        wanted=0 if arm=='before' else (1 if spec.get('role')!='guard' else 2)
        proof=bool((flags==wanted).all())
        canary=bool((item.out_storage[[0,-1]].cpu()==-76543).all() and
                    (item.mode_storage[[0,-1]].cpu()==-9876).all())
        immutable=all(torch.equal(x.cpu(),fixture[name]) for name,x in item.inputs.items())
        nonalias=actual.untyped_storage().data_ptr() not in [x.untyped_storage().data_ptr() for x in item.inputs.values()]
        return dict(passed=valid and proof and canary and immutable and nonalias,
            bitwise_numeric=valid,observed_modes=flags.tolist(),expected_mode=wanted,
            canaries=canary,inputs_immutable=immutable,nonalias=nonalias)


def factory():
    # The default CLI calls this before loading a fixture or importing NPU code.
    raise StudyUnsupported(FAILURE)
