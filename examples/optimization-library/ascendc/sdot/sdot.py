"""Native SDOT runtime-failed study; numeric workflows are disabled."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from sdot_compile import build,HERE

FAILURE=('R11 native SDOT runtime study failed: the first baseline correctness API reached output-transfer '
         'synchronization and raised ACL507035. No numerical gate passed; cause unresolved. '
         'See experiments/optimization-wave009/R11-result.json. Numeric execution is disabled.')


class StudyUnqualified(ValueError):
    """The retained native code failed its first bounded runtime gate."""


class Adapter:
    case_sources=[HERE/'sdot_contract.py']
    def __init__(self,directory=None,*,compile_inspection=False):
        self.directory=Path(directory or '.');self.builds={};self.compile_inspection=compile_inspection

    @staticmethod
    def make_case(spec):raise StudyUnqualified(FAILURE)
    def prepare(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def validate(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def capture(self,*args,**kwargs):raise StudyUnqualified(FAILURE)

    def _build(self,arm):
        if arm not in self.builds:
            candidate=build(arm,self.directory);self.builds[arm]=candidate
        return self.builds[arm]

    def compile_probe(self,arm,spec,output):
        """Explicit native compilation/ABI inspection, without numerical buffers or launches."""
        if not self.compile_inspection:raise StudyUnqualified(FAILURE)
        if arm not in ('before','after'):raise ValueError('arm')
        from sdot_contract import dimensions,tiling
        n=dimensions(spec);raw=tiling(n);compiled=[]
        def compile_only():
            if not compiled:
                b=self._build(arm);data=b['object'].read_bytes()
                compiled.append(SimpleNamespace(hash=hashlib.sha256(data).hexdigest(),asm={'device_vec_o':data,'host_shared_so':b['binary'].read_bytes()},metadata=b['resources']))
            return compiled
        def run():raise StudyUnqualified(FAILURE)
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:[dict(hash=a.hash,binaries={k:hashlib.sha256(v).hexdigest() for k,v in a.asm.items()}) for a in compiled],
            expected_kernels=['sdot_materialize','_Z17sdot_materializedPhS_S_S_S_'] if arm=='before' else ['_Z4sdotPhS_S_S_S_'],
            sources=[__file__,*self.case_sources,HERE/'sdot_compile.py',HERE/'native.cpp',*sorted(p for p in (HERE/'vendor').rglob('*') if p.is_file())],
            metadata=dict(arm=arm,prototype=True,n=n,grid=[8],tiling_bytes=len(raw),tiling_sha256=hashlib.sha256(raw).hexdigest(),source_declared_UB_high_water_bytes=194816,
                automatic_UB_scratch='unknown',numeric_execution_enabled=False,scope='Native compile/ABI metadata only; no scientific buffers or expected values',
                ABI=dict(pointer_count=6,input_n=n,workspace_bytes=1024,logical_result_floats=1,alignment=128)))


def factory():raise StudyUnqualified(FAILURE)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('case','output','cache','provenance'):p.add_argument('--'+k,required=True)
    p.add_argument('--correctness-only',action='store_true');p.parse_args()
    factory() # Reject before case/provenance reads, provider/backend imports or expected generation.


if __name__=='__main__':raise SystemExit(main())
