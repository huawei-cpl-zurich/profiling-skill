"""Native SASUM runtime-failed study: public numerical workflows disabled."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from sasum_compile import build,HERE,sections
from optimization_native_bridge import digest

FAILURE=('P04 native SASUM first baseline output synchronization failed ACL507015; '
         'cause and native device completion unresolved, no numerical gate passed. '
         'See experiments/optimization-wave010/P04-result.json. Numeric execution disabled.')


class StudyUnqualified(ValueError):
    """This retained native attempt did not pass numerical qualification."""


def device_objects(b):
    cubes=list(b['object'].parent.glob('*-cube.o'))
    if len(cubes)!=1:raise ValueError('actual MIX AIC object required alongside AIV object')
    return [('device_aic_elf',cubes[0]),('device_aiv_elf',b['object'])]


def binary_state(b):
    """Canonical identities bind BOTH actual MIX device ELF components."""
    return [dict(hash=digest(p),binaries={role:digest(p)}) for role,p in device_objects(b)]


class Adapter:
    case_sources=[HERE/'sasum_contract.py']
    def __init__(self,directory=None,*,compile_inspection=False):
        self.directory=Path(directory or '.');self.builds={};self.compile_inspection=compile_inspection
    @staticmethod
    def make_case(spec):raise StudyUnqualified(FAILURE)
    def prepare(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def validate(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def capture(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def _build(self,arm):
        if arm not in self.builds:
            candidate=build(arm,self.directory)
            for role,p in device_objects(candidate):
                meta=sections(p.read_bytes())
                if not meta.get('.text',{}).get('bytes') or not any(k.startswith('.ascend.meta.') for k in meta):raise ValueError('MIX component code/registration required')
            self.builds[arm]=candidate
        return self.builds[arm]

    def compile_probe(self,arm,spec,output):
        if not self.compile_inspection:raise StudyUnqualified(FAILURE)
        from sasum_contract import dimensions,tiling
        if arm not in ('before','after'):raise ValueError('arm')
        n,inc=dimensions(spec);raw=tiling(n,inc);compiled=[]
        def compile_only():
            if not compiled:
                b=self._build(arm);pending=[]
                for role,p in device_objects(b):
                    data=p.read_bytes();meta=sections(data)
                    names=[k.removeprefix('.ascend.meta.') for k in meta if k.startswith('.ascend.meta.')]
                    if not names:raise ValueError('actual native component registration required')
                    pending.append(SimpleNamespace(hash=digest(p),asm={role:data},metadata=dict(device_role=role,registered_entrypoints=names,sections=meta,declared_resources=b['resources'])))
                compiled.extend(pending)
            return compiled
        def run():raise RuntimeError('input-only native prototype cannot execute')
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:binary_state(self.builds[arm]) if compiled else [],
            expected_kernels=[],sources=[__file__,*self.case_sources,HERE/'sasum_compile.py',HERE/'native.cpp',*sorted(p for p in (HERE/'vendor').rglob('*') if p.is_file())],
            metadata=dict(arm=arm,prototype=True,n=n,incx=inc,grid=[8],tiling_bytes=len(raw),tiling_sha256=hashlib.sha256(raw).hexdigest(),source_declared_UB_bytes=165216+(32 if arm=='after' else 0),automatic_scratch='unknown',ABI=dict(pointer_count=4,input_span_floats=(n-1)*inc+1,workspace_bytes=1024,logical_result_floats=1,alignment=128)))



def factory():raise StudyUnqualified(FAILURE)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('case','output','cache','provenance'):p.add_argument('--'+k,required=True)
    p.add_argument('--correctness-only',action='store_true');p.parse_args()
    factory() # Before case/provenance reads, backend/provider imports or scientific expected.


if __name__=='__main__':raise SystemExit(main())
