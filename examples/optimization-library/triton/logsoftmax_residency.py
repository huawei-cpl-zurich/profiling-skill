"""Source-backed row residency study. Forward performs only the declared API."""
import hashlib, ctypes, re, os, shutil
from pathlib import Path
from types import SimpleNamespace
from logsoftmax_contract import CASE_SOURCES, dimensions, make_case, diagnostics
FLAGS = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)


def identity(compiled):
    return [dict(hash=c.hash, binaries={k: hashlib.sha256(v.encode() if isinstance(v, str) else v).hexdigest()
                  for k, v in c.asm.items() if isinstance(v, (str, bytes, bytearray))}) for c in compiled]


def query_resources():
 import ctypes,re,shutil
 compiler=Path(shutil.which('bisheng')).resolve()
 roots=[Path(os.environ[k]).resolve() for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME') if os.environ.get(k)]
 cann=next(p for p in roots+list(compiler.parents) if (p/'include/acl/acl.h').is_file())
 header=cann/'include/acl/acl_platform.h';text=header.read_text()
 libs=[]
 for p in sorted((cann/'lib64').glob('*.so')):
  try:
   lib=ctypes.CDLL(str(p));fn=lib.aclplatformGetDeviceInfo
  except (OSError,AttributeError):continue
  libs.append((p,lib,fn));break
 assert libs,'selected SDK aclplatformGetDeviceInfo library unavailable'
 path,lib,fn=libs[0];fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];fn.restype=ctypes.c_int
 values={}
 for key in ('ACL_PLATFORM_VECTOR_CORE_CNT','ACL_PLATFORM_AICORE_UB_SIZE'):
  match=re.search(r'\b'+key+r'\s*=\s*(\d+)',text);assert match,key
  output=ctypes.create_string_buffer(128);status=fn(int(match.group(1)),output,128)
  assert status==0,(key,status)
  values[key]=output.value.decode()
 return dict(aiv_count=int(values['ACL_PLATFORM_VECTOR_CORE_CNT']),ub_bytes=int(values['ACL_PLATFORM_AICORE_UB_SIZE']),raw=values,library=str(path),library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())

class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        import torch_npu
        import triton
        from logsoftmax_kernels import wave007_logsoftmax_streaming, wave007_logsoftmax_resident
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        rows, width, seed, contexts = dimensions(spec)
        if len(fixture['inputs']) != (len(contexts) if contexts else 1):
            raise ValueError('input component count mismatch')
        for x, n in zip(fixture['inputs'], contexts or [rows]):
            if x.device.type != 'cpu' or x.dtype != torch.float32 or x.shape != (n, width) or not x.is_contiguous():
                raise ValueError('contiguous CPU FP32 inputs of specified shape required')
            if not torch.isfinite(x).all() or not ((x >= -20) & (x <= 20)).all():
                raise ValueError('finite inputs in [-20,20] required')
        core_count = query_resources()["aiv_count"]
        if core_count <= 0:
            raise ValueError('actual positive AIV core count required')
        resident = arm == 'after' and width <= 4096
        kernel = wave007_logsoftmax_resident if resident else wave007_logsoftmax_streaming
        block = triton.next_power_of_2(width) if resident else 2048
        inputs = tuple(x.npu() for x in fixture['inputs'])
        storage = tuple(torch.full((x.numel()+32,), 12345.0, device=x.device, dtype=x.dtype) for x in inputs)
        outputs = tuple(s[16:-16].view_as(x) for s, x in zip(storage, inputs))
        args = [(x, y, width, width, x.shape[0], width, block) for x, y in zip(inputs, outputs)]
        compiled = [kernel.warmup(*a, grid=(core_count,), **FLAGS) for a in args]
        for index, c in enumerate(compiled):
            for key, value in c.asm.items():
                if isinstance(value, (str, bytes, bytearray)):
                    (output / f'component-{index}.{key}').write_bytes(value.encode() if isinstance(value, str) else value)
        def run():
            for a in args:
                kernel[(core_count,)](*a, **FLAGS)
            return outputs
        item = SimpleNamespace(run=run, inputs=inputs, outputs=outputs, storage=storage,
            sources=[Path(__file__), Path(__file__).with_name('logsoftmax_kernels.py'), *CASE_SOURCES],
            expected_kernels=[kernel.fn.__name__], compilation_state=lambda: identity(compiled),
            metadata=dict(arm=arm, rows=rows, width=width, grid=core_count, resident=resident,
                          block=block, api_components=len(inputs), compiler_flags=FLAGS,
                          resources=[repr(c.metadata) for c in compiled], device_resources=query_resources(),
                          torch=torch.__version__, triton=triton.__version__))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = diagnostics(tuple(x.cpu() for x in actual), fixture)
        for x, original, y, storage in zip(item.inputs, fixture['inputs'], item.outputs, item.storage):
            if x.data_ptr() == y.data_ptr() or not torch.equal(x.cpu(), original):
                result.update(passed=False, reason='alias/input mutation')
            if not bool((storage[:16] == 12345).all() and (storage[-16:] == 12345).all()):
                result.update(passed=False, reason='output canary changed')
        return result


def factory():
    return Adapter()
