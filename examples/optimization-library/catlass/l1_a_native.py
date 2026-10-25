"""Opaque native DeviceGemm handles; no parameter struct is copied into Python."""
import ctypes,json,sys,struct,hashlib
from pathlib import Path
from types import SimpleNamespace
from l1_a_build import compile_arm,sha,BASE
import torch
sys.path.insert(0,str(BASE.parents[2]/'scripts'))
from optimization_native_bridge import audit_tensor
_LIBRARIES=[];_CONTEXTS={};_REGISTERED=False


def backend():
    import torch_npu
    return 'npu:0',lambda:torch.npu.current_stream().npu_stream


def register():
    global _REGISTERED
    if _REGISTERED:return
    library=torch.library.Library('catlass_r12','DEF');library.define('launch(Tensor[] inputs, Tensor(a!) output, str token) -> ()')
    def launch(inputs,output,token):_CONTEXTS[token]._launch()
    library.impl('launch',launch,'PrivateUse1');library.impl('launch',launch,'CPU');_LIBRARIES.append(library);_REGISTERED=True


def entrypoints(manifest):
    directory=Path(manifest['binary']).parent
    def sections(path):
        data=path.read_bytes()
        if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('native ELF64LE required')
        offset=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58)
        if size!=64 or offset+size*count>len(data):raise ValueError('invalid native section table')
        headers=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)];t=headers[names];strings=data[t[4]:t[4]+t[5]]
        return {strings[t[0]:].split(b'\0',1)[0].decode():data[t[4]:t[4]+t[5]] for t in headers if t[1]!=8}
    result={}
    for kind,suffix in [('aic','-cube.o'),('aiv','-vec.o')]:
        files=list(directory.glob('*'+suffix))
        if len(files)!=1:raise RuntimeError('selected native component unavailable '+kind)
        path=files[0];sections_=sections(path);names=[n.removeprefix('.ascend.meta.') for n in sections_ if n.startswith('.ascend.meta.') and n.endswith('ParamsE_mix_'+kind)]
        if len(names)!=1 or '13KernelAdapter' not in names[0]:raise RuntimeError('ambiguous Params-only source registration')
        selected=names[0];code=sections_.get('.text.'+selected)
        if not code:raise RuntimeError('selected native function section empty')
        result[kind]=dict(selector=selected,object_sha256=sha(path),text_sha256=hashlib.sha256(code).hexdigest(),text_bytes=len(code),stack_size_record_hex=sections_.get('.ascend.stack.size.record',b'').hex())
    base=result['aic']['selector'].removesuffix('_mix_aic')
    if result['aiv']['selector']!=base+'_mix_aiv' or base.encode()+b'\0' not in Path(manifest['binary']).read_bytes():raise RuntimeError('host registration/device components disagree')
    result['host_registered_selector']=base;return result


def binary_identity(manifest):
    entries=entrypoints(manifest);host=sha(manifest['binary'])
    return [dict(hash=entries[k]['object_sha256'],sha256=entries[k]['object_sha256'],format='ELF64LE-hiipu-device',identity_source='actual selected SDK device ELF plus .ascend.meta and function section',name=entries[k]['selector'],component=k,binaries=dict(elf=entries[k]['object_sha256'],selected_text=entries[k]['text_sha256']),host_shared_library_sha256=host) for k in ('aic','aiv')]


def inspect_build(arm,p,directory):
    manifest=compile_arm(arm,directory);library=ctypes.CDLL(manifest['binary'])
    library.r12_inspect.argtypes=[ctypes.c_uint32]*3+[ctypes.POINTER(ctypes.c_uint64),ctypes.c_uint32];library.r12_inspect.restype=ctypes.c_int
    proof=(ctypes.c_uint64*16)()
    if library.r12_inspect(*[p[k] for k in ('M','N','K')],proof,16):raise RuntimeError('source ABI/resource inspection rejected')
    if list(proof)[9:]!=[128*p['K']*2,262144,32768,65536,131072,4,0]:raise RuntimeError('FP32 accumulator/rounded resource/workspace mismatch')
    return dict(build=manifest,case_abi_resources=list(proof),entrypoints=entrypoints(manifest))


class Native:
    def __init__(self,arm,p,inputs,out,directory,stream_provider):
        self.layouts=[audit_tensor(t) for t in [*inputs,out]]
        self.inputs=list(inputs);self.output=out;self.stream=stream_provider;self.context=ctypes.c_void_p();self.manifest=compile_arm(arm,directory);self.directory=Path(directory)
        self.library=ctypes.CDLL(self.manifest['binary'])
        self.library.r12_inspect.argtypes=[ctypes.c_uint32]*3+[ctypes.POINTER(ctypes.c_uint64),ctypes.c_uint32];self.library.r12_inspect.restype=ctypes.c_int
        proof=(ctypes.c_uint64*16)();dims=[p[k] for k in ('M','N','K')]
        if self.library.r12_inspect(*dims,proof,16):raise RuntimeError('source layout/resource inspection rejected')
        self.proof=list(proof)
        if self.proof[9:]!=[128*p['K']*2,262144,32768,65536,131072,4,0]:raise RuntimeError('source resource/FP32 accumulator/workspace proof mismatch')
        self.library.r12_create.argtypes=[ctypes.c_uint32]*3+[ctypes.c_void_p]*5+[ctypes.POINTER(ctypes.c_void_p)];self.library.r12_create.restype=ctypes.c_int
        self.library.r12_launch.argtypes=[ctypes.c_void_p,ctypes.c_void_p];self.library.r12_launch.restype=ctypes.c_int
        self.library.r12_destroy.argtypes=[ctypes.c_void_p];self.library.r12_destroy.restype=None
        pointers=[v.data_ptr() for v in [*self.inputs,self.output]]
        if self.library.r12_create(*dims,*pointers,None,self.stream(),ctypes.byref(self.context)):raise RuntimeError('native CanImplement/Initialize failed')
        self.token=str(id(self));register();_CONTEXTS[self.token]=self

    def _launch(self):
        if self.library.r12_launch(self.context,self.stream()):raise RuntimeError('source DeviceGemm launch status failure')

    def run(self):torch.ops.catlass_r12.launch(self.inputs,self.output,self.token);return self.output

    def close(self):
        _CONTEXTS.pop(self.token,None)
        if self.context:self.library.r12_destroy(self.context);self.context=ctypes.c_void_p()
