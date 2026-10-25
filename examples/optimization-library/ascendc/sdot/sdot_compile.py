"""Extracted native SDK build; no full ops-blas build or test stubs."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts'))
from optimization_native_bridge import build_shared,digest
HERE=Path(__file__).resolve().parent


def derivative(source):
    y='    copy_vec_gm2ub(ub_y, gm_y, len);\n';mul='    mul_v<ArchType::ASCEND_V220, float>(ub_x, ub_x, ub_y, repeatTime, 1, 1, 1, 8, 8, 8);\n'
    if source.count(y)!=1 or source.count(mul)!=1:raise ValueError('exact source transform mismatch')
    source=source.replace(y,'').replace(mul,'')
    return re.sub(r'\bsdot\b','sdot_materialized',source).replace('void sdot_kernel_do(', 'void sdot_materialized_kernel_do(')


def sections(data):
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('ELF64LE required')
    offset=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58)
    if size!=64 or offset+size*count>len(data):raise ValueError('ELF sections invalid')
    table=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)];s=table[names];strings=data[s[4]:s[4]+s[5]]
    return {strings[t[0]:].split(b'\0',1)[0].decode():dict(bytes=t[5],sha256=hashlib.sha256(data[t[4]:t[4]+t[5]]).hexdigest(),raw_hex=data[t[4]:t[4]+t[5]].hex() if t[5]<=64 else None) for t in table}


def registered_selectors(meta,logical):
    registered=[k.removeprefix('.ascend.meta.') for k in meta if k.startswith('.ascend.meta.')]
    result=[]
    for name in logical:
        matches=[n for n in registered if n==name or n.startswith('_Z'+str(len(name))+name)]
        if len(matches)!=1:raise RuntimeError('ambiguous or absent native registration: '+name)
        result.append(matches[0])
    return result


def build(arm,output):
    if arm not in ('before','after'):raise ValueError('arm')
    output=Path(output).resolve()/('native-'+arm+'-'+uuid.uuid4().hex[:8]);output.mkdir(parents=True,exist_ok=False)
    compiler=shutil.which('bisheng')
    if not compiler:raise RuntimeError('selected runtime native compiler unavailable')
    compiler=Path(compiler).resolve();roots=[Path(os.environ[k]).resolve() for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME') if os.environ.get(k)]
    sdk=next(p for p in roots+list(compiler.parents) if (p/'include/acl/acl.h').is_file());headers=list((sdk/'include').rglob('kernel_operator.h'))
    if len(headers)!=1:raise RuntimeError('ambiguous SDK kernel_operator.h')
    api=headers[0].resolve().parent;includes=[sdk/'include',api.parent,api,api/'interface',api/'impl',api/'basic_api',api.parent/'impl/basic_api',HERE,HERE/'vendor/blas',HERE/'vendor/include']
    source=HERE/'vendor/blas/dot/arch22/sdot_kernel.cpp';(output/'sdot_reduce.cpp').write_text(derivative(source.read_text()))
    tu=output/'sdot.cpp';tu.write_bytes((HERE/'native.cpp').read_bytes());binary=output/'sdot.so'
    command=[str(compiler),'-xcce','-std=c++17','-O2','-fPIC','-shared','-save-temps=obj','--cce-aicore-arch=dav-c220','-DBEFORE='+str(int(arm=='before')),*['-I'+str(p) for p in includes],str(tu),'-o',str(binary)]
    sources=[HERE/'native.cpp',Path(__file__),source,output/'sdot_reduce.cpp',*sorted(p for p in (HERE/'vendor').rglob('*') if p.is_file())]
    manifest=build_shared(command,source_paths=sources,binary=binary,output=output/'build',source_revision='c2332551586763f8c2d27621f8bc119bd1c3a301')
    objects=list(output.glob('*-vec.o'))
    if len(objects)!=1:raise RuntimeError('actual vector device object unavailable')
    obj=objects[0];meta=sections(obj.read_bytes())
    if not meta.get('.text',{}).get('bytes'):raise RuntimeError('device executable text missing')
    selectors=registered_selectors(meta,['sdot_materialize','sdot_materialized'] if arm=='before' else ['sdot'])
    symbols=subprocess.run(['nm','--defined-only',str(obj)],capture_output=True,text=True,check=True).stdout
    if not all(any(line.split()[-1]==name for line in symbols.splitlines()) for name in selectors):raise RuntimeError('selected native symbols missing')
    disasm=subprocess.run(['llvm-objdump','-d',str(obj)],capture_output=True,text=True);(output/'device-disassembly.txt').write_text(disasm.stdout+'\n'+disasm.stderr)
    import torch_npu # Load the selected native runtime symbols before host registration.
    library=ctypes.CDLL(str(binary));library.catlass_bridge_kernel_names.restype=ctypes.c_char_p
    if json.loads(library.catlass_bridge_kernel_names().decode())!=selectors:raise RuntimeError('bridge selectors mismatch')
    library.catlass_bridge_abi_version.restype=ctypes.c_uint32
    if library.catlass_bridge_abi_version()!=1:raise RuntimeError('native ABI mismatch')
    resource=dict(device_object_sha256=digest(obj),device_sections=meta,selected_symbols=selectors,symbol_table=symbols,disassembly_exit=disasm.returncode,declared_UB_high_water_bytes=194816,compiler_scratch_qualification='requires read-only generated device artifact audit before scientific expected')
    (output/'resources.json').write_text(json.dumps(resource,indent=2)+'\n')
    return dict(binary=binary,object=obj,manifest=manifest,resources=resource,selectors=selectors,sources=sources+[output/'resources.json'],directory=output)
