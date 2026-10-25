"""SDK-only source SROTM build and narrowly checked DMA derivative."""
import ctypes,hashlib,json,os,re,shutil,struct,subprocess,sys,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[3]/'scripts'))
from optimization_native_bridge import build_shared,digest
PIN='c2332551586763f8c2d27621f8bc119bd1c3a301'


def derivative(source):
    begin='    // 对齐扩展部分补零，避免向量计算读到未定义值。'
    if source.count(begin)!=1:raise ValueError('pinned CopyIn mismatch')
    pos=source.index(begin)
    source=source[:pos]+'''    if (incx == 1 && incy == 1) {
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
        DataCopyExtParams p{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(computeCount-dataCount), 0};
        DataCopyPad(xLocal, xGM[curOffset], p, pad);
        DataCopyPad(yLocal, yGM[curOffset], p, pad);
        xQueue.EnQue<T>(xLocal); yQueue.EnQue<T>(yLocal);
        return;
    }
'''+source[pos:]
    marker='    LocalTensor<T> yOutLocal = yOutBuf.Get<T>();\n    for (uint32_t i = 0; i < dataCount; ++i) {'
    if source.count(marker)!=1:raise ValueError('pinned CopyOut mismatch')
    dma='''    LocalTensor<T> yOutLocal = yOutBuf.Get<T>();
    if (incx == 1 && incy == 1) {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id); WaitFlag<HardEvent::V_MTE3>(id);
        DataCopyExtParams p{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(xGM[curOffset], xOutLocal, p);
        DataCopyPad(yGM[curOffset], yOutLocal, p);
        id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(id); WaitFlag<HardEvent::MTE3_V>(id);
        return;
    }
    for (uint32_t i = 0; i < dataCount; ++i) {'''
    return source.replace(marker,dma).replace('srotm_kernel','srotm_kernel_dma')


def sections(data):
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('ELF64LE required')
    offset=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58)
    if size!=64 or offset+size*count>len(data):raise ValueError('ELF bounds')
    rows=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)];s=rows[names];strings=data[s[4]:s[4]+s[5]]
    return [dict(index=i,name=strings[t[0]:].split(b'\0',1)[0].decode(),bytes=t[5],sha256=hashlib.sha256(data[t[4]:t[4]+t[5]]).hexdigest(),type=t[1],info=t[7]) for i,t in enumerate(rows)]


def registered_selectors(meta,names):
    entries=set(k.removeprefix('.ascend.meta.') for k in (meta if isinstance(meta,dict) else [v['name'] for v in meta]) if k.startswith('.ascend.meta.'));result=[]
    for name in names:
        found=[n for n in entries if n in (name,name+'_mix_aiv',name+'_mix_aic') or n.startswith('_Z'+str(len(name))+name)]
        if len(found)!=1:raise RuntimeError('native selector missing/ambiguous: '+name)
        result+=found
    return result


CMAKE='''cmake_minimum_required(VERSION 3.16)
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
set(CMAKE_ASC_FLAGS_RELEASE "-O3" CACHE STRING "Source Release flags" FORCE)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Source arch22" FORCE)
find_package(ASC CONFIG REQUIRED)
project(srotm_build_qualification LANGUAGES ASC CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
ascendc_library(srotm_before SHARED native.cpp)
target_compile_definitions(srotm_before PRIVATE BEFORE=@BEFORE@)
target_include_directories(srotm_before PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
get_target_property(sdk_includes srotm_before INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(sdk_linkdirs srotm_before INTERFACE_LINK_DIRECTORIES)
get_target_property(sdk_libraries srotm_before INTERFACE_LINK_LIBRARIES)
target_include_directories(srotm_before PRIVATE ${sdk_includes})
target_link_directories(srotm_before PRIVATE ${sdk_linkdirs})
target_link_libraries(srotm_before PRIVATE ${sdk_libraries})
ascendc_compile_options(srotm_before PRIVATE --cce-auto-sync=on "--cce-auto-sync-log=@SYNC_LOG@" -g)
target_compile_options(srotm_before PRIVATE -save-temps=obj)
'''


def build(arm,output):
    if arm not in ('before','after'):raise ValueError('arm')
    out=Path(output).resolve()/('native-'+arm+'-'+uuid.uuid4().hex[:8]);source=out/'source';source.mkdir(parents=True,exist_ok=False)
    shutil.copytree(HERE/'vendor',source/'vendor');shutil.copy2(HERE/'native.cpp',source/'native.cpp')
    (source/'srotm_dma.cpp').write_text(derivative((HERE/'vendor/srotm_kernel.cpp').read_text()))
    (source/'CMakeLists.txt').write_text(CMAKE.replace('@BEFORE@',str(int(arm=='before'))).replace('@SYNC_LOG@',str(out/'auto-sync.log')))
    for name,cmd,seconds in [('configure',['cmake','-S',str(source),'-B',str(out/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],120),('build',['cmake','--build',str(out/'build'),'--parallel','1','--verbose'],300)]:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=seconds);(out/(name+'.stdout')).write_text(result.stdout);(out/(name+'.stderr')).write_text(result.stderr)
        if result.returncode:raise RuntimeError('source ASC '+name+' failed; retained '+str(out))
    return qualify(arm,out,out)


def qualify(arm,out,resource_directory):
    out=Path(out);source=out/'source';build_dir=out/'build';binary=build_dir/'libsrotm_before.so'
    evidence=build_dir/'compile_commands.json'
    if evidence.is_file():command=json.loads(evidence.read_text())[0]['command'];command_kind='compile_commands'
    else:
        evidence=out/'build.stdout';candidates=[line.strip() for line in evidence.read_text().splitlines() if '--asc-aicore-lang' in line and ' -c ' in line]
        if len(candidates)!=1:raise RuntimeError('unique actual ASC verbose compile command required')
        command=candidates[0];command_kind='retained-verbose-build; compile_commands unavailable for SDK ASC generator'
    argv=__import__('shlex').split(command);compiler=Path(argv[0]).resolve()
    for required in ('--asc-aicore-lang','--npu-arch=dav-2201','-O3','--cce-auto-sync=on','-g','-DBEFORE='+str(int(arm=='before'))):
        if required not in argv:raise RuntimeError('unqualified ASC command: '+required)
    source_paths=[p for p in source.rglob('*') if p.is_file()]+[Path(__file__),evidence]+[p for p in (build_dir/'CMakeFiles/srotm_before.dir').glob('*.make')]+list((build_dir/'CMakeFiles/srotm_before.dir').glob('link.txt'))
    sdk=Path(os.environ['ASCEND_HOME_PATH'])/'compiler/tikcpp/ascendc_kernel_cmake'
    source_paths += [p for folder in ('asc_modules','fwk_modules') for p in (sdk/folder).glob('*.cmake')]
    manifest=dict(command=command,command_source=dict(kind=command_kind,path=str(evidence),sha256=digest(evidence)),source_revision=PIN,compiler=dict(path=str(compiler),sha256=digest(compiler)),binary=str(binary),binary_sha256=digest(binary),sources=[dict(path=str(p),sha256=digest(p)) for p in source_paths])
    objects=[];details=[];selected=[]
    names=['srotm_reset','srotm_kernel' if arm=='before' else 'srotm_kernel_dma']
    for kind,suffix in [('device_vec_o','-vec.o'),('device_cube_o','-cube.o')]:
        matches=list(build_dir.rglob('*'+suffix))
        if len(matches)!=1:raise RuntimeError('actual device object missing/ambiguous: '+kind)
        obj=matches[0];meta=sections(obj.read_bytes());entries=[v for v in meta if v['name'].startswith('.ascend.meta.')];symbols=subprocess.run(['nm','--defined-only',str(obj)],capture_output=True,text=True,check=True).stdout
        selectors=registered_selectors(meta,names) if entries else []
        if not all(any(line.split()[-1]==n for line in symbols.splitlines()) for n in selectors):raise RuntimeError('device defined symbol missing')
        if kind=='device_vec_o':
            if not selectors or not any(v['name']=='.text' and v['bytes'] for v in meta):raise RuntimeError('vector executable missing')
            selected=selectors
        objects.append((kind,obj));details.append(dict(kind=kind,sha256=digest(obj),sections=meta,registrations=selectors,symbol_table=symbols))
    exports=subprocess.run(['nm','-D','--defined-only',str(binary)],capture_output=True,text=True,check=True).stdout
    if any(not any(line.split()[-1]==name for line in exports.splitlines()) for name in ('catlass_bridge_abi_version','catlass_bridge_kernel_names','catlass_bridge_launch')):raise RuntimeError('host ABI exports missing')
    resources=dict(device_objects=details,selected_symbols=selected,host_binary_sha256=digest(binary),source_UB_bytes=196608,reset_UB_bytes=32768,compiler_managed_scratch='UNKNOWN if unreported',tiling_size=80,CDLL_loaded=False,actual_executed_engine='not inferred from compile; require actual capture')
    resource_directory=Path(resource_directory);resource_directory.mkdir(parents=True,exist_ok=True)
    for name,value in [('resources.json',resources),('build.json',manifest)]: (resource_directory/name).write_text(json.dumps(value,indent=2)+'\n')
    return dict(binary=binary,object=objects[0][1],objects=objects,manifest=manifest,resources=resources,selectors=selected,sources=source_paths+[resource_directory/'resources.json',resource_directory/'build.json'],directory=out)
