import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch,ctypes,hashlib,shutil,weakref
from contextlib import contextmanager,ExitStack
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'examples/optimization-library/ascendc/sasum'
sys.path[:0]=[str(HERE),str(ROOT/'scripts'),str(ROOT/'tests')]
import sasum
import optimization_native_bridge as bridge
import sasum_contract as provider
import sasum_compile as compiler
import test_optimization_profile_evidence as evidence_tests
import optimization_profile_evidence as evidence


class SasumTests(unittest.TestCase):
    def test_source_partition_and_independent_nonfrozen_expected(self):
        for n in (1,7,8,23,67,1059,164003):
            raw=struct.unpack('<Ii81I',provider.tiling(n,2));offset=raw[3:43];count=raw[43:83]
            owner=[j for a,c in zip(offset[:8],count[:8]) for j in range(a,a+c)]
            self.assertEqual(owner,list(range(n)))
            self.assertTrue(all(a%8==0 for a,c in zip(offset,count) if c))
        old=torch.get_default_dtype()
        try:
            for dt in (torch.float64,torch.float16):
                torch.set_default_dtype(dt);f=provider.make_case(dict(n=37,incx=2,seed=214))
                self.assertEqual(f['x'].dtype,torch.float32);self.assertEqual(f['x'].stride(),(2,))
                self.assertEqual(float(f['expected'][0]),sum(abs(float(v)) for v in f['x']))
                provider.check_fixture(dict(n=37,incx=2),f)
        finally:torch.set_default_dtype(old)

    def test_domain_alias_holes_and_default_strictness(self):
        for s in ({'n':True},{'n':0},{'n':9,'incx':True},{'n':9,'incx':3}):
            with self.assertRaises(ValueError):sasum.Adapter(compile_inspection=True).compile_probe('before',s,'.')
        s=dict(n=19,incx=2);f=provider.make_case(s)
        for key,value in [('x',f['x'].double()),('expected',f['x'][:1])]:
            with self.assertRaises(ValueError):provider.check_fixture(s,dict(f,**{key:value}))
        provider.full(f['x'])[33]=0
        with self.assertRaisesRegex(ValueError,'hole'):provider.check_fixture(s,f)

    def test_probe_zero_launch_atomic_failure_retry_and_repeated_adapters(self):
        with tempfile.TemporaryDirectory() as t:
            obj=Path(t)/'device-vec.o';obj.write_bytes(b'device-ELF');(Path(t)/'device-cube.o').write_bytes(b'cube-ELF');b=dict(object=obj,resources={},binary=obj)
            a=sasum.Adapter(compile_inspection=True);p=a.compile_probe('after',dict(n=11),t)
            with patch.object(sasum,'build',side_effect=[RuntimeError('compile failed'),dict(b,objects=[('device_aic_elf',Path(t)/'device-cube.o'),('device_aiv_elf',obj)])]) as build:
                with self.assertRaisesRegex(RuntimeError,'cannot execute'):p.run()
                self.assertEqual(build.call_count,0)
                with self.assertRaisesRegex(RuntimeError,'compile failed'):p.compile_only()
                self.assertEqual(a.builds,{})
                self.assertEqual(len(p.compile_only()),2);p.compile_only();self.assertEqual(build.call_count,2)
                with self.assertRaisesRegex(RuntimeError,'cannot execute'):p.run()
                self.assertEqual(p.compilation_state(),sasum.binary_state(a.builds['after']));self.assertEqual(sasum.Adapter().builds,{})

    def test_actual_collector_normalization_accepts_device_identity_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);obj=root/'selected-aiv.o';obj.write_bytes(b'selected device ELF');(root/'selected-cube.o').write_bytes(b'actual cube component')
            path,report=evidence_tests.PairTests().fixture(root)
            for row in report['arms'].values():
                row['compilation_state_before_capture']=sasum.binary_state(dict(object=obj,objects=[('device_aic_elf',root/'selected-cube.o'),('device_aiv_elf',obj)]))
                row['compilation_state_after_capture']=sasum.binary_state(dict(object=obj,objects=[('device_aic_elf',root/'selected-cube.o'),('device_aiv_elf',obj)]))
            path.write_text(json.dumps(report)+'\n');out=evidence.collect_pair(path)
            self.assertEqual(out['status'],'complete');self.assertEqual(out['arms']['after']['event_latency_ms']['count'],10)
            report['arms']['after']['compilation_state_after_capture'][0]['binaries']['device_aic_elf']='f'*64
            path.write_text(json.dumps(report)+'\n')
            with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path)

    def test_registration_suffix_and_external_cwd_cli_strict_json(self):
        name='_Z12sasum_kernelPhS_S_S__mix_aiv'
        self.assertEqual(compiler.registered_selectors({'.ascend.meta.'+name:{}},['sasum_kernel']),[name])
        with self.assertRaises(RuntimeError):compiler.registered_selectors({'.ascend.meta.sasum_kernel':{},'.ascend.meta.sasum_kernel_mix_aiv':{}},['sasum_kernel'])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);case=p/'case.json';case.write_text('{"n":false}\n');prov=p/'provenance.json';prov.write_text('{}\n')
            out=subprocess.run([sys.executable,str(HERE/'sasum.py'),'--case',str(case),'--output',str(p/'out'),'--cache',str(p/'cache'),'--provenance',str(prov)],cwd=t,capture_output=True,text=True)
            self.assertNotEqual(out.returncode,0);self.assertIn('profile accounting',out.stderr);self.assertFalse((p/'cache').exists())

    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.binary=Path(cls.tmp.name)/'mock.so';src=cls.binary.with_suffix('.cpp')
        src.write_text('''#include <cstdint>
#include <cmath>
static uintptr_t last=0;
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\\"sasum_kernel\\\"]";}
extern "C" uintptr_t stream_seen(){return last;}
extern "C" int catlass_bridge_launch(void**p,uint64_t count,void*stream){
 if(count!=4)return 1;last=(uintptr_t)stream;auto*t=(uint32_t*)p[2];float sum=0;auto*x=(float*)p[0];
 for(uint32_t j=0;j<t[0];j++)sum+=std::fabs(x[j*t[1]]);*(float*)p[3]=sum;return 0;}
''')
        subprocess.run([shutil.which('g++') or 'g++','-shared','-fPIC',str(src),'-o',str(cls.binary)],check=True)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    @contextmanager
    def mock_device(self):
        state=SimpleNamespace(current=1,allocations=[])
        @contextmanager
        def device(index):
            old=state.current;state.current=index
            try:yield
            finally:state.current=old
        npu=SimpleNamespace(current_device=lambda:state.current,device=device,current_stream=lambda:SimpleNamespace(npu_stream=0x1000+state.current))
        real_full=torch.full;real_to=torch.Tensor.to
        def full(*args,**kw):
            if kw.get('device')=='cpu':return real_full(*args,**kw)
            state.allocations.append((kw['device'],state.current));kw['device']='cpu';size=args[0][0];store=real_full((size+32,),args[1],**kw);offset=((128-store.data_ptr()%128)%128)//4;return store[offset:offset+size]
        def to(x,*args,**kw):
            if args and str(args[0]).startswith('npu'):
                state.allocations.append((args[0],state.current));data=bytearray(x.numel()*x.element_size()+127);offset=(-ctypes.addressof(ctypes.c_char.from_buffer(data)))%128;out=torch.frombuffer(memoryview(data)[offset:offset+x.numel()*x.element_size()],dtype=x.dtype);out.copy_(x);return out
            return real_to(x,*args,**kw)
        with ExitStack() as stack:
            for q in (patch.dict(sys.modules,{'torch_npu':object()}),patch.object(torch,'full',full),patch.object(torch.Tensor,'to',to),patch.object(torch,'npu',npu,create=True),patch.object(bridge,'audit_tensor',lambda x,**kw:{})):stack.enter_context(q)
            yield state
    def adapter(self):
        a=sasum._NumericalAdapter();b=dict(binary=self.binary,object=self.binary,manifest=dict(binary_sha256=bridge.digest(self.binary)),resources={},sources=[],selectors=['sasum_kernel']);a.builds=dict(before=b,after=b);return a

    def test_real_dispatch_device_switch_close_and_run_pair(self):
        from optimization_pair import run_pair
        c=dict(parameters=dict(n=31,incx=2),seed=210410);f=provider.make_case(c);a=self.adapter()
        with self.mock_device() as state:
            first=a.prepare('before',c,f,None);token=first.native._token;refs=[weakref.ref(x) for x in (first.x,*first.stores)]
            state.current=2;other=a.prepare('after',c,f,None);first.run();first.native._library.stream_seen.restype=ctypes.c_uint64
            self.assertEqual(first.native._library.stream_seen(),0x1001);self.assertEqual(state.current,2)
            self.assertTrue(a.validate('before',first.result,f,c)['passed']);self.assertEqual(state.current,2)
            first.close();first.close();self.assertNotIn(token,bridge._LAUNCHES);self.assertIn(other.native._token,bridge._LAUNCHES);self.assertTrue(all(x() is None for x in refs));self.assertEqual(state.current,2)
            with self.assertRaises(RuntimeError):first.run()
            other.close()
            old=a.prepare('before',c,f,None);new=a.prepare('before',c,f,None);old.close();self.assertIs(a.items['before'],new);new.close()
            self.assertTrue(all(request=='npu:'+str(current) for request,current in state.allocations))
            for order in (('before','after'),('after','before')):
                a=self.adapter()
                for arm in order:
                    item=a.prepare(arm,c,f,None);r=a.validate(arm,item.run(),f,c);self.assertTrue(r['passed']);self.assertIsNone(r['paired_bitwise_secondary']) if arm==order[0] else self.assertTrue(r['paired_bitwise_secondary'])
                for item in list(a.items.values()):item.close()
            for fail in (False,True):
                a=self.adapter();saved=[];prepare=a.prepare
                def record(*args):
                    item=prepare(*args);saved.append(item);return item
                a.prepare=record
                if fail:a.validate=lambda *args:dict(passed=False)
                with tempfile.TemporaryDirectory() as t:
                    report=run_pair(a,c,Path(t)/'out',Path(t)/'cache',{},torch=torch,correctness_only=True)
                    self.assertEqual(report['status'],'failed' if fail else 'complete',report);self.assertEqual(a.items,{})
                    self.assertTrue(all(i.native.closed and i.native._token not in bridge._LAUNCHES for i in saved))

    def test_default_rejects_before_provider_cache_backend_or_compile(self):
        with patch.object(sasum,'build',side_effect=AssertionError('compiler forbidden')),patch.object(sasum,'_provider_make_case',side_effect=AssertionError('oracle forbidden')):
            a=sasum.Adapter()
            for fn,args in ((sasum.factory,()),(sasum.make_case,({'n':13},)),(a.make_case,({'n':13},)),(a.prepare,('before',{},None,'.')),(a.validate,()),(a.capture,()),(a.compile_probe,('before',{'n':13},'.'))):
                with self.assertRaises(sasum.StudyUnqualified):fn(*args)
            self.assertEqual(a.builds,{})

    def test_actual_ASC_inspection_without_compile_commands_and_empty_cube(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);(out/'source').mkdir();(out/'source/native.cpp').write_text('mock source');build=out/'build';build.mkdir()
            (build/'libsasum_before.so').write_bytes(self.binary.read_bytes());(build/'native-vec.o').write_bytes(b'VEC');(build/'native-cube.o').write_bytes(b'CUBE')
            command=str(Path(sys.executable).resolve())+' -DBEFORE=1 --asc-aicore-lang --npu-arch=dav-2201 -O3 --cce-auto-sync=on -g -c native.cpp'
            (out/'build.stdout').write_text(command+'\n')
            def sections(data):return [dict(name='.text',bytes=4 if data==b'VEC' else 0)]+([dict(name='.ascend.meta._Z12sasum_kernelPhS_S_S_')] if data==b'VEC' else [])
            symbols='\n'.join('0 T '+n for n in ('_Z12sasum_kernelPhS_S_S_','catlass_bridge_abi_version','catlass_bridge_kernel_names','catlass_bridge_launch'))
            with patch.object(compiler,'sections',sections),patch.object(compiler.subprocess,'run',return_value=SimpleNamespace(stdout=symbols)),patch.dict(__import__('os').environ,{'ASCEND_HOME_PATH':str(out/'SDK')}):
                b=compiler.qualify('before',out,out/'qualified');self.assertEqual(len(b['objects']),2);self.assertIn('retained-verbose-build',b['manifest']['command_source']['kind']);self.assertFalse(b['resources']['CDLL_loaded'])
                (out/'build.stdout').write_text(command.replace('-O3','-O2')+'\n')
                with self.assertRaisesRegex(RuntimeError,'-O3'):compiler.qualify('before',out,out/'bad')


if __name__=='__main__':unittest.main()
