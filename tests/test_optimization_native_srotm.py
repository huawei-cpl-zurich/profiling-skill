"""NONFROZEN CPU/C-ABI/mock tests; no Ascend kernel builds or device calls."""
import ctypes,hashlib,json,shutil,struct,subprocess,sys,tempfile,unittest,weakref
from contextlib import contextmanager,ExitStack
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'examples/optimization-library/ascendc/srotm';sys.path.insert(0,str(HERE))
import torch
import srotm
from srotm_compile import registered_selectors,qualify
from srotm_contract import dimensions,tiling,ranges,make_case,check_inputs,check_output


def spec(n=29,inc=1,flag=-1.):return dict(seed=201010,parameters=dict(n=n,incx=inc,incy=inc,sparam=[flag,.75,-.25,.5,1.25]))


def collector_boundary(compiled,root):
    # Execute the real evidence consumer with NONFROZEN fixture events; no profiler/device calls.
    sys.path.insert(0,str(ROOT/'tests'))
    from test_optimization_profile_evidence import PairTests
    from optimization_profile_evidence import collect_pair
    if not isinstance(compiled,list) or compiled[0]['hash']!=compiled[0]['binaries']['device_vec_o']:raise ValueError('device ELF must be the primary identity')
    root=Path(root);root.mkdir(parents=True,exist_ok=False);path,report=PairTests().fixture(root)
    for row in report['arms'].values():row['compilation_state_before_capture']=row['compilation_state_after_capture']=compiled
    path.write_text(json.dumps(report));result=collect_pair(path,root=root)
    if any(row['status']!='complete' for row in result['arms'].values()):raise RuntimeError('mock collector boundary failed')
    return dict(status='complete',kind='NONFROZEN mock-profiler consumer boundary',new_NPU_calls=0,new_actual_captures=0,device_ELF_identity=compiled[0]['hash'])


class SrotmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.folder=Path(cls.tmp.name);cls.binary=cls.folder/'mock.so';src=cls.folder/'mock.cpp'
        src.write_text('''#include <cstring>
#include "srotm_tiling_data.h"
static_assert(sizeof(SrotmTilingData)==80 && alignof(SrotmTilingData)==8);
static unsigned calls=0; static uintptr_t last=0; static int fail=0;
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" unsigned counter(){return calls;} extern "C" uintptr_t stream_seen(){return last;}
extern "C" void set_fail(int v){fail=v;}
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream){
 if(count!=5)return 1;if(fail)return fail;const auto&t=*(SrotmTilingData*)p[4];
 if(t.x!=(uintptr_t)p[2]||t.y!=(uintptr_t)p[3]||t.xStorageSize!=t.yStorageSize)return 2;
 last=(uintptr_t)stream;calls+=2;float*x=(float*)p[2],*y=(float*)p[3];
 memcpy(x,p[0],t.xStorageSize*4);memcpy(y,p[1],t.yStorageSize*4);
 for(unsigned j=0;j<t.n;j++){unsigned i=j*t.incx;float a=x[i],b=y[i];x[i]=t.h11*a+t.h12*b;y[i]=t.h21*a+t.h22*b;}return 0;
}''')
        subprocess.run([shutil.which('g++') or 'g++','-std=c++17','-shared','-fPIC','-I'+str(HERE/'vendor'),str(src),'-o',str(cls.binary)],check=True)
        cls.b=dict(binary=cls.binary,manifest=dict(binary_sha256=hashlib.sha256(cls.binary.read_bytes()).hexdigest()))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_cpu_oracle_flag_and_stride_exactness(self):
        for flag in (-1.,0.):
            c=spec(31,2,flag);f=make_case(c);a,b=(v.double() for v in f['inputs'].values());x,y=f['expected']
            self.assertTrue(torch.equal(x[1::2],a.float()[1::2]));self.assertTrue(torch.equal(y[1::2],b.float()[1::2]))
            self.assertTrue(torch.equal(x[::2],((.75*a[::2]+.5*b[::2]) if flag else (a[::2]+.5*b[::2])).float()))
            self.assertTrue(torch.equal(y[::2],((-.25*a[::2]+1.25*b[::2]) if flag else (-.25*a[::2]+b[::2])).float()))
        c=spec(2051);raw=tiling(c,0x1000,0x2000);self.assertEqual(len(raw),80)
        self.assertEqual(struct.unpack_from('<QQ',raw),(0x1000,0x2000));self.assertEqual(struct.unpack_from('<II',raw,32),(2051,2))
        self.assertEqual(struct.unpack_from('<5f',raw,56),(-1.,.75,.5,-.25,1.25))
        for n in (9,2065,50111):
            rr=[(s,k) for s,k in ranges(spec(n)) if k];self.assertEqual(rr[0][0],0);self.assertEqual(sum(k for _,k in rr),n)
            self.assertTrue(all(a+k==b for (a,k),(b,_) in zip(rr,rr[1:])));self.assertTrue(all(s%128==0 for s,_ in rr))

    def test_native_real_C_ABI_two_mutable_outputs_reset_stream_and_errors(self):
        for flag in (-1.,0.):
            c=spec(19,2,flag);f=make_case(c);originals=tuple(f['inputs'].values());out=tuple(torch.full_like(x,999) for x in originals);stream=[0x1234]
            n=srotm.Native(self.b,originals,out,tiling(c,*[x.data_ptr() for x in out]),lambda:stream[0]);lib=n.library;lib.counter.restype=ctypes.c_uint;lib.stream_seen.restype=ctypes.c_uint64
            before=lib.counter();self.assertIs(n.run(),n.outputs);self.assertTrue(check_output(out,f['expected'])['passed'])
            stream[0]=0x5678;n.run();self.assertEqual(lib.stream_seen(),0x5678);self.assertEqual(lib.counter()-before,4);self.assertTrue(check_output(out,f['expected'])['passed'])
            lib.set_fail(7)
            with self.assertRaises(RuntimeError):n.run()
            lib.set_fail(0)
        with self.assertRaises(ValueError):srotm.Native(self.b,originals,out,b'bad',lambda:0)

    def test_domains_inputs_layout_alias_diagnostics_before_build(self):
        for key,value in [('n',0),('n',1048580),('incx',-1),('incx',0),('incy',2),('sparam',[1.,.75,-.25,.5,1.25])]:
            c=spec();c['parameters'][key]=value
            with self.assertRaises(ValueError):dimensions(c)
        c=spec();f=make_case(c)
        for name,value in [('X',f['inputs']['X'].double()),('X',torch.zeros(58)[::2]),('Y',f['inputs']['X']),('Y',torch.full((29,),float('inf')))]:
            bad=dict(f['inputs']);bad[name]=value
            with patch.object(srotm,'build') as build:
                with self.assertRaises(ValueError):srotm.Adapter().prepare('before',c,dict(f,inputs=bad),None)
                build.assert_not_called()
        x,y=(v.clone() for v in f['expected']);x[3]=float('nan');r=check_output((x,y),f['expected']);self.assertFalse(r['passed']);self.assertEqual(r['outputs'][0]['first_mismatch']['coordinates'],[3]);json.dumps(r,allow_nan=False)
        for actual in ((x,),[x,y],(x.double(),y)):self.assertFalse(check_output(actual,f['expected'])['passed'])

    def test_actual_collect_pair_canonical_identity_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=hashlib.sha256(self.binary.read_bytes()).hexdigest();state=[dict(hash=h,binaries=dict(device_vec_o=h))]
            self.assertEqual(collector_boundary(state,Path(tmp)/'mock')['status'],'complete')
            with self.assertRaises(ValueError):collector_boundary(dict(binary_sha256=h),Path(tmp)/'bad')

    def test_qualify_actual_verbose_boundary_without_compile_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'source').mkdir();(out/'source/native.cpp').write_text('mock source input')
            build=out/'build';build.mkdir();(build/'libsrotm_before.so').write_bytes(self.binary.read_bytes())
            (build/'native-vec.o').write_bytes(b'VEC');(build/'native-cube.o').write_bytes(b'CUBE')
            (out/'build.stdout').write_text(str(Path(sys.executable).resolve())+' -DBEFORE=1 --asc-aicore-lang --npu-arch=dav-2201 -O3 --cce-auto-sync=on -g -c native.cpp\n')
            def section(data):return [dict(name='.text',bytes=4 if data==b'VEC' else 0)]+([dict(name='.ascend.meta.'+name) for name in ('srotm_reset','_Z12srotm_kernel15SrotmTilingData')] if data==b'VEC' else [])
            stdout='\n'.join('0 T '+name for name in ('srotm_reset','_Z12srotm_kernel15SrotmTilingData','catlass_bridge_abi_version','catlass_bridge_kernel_names','catlass_bridge_launch'))
            with patch('srotm_compile.sections',section),patch('srotm_compile.subprocess.run',return_value=SimpleNamespace(stdout=stdout)),patch.dict(__import__('os').environ,{'ASCEND_HOME_PATH':str(out/'SDK')}):
                b=qualify('before',out,out/'qualification');self.assertEqual(len(b['objects']),2);self.assertIn('retained-verbose-build',b['manifest']['command_source']['kind'])
                self.assertEqual(b['selectors'],['srotm_reset','_Z12srotm_kernel15SrotmTilingData']);self.assertFalse((build/'compile_commands.json').exists())

    def test_registered_SDK_names_are_unique_and_exact(self):
        meta={'.ascend.meta.srotm_reset_mix_aiv':{},'.ascend.meta._Z12srotm_kernel15SrotmTilingData_mix_aiv':{}}
        self.assertEqual(registered_selectors(meta,['srotm_reset','srotm_kernel']),['srotm_reset_mix_aiv','_Z12srotm_kernel15SrotmTilingData_mix_aiv'])
        for wrong in ({'.ascend.meta.srotm_reset_aux':{}},dict(meta,**{'.ascend.meta.srotm_reset':{}})):
            with self.assertRaises(RuntimeError):registered_selectors(wrong,['srotm_reset'])

    def test_atomic_compile_retry_probe_zero_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=Path(tmp)/'vec.o';obj.write_bytes(b'mocked-device-object');b=dict(self.b,object=obj,resources={},sources=[],selectors=['reset','rotation'])
            a=srotm.Adapter(tmp);p=a.compile_probe('before',spec())
            with patch.object(srotm,'build',side_effect=[RuntimeError('late component/ABI qualification failure'),b]) as build:
                with self.assertRaises(RuntimeError):p.run()
                self.assertEqual(build.call_count,0)
                with self.assertRaises(RuntimeError):p.compile_only()
                self.assertEqual(a.builds,{});self.assertEqual(p.compilation_state(),[])
                p.compile_only();p.compile_only();self.assertEqual(build.call_count,2);self.assertEqual(len(p.compilation_state()),1)

    @contextmanager
    def mock_device(self):
        state=SimpleNamespace(current=1,allocations=[],guards=[])
        @contextmanager
        def device(index):
            old=state.current;state.current=index;state.guards.append(index)
            try:yield
            finally:state.current=old
        npu=SimpleNamespace(current_device=lambda:state.current,device=device,current_stream=lambda:SimpleNamespace(npu_stream=0x1000+state.current))
        real_full=torch.full;real_to=torch.Tensor.to
        def full(*args,**kw):
            state.allocations.append((kw['device'],state.current));kw['device']='cpu';size=args[0][0];store=real_full((size+32,),args[1],**kw);offset=((128-store.data_ptr()%128)%128)//4
            return store[offset:offset+size]
        def to(x,*args,**kw):
            if args and str(args[0]).startswith('npu'):
                state.allocations.append((args[0],state.current));return x.clone()
            return real_to(x,*args,**kw)
        with ExitStack() as stack:
            for patcher in (patch.dict(sys.modules,{'torch_npu':object()}),patch.object(torch,'full',full),patch.object(torch.Tensor,'to',to),patch.object(torch,'npu',npu,create=True),patch.object(srotm,'audit_tensor',lambda x:{})):stack.enter_context(patcher)
            yield state

    def mocked_adapter(self):
        a=srotm.Adapter();b=dict(self.b,object=self.binary,resources={},sources=[],selectors=['reset','rotation']);a.builds=dict(before=b,after=b);return a

    def test_reused_adapter_orders_defaultdtype_canary_immutable(self):
        old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);self.addCleanup(torch.set_default_dtype,old)
        c=spec();f=make_case(c);f['inputs']=dict(Y=f['inputs']['Y'],X=f['inputs']['X']);a=self.mocked_adapter()
        with self.mock_device():
            for order in (('before','after'),('after','before'),('before','after')):
                for arm in order:
                    item=a.prepare(arm,c,f,None);actual=item.run();r=a.validate(arm,actual,f,c);self.assertTrue(r['passed'],r);self.assertEqual(actual[0].dtype,torch.float32);self.assertEqual(r['paired_bits_secondary'],None if arm==order[0] else True)
                    item.stores[0][0]=0;self.assertFalse(a.validate(arm,actual,f,c)['passed']);item.stores[0][0]=12345
                    item.originals[0][0]+=1;self.assertFalse(a.validate(arm,actual,f,c)['passed'])
                for item in list(a.items.values()):item.close()

    def test_device_switch_own_close_and_real_run_pair_cleanup(self):
        from optimization_pair import run_pair
        c=spec();f=make_case(c);a=self.mocked_adapter()
        with self.mock_device() as state:
            first=a.prepare('before',c,f,None);token=first.native.token;refs=[weakref.ref(x) for x in (*first.originals,*first.stores)]
            state.current=2;other=a.prepare('after',c,f,None);other_token=other.native.token
            first.run();self.assertEqual(first.native.library.stream_seen(),0x1001);self.assertEqual(state.current,2)
            self.assertTrue(all(request=='npu:'+str(current) for request,current in state.allocations))
            first.close();first.close();self.assertNotIn(token,srotm._LAUNCHES);self.assertIn(other_token,srotm._LAUNCHES);self.assertEqual(state.current,2);self.assertTrue(all(x() is None for x in refs))
            with self.assertRaises(RuntimeError):first.run()
            other.close();self.assertEqual(a.items,{})
            old=a.prepare('before',c,f,None);new=a.prepare('before',c,f,None);old.close()
            self.assertIs(a.items['before'],new);self.assertIn(new.native.token,srotm._LAUNCHES);new.close()
            for reject in (False,True):
                a=self.mocked_adapter();saved=[];prepare=a.prepare
                def record(*args):
                    item=prepare(*args);saved.append(item);return item
                a.prepare=record
                if reject:a.validate=lambda *args:dict(passed=False)
                with tempfile.TemporaryDirectory() as tmp:
                    report=run_pair(a,c,Path(tmp)/'out',Path(tmp)/'cache',{'physical_die':0},torch=torch,correctness_only=True)
                    self.assertEqual(report['status'],'failed' if reject else 'complete')
                    self.assertTrue(all(row['cleanup']['status']=='complete' for row in report['arms'].values()))
                    self.assertEqual(a.items,{})
                    self.assertTrue(all(item.native.closed and not item.stores and item.native.token not in srotm._LAUNCHES for item in saved))

    def test_externalcwd_CLI_strict_JSON_and_domain_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);case=t/'case.json';case.write_text(json.dumps(dict(spec(),role='guard')));prov=t/'prov.json';prov.write_text('{}');cli=HERE/'srotm.py';args=['--case',str(case),'--output',str(t/'out'),'--cache',str(t/'cache'),'--provenance',str(prov)]
            result=subprocess.run([sys.executable,str(cli),*args,'--validate-only'],cwd=tmp,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['role'],'guard')
            wrapper="import runpy,sys,types,json,pathlib; sys.path.insert(0,str(pathlib.Path(sys.argv[1]).parent)); m=types.ModuleType('optimization_pair'); m.run_pair=lambda adapter,spec,out,cache,provenance:(pathlib.Path(out).mkdir(),(pathlib.Path(out)/'result.json').write_text(json.dumps({'status':'complete'})+'\\n'),{'status':'complete'})[-1]; sys.modules['optimization_pair']=m; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
            result=subprocess.run([sys.executable,'-c',wrapper,str(cli),*args],cwd=tmp,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads((t/'out/result.json').read_text())['status'],'complete')
            case.write_text(json.dumps(spec(0)));result=subprocess.run([sys.executable,str(cli),*args],cwd=tmp,capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse((t/'cache').exists())


if __name__=='__main__':unittest.main()
