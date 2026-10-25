"""NONFROZEN CPU/CABI/mock behavior tests: zero Ascend compilation or NPU calls."""
import ctypes,hashlib,json,shutil,struct,subprocess,sys,tempfile,unittest,weakref
from contextlib import contextmanager,ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'examples/optimization-library/ascendc/sger';sys.path.insert(0,str(HERE))
import torch
import sger
from sger_contract import dimensions,tiling,make_case,check_inputs,check_output
from sger_compile import registered_selectors


def spec(m=13,n=7,lda=16,alpha=.5):return dict(seed=105005,parameters=dict(m=m,n=n,lda=lda,alpha=alpha,incx=1,incy=1,aiv_cores=48))


class SgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.folder=Path(cls.tmp.name);cls.binary=cls.folder/'mock.so';src=cls.folder/'mock.cpp'
        src.write_text('''#include <cstring>
#include <cstddef>
#include "sger_tiling.h"
static_assert(sizeof(SgerTilingData)==56 && alignof(SgerTilingData)==8);
static_assert(offsetof(SgerTilingData,m)==24 && offsetof(SgerTilingData,alpha)==52);
static unsigned calls=0; static uintptr_t last=0; static int fail=0;
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" unsigned counter(){return calls;} extern "C" uintptr_t stream_seen(){return last;}
extern "C" void set_fail(int v){fail=v;}
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream){
 if(count!=6)return 1;if(fail)return fail;const auto&t=*(SgerTilingData*)p[5];
 if(t.A!=(uintptr_t)p[3]||t.x!=(uintptr_t)p[1]||t.y!=(uintptr_t)p[2])return 2;
 last=(uintptr_t)stream;calls++;float*a=(float*)p[3],*x=(float*)p[1],*y=(float*)p[2];memcpy(a,p[0],t.lda*t.n*4);
 for(unsigned j=0;j<t.n;j++)for(unsigned i=0;i<t.m;i++)a[j*t.lda+i]+=x[i]*(t.alpha*y[j]);return 0;
}''')
        subprocess.run([shutil.which('g++') or 'g++','-std=c++17','-shared','-fPIC','-ffp-contract=off','-I'+str(HERE/'vendor'),str(src),'-o',str(cls.binary)],check=True)
        cls.b=dict(binary=cls.binary,manifest=dict(binary_sha256=hashlib.sha256(cls.binary.read_bytes()).hexdigest()),object=cls.binary,resources={},sources=[],selectors=['reset','update'])
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_independent_oracle_padding_and_real_C_ABI_reset_stream_errors(self):
        for alpha in (.5,-.25):
            c=spec(alpha=alpha);f=make_case(c);raw=tiling(c,0x1000,0x2000,0x3000);self.assertEqual(len(raw),56);self.assertEqual(struct.unpack_from('<QQQ',raw),(0x1000,0x2000,0x3000))
            a=f['expected'].reshape(7,16);original=f['inputs']['A'].reshape(7,16);self.assertTrue(torch.equal(a[:,13:],original[:,13:]))
            self.assertTrue(torch.equal(a[:,:13],original[:,:13]+torch.outer(alpha*f['inputs']['Y'],f['inputs']['X'])))
            inputs=tuple(f['inputs'][k]for k in ('A','X','Y'));outputs=tuple(torch.full_like(inputs[0],999)for _ in range(2));stream=[123]
            native=sger.Native(self.b,inputs,outputs,tiling(c,outputs[0].data_ptr(),inputs[1].data_ptr(),inputs[2].data_ptr()),lambda:stream[0]);native.library.stream_seen.restype=ctypes.c_uint64
            self.assertTrue(check_output(native.run(),f['expected'])['passed']);stream[0]=456
            self.assertTrue(check_output(native.run(),f['expected'])['passed']);self.assertEqual(native.library.stream_seen(),456)
            native.library.set_fail(7)
            with self.assertRaises(RuntimeError):native.run()
            native.library.set_fail(0);native.close();native.close()
            with self.assertRaises(RuntimeError):native.run()

    def test_reject_unsafe_source_partition_invalid_inputs_before_build(self):
        for m,n,lda in [(1,65,8),(512,256,520),(1024,512,1024),(0,1,8),(1,1,0)]:
            with self.assertRaises(ValueError):dimensions(spec(m,n,lda))
        f=make_case(spec())
        for name,x in [('X',f['inputs']['X'].double()),('A',torch.zeros(16*14)[::2]),('Y',torch.full((7,),float('inf'))),('Y',torch.ones(7)/3)]:
            bad=dict(f['inputs']);bad[name]=x
            with patch.object(sger,'build')as build:
                with self.assertRaises(ValueError):sger.Adapter().prepare('before',spec(),dict(f,inputs=bad),None)
                build.assert_not_called()
        actual=f['expected'].clone();actual[5]=float('nan');row=check_output(actual,f['expected']);self.assertFalse(row['passed']);self.assertEqual(row['first_mismatch']['coordinates'],[5]);json.dumps(row,allow_nan=False)
        x=torch.tensor([-0.,1.],dtype=torch.float32);self.assertTrue(check_output(x,torch.tensor([0.,1.]))['passed'])

    @contextmanager
    def device_mock(self):
        state=SimpleNamespace(current=1,allocations=[])
        @contextmanager
        def device(index):
            old=state.current;state.current=index
            try:yield
            finally:state.current=old
        npu=SimpleNamespace(current_device=lambda:state.current,device=device,current_stream=lambda:SimpleNamespace(npu_stream=0x1000+state.current))
        full=torch.full;to=torch.Tensor.to
        def allocate(*args,**kw):
            state.allocations.append((kw['device'],state.current));kw['device']='cpu';size=args[0][0];buf=full((size+32,),args[1],**kw);offset=((128-buf.data_ptr()%128)%128)//4;return buf[offset:offset+size]
        def move(x,*args,**kw):
            if args and str(args[0]).startswith('npu'):state.allocations.append((args[0],state.current));return x.clone()
            return to(x,*args,**kw)
        with ExitStack()as stack:
            for p in (patch.dict(sys.modules,{'torch_npu':object()}),patch.object(torch,'npu',npu,create=True),patch.object(torch,'full',allocate),patch.object(torch.Tensor,'to',move),patch.object(sger,'audit_tensor',lambda x:{}),patch.object(sger,'runtime_cores',return_value={'num_vectorcore':48})):stack.enter_context(p)
            yield state

    def test_device_restore_entire_capture_and_cleanup(self):
        c=spec();f=make_case(c);a=sger.Adapter();a.builds=dict(before=self.b,after=self.b)
        with self.device_mock()as state:
            first=a.prepare('before',c,f,None);token=first.native.token;refs=[weakref.ref(x)for x in (*first.originals,*first.stores)]
            state.current=2;second=a.prepare('after',c,f,None);first.run();self.assertEqual(first.native.library.stream_seen(),0x1001);self.assertEqual(state.current,2)
            self.assertTrue(a.validate('before',first.outputs[0],f,c)['passed'])
            def capture(run,output,**kw):
                self.assertEqual(state.current,1);self.assertEqual(kw['expected_kernel'],['reset','update']);run();self.assertEqual(state.current,1);return {'status':'mocked-complete'}
            with patch.dict(sys.modules,{'profile_ten':SimpleNamespace(capture_ten=capture)}):self.assertEqual(a.capture(first.run,first,Path('mock'))['status'],'mocked-complete')
            self.assertEqual(state.current,2);self.assertTrue(all(target=='npu:'+str(current)for target,current in state.allocations))
            first.close();self.assertNotIn(token,sger._LAUNCHES);self.assertIn(second.native.token,sger._LAUNCHES);self.assertTrue(all(x()is None for x in refs));self.assertEqual(state.current,2);second.close();self.assertEqual(a.items,{})

    def test_actual_core_mismatch_before_build_or_movement_and_pair_cleanup(self):
        c=spec();f=make_case(c)
        with self.device_mock():
            with patch.object(sger,'runtime_cores',side_effect=RuntimeError('actualdifferentcores')),patch.object(sger,'build')as build:
                with self.assertRaises(RuntimeError):sger.Adapter().prepare('before',c,f,None)
                build.assert_not_called()
            from optimization_pair import run_pair
            for fail in (False,True):
                a=sger.Adapter();a.builds=dict(before=self.b,after=self.b);saved=[];prepare=a.prepare
                def record(*args):
                    item=prepare(*args);saved.append(item);return item
                a.prepare=record
                if fail:a.validate=lambda *args:dict(passed=False)
                with tempfile.TemporaryDirectory()as tmp:
                    report=run_pair(a,c,Path(tmp)/'out',Path(tmp)/'cache',{'physical_die':0},torch=torch,correctness_only=True)
                    self.assertEqual(report['status'],'failed'if fail else 'complete');self.assertEqual(a.items,{})
                    self.assertTrue(all(item.native.closed and not item.stores and item.native.token not in sger._LAUNCHES for item in saved))

    def test_compile_only_failure_atomic_cache_and_selectors(self):
        a=sger.Adapter();p=a.compile_probe('before',spec())
        with patch.object(sger,'build',side_effect=[RuntimeError('mockwholeprogramfailure'),self.b])as build:
            with self.assertRaises(RuntimeError):p.run()
            self.assertEqual(build.call_count,0)
            with self.assertRaises(RuntimeError):p.compile_only()
            self.assertEqual(a.builds,{});self.assertEqual(p.compilation_state(),[]);p.compile_only();p.compile_only();self.assertEqual(build.call_count,2)
        meta=[{'name':'.ascend.meta._Z11sger_kernel14SgerTilingData'}];self.assertEqual(registered_selectors(meta,['sger_kernel']),['_Z11sger_kernel14SgerTilingData'])
        with self.assertRaises(RuntimeError):registered_selectors(meta,['sger_product'])


if __name__=='__main__':unittest.main()
