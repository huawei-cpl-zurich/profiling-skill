from contextlib import contextmanager
import math
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'examples/optimization-library/triton/wy_shared';sys.path[:0]=[str(HERE),str(ROOT/'scripts'),str(ROOT/'tests')]
import wy_shared as api
import wy_shared_contract as provider
import wy_shared_codegen as codegen
import test_optimization_profile_evidence as evidence_tests
import optimization_profile_evidence as evidence
SPEC=dict(T=5,packed=True,cu_seqlens=[0,5])
IR='%x = tt.dot %a, %b, %c, inputPrecision = ieee : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>'


class Kernel:
    def __init__(self,name,mode,f,state):self.fn=SimpleNamespace(__name__=name);self.mode=mode;self.f=f;self.state=state;self.compiles=0;self.launches=0;self.fail=False
    def warmup(self,*a,**kw):
        if self.state[0]!=self.f['inputs']['k'].device:raise AssertionError('wrong selected device')
        self.compiles+=1
        if self.fail:self.fail=False;raise RuntimeError('compile failed')
        return SimpleNamespace(hash='a'*64,asm={'npubin':b'\x7fELF'+self.mode.encode(),'ttir':IR})
    def __getitem__(self,grid):
        def run(*args,**kw):
            if self.state[0]!=self.f['inputs']['k'].device:raise AssertionError('wrong launch device')
            self.launches+=1
            for name,idx in [('w',3),('u',4)]:
                if self.mode in (name,'fused'):args[idx].copy_(self.f['expected'][name])
        return run


class WYTests(unittest.TestCase):
    def test_typed_reference_defaults_and_domain_hazards(self):
        old=torch.get_default_dtype()
        try:
            for dt in (torch.float64,torch.float16):
                torch.set_default_dtype(dt);f=provider.make_case(dict(SPEC,seed=315));self.assertEqual(f['expected']['u'].dtype,torch.float16)
                # A diag1 and dyadic offdiag/reference products give this manually summed CPU row.
                x=f['inputs'];j=2;h=1;d=3
                val=sum(float(x['A'][0,j,h,i])*float(x['v'][0,i,h,d])*float(x['beta'][0,h,i]) for i in range(j+1))
                self.assertEqual(float(f['expected']['u'][0,j,h,d]),float(torch.tensor(val,dtype=torch.float16,device='cpu')))
                def fp32(v):return float(torch.tensor(v,dtype=torch.float32,device='cpu'))
                val=sum(float(x['A'][0,j,h,i])*fp32(fp32(float(x['k'][0,i,h,d])*float(x['beta'][0,h,i]))*fp32(math.exp(float(x['g'][0,h,i])))) for i in range(j+1))
                self.assertEqual(float(f['expected']['w'][0,j,h,d]),float(torch.tensor(val,dtype=torch.float16,device='cpu')))
        finally:torch.set_default_dtype(old)
        for patching in ({'B':2},{'packed':False},{'cu_seqlens':[0,0,5]},{'cu_seqlens':None},{'H':16},{'Hg':4},{'K':64},{'V':64},{'BT':32},{'T':0},{'precision':'tf32'}):
            with self.assertRaises(ValueError):api.Adapter().compile_probe('before',dict(SPEC,**patching),'.')
        for mode in ('A','g','cu','indices','layout','dtype','alias'):
            f=provider.make_case(SPEC)
            if mode=='A':f['inputs']['A'][0,0,0,63]=1
            if mode=='g':f['inputs']['g'][0,0,0]=0
            if mode=='cu':f['inputs']['cu'][1]=4
            if mode=='indices':f['inputs']['indices'][0,0]=1
            if mode=='layout':f['inputs']['v']=f['inputs']['v'].transpose(-1,-2)
            if mode=='dtype':f['inputs']['beta']=f['inputs']['beta'].float()
            if mode=='alias':f['inputs']['v']=f['inputs']['k']
            with self.subTest(mode=mode),patch.dict(sys.modules,{'torch_npu':None}),self.assertRaises(ValueError):api.Adapter().prepare('after',SPEC,f,'.')

    def test_compile_atomic_ieee_probe_no_launch_device_and_dual_output_validation(self):
        f=provider.make_case(dict(SPEC,seed=316));state=['caller1']
        @contextmanager
        def device(d):
            old=state[0];state[0]=d
            try:yield
            finally:state[0]=old
        a,b,c=[Kernel('wy_'+mode+'_kernel',mode,f,state) for mode in ('w','u','fused')]
        modules={'torch_npu':SimpleNamespace(get_npu_format=lambda t:2,get_storage_size=lambda t:t.untyped_storage().nbytes()//t.element_size()),'wy_shared_generated':SimpleNamespace(wy_w_kernel=a,wy_u_kernel=b,wy_fused_kernel=c)}
        with patch.dict(sys.modules,modules),patch.object(torch,'npu',SimpleNamespace(device=device),create=True),patch.object(torch.Tensor,'data_ptr',return_value=128):
            adapter=api.Adapter();p=adapter._prepare('before',SPEC,f['inputs'],True)
            with self.assertRaisesRegex(RuntimeError,'cannot execute'):p.run()
            self.assertEqual(a.launches+b.launches,0);b.fail=True
            with self.assertRaisesRegex(RuntimeError,'compile failed'):p.compile_only()
            self.assertEqual(p.compilation_state(),[]);self.assertEqual(state[0],'caller1');self.assertEqual(len(p.compile_only()),2)
            with self.assertRaisesRegex(RuntimeError,'cannot execute'):p.run()
            self.assertEqual(a.launches+b.launches,0)
            for order in (('before','after'),('after','before')):
                adapter=api.Adapter()
                for arm in order:
                    item=adapter._prepare(arm,SPEC,f['inputs'],False);out=item.run();d=adapter.validate(arm,out,f,SPEC);self.assertTrue(d['passed'])
                self.assertEqual(state[0],'caller1');self.assertEqual(d['paired_bits_secondary'],dict(w=True,u=True));self.assertEqual(api.Adapter().items,{})
                item.stores[1][0]=0;self.assertFalse(adapter.validate(order[-1],out,f,SPEC)['passed'])
                with self.assertRaises(ValueError):adapter.validate(order[-1],out[:1],f,SPEC)
                bad=out[0].as_strided(out[0].shape,(*out[0].stride()[:-1],0))
                with self.assertRaises(ValueError):adapter.validate(order[-1],(bad,out[1]),f,SPEC)
        for bad in ('',IR.replace('ieee','tf32'),IR.replace('64x64xf32','64x64xf16',1)):
            with self.assertRaises(ValueError):api.ieee_policy(SimpleNamespace(asm={'ttir':bad}))
        self.assertEqual(api.ieee_policy(SimpleNamespace(asm={'ttir':IR}))['status'],'ieee')
        elided=IR.replace(', inputPrecision = ieee','')
        c=SimpleNamespace(asm={'ttir':elided},metadata=SimpleNamespace(default_dot_input_precision='ieee'))
        self.assertEqual(api.ieee_policy(c)['policy_origins'],['actualCompilerMetadataDefaultIEEE'])
        for default in (None,'hf32','tf32'):
            c.metadata.default_dot_input_precision=default
            with self.assertRaises(ValueError):api.ieee_policy(c)
        c.metadata.default_dot_input_precision='ieee'
        for ir in (IR.replace('ieee','hf32'),IR.replace('ieee','#unknown_precision')):
            c.asm['ttir']=ir
            with self.assertRaises(ValueError):api.ieee_policy(c)

    def test_generated_source_reproducible_and_real_collector_boundary(self):
        text,patchtext=codegen.generate()
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'generated.py';path.write_text(text);compile(path.read_text(),str(path),'exec')
            path,report=evidence_tests.PairTests().fixture(Path(t));compiled=api.identity([SimpleNamespace(hash='a'*64,asm={'npubin':b'\x7fELF','ttir':IR})]*2)
            for row in report['arms'].values():row['compilation_state_before_capture']=compiled;row['compilation_state_after_capture']=json.loads(json.dumps(compiled))
            path.write_text(json.dumps(report));self.assertEqual(evidence.collect_pair(path)['status'],'complete')
            report['arms']['after']['compilation_state_after_capture'][1]['binaries']['npubin']='f'*64;path.write_text(json.dumps(report))
            with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path)

    def test_capture_setup_forward_teardown_use_prepared_device_and_restore_caller(self):
        state=['caller1'];seen=[]
        @contextmanager
        def device(value):
            old=state[0];state[0]=value
            try:yield
            finally:state[0]=old
        def run():seen.append(('forward',state[0]))
        def capture(run,output,**kw):
            seen.append(('setup',state[0]));run();seen.append(('teardown',state[0]));return dict(status='complete',kernels=kw['expected_kernel'])
        item=SimpleNamespace(outputs=(SimpleNamespace(device='prepared0'),),expected_kernels=['wy_fused_kernel'])
        with patch.object(torch,'npu',SimpleNamespace(device=device),create=True),patch.dict(sys.modules,{'profile_ten':SimpleNamespace(capture_ten=capture)}):
            result=api.Adapter().capture(run,item,None)
            self.assertEqual(result,dict(status='complete',kernels=['wy_fused_kernel']));self.assertEqual(seen,[('setup','prepared0'),('forward','prepared0'),('teardown','prepared0')]);self.assertEqual(state[0],'caller1')
            def failed(*a,**kw):
                self.assertEqual(state[0],'prepared0');raise RuntimeError('export failed')
            with patch.dict(sys.modules,{'profile_ten':SimpleNamespace(capture_ten=failed)}),self.assertRaisesRegex(RuntimeError,'export failed'):api.Adapter().capture(run,item,None)
            self.assertEqual(state[0],'caller1')

    def test_external_cwd_cli_both_modules_and_strict_JSON(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'examples/optimization-library/triton/wy_shared';folder.mkdir(parents=True);(root/'scripts').mkdir()
            for name in ('wy_shared.py','wy_shared_contract.py'):shutil.copy2(HERE/name,folder/name)
            (root/'scripts/optimization_pair.py').write_text("import json\nfrom pathlib import Path\ndef run_pair(a,s,o,*args,**kw):\n f=a.make_case(s)\n Path(o).write_text(json.dumps({'outputs':sorted(f['expected'])}))\n return {'status':'complete'}\n")
            case=root/'case.json';case.write_text(json.dumps(SPEC)+'\n');prov=root/'prov.json';prov.write_text('{}\n');output=root/'output.json'
            args=[sys.executable,str(folder/'wy_shared.py'),'--case',str(case),'--output',str(output),'--cache',str(root/'cache'),'--provenance',str(prov)]
            p=subprocess.run(args,cwd='/',capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(output.read_text())['outputs'],['u','w'])
            case.write_text('{"T":NaN}\n');p=subprocess.run(args,cwd='/',capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('nonfinite JSON',p.stderr)


if __name__=='__main__':unittest.main()
