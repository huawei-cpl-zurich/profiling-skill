from contextlib import contextmanager
import hashlib
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
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'examples/optimization-library/triton'
sys.path[:0]=[str(HERE),str(ROOT/'scripts'),str(ROOT/'tests')]
import request_sentinel as api
import request_sentinel_contract as provider
import test_optimization_profile_evidence as evidence_tests
import optimization_profile_evidence as evidence
SPEC=dict(batch=5,length_mode='explicit',lengths=[0,1,2,3,0],num_tokens=6,replace_from=-9,replace_to=42)


class Kernel:
    def __init__(self,name,replace=False,token=False):self.fn=SimpleNamespace(__name__=name);self.replace=replace;self.token=token;self.compiles=0;self.launches=0;self.fail=False;self.check_device=lambda:None
    def warmup(self,*args,**kwargs):
        self.check_device();self.compiles+=1
        if self.fail:self.fail=False;raise RuntimeError('compile failed')
        return SimpleNamespace(hash=hashlib.sha256(self.fn.__name__.encode()).hexdigest(),asm={'npubin':b'\x7fELF'+self.fn.__name__.encode()})
    def __getitem__(self,grid):
        def run(out,*args,**kwargs):
            self.check_device();self.launches+=1
            if self.token:
                rf,rt=args;out.masked_fill_(out==rf,rt);return
            x,ends,rf,rt=args;start=0
            for value,end in zip(x.tolist(),ends.tolist()):
                out[start:end]=rt if self.replace and value==rf else value;start=end
        return run


class SentinelTests(unittest.TestCase):
    def modules(self):
        a,b,c=Kernel('expand_kernel',True),Kernel('expand_unreplaced_kernel'),Kernel('replace_tokens_kernel',token=True)
        return dict(torch_npu=SimpleNamespace(get_npu_format=lambda t:2,get_storage_size=lambda t:t.untyped_storage().nbytes()//4),request_sentinel_kernels=SimpleNamespace(expand_kernel=a,expand_unreplaced_kernel=b,replace_tokens_kernel=c)),(a,b,c)

    def test_oracle_dtype_domain_prefix_layout_alias(self):
        old=torch.get_default_dtype()
        try:
            for dt in (torch.float64,torch.float16):
                torch.set_default_dtype(dt);f=provider.make_case(dict(SPEC,seed=246));self.assertEqual(f['expected'].dtype,torch.int32)
                self.assertEqual(f['expected'].tolist(),[f['inputs']['x'][1].item()]+[f['inputs']['x'][2].item()]*2+[42]*3)
        finally:torch.set_default_dtype(old)
        bad=[dict(SPEC,batch=0),dict(SPEC,num_tokens=7),dict(SPEC,lengths=[0,1,2,33,0]),dict(SPEC,lengths=[0]*5,num_tokens=0),dict(SPEC,replace_from=True),dict(SPEC,replace_to=2**31),dict(SPEC,replace_to=-9)]
        for spec in bad:
            with self.assertRaises(ValueError):api.Adapter().compile_probe('before',spec,'.')
        for mode in ('negative','decrease','large','trailing','dtype','layout','alias'):
            f=provider.make_case(SPEC)
            if mode=='negative':f['inputs']['ends'][0]=-1
            if mode=='decrease':f['inputs']['ends'][3]=1
            if mode=='large':f['inputs']['ends'][3]=100
            if mode=='trailing':f['inputs']['ends'][-1]=7
            if mode=='dtype':f['inputs']['x']=f['inputs']['x'].long()
            if mode=='layout':f['inputs']['x']=f['inputs']['x'].as_strided((5,),(2,))
            if mode=='alias':f['inputs']['ends']=f['inputs']['x']
            with self.subTest(mode=mode),patch.dict(sys.modules,{'torch_npu':None}),self.assertRaises(ValueError):api.Adapter().prepare('after',SPEC,f,'.')

    def test_compile_transaction_probe_zero_launch_and_both_validation_orders(self):
        modules,kernels=self.modules();f=provider.make_case(SPEC);device_state=['caller-device-1']
        @contextmanager
        def selected(device):
            old=device_state[0];device_state[0]=device
            try:yield
            finally:device_state[0]=old
        def check_device():self.assertEqual(device_state[0],f['inputs']['x'].device)
        for k in kernels:k.check_device=check_device
        # CPU mocks supply physical metadata; no device pointer is dereferenced by a mock kernel.
        with patch.dict(sys.modules,modules),patch.object(torch.Tensor,'data_ptr',return_value=128),patch.object(torch,'npu',SimpleNamespace(device=selected),create=True):
            adapter=api.Adapter();p=adapter._prepare('before',SPEC,**f['inputs'],probe=True)
            with self.assertRaisesRegex(RuntimeError,'cannot execute'):p.run()
            self.assertEqual(sum(k.compiles+k.launches for k in kernels),0)
            kernels[2].fail=True
            with self.assertRaisesRegex(RuntimeError,'compile failed'):p.compile_only()
            self.assertEqual(p.compilation_state(),[]);self.assertEqual(device_state[0],'caller-device-1');self.assertEqual(len(p.compile_only()),2);p.compile_only()
            self.assertEqual([k.compiles for k in kernels],[0,2,2])
            for order in (('before','after'),('after','before')):
                adapter=api.Adapter()
                for arm in order:
                    item=adapter._prepare(arm,SPEC,**f['inputs'],probe=False);out=item.run();d=adapter.validate(arm,out,f,SPEC);self.assertTrue(d['passed'])
                self.assertTrue(d['paired_bitwise_equal']);self.assertEqual(api.Adapter().items,{});self.assertEqual(device_state[0],'caller-device-1')
                # Whole output halos and exact buffer identity are part of validation.
                item.store[0]=0;self.assertFalse(adapter.validate(order[-1],out,f,SPEC)['passed'])
        with self.assertRaises(ValueError):adapter.validate(order[-1],out.clone(),f,SPEC)

    def test_real_collector_accepts_canonical_multi_kernel_identity(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path,report=evidence_tests.PairTests().fixture(root)
            c=[SimpleNamespace(hash='a'*64,asm={'npubin':b'\x7fELFbefore1'}),SimpleNamespace(hash='b'*64,asm={'npubin':b'\x7fELFbefore2'})]
            for row in report['arms'].values():row['compilation_state_before_capture']=api.identity(c);row['compilation_state_after_capture']=api.identity(c)
            path.write_text(json.dumps(report));self.assertEqual(evidence.collect_pair(path)['status'],'complete')
            report['arms']['after']['compilation_state_after_capture'][1]['binaries']['npubin']='f'*64
            path.write_text(json.dumps(report))
            with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path)

    def test_external_cwd_cli_with_both_modules_and_strict_json(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'examples/optimization-library/triton';folder.mkdir(parents=True);(root/'scripts').mkdir()
            for name in ('request_sentinel.py','request_sentinel_contract.py'):shutil.copy2(HERE/name,folder/name)
            (root/'scripts/optimization_pair.py').write_text("import json\nfrom pathlib import Path\ndef run_pair(adapter,spec,output,cache,provenance,**kwargs):\n data=adapter.make_case(spec)\n Path(output).write_text(json.dumps({'expected':data['expected'].tolist()}))\n return {'status':'complete'}\n")
            case=root/'case.json';case.write_text(json.dumps(SPEC)+'\n');prov=root/'provenance.json';prov.write_text('{}\n');output=root/'result.json'
            args=[sys.executable,str(folder/'request_sentinel.py'),'--case',str(case),'--provenance',str(prov),'--output',str(output),'--cache',str(root/'cache')]
            p=subprocess.run(args,cwd='/',capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(len(json.loads(output.read_text())['expected']),6)
            case.write_text('{"batch":NaN}\n');p=subprocess.run(args,cwd='/',capture_output=True,text=True);self.assertNotEqual(p.returncode,0);self.assertIn('nonfinite JSON',p.stderr)


if __name__=='__main__':unittest.main()
