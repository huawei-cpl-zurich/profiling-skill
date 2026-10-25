"""NONFROZEN CPU/mock typed qualification, codec, contracts and artifact tests."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import torch
from awq_inline import Adapter
from awq_inline_contract import dimensions,decode,make_case,validate_inputs,check_output
from awq_inline_types import classify
IR='%loop = scf.for %k = %a to %b step %c iter_args(%acc = %init) -> (tensor<16x32xf32>) {\n %d = tt.dot %a, %b, %acc : tensor<16x32xf16> * tensor<32x32xf16> -> tensor<16x32xf32>\n scf.yield %d : tensor<16x32xf32>\n}'
ENGINE='mix_mode = "mix"; linalg.matmul'
def spec():return dict(seed=190909,parameters=dict(M=3,K=35,N=24,group=35))
class Kernel:
    def __init__(self,kind):self.kind=kind;self.fn=SimpleNamespace(__name__=kind);self.compiles=0;self.launches=0;self.fail=False
    def warmup(self,*args,**kw):
        self.compiles+=1
        if self.fail:self.fail=False;raise RuntimeError('late compile failure')
        return SimpleNamespace(hash='a'*64,asm=dict(ttir=IR,ttadapter=ENGINE,npubin=b'code'))
    def __getitem__(self,grid):
        def run(*args,**kw):
            self.launches+=1
            if self.kind=='dequant':
                q,s,z,g,out=args[:5];out.copy_(((decode(q)-decode(z).repeat_interleave(g,0))*s.repeat_interleave(g,0)).half())
            elif self.kind=='materialized':
                a,b,c=args[:3];c.copy_((a.double()@b.double()).half())
            else:
                a,q,c,z,s,m,n,k,g=args
                b=(decode(q)-decode(z).repeat_interleave(g,0)).double()*s.double().repeat_interleave(g,0);c.copy_((a.double()@b).half())
        return run
class AWQInlineTests(unittest.TestCase):
    def setUp(self):
        self.d,self.g,self.f=Kernel('dequant'),Kernel('materialized'),Kernel('inline')
        p=patch.dict(sys.modules,{'triton':SimpleNamespace(),'torch_npu':SimpleNamespace(),'awq_inline_kernel':SimpleNamespace(awq_dequantize_kernel=self.d,materialized_gemm=self.g,awq_gemm_kernel=self.f)});p.start();self.addCleanup(p.stop)
    def test_independent_models_domain_and_signed_codec(self):
        c=spec();f=make_case(c);self.assertTrue((f['inputs']['Q']<0).any())
        for key,value in [('K',513),('N',23),('group',7)]:
            bad=spec();bad['parameters'][key]=value
            with self.assertRaises(ValueError):dimensions(bad)
        for name in ('A','S'):
            v={n:x.clone() for n,x in f['inputs'].items()};v[name].view(-1)[0]=float('nan')
            with self.assertRaises(ValueError):validate_inputs(v,c)
        for name,value in [('Q',f['inputs']['Q'].long()),('A',f['inputs']['A'].t().contiguous().t())]:
            bad=dict(f['inputs']);bad[name]=value
            with self.assertRaises(ValueError):validate_inputs(bad,c)
        alias_spec=dict(seed=190910,parameters=dict(M=4,K=64,N=128,group=32));alias=make_case(alias_spec)['inputs']
        alias['A'].fill_(1/64);alias['S']=alias['A'].view(2,128)
        with self.assertRaises(ValueError):validate_inputs(alias,alias_spec)
        invalid,_=check_output((f['expected']['fp32'],),f['expected'],'fp32');self.assertFalse(invalid['passed'])
        a=f['inputs']['A'].double();b=(decode(f['inputs']['Q'])-decode(f['inputs']['Z']).repeat_interleave(35,0)).double()*f['inputs']['S'].double()
        exact=torch.zeros((3,24),dtype=torch.float64)
        for k in range(35):exact+=a[:,k,None]*b[k]
        self.assertTrue(torch.equal(exact,f['expected']['fp64']))
        self.assertTrue(torch.equal((a[:,:32]@b[:32]).half().double().add(a[:,32:]@b[32:]).half(),f['expected']['fp16']))
    def test_type_and_engine_classifier_fail_closed(self):
        self.assertEqual(classify(IR,ENGINE)['model'],'fp32')
        half=IR.replace('iter_args(%acc = %init) -> (tensor<16x32xf32>)','iter_args(%acc = %init) -> (tensor<16x32xf16>)').replace('scf.yield %d','%h = arith.truncf %d : tensor<16x32xf32> to tensor<16x32xf16>\n scf.yield %h')
        self.assertEqual(classify(half,ENGINE)['model'],'fp16')
        for ir,engine in ((IR,'mix_mode = "aiv"; linalg.matmul'),(IR.replace('tt.dot','unknown'),ENGINE),(half.replace('arith.truncf','arith.extf'),ENGINE)):
            with self.assertRaises(ValueError):classify(ir,engine)
    def test_probe_transactional_late_failure_and_case_qualification(self):
        c=spec();f=make_case(c);a=Adapter()
        with self.assertRaises(RuntimeError):a.make_case(c)
        with self.assertRaises(RuntimeError):a.prepare('before',c,f,None)
        item=a._device_item('before',c,f['inputs'],True)
        with self.assertRaises(RuntimeError):item.run()
        self.assertEqual(self.d.compiles,0);self.g.fail=True
        with self.assertRaises(RuntimeError):item.compile_only()
        self.assertEqual(item.compilation_state(),[]);self.assertEqual(len(item.attempted_artifacts),1)
        item.compile_only();self.assertEqual((self.d.compiles,self.g.compiles),(2,2));self.assertEqual(self.g.launches,0)
    def test_models_one_transfer_diagnostics_and_adapter_orders(self):
        c=spec();f=make_case(c);got=f['expected']['fp32'].clone();got[0,0]=float('inf')
        calls=[];cpu=torch.Tensor.cpu
        with patch.object(torch.Tensor,'cpu',lambda x:(calls.append(1),cpu(x))[1]):result,_=check_output(got,f['expected'],'fp32')
        self.assertEqual(len(calls),1);self.assertEqual(result['first_mismatch']['index'],[0,0]);self.assertEqual(result['nonfinite_count'],1)
        zeros={x:torch.zeros((1,1),dtype=torch.float64 if x=='fp64' else torch.float16) for x in ('fp32','fp16','fp64')}
        zero_result,_=check_output(torch.full((1,1),-0.,dtype=torch.float16),zeros,'fp32')
        self.assertTrue(zero_result['passed']);self.assertEqual(zero_result['zero_sign_differences'],1)
        old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
        try:
            a=Adapter();a.model='fp32';a.qualification_key=json.dumps(c,sort_keys=True)
            with patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True):
                for order in (('before','after'),('after','before')):
                    for arm in order:
                        item=a._prepare_historical(arm,c,f,None);out=item.run();self.assertEqual(out.dtype,torch.float16);self.assertTrue(a.validate(arm,out,f,c)['passed'])
                        item.output_storage[0]=0;self.assertFalse(a.validate(arm,out,f,c)['passed'])
        finally:torch.set_default_dtype(old)
    def test_public_fail_closed_and_explicit_inspection(self):
        c=spec();f=make_case(c);a=Adapter()
        with self.assertRaises(RuntimeError):a.make_case(c)
        with self.assertRaises(RuntimeError):a.prepare('before',c,f,None)
        with self.assertRaises(RuntimeError):a.probe('before',c)
        self.assertEqual((self.d.compiles,self.g.compiles,self.f.compiles),(0,0,0))
        empty=torch.empty
        with patch.object(torch,'empty',lambda *args,**kw:empty(*args,**dict(kw,device='cpu'))):
            probe=Adapter(compile_inspection=True).probe('before',c)
            with self.assertRaises(RuntimeError):probe.run()
            probe.compile_only()
        self.assertEqual((self.d.launches,self.g.launches,self.f.launches),(0,0,0))
    def test_externalcwd_cli_json(self):
        cli=ROOT/'examples/optimization-library/triton/awq_inline_cli.py'
        wrapper="import sys,runpy;from types import SimpleNamespace as S;sys.modules['awq_inline']=S(Adapter=lambda **kw:S(probe=lambda *a:S(compile_only=lambda:[],metadata={},compilation_state=lambda:[])));runpy.run_path(sys.argv.pop(1),run_name='__main__')"
        with tempfile.TemporaryDirectory() as tmp:
            c=Path(tmp)/'case.json';c.write_text(json.dumps(dict(spec(),role='guard')))
            command=[sys.executable,'-c',wrapper,str(cli),'--case',str(c),'--output',tmp+'/out','--cache',tmp+'/cache','--provenance',tmp+'/p','--compile-inspection']
            result=subprocess.run(command,cwd=tmp,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads((Path(tmp)/'out/inspection.json').read_text());self.assertEqual([x['arm'] for x in data['arms']],['before','after']);self.assertFalse(data['numeric_qualified'])
            denied=subprocess.run(command[:-1],cwd=tmp,capture_output=True,text=True);self.assertEqual(denied.returncode,2)
            self.assertFalse((Path(tmp)/'cache').exists())
if __name__=='__main__':unittest.main()
