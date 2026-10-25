"""NONFROZEN host functional tests; no NPU operations or frozen fixtures."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import torch
from rotary_trig import Adapter
from rotary_trig_contract import dimensions,make_case,validate_inputs,check_output


def spec(stride=1,zero=False):return dict(seed=190903,parameters=dict(B=2,S=7,Hq=3,Hk=1,D=64,scaling=2.,position_stride=stride,zero_position=zero))


def rotate(x,p,i,scale):
    angle=p.double()[:,:,None,None]/scale*i.double()[None,None,None,:]
    a,b=x.double().chunk(2,-1)
    return torch.cat((a*angle.cos()-b*angle.sin(),b*angle.cos()+a*angle.sin()),-1).float()


class Kernel:
    def __init__(self,fused=False):self.fused=fused;self.fn=SimpleNamespace(__name__='fused' if fused else 'head');self.compiles=0;self.launches=0;self.fail_at=None
    def warmup(self,*args,**kw):
        self.compiles+=1
        if self.compiles==self.fail_at:raise RuntimeError('late compile failure')
        return SimpleNamespace(hash='a'*64,asm={'npubin':b'code'})
    def __getitem__(self,grid):
        def run(*args,**kw):
            self.launches+=1
            if self.fused:
                q,k,p,i,scale,oq,ok=args[:7];oq.copy_(rotate(q,p,i,scale));ok.copy_(rotate(k,p,i,scale))
            else:
                x,p,i,scale,out=args[:5];out.copy_(rotate(x,p,i,scale))
        return run


class RotaryTests(unittest.TestCase):
    def setUp(self):
        self.head,self.fused=Kernel(),Kernel(True)
        p=patch.dict(sys.modules,{'triton':SimpleNamespace(__version__='mock'),'torch_npu':SimpleNamespace(),
            'rotary_trig_kernel':SimpleNamespace(rotary_head_kernel=self.head,_fused_rotary_emb_kernel=self.fused)})
        p.start();self.addCleanup(p.stop)

    def test_fp64_provider_stride_and_zero_sign_contract(self):
        case=spec(2);f=make_case(case);self.assertEqual(f['inputs']['P'].stride(),(14,2))
        self.assertTrue(torch.equal(f['inputs']['P'][0],torch.arange(7)+case['seed']%17))
        for name,expected in zip(('Q','K'),f['expected']):self.assertTrue(check_output((rotate(f['inputs'][name],f['inputs']['P'],f['inputs']['I'],2.),)*2,(expected,expected))['passed'])
        z=make_case(spec(zero=True));self.assertTrue(check_output((z['inputs']['Q'],z['inputs']['K']),z['expected'])['passed'])
        self.assertTrue(torch.signbit(z['inputs']['Q']).any())

    def test_domains_layout_alias_arity_and_finite(self):
        for key,value in [('D',65),('D',96),('scaling',3.),('S',3000),('position_stride',3)]:
            c=spec();c['parameters'][key]=value
            with self.assertRaises(ValueError):dimensions(c)
        c=spec(2);f=make_case(c)
        for name,wrong in [('P',f['inputs']['P'].contiguous()),('I',f['inputs']['I'].double()),('Q',f['inputs']['Q'].transpose(2,3))]:
            v=dict(f['inputs']);v[name]=wrong
            with self.assertRaises(ValueError):validate_inputs(v,c)
        v={n:x.clone() if n!='P' else x for n,x in f['inputs'].items()};v['Q'][0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):validate_inputs(v,c)
        for actual in (f['expected'],(f['inputs']['Q'],),[f['inputs']['Q'],f['inputs']['K']]):self.assertFalse(check_output(actual,f['expected'])['passed'])
        with self.assertRaises(ValueError):Adapter().prepare('bad',c,f,None)

    def test_atomic_compile_retry_and_probe_zero_launch(self):
        c=spec();f=make_case(c);a=Adapter();probe=a._device_item('before',c,f['inputs'],probe=True)
        with self.assertRaises(RuntimeError):probe.run()
        self.assertEqual((self.head.compiles,self.head.launches),(0,0))
        self.head.fail_at=2
        with self.assertRaises(RuntimeError):probe.compile_only()
        self.assertEqual(probe.compilation_state(),[]);probe.compile_only();self.assertEqual(self.head.compiles,4)
        item=a._device_item('before',c,f['inputs']);item.run();item.run()
        self.assertEqual((self.head.compiles,self.head.launches),(6,4));self.assertEqual(len(item.compilation_state()),2)

    def test_adapter_orders_private_outputs_canaries_fullbacking_immutable(self):
        a=Adapter()
        with patch.object(torch.Tensor,'npu',lambda x:x.clone(),create=True):
            for order in (('before','after'),('after','before'),('before','after')):
                c=spec(2);f=make_case(c)
                for arm in order:
                    item=a.prepare(arm,c,f,None);self.assertEqual(item.inputs['P'].stride(),(14,2))
                    actual=item.run();r=a.validate(arm,actual,f,c);self.assertTrue(r['passed']);self.assertEqual(r['paired_output_bits'],None if arm==order[0] else True)
                    item.buffers[0][0]=0;self.assertFalse(a.validate(arm,actual,f,c)['passed']);item.buffers[0][0]=-701
                    item.inputs['P'].as_strided((28,),(1,))[1]=0
                    self.assertFalse(a.validate(arm,actual,f,c)['passed'])

    def test_flat_nested_externalcwd_cli(self):
        c=spec();self.assertEqual(dimensions(c),dimensions(dict(c['parameters'],seed=c['seed'])))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'case.json';p.write_text(json.dumps(dict(c,role='guard')))
            command=[sys.executable,str(ROOT/'examples/optimization-library/triton/rotary_trig_cli.py'),'--case',str(p),'--output',tmp+'/o','--cache',tmp+'/c','--provenance',tmp+'/p','--validate-only']
            result=subprocess.run(command,cwd=tmp,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['role'],'guard')


if __name__=='__main__':unittest.main()
