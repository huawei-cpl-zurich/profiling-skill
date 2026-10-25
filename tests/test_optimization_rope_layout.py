from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch,sys,unittest,tempfile,json,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import rope_layout as op
import rope_layout_contract as c


class LayoutTests(unittest.TestCase):
    def spec(self,**kw):return dict(B=2,S=19,Hq=5,Hk=2,D=46,cos_B=1,producer='BHSD',seed=190401,**kw)

    def test_reference_domain_alias_layout_and_codec(self):
        p=self.spec();f=c.make_case(p);half=p['D']//2
        self.assertTrue(torch.equal(f['expected'][0][...,:half],(f['inputs']['q'][...,:half].double()*f['inputs']['cos'][:,None,:,:half]-f['inputs']['q'][...,half:].double()*f['inputs']['sin'][:,None,:,:half]).float()))
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            for spec in (p,dict(parameters=dict(p,producer='BSHD-backing'),seed=190402,role='guard')):
                a,x=fixture(op.Adapter(),spec,Path(tmp),torch);b,y=fixture(op.Adapter(),spec,Path(tmp),torch)
                self.assertEqual((x['oracle_calls'],y['oracle_calls']),(1,0));self.assertEqual(a['inputs']['q'].stride(),b['inputs']['q'].stride());self.assertTrue(c.equal(a['expected'][0],b['expected'][0],bits=True))
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for name,t in [('q',f['inputs']['q'].double()),('q',f['inputs']['q'].transpose(1,2).contiguous().transpose(1,2)),('cos',f['inputs']['cos'][:,:,:-1]),('sin',f['inputs']['cos'])]:
                with self.assertRaises(ValueError):op.Adapter().prepare('after',p,dict(f,inputs=dict(f['inputs'],**{name:t})),None)
            for v in (float('nan'),float('inf'),.01,2.):
                q=f['inputs']['q'].clone();q[0,0,0,0]=v
                with self.assertRaises(ValueError):op.Adapter().prepare('before',p,dict(f,inputs=dict(f['inputs'],q=q)),None)
            with self.assertRaises(ValueError):op.Adapter().compile_probe('after',dict(p,D=45),None)

    def test_prepared_layout_routes_atomic_compile_and_global_dtype(self):
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.calls=0;self.fail=False
            def warmup(self,*a,**kw):
                if self.fail:raise RuntimeError('mock compile failure')
                return NS(hash='a'*64,name=self.fn.__name__,asm=dict(npubin=b'ELF'))
            def __getitem__(self,grid):
                def run(*a,**kw):
                    self.calls+=1
                    if self.fn.__name__=='_triton_rope':q,k,cos,sin=a[0],a[2],a[4],a[6];oq,ok=q,k
                    else:q,k,oq,ok,cos,sin=a[0].transpose(1,2),a[2].transpose(1,2),a[4],a[5],a[14],a[16]
                    for value,out in ((q,oq),(k,ok)):
                        left,right=value.chunk(2,-1);cc=cos[...,:value.shape[-1]//2][:,:,None];ss=sin[...,:value.shape[-1]//2][:,:,None]
                        rotated=torch.cat((left*cc-right*ss,right*cc+left*ss),-1);out.copy_(rotated)
                return run
        source,after=Kernel('_triton_rope'),Kernel('_strided_rope');original=torch.get_default_dtype()
        try:
            for dtype in (torch.float64,torch.float16):
                torch.set_default_dtype(dtype)
                with patch.object(op,'backend',return_value=('cpu',source,after)):
                    for producer,hk in (('BHSD',2),('BHSD',1),('BSHD-backing',2)):
                        spec=dict(self.spec(),producer=producer,Hk=hk);f=c.make_case(spec);adapter=op.Adapter()
                        for arm in ('before','after'):
                            probe=adapter.compile_probe(arm,spec,None);calls=source.calls+after.calls
                            with self.assertRaises(RuntimeError):probe.run()
                            probe.compile_only();self.assertEqual(source.calls+after.calls,calls)
                        for order in [('before','after'),('after','before')]:
                            for arm in order:
                                item=adapter.prepare(arm,spec,f,None);kernel=source if arm=='before' or producer=='BSHD-backing' else after;kernel.fail=True
                                with self.assertRaises(RuntimeError):item.run()
                                self.assertEqual(item.compilation_state(),[]);kernel.fail=False
                                copy=torch.Tensor.copy_;copied=[];input_ptrs=[v.data_ptr() for n,v in adapter.state[arm]['inputs'].items() if n in ('q','k')]
                                def tracked(destination,original,*args,**kw):
                                    if original.data_ptr() in input_ptrs:copied.append(original.data_ptr())
                                    return copy(destination,original,*args,**kw)
                                with patch.object(torch.Tensor,'copy_',tracked):actual=item.run()
                                self.assertEqual(len(copied),item.metadata['charged_torch_copies'])
                                self.assertEqual([v.dtype for v in actual],[torch.float32]*4)
                                self.assertTrue(adapter.validate(arm,actual,f,spec)['passed']);self.assertEqual(item.metadata['source_aliases'],dict(q=producer=='BSHD-backing',k=producer=='BSHD-backing' or hk==1))
                                if producer=='BHSD' and hk>1:
                                    self.assertIs(item.run(),actual);self.assertTrue(adapter.validate(arm,actual,f,spec)['passed'])
                                self.assertFalse(adapter.validate(arm,actual[:2],f,spec)['passed'])
                                state=adapter.state[arm];state['buffers'][0][0]=0
                                self.assertFalse(adapter.validate(arm,actual,f,spec)['passed']);state['buffers'][0][0]=12345
                                for name,alias in item.metadata['source_aliases'].items():
                                    self.assertEqual(c.equal(state['inputs'][name],f['inputs'][name],bits=True),not alias)
        finally:torch.set_default_dtype(original)

    def test_external_cli_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'bad.json';case.write_text(json.dumps(dict(self.spec(),D=45)))
            p=subprocess.run([sys.executable,op.__file__,'--case',str(case),'--cache',str(root/'cache'),'--output',str(root/'out'),'--provenance',str(root/'absent')],cwd=tmp,capture_output=True)
            self.assertNotEqual(p.returncode,0);self.assertFalse((root/'cache').exists())


if __name__=='__main__':unittest.main()
