from pathlib import Path
import sys,unittest,tempfile,struct
from unittest.mock import patch
from types import SimpleNamespace as NS
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import clipped_zero as op
import clipped_zero_contract as c
from clipped_zero_screen import elf_state,compare


def elf(code=b'\x01',ro=b'A',tag=b'x',symbol=None):
    names=b'\0.text\0.rodata\0.comment\0.symtab\0.strtab\0.rela.text\0';parts=[names,code,ro,tag]
    specs=[(0,3,0,0,1,0),(1,1,6,0,1,0),(7,1,2,0,1,0),(15,1,0,0,1,0)]
    if symbol is not None:
        parts += [bytes(24)+struct.pack('<IBBHQQ',1,1,0,2,0,len(ro)),b'\0'+symbol.encode()+b'\0',struct.pack('<QQq',0,(1<<32)|1,0)]
        specs += [(24,2,0,5,8,24),(32,3,0,0,1,0),(40,4,0,4,8,24)]
    data=bytearray(64+64*len(parts));data[:6]=b'\x7fELF\x02\x01';struct.pack_into('<HHIQQQIHHHHHH',data,16,1,183,1,0,0,64,0,64,0,0,64,len(parts),0)
    for i,(part,(name,kind,flags,link,align,entsize)) in enumerate(zip(parts,specs)):
        offset=len(data);data.extend(part);struct.pack_into('<IIQQQQIIQQ',data,64+i*64,name,kind,flags,0,offset,len(part),link,1 if kind==4 else 0,align,entsize)
    return bytes(data)


class ZeroTests(unittest.TestCase):
    def test_effective_screen_checks_constants_semantic_relocations_and_abi(self):
        abi=dict(known=True,signature={'0':'*fp32'},metadata={})
        a=elf_state(elf(), 'before');b=elf_state(elf(tag=b'y'),'after')
        self.assertEqual(compare(a,a,abi,abi),'compiler-noop-full-ELF')
        self.assertEqual(compare(a,b,abi,abi),'compiler-noop-effective')
        self.assertTrue(compare(a,elf_state(elf(ro=b'B'),'after'),abi,abi).startswith('inconclusive'))
        self.assertTrue(compare(a,b,abi,dict(known=False)).startswith('inconclusive'))
        self.assertEqual(compare(a,elf_state(elf(code=b'\x02'),'after'),abi,abi),'code-distinct')
        x=elf_state(elf(symbol='before'),'before');y=elf_state(elf(symbol='after'),'after')
        self.assertEqual(compare(x,y,abi,abi),'compiler-noop-effective')
        self.assertTrue(compare(x,elf_state(elf(symbol='other'),'after'),abi,abi).startswith('inconclusive'))

    def test_cpu_formula_signed_zero_and_input_rejection(self):
        f=c.make_case(dict(M=2,K=19,limit=0,signed_zero=True,seed=1211));g=f['inputs']['gup'][:,:19].double().clamp(max=0)
        self.assertTrue(torch.equal(f['expected'].view(torch.int32),(g*torch.sigmoid(g*1.702)).float().view(torch.int32)))
        self.assertTrue(c.validate_actual(f['expected'],f)['passed'])
        for value in (float('nan'),float('inf'),-float('inf')):
            for col in (0,19,22):
                bad=f['inputs']['gup'].clone();bad[0,col]=value
                with self.assertRaises(ValueError):op.Adapter().prepare('after',dict(M=2,K=19,limit=0),dict(f,inputs=dict(gup=bad)),None)
        for bad in (f['inputs']['gup'].double(),f['inputs']['gup'][:,:-1],f['inputs']['gup'].T.contiguous().T):
            with self.assertRaises(ValueError):op.Adapter().prepare('before',dict(M=2,K=19),dict(f,inputs=dict(gup=bad)),None)
        for spec in (dict(M=1,K=19,two_k=37),dict(M=1,K=19,limit=float('nan'))):
            with self.assertRaises(ValueError):c.make_case(spec)

    def test_probe_cannot_run_transactional_retry_and_both_orders(self):
        f=c.make_case(dict(M=2,K=21,limit=0,seed=1212))
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.binary=elf();self.compiles=0;self.launches=0
            def warmup(self,*a,**kw):
                self.compiles+=1;return NS(name=self.fn.__name__,asm=dict(npubin=self.binary),hash='a'*64,metadata=NS(),src=NS(signature={0:'*fp32',1:'*fp32'}))
            def __getitem__(self,grid):
                def run(*a,**kw):self.launches+=1;a[1].copy_(f['expected'])
                return run
        before,after=Kernel('before'),Kernel('after');check=c.check_input
        with patch.dict(sys.modules,dict(torch_npu=NS(),clipped_zero_kernels=NS(_clipped_swiglu_kernel=before,_zero_limit_swiglu_kernel=after))),patch.object(torch.Tensor,'to',lambda self,*a,**kw:self.clone()),patch.object(op,'check_input',lambda t,p,device:check(t,p,'cpu')):
            for arm,k in [('before',before),('after',after)]:
                probe=op.Adapter().compile_probe(arm,dict(M=2,K=21),None);calls=k.launches
                with self.assertRaises(RuntimeError):probe.run()
                probe.compile_only()
                with self.assertRaises(RuntimeError):probe.run()
                self.assertEqual(k.launches,calls)
            after.binary=b'invalid';item=op.Adapter().prepare('after',dict(M=2,K=21),f,None);calls=after.launches
            with self.assertRaises(ValueError):item.run()
            self.assertEqual(item.compilation_state(),[]);self.assertEqual(after.launches,calls)
            after.binary=elf();item.run();self.assertEqual(after.launches,calls+1)
            adapter=op.Adapter()
            for order in [('before','after'),('after','before')]:
                for index,arm in enumerate(order):
                    item=adapter.prepare(arm,dict(M=2,K=21),f,None);a=item.run();self.assertIs(a,item.run())
                    result=adapter.validate(arm,a,f,dict(M=2,K=21));self.assertTrue(result['passed'])
                    self.assertEqual('paired_exact_bits' in result,index==1)

    def test_cache_flat_nested_role_and_external_cli_before_cache(self):
        import subprocess,json
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for spec in (dict(M=2,K=23,limit=0,seed=1213),dict(parameters=dict(M=1,K=25,limit=7),seed=1214,role='guard')):
                f,a=fixture(op.Adapter(),spec,root/'cache',torch);g,b=fixture(op.Adapter(),spec,root/'cache',torch)
                self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0));self.assertTrue(c.validate_actual(g['expected'],f)['passed'])
            case=root/'case.json';case.write_text(json.dumps(dict(M=1,K=19,two_k=37)))
            p=subprocess.run([sys.executable,op.__file__,'--case',str(case),'--output',str(root/'out'),'--cache',str(root/'invalid-cache'),'--provenance',str(root/'missing.json')],cwd=tmp,capture_output=True)
            self.assertNotEqual(p.returncode,0);self.assertFalse((root/'invalid-cache').exists())


if __name__=='__main__':unittest.main()
