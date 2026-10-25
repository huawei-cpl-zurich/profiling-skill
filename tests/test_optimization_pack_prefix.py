from pathlib import Path
import sys,tempfile,unittest,json,subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'))
import pack_prefix as op
import pack_prefix_contract as c


class PackTests(unittest.TestCase):
    def test_serial_codec_padding_bits_and_signed_zero(self):
        spec=dict(B=3,Lmax=19,D=67,lengths=[0,19,2],seed=190201)
        f=c.make_case(spec);packed,rt=f['expected']
        self.assertTrue(c.bits_equal(rt,f['inputs']['x']))
        self.assertTrue((packed[0]==-9.5).all());self.assertTrue((packed[2,2:]==-9.5).all())
        self.assertEqual(f['expected_prefix'].tolist(),[0,0,19,21])
        z=c.make_case(dict(B=1,Lmax=2,D=3,lengths=[2],signed_zero=True,seed=190202))
        self.assertTrue(torch.signbit(z['expected'][0]).all());self.assertTrue(c.bits_equal(z['expected'][1],z['inputs']['x']))
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import clone,fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            g=clone(f,torch);self.assertEqual(fingerprint(g,torch),fingerprint(f,torch))
            self.assertTrue(c.bits_equal(g['expected'][0],packed));self.assertTrue(c.bits_equal(g['expected_prefix'],f['expected_prefix']))

    def test_admission_rejects_bad_lengths_shapes_and_values_before_backend(self):
        spec=dict(B=2,Lmax=5,D=3,lengths=[0,5],seed=190203);f=c.make_case(spec)
        invalid=[dict(x=f['inputs']['x'],lengths=t) for t in (torch.tensor([0.,5.]),torch.tensor([-1,6],dtype=torch.int32),torch.tensor([0,4],dtype=torch.int32))]
        invalid += [dict(f['inputs'],x=t) for t in (f['inputs']['x'].double(),f['inputs']['x'][:-1],f['inputs']['x'].T.contiguous().T)]
        for v in (float('nan'),float('inf'),.01,3.):
            x=f['inputs']['x'].clone();x[0,0]=v;invalid.append(dict(f['inputs'],x=x))
        with patch.object(op,'backend',side_effect=AssertionError('backend touched')):
            for inputs in invalid:
                with self.assertRaises(ValueError):op.Adapter().prepare('after',spec,dict(f,inputs=inputs),None)
            for lengths in ([0,0],[-1,5],[0,5.0],[5],[0,6]):
                with self.assertRaises(ValueError):op.Adapter().compile_probe('before',dict(spec,lengths=lengths),None)

    def test_compile_atomic_retry_probe_no_launch_and_repeated_both_orders(self):
        spec=dict(B=2,Lmax=7,D=5,lengths=[0,7],seed=190204);f=c.make_case(spec)
        class Kernel:
            def __init__(self,name):self.fn=NS(__name__=name);self.launches=0;self.compiles=0;self.fail=False
            def warmup(self,*a,**kw):
                self.compiles+=1
                if self.fail:raise RuntimeError('mock compile failure')
                return NS(name=self.fn.__name__,hash='a'*64,asm=dict(npubin=b'elf'),metadata=NS())
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    self.launches+=1
                    if self.fn.__name__=='_exclusive_prefix':a[1].copy_(f['expected_prefix'])
                    elif 'unpack' in self.fn.__name__:a[1].copy_(f['expected'][1])
                    else:a[1].copy_(f['expected'][0])
                return launch
        names=['_pack_seq_kernel','_unpack_seq_triton_kernel','_exclusive_prefix','_pack_seq_kernel_prefix','_unpack_seq_triton_kernel_prefix']
        kernels=NS(**{name:Kernel(name) for name in names})
        with patch.object(op,'backend',return_value=('cpu',kernels)):
            for arm,failname in [('before',names[1]),('after',names[4])]:
                adapter=op.Adapter();probe=adapter.compile_probe(arm,spec,None)
                with self.assertRaises(RuntimeError):probe.run()
                self.assertEqual(sum(getattr(kernels,n).launches for n in names),0)
                probe.compile_only()
                with self.assertRaises(RuntimeError):probe.run()
                self.assertEqual(sum(getattr(kernels,n).launches for n in names),0)
                item=adapter.prepare(arm,spec,f,None);bad=getattr(kernels,failname);bad.fail=True
                with self.assertRaises(RuntimeError):item.run()
                self.assertEqual(item.compilation_state(),[]);self.assertEqual(sum(getattr(kernels,n).launches for n in names),0)
                bad.fail=False;item.compile_only();self.assertEqual(len(item.compilation_state()),2 if arm=='before' else 3)
            adapter=op.Adapter()
            for order in [('before','after'),('after','before')]:
                for i,arm in enumerate(order):
                    item=adapter.prepare(arm,spec,f,None);actual=item.run();again=item.run()
                    self.assertTrue(all(a is b for a,b in zip(actual,again)))
                    result=adapter.validate(arm,actual,f,spec);self.assertTrue(result['passed']);self.assertEqual('paired_exact_bits' in result,i==1)
                    if arm=='after':self.assertTrue(result['prefix_exact'])
            self.assertFalse(adapter.validate('after',(actual[0],),f,spec)['passed'])
            state=adapter.state['after'];state['buffers'][0][0]=0
            self.assertFalse(adapter.validate('after',actual,f,spec)['passed']);state['buffers'][0][0]=12345
            state['inputs']['x'][0,0]=3
            self.assertFalse(adapter.validate('after',actual,f,spec)['passed']);state['inputs']['x'].copy_(f['inputs']['x'])
            adapter.state['after']['starts'][0]=1
            self.assertFalse(adapter.validate('after',actual,f,spec)['passed'])

    def test_cache_nested_role_and_external_cli_admission(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from optimization_pair import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for spec in (dict(B=2,Lmax=9,D=3,seed=190205),dict(parameters=dict(B=3,Lmax=11,D=7,lengths=[0,11,1]),seed=190206,role='guard')):
                f,a=fixture(op.Adapter(),spec,root/'cache',torch);g,b=fixture(op.Adapter(),spec,root/'cache',torch)
                self.assertEqual((a['oracle_calls'],b['oracle_calls']),(1,0));self.assertTrue(c.bits_equal(f['expected'][0],g['expected'][0]))
            case=root/'case.json';case.write_text(json.dumps(dict(B=2,Lmax=9,D=3,lengths=[0,2])))
            p=subprocess.run([sys.executable,op.__file__,'--case',str(case),'--cache',str(root/'badcache'),'--output',str(root/'out'),'--provenance',str(root/'missing')],cwd=tmp,capture_output=True)
            self.assertNotEqual(p.returncode,0);self.assertFalse((root/'badcache').exists())


if __name__=='__main__':unittest.main()
