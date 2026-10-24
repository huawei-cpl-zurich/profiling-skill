import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'examples/optimization-library/triton'
sys.path[:0]=[str(DIRECTORY),str(ROOT/'scripts')]
from sinkhorn_contract import make_case, check_outputs, validate_inputs, reference
from sinkhorn_fusion import Adapter
from optimization_pair import fixture, run_pair


def elf():
    import struct
    data=bytearray(274)
    data[:6]=b'\x7fELF\x02\x01'
    struct.pack_into('<Q',data,40,64)
    struct.pack_into('<HHH',data,58,64,3,1)
    struct.pack_into('<IIQQQQIIQQ',data,128,1,3,0,0,256,17,0,0,1,0)
    struct.pack_into('<IIQQQQIIQQ',data,192,11,1,6,0,273,1,0,0,1,0)
    data[256:273]=b'\0.shstrtab\0.text\0'
    return bytes(data)

def spec(**changes):
    return dict(seed=97,parameters=dict(BS=5,iters=3,eps=1e-5,**changes))


class SinkhornTests(unittest.TestCase):
    def test_source_initial_smoothing_equal_logits(self):
        case=dict(seed=62,BS=2,iters=1,eps=1e-3,zero_scale_equal_logits=True)
        value=make_case(case); pre,post,comb=value['expected']
        self.assertTrue(torch.equal(pre,torch.full_like(pre,.501)))
        self.assertTrue(torch.equal(post,torch.ones_like(post)))
        wanted=torch.full_like(comb,.251/(1.004+.001))
        self.assertTrue(torch.equal(comb,wanted))
        self.assertFalse(torch.equal(comb,torch.full_like(comb,.25)))

    def test_independent_scalar_oracle(self):
        import math
        value=make_case(spec()); x,s,b=(value['inputs'][k] for k in ('mixes','scale','base'))
        pre,post,comb=value['expected']
        for batch in range(5):
            rows=[]
            for r in range(4):
                z=[float(x[batch,8+r*4+c])*float(s[2])+float(b[8+r*4+c]) for c in range(4)]
                ex=[math.exp(v-max(z)) for v in z]
                rows.append([v/sum(ex)+1e-5 for v in ex])
            def colnorm(a):
                return [[a[r][c]/(sum(a[k][c] for k in range(4))+1e-5) for c in range(4)] for r in range(4)]
            rows=colnorm(rows)
            for _ in range(2):
                rows=colnorm([[v/(sum(row)+1e-5) for v in row] for row in rows])
            self.assertTrue(torch.equal(comb[batch],torch.tensor(rows,dtype=torch.float32)))
            for j in range(4):
                expected=1/(1+math.exp(-(float(x[batch,j])*float(s[0])+float(b[j]))))+1e-5
                self.assertEqual(pre[batch,j],torch.tensor(expected,dtype=torch.float32))

    def test_contract_and_cache(self):
        expected=make_case(spec())['expected']
        for bad in (expected[:2],expected+expected,list(expected),(expected[0].double(),*expected[1:]),(expected[0]+.01,*expected[1:])):
            self.assertFalse(check_outputs(bad,expected)['passed'])
        with tempfile.TemporaryDirectory() as folder:
            a,first=fixture(Adapter(),spec(),Path(folder)/'cache',torch)
            b,second=fixture(Adapter(),spec(),Path(folder)/'cache',torch)
            self.assertEqual((first['oracle_calls'],second['oracle_calls']),(1,0))
            self.assertTrue(torch.equal(a['expected'][2],b['expected'][2]))
            with self.assertRaises(ValueError):
                fixture(Adapter(),dict(seed=5,BS=0,iters=1,eps=1e-6),Path(folder)/'bad',torch)
            self.assertFalse((Path(folder)/'bad').exists())
        for key,replacement in [('scale',torch.ones(3,dtype=torch.float64)),('base',torch.full((24,),float('nan'))),('mixes',torch.zeros(5,25))]:
            value=make_case(spec());value['inputs'][key]=replacement
            with self.assertRaises(ValueError):Adapter().prepare('after',spec(),value,Path('.'))

    def test_prepared_interface_reuses_outputs_and_two_launches(self):
        launches=[]
        artifact=SimpleNamespace(hash='b'*64,asm={'npubin':elf()})
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*args,**kwargs):return artifact
            def __getitem__(self,grid):
                def run(*args,**kwargs):launches.append((self.fn.__name__,grid,kwargs));return artifact
                return run
        original=torch.empty
        def allocate(*args,**kwargs):kwargs.pop('device',None);return original(*args,**kwargs)
        modules=dict(torch_npu=SimpleNamespace(),triton=SimpleNamespace(__version__='fake'),
            sinkhorn_kernel=SimpleNamespace(hc_split_sinkhorn_forward_prepost_kernel=Kernel('prepost'),hc_split_sinkhorn_forward_comb_kernel=Kernel('comb')), sinkhorn_fusion_kernel=SimpleNamespace(hc_sinkhorn_forward_fused_kernel=Kernel('fused')))
        with patch.dict(sys.modules,modules),patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True),patch.object(torch,'empty',allocate):
            adapter=Adapter(); before=adapter.prepare('before',spec(),make_case(spec()),Path('.'));after=adapter.prepare('after',spec(),make_case(spec()),Path('.'))
            self.assertEqual(len(before.compile_only()),2);self.assertEqual(len(after.compile_only()),1);self.assertEqual(launches,[])
            a=before.run();self.assertIs(a,before.run());b=after.run()
            self.assertEqual(len(launches),5);self.assertEqual(launches[1][2]['BLOCK_ALIGN'],8);self.assertEqual(launches[-1][2]['BLOCK_ALIGN'],8)
            self.assertEqual(a[2].stride(),(32,8,1));self.assertEqual(b[2].stride(),(32,8,1))
            self.assertNotEqual(a[0].data_ptr(),b[0].data_ptr())
            self.assertEqual(before.output_storages[0][[0,-1]].tolist(),[-701.,-701.])
            self.assertIsInstance(before.compilation_state(),list)
            for output,wanted in zip(b,make_case(spec())['expected']):output.copy_(wanted)
            after.backing[:,:,4:]=0
            self.assertTrue(adapter.validate('after',b,make_case(spec()),spec())['passed'])
            after.backing[0,0,4]=-0.
            self.assertFalse(adapter.validate('after',b,make_case(spec()),spec())['passed'])

    def test_binary_rejection(self):
        from sinkhorn_fusion import screen_binary
        self.assertEqual(len(screen_binary(elf())),64)
        for data in (b'',elf()[:128]):
            with self.assertRaises(ValueError):screen_binary(data)

    def test_external_cli_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);case=folder/'case.json'
            command=[sys.executable,str(DIRECTORY/'sinkhorn_fusion_cli.py'),'--case',str(case),'--output',str(folder/'out'),'--cache',str(folder/'cache'),'--provenance',str(folder/'prov'),'--validate-only']
            for value in (spec(),dict(seed=97,role='gallery',expected='metadata',**spec()['parameters'])):
                case.write_text(json.dumps(value));result=subprocess.run(command,cwd=folder,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            for change in (dict(BS=0),dict(iters=0),dict(eps=.01)):
                value=spec();value['parameters'].update(change);case.write_text(json.dumps(value))
                self.assertNotEqual(subprocess.run(command,cwd=folder,capture_output=True).returncode,0)
            self.assertFalse((folder/'out').exists());self.assertFalse((folder/'cache').exists())

class PublicPairLifecycleTests(unittest.TestCase):
    def test_orders_reuse_and_one_ulp_mismatch(self):
        artifact=SimpleNamespace(hash='c'*64,asm={'npubin':elf()})
        class Kernel:
            def __init__(self,name):self.fn=SimpleNamespace(__name__=name)
            def warmup(self,*args,**kwargs):return artifact
            def __getitem__(self,grid):
                def launch(*args,**flags):
                    mixes,scale,base=args[:3]
                    if self.fn.__name__ in ('prepost','fused'):
                        if self.fn.__name__=='fused':
                            _,_,_,pre,post,backing,bs,hc,eps,feat=args
                        else:
                            _,_,_,pre,post,bs,hc,eps,feat=args
                        values=reference(dict(mixes=mixes,scale=scale,base=base),1,eps)
                        pre.copy_(values[0]);post.copy_(values[1])
                    if self.fn.__name__ in ('comb','fused'):
                        if self.fn.__name__=='comb':
                            _,_,_,backing,bs,hc,eps,feat=args
                        value=reference(dict(mixes=mixes,scale=scale,base=base),flags['SINKHORN_ITERS'],eps)[2]
                        backing[:,:,:4].copy_(value);backing[:,:,4:].zero_()
                    return artifact
                return launch
        class BoundaryAdapter(Adapter):
            mismatch=None
            def prepare(self,arm,*args):
                item=super().prepare(arm,*args)
                original=item.run
                def run():
                    actual=original()
                    if self.mismatch is not None and arm==self.mismatch[0]:
                        output=actual[self.mismatch[1]]
                        index=(0,)*output.ndim
                        output[index]=torch.nextafter(output[index],torch.tensor(float('inf')))
                    return actual
                item.run=run
                return item
        original_empty=torch.empty
        def allocate(*args,**kwargs):
            kwargs.pop('device',None)
            return original_empty(*args,**kwargs)
        modules=dict(torch_npu=SimpleNamespace(),triton=SimpleNamespace(__version__='cpu-double'),
            sinkhorn_kernel=SimpleNamespace(hc_split_sinkhorn_forward_prepost_kernel=Kernel('prepost'),
                                          hc_split_sinkhorn_forward_comb_kernel=Kernel('comb')), sinkhorn_fusion_kernel=SimpleNamespace(hc_sinkhorn_forward_fused_kernel=Kernel('fused')))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,modules), \
                patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True), \
                patch.object(torch,'empty',allocate):
            root=Path(folder);adapter=BoundaryAdapter()
            for index,order in enumerate((('before','after'),('after','before'))):
                for case_index,bs in enumerate((2,7,3)):
                    case=dict(seed=120+case_index,BS=bs,iters=2,eps=1e-5,role='guard')
                    report=run_pair(adapter,case,root/f'good-{index}-{case_index}',root/'cache',{},order=order,torch=torch)
                    self.assertEqual(report['status'],'complete',report)
                    first=report['arms'][order[0]]['correctness']['diagnostics']
                    second=report['arms'][order[1]]['correctness']['diagnostics']
                    self.assertIsNone(first['paired_output_bitwise_equal'])
                    self.assertEqual(first['paired_output_comparison'],'pending')
                    self.assertTrue(second['paired_output_bitwise_equal'])
                    self.assertEqual(adapter.pair_bits,{})
                for output_index in (0,1,2):
                    adapter.mismatch=(order[0],output_index)
                    case=dict(seed=143+output_index,BS=4,iters=2,eps=1e-5,role='guard')
                    report=run_pair(adapter,case,root/f'bad-{index}-{output_index}',root/'cache',{},order=order,torch=torch)
                    self.assertEqual(report['status'],'failed')
                    second=report['arms'][order[1]]['correctness']
                    self.assertEqual(second['status'],'failed')
                    self.assertFalse(second['diagnostics']['paired_output_bitwise_equal'])
                    self.assertTrue(all(x['passed'] for x in second['diagnostics']['outputs']))
                    self.assertTrue(all(x['profiled_calls']==0 for x in report['arms'].values()))
                    self.assertEqual(adapter.pair_bits,{})
                adapter.mismatch=None


if __name__=='__main__':unittest.main()
