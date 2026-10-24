"""Functional independent oracle, host guards/cache, and compound native ABI."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/ascendc'),str(ROOT/'scripts')]
from affine import Adapter,dimensions,make_case,verify_binary
from optimization_pair import fixture, run_pair
from optimization_native_bridge import digest


def test_oracle_and_once_cache(tmp_path):
    for i,(r,w) in enumerate(((5,17),(67,33),(2,1024))):
        v=make_case(dict(seed=19100+i,parameters=dict(rows=r,width=w)))
        assert v['x'].shape==(r,w)
        assert torch.equal(v['expected'],((v['x'].double()*v['gamma'].double()).float().double()+v['beta'].double()).float())
    spec=dict(seed=19321,parameters=dict(rows=5,width=17,two_calls_same_coefficient_pointer=True))
    a,f=fixture(Adapter(tmp_path),spec,tmp_path/'cache',torch)
    b,s=fixture(Adapter(tmp_path),spec,tmp_path/'cache',torch)
    assert f['oracle_calls']==1 and s['oracle_calls']==0 and f['sha256']==s['sha256']
    assert isinstance(a['expected'],tuple) and len(a['expected'])==2
    assert not torch.equal(a['expected'][0],a['expected'][1])
    assert torch.equal(a['expected'][1],b['expected'][1])


@pytest.mark.parametrize('r,w',[(0,17),(2,0),(2,1025),(True,17),(1.5,17),(65537,8)])
def test_reject_before_cache(tmp_path,r,w):
    spec=dict(seed=19,parameters=dict(rows=r,width=w));a=Adapter(tmp_path)
    with pytest.raises(ValueError):fixture(a,spec,tmp_path/'cache',torch)
    for arm in ('before','after'):
        with pytest.raises(ValueError):a.prepare(arm,spec,{},tmp_path/arm)
    assert not (tmp_path/'cache').exists()


def test_cli_rejection(tmp_path):
    (tmp_path/'case.json').write_text(json.dumps(dict(seed=19201,parameters=dict(rows=3,width=1025))))
    (tmp_path/'p.json').write_text('{}')
    p=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','affine:factory',
        '--case',str(tmp_path/'case.json'),'--output',str(tmp_path/'out'),'--cache',str(tmp_path/'cache'),
        '--provenance',str(tmp_path/'p.json'),'--correctness-only'],env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/ascendc')),capture_output=True,text=True,cwd=tmp_path)
    assert p.returncode==1
    r=json.loads((tmp_path/'out/result.json').read_text())
    assert r['error']['type']=='ValueError' and r['counts']==dict(profiled_calls=0,capture_calls=0,correctness_calls=0)
    assert not (tmp_path/'cache').exists()


@pytest.fixture
def cpu_native(monkeypatch):
    original_to=torch.Tensor.to;original_full=torch.full
    def to(t,*args,**kw):
        if kw.get('device')=='npu:0':kw['device']='cpu'
        return original_to(t,*args,**kw)
    def full(*args,**kw):
        if kw.get('device')=='npu:0':kw['device']='cpu'
        return original_full(*args,**kw)
    monkeypatch.setattr(torch.Tensor,'to',to);monkeypatch.setattr(torch,'full',full)
    monkeypatch.setitem(sys.modules,'torch_npu',SimpleNamespace())
    monkeypatch.setattr(torch,'npu',SimpleNamespace(synchronize=lambda:None,current_stream=lambda:SimpleNamespace(npu_stream=0)),raising=False)


@pytest.mark.skipif(not shutil.which('c++'),reason='host compiler required')
def test_compound_native_abi_and_validation(tmp_path,cpu_native):
    source=tmp_path/'stub.cpp'
    source.write_text('''#include <stdint.h>
extern "C" uint32_t affine_rows(){return 5;}
extern "C" uint32_t affine_width(){return 17;}
extern "C" uint32_t affine_grid(){return 2;}
extern "C" uint32_t affine_after(){return 1;}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\"host_affine\\"]";}
extern "C" int catlass_bridge_launch(void**p,uint64_t c,void*){
 if(c!=4)return 1;
 for(int i=0;i<85;i++){volatile float v=((float*)p[0])[i]*((float*)p[1])[i%17];((float*)p[3])[i]=v+((float*)p[2])[i%17];}return 0;}
''')
    binary=tmp_path/'5-17-after.so'
    subprocess.run(['c++','-shared','-fPIC',str(source),'-o',str(binary)],check=True)
    assert verify_binary(binary,5,17,'after')==2
    with pytest.raises(ValueError):verify_binary(binary,5,17,'before')
    build=tmp_path/'build-5-17-after';build.mkdir()
    (build/'build.json').write_text(json.dumps(dict(binary_sha256=digest(binary))))
    (tmp_path/'sdk-declarations.json').write_text('[]')
    spec=dict(seed=19205,parameters=dict(rows=5,width=17,two_calls_same_coefficient_pointer=True))
    v=make_case(spec);a=Adapter(tmp_path);item=a.prepare('after',spec,v,tmp_path/'out')
    pointers=[t.data_ptr() for t in item.inputs]
    try:
        result=item.run();assert [t.data_ptr() for t in item.inputs]==pointers
        good=a.validate('after',result,v,spec)
        assert good['passed'] and good['exact_oracle'] and good['component_API_calls']==2 and good['caller_coefficient_copies']==2
        a.prepared['after'][1][0][0]=0
        assert not a.validate('after',result,v,spec)['gm_canaries']
        a.prepared['after'][1][0][0]=-8191;result[1][0,0]=99
        assert not a.validate('after',result,v,spec)['primary_accuracy']
        item.inputs[0][0,0]=99
        assert not a.validate('after',result,v,spec)['immutable_input']
    finally:item.close()


@pytest.mark.skipif(not shutil.which('c++'),reason='host compiler required')
def test_public_run_pair_reuses_adapter_without_stale_bits(tmp_path,cpu_native):
    for rows,width in ((5,17),(4,9)):
        for arm in ('before','after'):
            source=tmp_path/f'{rows}-{width}-{arm}.cpp'
            source.write_text(f'''#include <stdint.h>
extern "C" uint32_t affine_rows(){{return {rows};}}
extern "C" uint32_t affine_width(){{return {width};}}
extern "C" uint32_t affine_grid(){{return 2;}}
extern "C" uint32_t affine_after(){{return {int(arm=='after')};}}
extern "C" uint32_t catlass_bridge_abi_version(){{return 1;}}
extern "C" const char* catlass_bridge_kernel_names(){{return "[\\\"host_affine\\\"]";}}
extern "C" int catlass_bridge_launch(void**p,uint64_t c,void*){{
 if(c!=4)return 1;
 for(int i=0;i<{rows*width};i++){{volatile float v=((float*)p[0])[i]*((float*)p[1])[i%{width}];((float*)p[3])[i]=v+((float*)p[2])[i%{width}];}}return 0;}}
''')
            binary=source.with_suffix('.so')
            subprocess.run(['c++','-shared','-fPIC',str(source),'-o',str(binary)],check=True)
            build=tmp_path/f'build-{rows}-{width}-{arm}';build.mkdir()
            (build/'build.json').write_text(json.dumps(dict(binary_sha256=digest(binary))))
    (tmp_path/'sdk-declarations.json').write_text('[]')
    adapter=Adapter(tmp_path)
    for i,(seed,rows,width,order) in enumerate(((19401,5,17,('before','after')),
            (19402,5,17,('before','after')),(19403,4,9,('after','before')))):
        spec=dict(seed=seed,parameters=dict(rows=rows,width=width))
        report=run_pair(adapter,spec,tmp_path/f'pair-{i}',tmp_path/'cache',{},order=order,correctness_only=True)
        assert report['status']=='complete', report
        assert report['arms'][order[0]]['correctness']['diagnostics']['paired_bitwise'] is None
        assert report['arms'][order[1]]['correctness']['diagnostics']['paired_bitwise'] is True
        assert report['counts']==dict(profiled_calls=0,capture_calls=0,correctness_calls=2)
        assert not adapter.output_bits  # Completed pairs release their CPU snapshots.

    class Mismatch(Adapter):
        def prepare(self,arm,*args):
            item=super().prepare(arm,*args)
            if arm=='after':
                original=item.run
                def run():
                    output=original()
                    output[0,0]=torch.nextafter(output[0,0],torch.full((),float('inf')))
                    return output
                item.run=run
            return item
    report=run_pair(Mismatch(tmp_path),dict(seed=19404,parameters=dict(rows=5,width=17)),
        tmp_path/'mismatch',tmp_path/'cache',{},correctness_only=True)
    assert report['status']=='failed'
    diagnostics=report['arms']['after']['correctness']['diagnostics']
    assert diagnostics['primary_accuracy'] and diagnostics['paired_bitwise'] is False
