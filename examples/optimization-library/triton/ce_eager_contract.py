"""Independent FP64 mean-CE oracle; represented dyadic FP32 inputs."""
from pathlib import Path
import math
import torch
CASE_SOURCES=[Path(__file__)]


def dimensions(spec):
    p=spec.get('parameters',spec);r,v=p.get('rows'),p.get('V');need=p.get('needs_grad');u=p.get('upstream')
    if (type(r) is not int or type(v) is not int or not 1<=r<=128 or not 2<=v<=16384
        or type(need) is not bool or p.get('smoothing') not in (0.,.1)
        or type(p.get('smoothing')) not in (int,float) or type(spec.get('seed')) is not int or spec['seed']<0
        or (need and (type(u) not in (int,float) or not math.isfinite(u) or abs(u)>2)) or (not need and u is not None)
        or type(p.get('all_ignored',False)) is not bool):raise ValueError('unsupported CE domain')
    targets=p.get('targets')
    if targets is not None and (not isinstance(targets,list) or len(targets)!=r or any(type(y) is not int or (y!=-100 and not 0<=y<v) for y in targets)):
        raise ValueError('target bounds')
    return r,v


def validate_inputs(inputs,spec):
    r,v=dimensions(spec)
    if not isinstance(inputs,dict) or set(inputs)!={'X','Y'}:raise ValueError('X/Y required')
    for name,shape,dtype in [('X',(r,v),torch.float32),('Y',(r,),torch.int64)]:
        x=inputs[name]
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shape or not x.is_contiguous()
            or x.storage_offset()!=0 or x.untyped_storage().nbytes()!=x.numel()*x.element_size()):raise ValueError('owned contiguous CPU input required')
    x,y=inputs['X'],inputs['Y']
    if x.untyped_storage().data_ptr()==y.untyped_storage().data_ptr():raise ValueError('input alias')
    if not bool(torch.isfinite(x).all() and (x.abs()<=4).all() and (x*16==(x*16).trunc()).all()):raise ValueError('finite integer/16 logits in [-4,4]')
    if not bool(((y==-100)|((y>=0)&(y<v))).all()):raise ValueError('target bounds')
    p=spec.get('parameters',spec)
    if p.get('all_ignored') and bool((y!=-100).any()):raise ValueError('all-ignored target contract')
    if 'targets' in p and not p.get('all_ignored') and y.tolist()!=p['targets']:raise ValueError('explicit targets contract')
    return int((y!=-100).sum())


def make_case(spec):
    r,v=dimensions(spec);p=spec.get('parameters',spec);rng=torch.Generator().manual_seed(spec['seed'])
    x=torch.randint(-64,65,(r,v),generator=rng).float()/16
    y=torch.randint(v,(r,),generator=rng);y[::7]=-100
    if 'targets' in p:y=torch.tensor(p['targets'],dtype=torch.int64)
    if p.get('all_ignored'):y.fill_(-100)
    inputs=dict(X=x,Y=y);n=validate_inputs(inputs,spec);xd=x.double();lse=xd.logsumexp(1);prob=(xd-lse[:,None]).exp()
    valid=y!=-100;eps=p['smoothing'];loss=torch.zeros((),dtype=torch.float64);dx=torch.zeros_like(xd)
    if n:
        loss=((1-eps)*(lse[valid]-xd[valid,y[valid]])+eps*(lse[valid]-xd[valid].mean(1))).sum()/n
        if p['needs_grad']:
            dx[valid]=(prob[valid]-eps/v)/n
            dx[valid,y[valid]]-=(1-eps)/n
            dx*=p['upstream']
    return dict(inputs=inputs,expected=(loss,dx if p['needs_grad'] else None))


def check_output(actual,expected):
    if not isinstance(actual,tuple) or len(actual)!=2:return dict(passed=False,reason='(loss,dx-or-None) required')
    rows=[];passed=True
    for name,a,e,atol in zip(('loss','dx'),actual,expected,(2e-5,3e-6)):
        if e is None:
            passed=passed and a is None;rows.append(dict(name=name,expected_none=True,actual_none=a is None));continue
        if not isinstance(a,torch.Tensor) or a.dtype!=torch.float32 or a.shape!=e.shape or not a.is_contiguous():return dict(passed=False,reason='FP32 scalar/dense gradient required')
        got=a.detach().cpu().double();finite=torch.isfinite(got);bad=~torch.isclose(got,e,atol=atol,rtol=2e-5)|~finite;count=int(bad.sum());first=None
        if count:
            flat=int(bad.reshape(-1).nonzero()[0]);coords=[];remaining=flat
            for d in reversed(got.shape):coords.insert(0,remaining%d);remaining//=d
            first=dict(coordinates=coords,expected=float(e.reshape(-1)[flat]),actual=float(got.reshape(-1)[flat]) if bool(finite.reshape(-1)[flat]) else str(got.reshape(-1)[flat].item()))
        error=(got-e).abs();rows.append(dict(name=name,mismatch_count=count,first_mismatch=first,nonfinite_count=int((~finite).sum()),max_abs_error=float(error[finite].max()) if bool(finite.any()) else None,atol=atol,rtol=2e-5))
        passed=passed and count==0
    return dict(passed=passed,outputs=rows)
