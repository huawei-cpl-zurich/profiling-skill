"""Specified FP32 full-softmax and left-to-right strict crossing, without sampling."""
from pathlib import Path
import numpy as np
import torch


def dimensions(spec):
    rows, vocab, prefix, seed = (spec.get(k) for k in ('rows','V','prefix','seed'))
    if any(type(x) is not int for x in (rows,vocab,prefix,seed)):
        raise ValueError('integer dimensions and seed required')
    p=np.float32(spec['p'])
    if min(rows,vocab,prefix)<=0 or prefix>vocab or seed<0 or not 0<p<1:
        raise ValueError('positive dimensions, valid prefix, finite 0<p<1 required')
    return rows,vocab,prefix,seed


def reference(scores,p,prefix=None):
    """Host executable semantics; denominator and ordered prefix are FP32 scans."""
    rows=[]
    for row in scores.float().numpy():
        exp=np.exp(row-np.max(row),dtype=np.float32)
        denominator=np.float32(0)
        for x in exp: denominator=np.float32(denominator+x)
        probability=exp/denominator
        order=np.argsort(-row,kind='stable')
        total=np.float32(0); previous=np.float32(0)
        for count,index in enumerate(order,1):
            previous=total; total=np.float32(total+probability[index])
            if total>np.float32(p): break
        certified=(prefix is not None and count<=prefix and
                   np.float32(p)-previous>1e-5 and total-np.float32(p)>1e-5)
        rows.append(dict(count=count,previous=float(previous),mass=float(total),
                         tag=int(certified),values=torch.from_numpy(row[order[:count]].copy())))
    return rows


def make_case(spec):
    rows,vocab,prefix,seed=dimensions(spec)
    distribution=spec['distribution']
    scores=torch.zeros(rows,vocab,dtype=torch.float16)
    if distribution.endswith('-dominant-scores'):
        dominant=int(distribution.split('-')[0]);scores.fill_(-8);scores[:,:dominant]=8
    elif distribution!='uniform': raise ValueError('unknown frozen distribution')
    expected=reference(scores,spec['p'],prefix)
    if spec.get('role')!='guard' and any(min(spec['p']-r['previous'],r['mass']-spec['p'])<=1e-5 for r in expected):
        raise ValueError('frozen payload fails predeclared crossing margin')
    return dict(scores=scores,expected=expected)


def diagnostics(actual,fixture,arm):
    indices,counts,tags=(t.detach().cpu() for t in actual)
    scores=fixture['scores']
    if tuple(indices.shape)!=tuple(scores.shape) or counts.shape!=(scores.shape[0],) or tags.shape!=counts.shape:
        return dict(passed=False,reason='shape')
    details=[]
    for row,expected in enumerate(fixture['expected']):
        count=int(counts[row]);tag=int(tags[row]);ok=count==expected['count']
        if not 0<count<=scores.shape[1]:return dict(passed=False,reason='count')
        ids=indices[row,:count].long()
        if not bool(((ids>=0)&(ids<scores.shape[1])).all()):return dict(passed=False,reason='index range')
        ok=ok and ids.unique().numel()==count
        values=scores[row,ids].float()
        ok=ok and torch.equal(values,expected['values']) and tag==(expected['tag'] if arm=='after' else 0)
        details.append(dict(passed=bool(ok),count=count,dispatch='prefix' if tag else 'full_sort'))
    return dict(passed=all(r['passed'] for r in details),rows=details)


CASE_SOURCES=[Path(__file__)]
