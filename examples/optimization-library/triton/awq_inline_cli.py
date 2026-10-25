#!/usr/bin/env python3
"""Exploratory typed-source AWQ CLI; controlled all-case qualification is collector-owned."""
import argparse,json,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]/'scripts')]
from awq_inline import Adapter
from awq_inline_contract import dimensions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('case','output','cache','provenance'):p.add_argument('--'+n,required=True)
    p.add_argument('--validate-only',action='store_true');p.add_argument('--compile-inspection',action='store_true')
    a=p.parse_args();spec=json.loads(Path(a.case).read_text());dimensions(spec)
    if a.validate_only:print(json.dumps(dict(status='domain-valid',numeric_qualified=False,role=spec.get('role'))));return 0
    if not a.compile_inspection:p.error("R09 compiler/type qualification failed; numeric execution disabled; --compile-inspection permits input-only inspection")
    adapter=Adapter(compile_inspection=True)
    if a.compile_inspection:
        rows=[]
        for arm in ('before','after'):
            item=adapter.probe(arm,spec);item.compile_only()
            rows.append(dict(arm=arm,metadata=item.metadata,compilation=item.compilation_state()))
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        (out/'inspection.json').write_text(json.dumps(dict(status='compile-inspection-only',numeric_qualified=False,arms=rows),indent=2)+'\n')
        return 0

if __name__=='__main__':raise SystemExit(main())
