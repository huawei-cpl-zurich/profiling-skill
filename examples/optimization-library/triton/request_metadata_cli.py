#!/usr/bin/env python3
"""Exploratory request mapping adapter CLI; frozen qualification is collector-owned."""
import argparse
import json
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]/'scripts')]
from request_metadata import Adapter
from request_metadata_contract import dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--compile-inspection',action='store_true',help='Input-only compilation; numeric execution remains disabled')
    args = parser.parse_args()
    spec = json.loads(Path(args.case).read_text())
    dimensions(spec)
    if args.validate_only:
        print(json.dumps(dict(status='domain-valid',qualification='failed-study-no-numeric-execution',role=spec.get('role'))))
        return 0
    if not args.compile_inspection:
        parser.error('R05 qualification failed; numeric execution disabled; --compile-inspection permits input-only inspection')
    adapter=Adapter(compile_inspection=True);rows=[]
    for arm in ('before','after'):
        item=adapter.probe(arm,spec);item.compile_only()
        rows.append(dict(arm=arm,metadata=item.metadata,compilation=item.compilation_state()))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'inspection.json').write_text(json.dumps(dict(status='compile-inspection-only',numeric_qualified=False,arms=rows),indent=2)+'\\n')
    return 0



if __name__=='__main__':
    raise SystemExit(main())
