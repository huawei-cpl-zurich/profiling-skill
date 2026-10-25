#!/usr/bin/env python3
"""Exploratory rotary Q/K adapter CLI; frozen qualification is collector-owned."""
import argparse
import json
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]/'scripts')]
from rotary_trig import Adapter
from rotary_trig_contract import dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    spec = json.loads(Path(args.case).read_text())
    dimensions(spec)
    if args.validate_only:
        print(json.dumps(dict(status='admitted',qualification='exploratory',role=spec.get('role'))))
        return 0
    from optimization_pair import run_pair
    report = run_pair(Adapter(),spec,Path(args.output),Path(args.cache),
                      json.loads(Path(args.provenance).read_text()))
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
