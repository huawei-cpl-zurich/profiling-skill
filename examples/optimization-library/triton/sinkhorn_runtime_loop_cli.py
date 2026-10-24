#!/usr/bin/env python3
"""Exploratory one-case runner; no controlled pre-oracle pair qualification."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinkhorn_runtime_loop import Adapter
from sinkhorn_runtime_contract import dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'output', 'cache', 'provenance'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    spec = json.loads(Path(args.case).read_text())
    parameters, _, _, _ = dimensions(spec)
    if args.validate_only:
        print(json.dumps(dict(status='supported', parameters=parameters,
            qualification='exploratory/unqualified',pre_oracle_pair_screen=False,paired_codegen_screen_before_device_math=False)))
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
    from optimization_pair import run_pair
    provenance = json.loads(Path(args.provenance).read_text())
    provenance.update(qualification='exploratory/unqualified',pre_oracle_pair_screen=False,paired_codegen_screen_before_device_math=False)
    result = run_pair(Adapter(), spec, args.output, args.cache, provenance)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
