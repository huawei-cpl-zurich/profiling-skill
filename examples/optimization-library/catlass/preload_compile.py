"""Compile-only, six immutable native preload-family libraries."""
import argparse
from pathlib import Path
from compile_tile import compile_pair


def compile_variants(catlass,output):
    variants={f'{family}-{arm}':[f'-DFAMILY={int(family)}',f'-DAFTER={int(arm=="after")}']
              for family in ('08','09','11') for arm in ('before','after')}
    return compile_pair(catlass,output,source=Path(__file__).with_name('preload.cpp'),variants=variants)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    compile_variants(args.catlass,args.output)
