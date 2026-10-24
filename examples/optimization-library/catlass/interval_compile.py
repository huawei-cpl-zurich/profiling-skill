"""Compile the two copy-method arms from one pinned Catlass dependency closure."""
import argparse
from pathlib import Path
from compile_tile import compile_pair


def compile_variants(catlass,output):
    here=Path(__file__).parent
    return compile_pair(catlass,output,source=here/'interval.cpp',
        variants={f'E12-{arm}':[f'-DAFTER={int(arm=="after")}'] for arm in ('before','after')},
        extra_sources=[here/'interval_config.h'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    compile_variants(args.catlass,args.output)
