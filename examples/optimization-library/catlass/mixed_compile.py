"""Compile the four immutable mixed-kernel variants without launching them."""
import argparse
from pathlib import Path
from compile_tile import compile_pair


def compile_variants(catlass,output,families=('02','10')):
    if not families or any(f not in ('02','10') for f in families):
        raise ValueError('known nonempty native mixed families required')
    variants={f'{family}-{arm}':[f'-DFAMILY={int(family)}',f'-DAFTER={int(arm=="after")}']
              for family in families for arm in ('before','after')}
    return compile_pair(catlass,output,source=Path(__file__).with_name('mixed.cpp'),variants=variants)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    compile_variants(args.catlass,args.output)
