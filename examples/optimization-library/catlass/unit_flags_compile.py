"""Compile both unit-flag resources before any correctness/profile forward."""
import argparse
from pathlib import Path
from compile_tile import compile_pair


def compile_variants(catlass, output):
    return compile_pair(catlass, output, source=Path(__file__).with_name('unit_flags.cpp'),
                        variants={arm: [f'-DW006_UNIT_FLAGS={int(arm=="after")}']
                                  for arm in ('before', 'after')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compile_variants(args.catlass, args.output)
