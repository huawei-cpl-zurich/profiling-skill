"""Compile both controlled bodies before any device forwards."""
import argparse
from pathlib import Path
from compile_tile import compile_pair


def compile_variants(catlass, output):
    compile_pair(catlass, output, source=Path(__file__).with_name('single_stage.cpp'),
                 variants={name: [f'-DCONTROL_DIRECT={direct}']
                           for name, direct in [('general', 0), ('direct', 1)]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catlass', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compile_variants(args.catlass, args.output)
