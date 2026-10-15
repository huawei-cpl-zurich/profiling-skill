#!/usr/bin/env python3
"""Derived runtime-loop controls; never substitute these for original timings.

Each iteration rereads the same inputs and rewrites the same result. Volatile
loads prevent hoisting the input operations. The added loop and volatility can
change compiler scheduling; use only paired evidence with matching repetitions.
"""
import argparse
import json

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from a3_profile_microbench import a3_vector_probe, generate_inputs, reference_output, outputs_close
from compile_evidence import retain_compiler_evidence


@triton.jit
def a3_amplified_probe(x, selected, work_ptr, output, repeat_ptr,
                       n: tl.constexpr, work: tl.constexpr, mode: tl.constexpr):
    repetitions = tl.load(repeat_ptr)
    for _ in range(repetitions):
        a3_vector_probe(x, selected, work_ptr, output, n, work, 128, mode, 1,
                        volatile_loads=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True,
                        choices=['scalar-mask-decision', 'load-masking', 'sparse-mode-control'])
    parser.add_argument('--sweep', required=True)
    parser.add_argument('--variant', required=True, choices=['before', 'after'])
    parser.add_argument('--repetitions', type=int, choices=[1, 32, 128], required=True)
    args = parser.parse_args()
    inputs = generate_inputs(args.experiment, args.sweep)
    expected = reference_output(args.experiment, args.variant, inputs)
    dev = {key: value.npu() for key, value in inputs.items()}
    repeats = torch.tensor([args.repetitions], dtype=torch.int32).npu()
    output = torch.empty_like(expected, device='npu')
    modes = {'scalar-mask-decision': (2, 3), 'load-masking': (7, 4),
             'sparse-mode-control': (5, 6)}
    mode = modes[args.experiment][args.variant == 'after']
    compiled = a3_amplified_probe[(triton.cdiv(inputs['x'].numel(), 128),)](
        dev['x'], dev['selected'], dev['work'], output, repeats,
        n=inputs['x'].numel(), work=int(inputs['work']), mode=mode,
    )
    torch.npu.synchronize()
    correct = outputs_close(output.cpu(), expected)
    artifacts = retain_compiler_evidence(compiled.asm, 'compiler-evidence',
                                         args.experiment, args.variant, args.repetitions)
    print(json.dumps({'status': 'success' if correct else 'failure',
                      'derived': True, 'repetitions': args.repetitions,
                      'correctness': correct, 'compiler': artifacts}))
    return 0 if correct else 1


if __name__ == '__main__':
    raise SystemExit(main())
