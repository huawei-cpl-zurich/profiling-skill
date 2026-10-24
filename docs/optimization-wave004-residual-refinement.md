# W004 W13: two residual corrections

The independent FP32 implementation factors once in both arms. The after arm
adds exactly two residual/retained-factor solves and whole-vector finite
proposal commits. See the [contract](../skills/ascend-optimization-library/references/residual-refinement.md)
for source scope, fixed arithmetic and acceptance criteria, and the
[frozen cases](../examples/optimization-library/cases/wave004/W13.json).

The four functional CPU tests pass, covering all six fixtures, immutable inputs
and padding, exact zero corrections, whole-vector overflow rollback, a
small-backward-error/large-forward-error counterexample, and the real nested
component-guard JSON through the public CLI. NPU and profiler outcomes are
retained separately when available. No accuracy improvement or speedup is
assumed. Whole first-to-last device operation span, including stage gaps, is the
primary profiler comparison; active duration and gaps remain separate. These
profiled observations do not establish ordinary application latency or physical
memory traffic.

On fresh healthy idle Ascend910B4-1 physical die 5, both arms passed the fixed
FP64 gate in all four cases. Maximum forward error decreased in each case,
while full profiled device span increased by 3.10–3.65 times. This is an
accuracy/work tradeoff, with no speed claim.

| Case | Before max forward error | After max forward error | Before span (us) | After span (us) |
| --- | ---: | ---: | ---: | ---: |
| 128 moderate | 4.238e-6 | 8.699e-7 | 386.002 | 1284.346 |
| 32 easy | 1.824e-9 | 1.487e-9 | 355.734 | 1298.918 |
| 192 moderate | 2.470e-6 | 1.071e-6 | 422.574 | 1308.460 |
| 64 easy | 2.723e-9 | 1.760e-9 | 362.314 | 1300.874 |

Each span is the mean of ten profiled forwards, with all copies, factors,
solves and inter-stage gaps included. The [retained compact evidence](data/optimization-w004-residual-refinement.json)
contains all samples, active/gap separation, forward/backward diagnostics,
source and NPUBIN hashes, task PMU rows and guard results. GA had exactly zero
forward error and residual in both arms; GB verified whole-vector rollback on
NPU with zero captures. Six CPU oracle payloads were generated once, each arm
checked once, and all resources compiled before the first forward. Four remote
functional CPU tests also passed. There were 80 profiled forwards and no warmup,
autotuning or extra timing calls. A staging-only path error exited before the
collector started; its terminal handle and correction are retained in evidence.
