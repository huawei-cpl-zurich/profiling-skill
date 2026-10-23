# Preload the next output task

Status: source-backed example, awaiting A3 correctness and ten-forward captures.
Native family08 is in `examples/optimization-library/catlass/preload.cpp`.

Recognize several independent output tasks per Cube core, each followed by
operand movement for its successor. A transfer-bound gap is a hypothesis;
pipe-active fractions alone do not establish an overlap opportunity.

The before path uses `BasicMatmul` and `MmadAtlasA2Pingpong<true>`. The after
path uses `OptimizedMatmul<void,void,...>` and
`MmadAtlasA2Preload<true,false>`. A/B/output types, tiles128x256x256,
L0 tiles128x256x64, two operand stages and output scheduler remain the same;
ShuffleK and padding are disabled. This is a kernel/API change because the
preload operator needs the next task's pointers and first/last-task flags.
Simply substituting its policy into `BasicMatmul` does not implement it.

Test2048x2048x512 against128x256x512. The first supplies128tasks versus one
in the second; actual observed Cube-core count determines tasks per core.
Potential gain is hiding next-task transfer latency. Costs are fill/drain,
preload bookkeeping and retention of additional live operands; there may be
nothing to preload in the one-task case. Both paths reserve384KiB of L1 in
this assembly, so an extra L1 allocation is not an observed tradeoff here.
A future asynchronous-policy implementation must check its own
PRELOAD_STAGES<L1_STAGES contract rather than transplanting these arguments.

Compare the complete launch, its ten samples and per-kernel rows. Use genuine
temporal evidence for an overlap claim; a lower duration with changed pipe
ratios supports a more limited result. Unchanged duration or no next-task
reuse falsifies the proposed benefit. Keep the FP16 operands/output, FP32
CPU oracle,1e-3 absolute/relative tolerance and input immutability unchanged.

Sources: [preload assembly](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/06_optimized_matmul/optimized_matmul.cpp:L72-L101),
[next-task interface](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/optimized_matmul.hpp:L343-L409).
