# Wave006 L01: single-wave control specialization

This educational contrast specializes task scheduling and the outer K loop in
one shared native CATLASS block body. It keeps row-major mapping, L1 tile
128×256×256, L0 tile128×256×64, double buffering, FP32 accumulation, FP16
operands/output, events and terminal barrier fixed. Unit flags remain false.
It does not compare the stock general pingpong algorithm against a different
small-matmul copy/buffering algorithm, or establish a best vendor baseline.

The source cue is a short GEMM with at most one output task per available Cube
and at most one L1 K stage. General loop bookkeeping may matter in this domain;
compilers or dominant copy/math costs may erase its effect. Direct control
requires `ceil(M/128)*ceil(N/256) <= queried Cube count` and `K <= 256`.
The cost is an extra binary and host dispatch condition. Unsupported shapes
load the exact same general binary on both arms.

Pinned source: CATLASS `40aced927fdaaa0f3eaa9c6f5c5d859e06392f55`, CANN Open
Software License2.0. The native small example explicitly annotates its entry
`__attribute__((aic))`; this placement annotation is required in both arms.
The local block inherits upstream allocation and event initialization/drain;
only its operator is adapted for shared general/direct control. See
[small block](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_mmad_small.hpp),
[small kernel](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/small_matmul.hpp),
and [native example](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/31_small_matmul/small_matmul.cpp).

A3 validation passed all six fixed cases on the observed20Cube device. Dense ND
physical layouts/strides were audited. Six bounded represented-FP16 fixtures and
independent FP64 GEMM payloads were generated once. Every arm passed output
finiteness, NaN-sentinel overwrite, input immutability, and atol=rtol0.002.
Each ordinary case had one correctness call and exactly10 full API forwards per
arm under profiling, with no warmup, autotune, or separate timing calls. Both
guards had one correctness call per arm and no captures.

| Case | M×N×K | Before span, µs mean±sample SD | After span, µs mean±sample SD |
| --- | --- | --- | --- |
| Gallery01 |128×256×64|3.634±1.448|3.022±0.348|
| Gallery02 |512×1024×256|5.576±0.156|5.574±0.108|
| Heldout03 |64×256×32|2.416±0.069|2.338±0.075|
| Heldout04 |384×1024×256|5.442±0.096|5.410±0.118|

These are profiler-active device operation spans, not ordinary latency. The
largest difference also has higher before-arm variation; the other differences
are small. No statistical speedup or exact scalar-instruction reduction is
claimed. Native binaries, generated device ELF sections/disassembly, all10
samples per arm, task PMU exports, source/dependency/compiler digests and oracle
bindings are retained in compact evidence; full vendor reports remain remote.
The generated device text sections are5036bytes (general) and4656bytes (direct);
the stack records are identical. The installed disassembler prints every
instruction as unavailable, so smaller text does not establish an exact scalar
instruction count. All80 retained task rows report AI_CORE, Block Num20 and
Mix Block Num0, consistent with the Cube-only entry annotation.
Pipe fields do not establish physical traffic, device utilization or roofline.

The32-task guard (1024×1024×64) and multi-K-stage guard (128×256×272) both passed
through the identical general artifact. The latter executes a second outer K
stage with a final16-element tail. The64-row/32-K heldout case also passed.

An earlier source version compiled both artifacts but entered the first
correctness call on the wrong core type because the Cube-only annotation was
missing. Synchronization raised an AIVector exception; collection stopped with
one attempted correctness call and zero captures. That failed source, binary,
oracles, runtime error and counters remain retained. The corrected source
reused the unchanged six payloads and passed12 correctness calls/80 profiled
forwards. This failure establishes no hardware fault.
