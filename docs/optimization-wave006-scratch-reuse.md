# Owned scratch reuse (W006 L06)

This independent Ascend C adaptation changes explicit local storage ownership in
an otherwise identical four-stage FP32 vector operator. It is not the upstream
layer-normalization operator and does not establish a new unique technique.

Both arms compute four separately rounded operations:

```
a = fl(x*x)
y = fl(a+x)
b = fl(0.5*y)
y = fl(b-0.25)
```

BEFORE owns separate X, Y, A and B TBufs. AFTER owns X, Y and one scratch
TBuf obtained once through `Get<float>()`. A's final read completes at the
stage-two PIPE_V barrier before the same scratch tensor receives B. The input
copy, output copy, tile partition, four vector intrinsics and every barrier are
identical. X and Y never alias scratch. A separate `needs_first_late` mode
computes `y=fl(b+a)` and retains the same two-scratch implementation in both
arms because the content lifetimes overlap.

The mechanism is informed by successive temporary reuse in
[ops-rec ln_mul_kernel.h](https://gitcode.com/Ascend/ops-rec/tree/e489c0bf6a2d5f2d05b3205fcacc01b62a908b10/ascendc/activation/ln_mul/op_kernel/ln_mul_kernel.h),
lines 224–278 (Apache 2.0), and the direct TBuf/InitBuffer/Get API in
[CATLASS local_tensor_buffer.hpp](https://gitcode.com/cann/catlass/tree/40aced927fdaaa0f3eaa9c6f5c5d859e06392f55/include/catlass/arch/local_tensor_buffer.hpp),
lines 209–221 (CANN Open Software License Agreement 2.0). No whole CATLASS
Resource object is instantiated or reserved for this study.

The immutable matrix is [L06.json](../experiments/optimization-wave006/L06.json).
Inputs are seeded dyadic FP32 values in [-2,2]. The independent CPU oracle
performs each operation in FP64 and casts to FP32 after each stage; validation
requires bitwise equality, immutable input, nonalias output, and prefix/suffix
canaries on both GM allocations. The second guard runs 99 elements in two
iterations (64 then 35), with legal bounded DataCopyPad transfers. Both guards
are numerical and receive no profiler captures.

For the four main cases, BEFORE/AFTER request 131072/98304, 1024/768,
147456/110592 and 4096/3072 local bytes. These are explicit logical reservations,
not certified physical peak memory, occupancy, or memory traffic. Widths 1024
and 1536 belong to this study's domain; the parent row-DMA study's columns 128
policy does not apply. Both arms use explicit AIV entry annotations. Actual
selected SDK enum/ELF kind 3, nonempty vector entry, and absence of a Cube entry
must be audited before any scientific forward.

Use `scratch_reuse.compile_all(specs, fresh_directory)` before preparing either
arm, then point `SCRATCH_REUSE_DIRECTORY` at those binaries. The public adapter
factory is `scratch_reuse:factory` through `scripts/optimization_pair.py`.
Compilation preserves per-variant commands, source/compiler/binary hashes,
selected SDK declarations and intermediates. Every prepared variant checks its
rows, width, tile, lifetime mode and arm through the native ABI.

The protocol permits one correctness call per case/arm/source version, followed
by exactly ten full API profiled calls per main arm, without warmup, autotuning,
extra timing, or profiled tensor inspection. Six cached numerical fixtures are
created once. Profiling spans and instrumented event samples are diagnostics;
ordinary latency, physical bandwidth and roofline claims remain unavailable.

## Collected A3 evidence

The first conforming source passed all twelve numerical checks on an idle
Ascend 910B4-1 with CANN 9.1.0. All twelve native variants passed SDK/ELF AIV
admission before scientific forwards. Exactly eight capture contexts contain
80 profiled forwards; both numerical guards have zero captures. The six cached
fixtures, all ten samples per main arm, raw PMU rows, build manifests,
compiler/SDK/binary identities and resources are retained in the compact archive
bound by [L06-evidence.json](../experiments/optimization-wave006/L06-evidence.json).

Mean profiled device-operation spans, in microseconds:

| Case | BEFORE | AFTER | Logical requested bytes BEFORE/AFTER |
| --- | ---: | ---: | ---: |
| 8×1024 | 2.982 | 2.774 | 131072 / 98304 |
| 1×64 | 2.218 | 2.218 | 1024 / 768 |
| 6×1536 | 2.264 | 2.260 | 147456 / 110592 |
| 2×128 | 1.612 | 1.630 | 4096 / 3072 |

The requested local reservation falls by 25% for main cases. The span signs are
mixed; small differences overlap sample variation. These diagnostic spans do
not establish ordinary latency savings or a general performance win. Generic
CSV `AI_CORE` labeling is not evidence of Cube placement: actual SDK kind 3 and
vector-entry metadata establish the AIV engine independently.
