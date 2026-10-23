# Padding and physical-layout conversion

Status: source-backed experiment awaiting hardware correctness and profiling.
Format: Catlass C++; `mixed.cpp`, FAMILY=10. Native factory:
`mixed_adapter:padding_factory`.

Identify a strided ND operand with inconvenient K/N alignment and repeated
GM-to-L1 movement. Compare direct ND access against an explicit NZ conversion
prologue, retaining the same Preload<true,false> policy, tiles, scheduler and
numerical contract. This isolates conversion/layout from ShuffleK or an added
preload optimization. Source and physical strides must match the claimed input
layout; a logical tensor shape does not establish its physical storage.

The OptimizedMatmul after arm executes both operand prologues, barriers and
compute in its complete mixed launch. Report workspace and count conversion,
synchronization and output work. Force the selected prologues for the aligned
negative case as well; a runtime bypass would compare identical paths.

Proposed beneficial regime: M=1024, N=K=1000. Proposed cost regime: M=N=K=128,
where conversion adds work to already aligned inputs. Held-outs use
1536x1500x1004 and 192x256x256. Both arms use FP16 inputs/output with an
independent FP32 CPU accumulation followed by FP16 conversion, rtol=atol=1e-3.
A correctness failure rejects the example rather than proving a tradeoff.

Expect the converted layout to change movement and complete task duration;
fewer transfer stalls is a hypothesis until supported counters establish it.
A faster central MMAD with a slower complete launch rejects an end-to-end win.
Missing traffic counters remain unavailable; pipe activity does not measure
bytes or distance to roofline.

[PaddingBuilder and conversion selection](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/06_optimized_matmul/optimized_matmul.cpp:L124-L149),
[whole mixed prologue/compute execution](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/optimized_matmul.hpp:L313-L374).
