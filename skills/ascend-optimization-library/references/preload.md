# Preload the next output task

Status: both listed native gallery pairs passed A3 correctness and ten-forward
profile/identity collection.
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

## Code format

```cpp
// Before assembly
using BeforePolicy = Gemm::MmadAtlasA2Pingpong<true>;
using BeforeKernel = Gemm::Kernel::BasicMatmul<BeforeBlockMmad, void, Scheduler>;
// After assembly: no padding prologues; ShuffleK stays disabled.
using AfterPolicy = Gemm::MmadAtlasA2Preload<true, false>;
using AfterKernel = Gemm::Kernel::OptimizedMatmul<void, void, AfterBlockMmad,
                                            void, Scheduler>;
```

Use [preload.cpp](../../../examples/optimization-library/catlass/preload.cpp),
family `08`, with `preload_adapter:factory`; do not substitute the policy into
an incompatible block-call signature.


## What the gallery establishes

The multi-task case showed a modest device-span improvement in this capture.
The one-task case and both caller-event comparisons were inconclusive. Both
assemblies reserve the same L1 capacity, so this pair does not demonstrate a
larger L1 allocation as a cost. Next-task existence and fill/drain/bookkeeping
remain the relevant applicability questions; an overlap claim needs stronger
temporal evidence.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| 08-catlass-gallery-benefit (2048×2048×512) | 24.468 ± 0.9446 / 22.436 ± 2.191 | 1.091×; gain | 0.132794 ± 0.08124 / 0.177662 ± 0.2231 | inconclusive |
| 08-catlass-gallery-cost (128×256×512) | 6.54185 ± 2.791 / 6.01588 ± 0.7692 | 1.087×; inconclusive | 0.152598 ± 0.05258 / 0.177682 ± 0.2005 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mte2_ratio` | `aic_mac_ratio` | `cube_utilization(%)` |
| --- | ---: | ---: | ---: |
| 08-catlass-gallery-benefit/before/`wave1_preload_before` | 0.5811 | 0.751 | 87.967 |
| 08-catlass-gallery-benefit/after/`wave1_preload_after` | 0.6719 | 0.8122 | 88.44 |
| 08-catlass-gallery-cost/before/`wave1_preload_before` | 0.0758 | 0.0955 | 26.214 |
| 08-catlass-gallery-cost/after/`wave1_preload_after` | 0.0582 | 0.0689 | 32.81 |

Observed resource/numerical costs and limits:
- **08-catlass-gallery-benefit**: no additional resource/numerical cost identified by the retained fields.
- **08-catlass-gallery-cost**: no additional resource/numerical cost identified by the retained fields.
