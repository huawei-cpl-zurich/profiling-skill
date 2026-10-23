# Padding and physical-layout conversion

Status: both listed native gallery pairs passed A3 correctness and ten-forward
profile/identity collection.
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

Gallery large/unaligned regime: M=1024, N=K=1000. Gallery aligned regime: M=N=K=128,
where conversion adds work to already aligned inputs. Both arms use FP16 inputs/output with an
independent FP32 CPU accumulation followed by FP16 conversion, rtol=atol=1e-3.
A correctness failure rejects the example rather than proving a tradeoff.

Expect the converted layout to change movement and complete task duration;
fewer transfer stalls is a hypothesis until supported counters establish it.
A faster central MMAD with a slower complete launch rejects an end-to-end win.
Missing traffic counters remain unavailable; pipe activity does not measure
bytes or distance to roofline.

[PaddingBuilder and conversion selection](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/06_optimized_matmul/optimized_matmul.cpp:L124-L149),
[whole mixed prologue/compute execution](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/optimized_matmul.hpp:L313-L374).

## Code format

```cpp
// Before: no conversion prologue, original row-major operand types.
using BeforeKernel = Gemm::Kernel::OptimizedMatmul<void, void, BeforeBlockMmad,
                                            void, Scheduler>;
// After: instantiate BlockMmad with PadA/PadB::LayoutAfterPadding.
using PadA = Gemm::Kernel::PaddingBuilder<
    Gemm::Kernel::PaddingTag::PADDING_NZ, Arch::AtlasA2,
    half, InputLayout, 48 * 1024 / sizeof(half)>;
using PadB = PadA;
using AfterKernel = Gemm::Kernel::OptimizedMatmul<
    PadA::Padding, PadB::Padding, AfterBlockMmad, void, Scheduler>;
```

`PadB` uses the same builder in this row-major example. The runnable
[source](../../../examples/optimization-library/catlass/mixed.cpp), family `10`,
computes the complete conversion workspace from the chosen kernel API.


## What the gallery establishes

The large unaligned case improved device span and added 4,096,512 workspace
bytes. The aligned case regressed in device span because the conversion path
was deliberately retained. Both caller-event comparisons were inconclusive.
This is a measured device-level benefit/cost pair; it does not establish an
application-latency win or the number of physical bytes moved.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| catlass / benefit | 40.284 ± 3.049 / 32.502 ± 1.95 | 1.239×; gain | 0.147674 ± 0.1247 / 0.20107 ± 0.2329 | inconclusive |
| catlass / cost | 5.56 ± 2.385 / 12.484 ± 1.197 | 0.4454×; regression | 0.14381 ± 0.05379 / 0.140672 ± 0.1044 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mte2_ratio` | `aiv_mte3_ratio` | `cube_utilization(%)` |
| --- | ---: | ---: | ---: |
| catlass/benefit/before/`wave1_padding_before` | 0.7442 | 0.0009 | 81.035 |
| catlass/benefit/after/`wave1_padding_after` | 0.1807 | 0.0843 | 86.201 |
| catlass/cost/before/`wave1_padding_before` | 0.012 | 0.0008 | 72.385 |
| catlass/cost/after/`wave1_padding_after` | 0.0029 | 0.0116 | 90.041 |

Observed resource/numerical costs and limits:
- **catlass/benefit**: adapter-explicit workspace allocation 0->4096512 bytes.
- **catlass/cost**: adapter-explicit workspace allocation 0->65536 bytes.
