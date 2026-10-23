# Operand multibuffering

Status: both listed native gallery pairs passed A3 correctness and ten-forward
profile/identity collection.

Look for repeated K-stage transfer/compute dependence in a kernel with enough
K work to amortize fill and drain. In `parallelism.cpp`, family07 holds tile
geometry, types, scheduler and BasicMatmul fixed while changing
`MmadPingpong<AtlasA2,true,false,1,false,S,S,S,S>` from S=1 to S=2.
L1 tiles are 128×256×256 and L0 tiles 128×256×64 in both arms.

Two operand buffers consume more L1/L0 capacity and may worsen a short K case.
Use the retained complete-call samples and supported timeline evidence to test
whether transfer gaps shrink. Pipe activity ratios alone do not demonstrate
simultaneous overlap. A crossover is inconclusive if it is within sample
variability; do not call additional buffering universally beneficial.

Gallery: 2048×2048×4096 versus 256×256×256. FP16 input/output and FP32 accumulation retain rtol=atol=1e-3.
The adapter records the stage count and actual compiled source/binary identity.

Pinned [stage policy](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/dispatch_policy.hpp:L351-L365),
[dispatch and capacity checks](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_mmad_pingpong.hpp:L24-L139).

## Code format

```cpp
// Hold Kernel, geometry and types fixed; S is the only changed policy setting.
constexpr uint32_t S = 1;  // Change to 2 in the after assembly.
using Policy = Gemm::MmadPingpong<Arch::AtlasA2, true, false, 1, false,
                                S, S, S, S>;
```

The runnable [native source](../../../examples/optimization-library/catlass/parallelism.cpp),
family `07`, uses `parallelism_adapter:factory`. Both variants must fit the
policy's L1/L0 capacity checks. More stages are not evidence of actual overlap.


## What the gallery establishes

The long-K case improved device span and the instrumented caller interval.
The short-K difference was inconclusive. The extra operand stage is a real
capacity cost even when it helps timing; it reduces headroom for another tile
or resident operand. These data do not by themselves prove temporal overlap.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| 07-catlass-gallery-benefit (2048×2048×4096) | 315.436 ± 4.883 / 148.115 ± 1.624 | 2.13×; gain | 0.348434 ± 0.1011 / 0.221298 ± 0.0914 | gain |
| 07-catlass-gallery-cost (256×256×256) | 5.54 ± 1.25 / 4.646 ± 0.8524 | 1.192×; inconclusive | 0.129332 ± 0.06487 / 0.13597 ± 0.08024 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mac_ratio` | `aic_mte2_ratio` | `aic_mte1_ratio` |
| --- | ---: | ---: | ---: |
| 07-catlass-gallery-benefit/before/`wave1_buffering_before` | 0.5028 | 0.3638 | 0.3326 |
| 07-catlass-gallery-benefit/after/`wave1_buffering_after` | 0.9528 | 0.7919 | 0.7358 |
| 07-catlass-gallery-cost/before/`wave1_buffering_before` | 0.0924 | 0.0759 | 0.0629 |
| 07-catlass-gallery-cost/after/`wave1_buffering_after` | 0.0954 | 0.0823 | 0.0696 |

Observed resource/numerical costs and limits:
- **07-catlass-gallery-benefit**: operand_stages: 1 -> 2.
- **07-catlass-gallery-cost**: operand_stages: 1 -> 2.
