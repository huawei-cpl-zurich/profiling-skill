# SplitK and complete partial reduction

Status: both listed native gallery pairs passed A3 correctness and ten-forward
profile/identity collection.

When M/N output tiles leave most Cube cores idle and K is long, split K into
independent FP32 partial matrices, then reduce every partial before converting
the complete result to FP16. In `parallelism.cpp`, family06 changes BasicMatmul
into `SplitkMatmul<BlockMmad,void,SplitkGemmIdentityBlockSwizzle<3,0>,ReduceAdd>`
with `SplitkReduceAdd<AtlasA2,float,half,2,16384>`. The source heuristic chooses
the factor; the adapter records the actual value and workspace bytes.

Compare whole-call latency, per-kernel activity and workspace traffic. Extra
partial writes, AIV reduction and cross-core synchronization can cost more than
extra parallelism saves. With abundant M/N tasks the factor may be one while
the mixed reduction still runs. The negative case therefore remains meaningful
without forcing artificial extra slices. Changed summation order must pass the
same FP32 CPU accumulation → FP16 oracle, rtol=atol=1e-3. A factor above one
alone is not evidence of a speedup or reduced bandwidth pressure.

Gallery: 64×64×16384 versus 2048×2048×256. The complete mixed launch includes reduction and its barrier.
Workspace is allocated before capture and its capacity cost is retained.

Pinned [kernel and heuristic](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/splitk_matmul.hpp:L142-L276),
[reduction](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/splitk_matmul.hpp:L23-L139).
Roofline distance remains unestimated without a supported capacity and traffic
denominator; the paired timing evidence below does not supply those values.

## Code format

```cpp
// Split path: FP32 partial output, complete reduction, then FP16 store.
using Partial = Gemm::GemmType<float, layout::RowMajor>;
using Reduce = Gemm::Kernel::SplitkReduceAdd<Arch::AtlasA2,
                                          float, half, 2, 16384>;
using Kernel = Gemm::Kernel::SplitkMatmul<BlockMmad, void,
    Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>, Reduce>;
```

Instantiate `BlockMmad` with FP32 partial CType. The runnable
[source](../../../examples/optimization-library/catlass/parallelism.cpp), family
`06`, supplies the workspace and hardware synchronization address. Use
`parallelism_adapter:factory`; dropping the reduction changes the operation.


## What the gallery establishes

The sparse long-K grid improved device span with factor16 and 262,144 bytes of
partial workspace. The abundant-output-tile case regressed with factor1 and
16,777,216 bytes of workspace. The complete reduction still ran with factor1.
Both caller-event comparisons were inconclusive; the device tradeoff is already
clear without dropping the reduction or manufacturing extra slices.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| catlass / benefit | 42.7411 ± 1.484 / 13.7797 ± 2.504 | 3.102×; gain | 0.14644 ± 0.1125 / 0.141998 ± 0.06016 | inconclusive |
| catlass / cost | 15.196 ± 3.392 / 27.68 ± 3.898 | 0.549×; regression | 0.137164 ± 0.08454 / 0.20001 ± 0.1609 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `cube_utilization(%)` | `aic_mac_ratio` | `aiv_vec_ratio` |
| --- | ---: | ---: | ---: |
| catlass/benefit/before/`wave1_splitk_before` | 7.9523 | 0.2544 | 0.1209 |
| catlass/benefit/after/`wave1_splitk_after` | 51.353 | 0.1199 | 0.0188 |
| catlass/cost/before/`wave1_splitk_before` | 87.74 | 0.6341 | 0.1218 |
| catlass/cost/after/`wave1_splitk_after` | 60.889 | 0.4889 | 0.0409 |

Observed resource/numerical costs and limits:
- **catlass/benefit**: adapter-explicit workspace allocation 0->262144 bytes; split_factor: 1 -> 16.
- **catlass/cost**: adapter-explicit workspace allocation 0->16777216 bytes.
