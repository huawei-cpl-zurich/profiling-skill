# Stagger the per-core K starting tile

Status: all four native gallery pairs passed A3 correctness and ten-forward
profile/identity collection.
Native family11 is in `examples/optimization-library/catlass/preload.cpp`.

Recognize a large K traversal where cores concurrently start at the same
operand region. A conflict or cache-contention explanation needs source and
profile support; high MTE activity alone does not identify that cause.

Both arms use the same `OptimizedMatmul<void,void,...>`, scheduler,
128x256x256 tiles and FP16 operands/output. Change only
`MmadAtlasA2Preload<true,false>` to `<true,true>`. The policy changes
startTileIdx from zero to blockIdx and visits
(startTileIdx+kLoopIdx)%ceil(K/256). Each core still visits every K tile.
This is K accumulation order, distinct from output-tile program ordering.

Test 2048x2048x4096 against512x512x256. With sixteen K tiles, staggered starts
may reduce conflicting accesses; modulo/index work and changed accumulation
order are competing costs. With one K tile both arms visit the same tile;
compiler elimination can remove the overhead, so do not promise a slowdown
or invent a tradeoff from equal paths. The unchanged numerical contract is
FP32 CPU accumulation then FP16 output,1e-3 absolute/relative tolerance.
Rounding differences must be measured; bitwise equivalence is not required.

Compare complete-launch ten-sample distributions and supported memory/pipe
observations. A benefit restricted to the multi-tile case is consistent with
the hypothesis; equal timing, an absent contention signal or a one-tile
speedup demands another explanation. Pipe ratios are neither bandwidth nor
distance to roofline. Both policies reserve384KiB L1 in this assembly.

Sources: [rotation implementation](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_mmad_preload.hpp:L150-L174),
[design rationale](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L430-L453).

## Code format

```cpp
// Before / after, with all other assembly choices identical:
using Before = Gemm::MmadAtlasA2Preload<true, false>;
using After  = Gemm::MmadAtlasA2Preload<true, true>;
// Kernel traversal: (blockIdx + kLoopIdx) % kTileCount after the change.
```

Use [preload.cpp](../../../examples/optimization-library/catlass/preload.cpp),
family `11`, with `preload_adapter:factory`. If `kTileCount == 1`, every core
visits the same sole tile under both policies; compare data rather than
claiming an indexing penalty that the compiler may remove.


## What the gallery establishes

The original multi-tile and one-tile cases established no timing gain at either
boundary. The supplemental 8192³ pressure case regresses on device span:
4670.565 ± 21.98 µs becomes 4739.465 ± 20.85 µs (0.9855×).
The thin 128×16384×16384 case improves modestly within this capture:
573.3 ± 7.657 µs becomes 549.36 ± 25.40 µs (1.0436×), with a descriptive
mean-difference interval excluding zero. Its caller comparison remains
inconclusive; this does not establish a robust gain across captures.

Both policies reserve the same L1 capacity. The one-tile operation has no useful
stagger and may compile away its indexing difference. These opposite pressure
outcomes reject a rule based on large K or high movement ratios alone. Preserve
the source contract and compare the candidate at the actual operation boundary.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| 11-catlass-gallery-benefit (2048×2048×4096) | 145.947 ± 2.237 / 146.983 ± 4.339 | 0.993×; inconclusive | 0.283208 ± 0.2433 / 0.209842 ± 0.07063 | inconclusive |
| 11-catlass-gallery-cost (512×512×256) | 6.62587 ± 3.523 / 5.77192 ± 1.537 | 1.148×; inconclusive | 0.230126 ± 0.1676 / 0.170388 ± 0.1953 | inconclusive |
| 11-catlass-gallery-pressure-square (8192×8192×8192) | 4670.56 ± 21.98 / 4739.47 ± 20.85 | 0.9855×; regression | 4.72433 ± 0.1825 / 4.77918 ± 0.1373 | inconclusive |
| 11-catlass-gallery-pressure-thin (128×16384×16384) | 573.3 ± 7.657 / 549.36 ± 25.4 | 1.044×; gain | 0.623124 ± 0.1547 / 0.577552 ± 0.1059 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mte2_ratio` | `aic_mac_ratio` | `cube_utilization(%)` |
| --- | ---: | ---: | ---: |
| 11-catlass-gallery-benefit/before/`wave1_shuffle_before` | 0.8067 | 0.9632 | 90.971 |
| 11-catlass-gallery-benefit/after/`wave1_shuffle_after` | 0.788 | 0.9554 | 91.092 |
| 11-catlass-gallery-cost/before/`wave1_shuffle_before` | 0.188 | 0.1695 | 53.048 |
| 11-catlass-gallery-cost/after/`wave1_shuffle_after` | 0.1583 | 0.1892 | 50.191 |
| 11-catlass-gallery-pressure-square/before/`wave1_shuffle_before` | 0.9557 | 0.8919 | 99.248 |
| 11-catlass-gallery-pressure-square/after/`wave1_shuffle_after` | 0.9693 | 0.8788 | 99.401 |
| 11-catlass-gallery-pressure-thin/before/`wave1_shuffle_before` | 0.9916 | 0.5456 | 83.874 |
| 11-catlass-gallery-pressure-thin/after/`wave1_shuffle_after` | 0.9914 | 0.5712 | 83.656 |

Observed resource/numerical costs and limits:
- **11-catlass-gallery-benefit**: no additional resource/numerical cost identified by the retained fields.
- **11-catlass-gallery-cost**: no additional resource/numerical cost identified by the retained fields.
- **11-catlass-gallery-pressure-square**: no additional resource/numerical cost identified by the retained fields.
- **11-catlass-gallery-pressure-thin**: no additional resource/numerical cost identified by the retained fields.
