# Change output-tile assignment without changing reduction order

Status: all four Triton gallery pairs passed A3 correctness and ten-forward
profile/identity collection. Family `12` retains its original row-wise rotation
in [gemm.py](../../../examples/optimization-library/triton/gemm.py), adapter
`gemm:Adapter`, and adds grouped-M ordering in
[grouped_order.py](../../../examples/optimization-library/triton/grouped_order.py),
adapter `grouped_order:Adapter`. The paired runner is repository-root
`scripts/optimization_pair.py`; use the same family12 case/CPU-oracle contract.

## Recognize the opportunity

Look for many output programs reloading overlapping A/B regions, a working set
that plausibly strains shared cache, and a scheduling order that separates
programs sharing an operand tile. Reordering can cluster accesses to the same B
column tile across nearby M tiles. It also adds mapping work and can worsen A
reuse. High movement activity alone does not prove cache contention or traffic.

The [pinned Catlass group4 swizzle](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_swizzle.hpp:L79-L97)
motivates grouping M tiles; its final group shrinks to the remaining M tiles.
The source also reverses N in odd groups. This independent Triton adaptation
omits that reversal. It is distinct from the
[original threshold-gated mapping](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/matmul.py:L100-L117),
which motivated the separately retained row-wise rotation experiment.

## Code formats and guards

Before: `tile_m, tile_n = divmod(pid, columns)`. Grouped-M after:

```python
group_m = 4
group_programs = group_m * columns
first_m = (pid // group_programs) * group_m
group_size = min(rows - first_m, group_m)
inner = pid % group_programs
tile_m = first_m + inner % group_size
tile_n = inner // group_size
```

Use positive grid dimensions and `0 <= pid < rows*columns`; the final partial
M group must use its actual `group_size` in both division and remainder. A fixed
four-row modulus in the last group can duplicate or omit output tiles.
The [CPU contract](../../../examples/optimization-library/triton/gemm_contract.py)
and grouped source expose coverage checks separately from arithmetic validation.
Both grouped arms retain 64×128 output tiles, K blocks of 32, ascending K
accumulation, FP16 inputs/output and FP32 accumulation. Preserve matrix strides,
edge masks, unchanged independent CPU oracle and rtol=atol=1e-3. This changes
output assignment, never the K summation order. Neither arm changes explicit
workspace allocation; compiler scratch is not quantified here.

The original after mapping remains:

```python
row, column = divmod(program_id, columns)
# Apply only when rows >= 2 and columns >= 2.
column = (column + row) % columns
```

For each row this is a permutation, but it does not provide grouped-M locality.
A verified bijection establishes coverage, not improved scheduling or cache use.
Small grids, one-row grids and cache-resident operands can make added mapping
useless. Group size4 is this tested implementation choice, not an inferred
universal crossover. A row-major alternative remains appropriate when reuse
is absent, mapping dominates, or measurements are inconclusive.

## What the gallery establishes

The original 4096×4096×2048 and 256×512×64 row rotations remain inconclusive.
The new grouped-M case 2048×16384×8192 improves device span from
19127.028 ± 178.2 µs to 15463.446 ± 32.57 µs (1.2369×), and instrumented caller
events from 19.2299 ± 0.3205 ms to 15.5161 ± 0.1855 ms (1.2394×).
This is a within-capture scheduling benefit; it does not prove reduced bytes or
cache misses. The 320×384×32 case exercises a final partial M group and shows
2.796 ± 0.1006 µs versus 2.878 ± 1.275 µs: a 0.9715× point ratio with an interval
crossing zero. Its cost regime is inconclusive, not a demonstrated slowdown.

Falsify the locality hypothesis with unchanged supported cache/traffic evidence,
no span improvement, or a regression at the intended working set. Attribute a
cache mechanism only when supported counters or traffic measurements establish
it. The timing evidence alone cannot estimate roofline distance.

Shared GEMM adapter metadata mentions an unfused intermediate even for this
family; the actual ordering sources do not change allocations. This recorded
string is not an allocation-saving result.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| 12-triton-gallery-benefit (4096×4096×2048) | 1436.3 ± 9.048 / 1436.53 ± 4.622 | 0.9998×; inconclusive | 1.50274 ± 0.2151 / 1.4825 ± 0.1472 | inconclusive |
| 12-triton-gallery-cost (256×512×64) | 3.85 ± 0.2598 / 4.222 ± 0.7394 | 0.9119×; inconclusive | 0.237184 ± 0.146 / 0.192532 ± 0.07155 | inconclusive |
| 12-triton-grouped-gallery-benefit (2048×16384×8192) | 19127 ± 178.2 / 15463.4 ± 32.57 | 1.237×; gain | 19.2299 ± 0.3205 / 15.5161 ± 0.1855 | gain |
| 12-triton-grouped-gallery-cost (320×384×32) | 2.796 ± 0.1006 / 2.878 ± 1.275 | 0.9715×; inconclusive | 0.255272 ± 0.1962 / 0.300846 ± 0.3631 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mte2_ratio` | `aic_mac_ratio` | `cube_utilization(%)` |
| --- | ---: | ---: | ---: |
| 12-triton-gallery-benefit/before/`gallery_gemm` | 0.9864 | 0.3145 | 98.692 |
| 12-triton-gallery-benefit/after/`gallery_gemm` | 0.9853 | 0.315 | 98.504 |
| 12-triton-gallery-cost/before/`gallery_gemm` | 0.421 | 0.0556 | 51.586 |
| 12-triton-gallery-cost/after/`gallery_gemm` | 0.4145 | 0.0524 | 50.625 |
| 12-triton-grouped-gallery-benefit/before/`gallery_gemm` | 0.999 | 0.2017 | 93.613 |
| 12-triton-grouped-gallery-benefit/after/`gallery_grouped_gemm` | 0.999 | 0.2341 | 99.784 |
| 12-triton-grouped-gallery-cost/before/`gallery_gemm` | 0.1131 | 0.0444 | 40.111 |
| 12-triton-grouped-gallery-cost/after/`gallery_grouped_gemm` | 0.1941 | 0.0484 | 41.601 |

Observed resource/numerical costs and limits:
- **12-triton-gallery-benefit**: no additional resource/numerical cost identified by the retained fields.
- **12-triton-gallery-cost**: no additional resource/numerical cost identified by the retained fields.
- **12-triton-grouped-gallery-benefit**: recorded allocation-boundary strings differ, but source inspection shows no allocation change; implicit compiler scratch is unquantified.
- **12-triton-grouped-gallery-cost**: recorded allocation-boundary strings differ, but source inspection shows no allocation change; implicit compiler scratch is unquantified.
