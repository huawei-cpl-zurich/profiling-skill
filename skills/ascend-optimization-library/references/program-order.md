# Change output-tile assignment without changing reduction order

Status: both listed Triton gallery pairs passed A3 correctness and ten-forward
profile/identity collection. Family `12` is
implemented in `examples/optimization-library/triton/gemm.py` (`gemm:Adapter`).

The [pinned threshold-gated source mapping](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/matmul.py:L100-L117)
motivates testing different output-tile assignments. The source has a suspicious
tuple assignment elsewhere; our mapping is independently implemented and
bijective rather than copying it as a verified operator.

Before, assign row-major tile coordinates. After, rotate the column within each
row; keep the K loop ascending and identical:

```python
row, column = divmod(program_id, columns)
# After only, when rows >= 2 and columns >= 2:
column = (column + row) % columns
# Both: for k_tile in range(ceildiv(K, BK)): accumulate(...)
```

For each row, addition modulo columns is a permutation; rows are unchanged.
This covers rectangular grids exactly once. Masks still handle matrix elements
outside the final M/N/K tiles. The CPU tests additionally enumerate real matrix
elements on a tail grid to detect duplicates and missing output coverage.

Recognize many output tiles reusing the same operand regions, with a plausible
cache or conflicting-access hypothesis. A changed assignment can alter which
operand tiles concurrent programs request, at the cost of extra mapping work
and possibly worse reuse of the other operand. Small grids and cache-resident
inputs can gain nothing. The guard prevents meaningless rotation of one-row
or one-column grids; it is a correctness/applicability guard, not a measured
crossover threshold.

The gallery compares 4096×4096×2048 and 256×512×64.
Both arms use FP16 inputs/output, FP32 accumulation, and the same CPU oracle
converted to FP16 at rtol=atol=1e-3. Preserve all input strides and dispatch only
within the adapter's declared matrix/layout contract.

Compare ten owned complete-call timings and kernel identities. Attribute a cache
or traffic improvement only with supported evidence, not elapsed time alone.
Falsifiers include equal times within variability, slower rotation, or unchanged
measured traffic. This transformation is distinct from ShuffleK: it reorders
M/N program assignment, never the sequence of K accumulation.

The measured regimes below are not a universal locality threshold.

Runnable [GEMM source](../../../examples/optimization-library/triton/gemm.py),
adapter `gemm:Adapter`, family `12`;
[CPU contract](../../../examples/optimization-library/triton/gemm_contract.py).


## What the gallery establishes

Neither listed case established a timing gain at either boundary. The measured
row-wise column rotation is a verified bijection, but that does not establish
better cache behavior. Concurrent reuse, working-set size and the target's
mapping still need evidence. Keep this recipe as a conditional locality
experiment, not an automatically enabled optimization.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| triton / benefit | 1436.3 ± 9.048 / 1436.53 ± 4.622 | 0.9998×; inconclusive | 1.50274 ± 0.2151 / 1.4825 ± 0.1472 | inconclusive |
| triton / cost | 3.85 ± 0.2598 / 4.222 ± 0.7394 | 0.9119×; inconclusive | 0.237184 ± 0.146 / 0.192532 ± 0.07155 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aic_mte2_ratio` | `aic_mac_ratio` | `cube_utilization(%)` |
| --- | ---: | ---: | ---: |
| triton/benefit/before/`gallery_gemm` | 0.9864 | 0.3145 | 98.692 |
| triton/benefit/after/`gallery_gemm` | 0.9853 | 0.315 | 98.504 |
| triton/cost/before/`gallery_gemm` | 0.421 | 0.0556 | 51.586 |
| triton/cost/after/`gallery_gemm` | 0.4145 | 0.0524 | 50.625 |

Observed resource/numerical costs and limits:
- **triton/benefit**: no additional resource/numerical cost identified by the retained fields.
- **triton/cost**: no additional resource/numerical cost identified by the retained fields.
