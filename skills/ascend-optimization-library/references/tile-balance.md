# Balance output tasks against tile movement costs

Status: all four listed Triton/native gallery pairs passed A3 correctness and
ten-forward profile/identity collection. Use family `05`
in `examples/optimization-library/triton/gemm.py` with adapter `gemm:Adapter`.

The [pinned layout-sensitive tile example](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-v130-design-550bbffa8b36#docs/contents/advanced/catlass_optimize_guidance.md:L114-L150)
shows why fewer output tasks can help while a different B layout can reverse
the result. Its historical target count is not the current target's core count.

Recognize output dimensions whose ceiling-divided tile grid narrowly exceeds
the observed Cube core count. Compare these configurations while holding dtype,
K blocking, arithmetic, mapping and matrix values fixed:

```python
# Before
BM, BN, BK = 128, 256, 32
# After
BM, BN, BK = 256, 128, 32
tasks = ceildiv(M, BM) * ceildiv(N, BN)
```

The same accumulator element count does not imply equal operand movement or
compiled resource use. A taller A tile and narrower B tile change transfers,
live operand sizes and alignment behavior. Compare both supported physical B
layouts: row-major strides [N,1] and column-major strides [1,K]. The adapter
preserves and records those strides; it does not silently transpose in the call.

Generate gallery geometry from actual C: choose positive a and odd b∈[3,31]
with a*b≤C<a*(b+1), maximize a*b and then minimize a*(b+1). Set M=256a,
N=128b, K=6144. At C=24, M=2048,N=384 gives 32 before tasks and 24 after.
This is task-count arithmetic, not an elapsed-time prediction. Do not infer
that one logical program always occupies one physical core for one wave.

Both arms use FP16 input/output and the same FP32 CPU accumulation followed
by FP16 conversion, rtol=atol=1e-3. Mask M/N/K tails independently of strides.
Compile resource evidence must cover both 128×256 and 256×128 tiles; a rejected
resource configuration is not a correct adverse performance example.

Inspect complete-call duration, observed kernel identity, actual task geometry,
and supported compiler resource or traffic evidence. Low global utilization
alone cannot distinguish too few tasks from memory stalls. The diagnosis is
falsified when the tile with fewer tasks is not faster, or layout movement
overwhelms task balance. Record these outcomes without assigning a saturated
pipe or roofline distance from activity percentages alone.

Treat each physical layout separately when interpreting the paired data below.

## Native format and physical-layout guard

The [native source](../../../examples/optimization-library/catlass/tile_matmul.cpp)
changes `GemmShape<128,256,256>` to `GemmShape<256,128,256>`, holding its K tile
fixed at 256. The Triton format above keeps K blocking at 32. Do not present
cross-format results as a control of K blocking.

Select `tile_adapter:adapter`. The native adapter checks the compiled B-layout
getter before accepting row-major `[N,1]` or column-major `[1,K]` strides. The
column path uses a dense ND transpose view prepared outside the call. The
bridge audits storage span and format; a row binary cannot be reused for a
column pointer merely because its logical shape matches.

Runnable [Triton source](../../../examples/optimization-library/triton/gemm.py),
`gemm:Adapter`, family `05`, shares the
[CPU contract](../../../examples/optimization-library/triton/gemm_contract.py)
with the native adapter.


## What the gallery establishes

The taller/narrower tile improved device span in both tested B layouts and in
both languages. The alternate layout did not reverse that gain. Caller-event
comparisons were inconclusive. The results support this target-derived geometry,
not a rule that fewer tasks always wins; another layout/shape can change operand
movement and capacity even at the same accumulator area.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| triton / benefit | 152.131 ± 0.2516 / 122.294 ± 1.23 | 1.244×; gain | 0.327778 ± 0.1776 / 0.28848 ± 0.1108 | inconclusive |
| catlass / benefit | 66.0186 ± 0.9286 / 35.9733 ± 1.551 | 1.835×; gain | 0.13734 ± 0.0787 / 0.127346 ± 0.05163 | inconclusive |
| triton / cost | 274.912 ± 2.292 / 153.068 ± 1.746 | 1.796×; gain | 0.332506 ± 0.1254 / 0.282858 ± 0.07292 | inconclusive |
| catlass / cost | 64.4647 ± 1.425 / 36.0293 ± 1.882 | 1.789×; gain | 0.199076 ± 0.1978 / 0.121868 ± 0.04899 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `cube_utilization(%)` | `aic_mac_ratio` | `aic_mte2_ratio` |
| --- | ---: | ---: | ---: |
| triton/benefit/before/`gallery_gemm` | 59.007 | 0.4283 | 0.9711 |
| triton/benefit/after/`gallery_gemm` | 93.527 | 0.3246 | 0.9805 |
| catlass/benefit/before/`wave1_tile_before` | 54.114 | 0.8535 | 0.8653 |
| catlass/benefit/after/`wave1_tile_after` | 95.932 | 0.8743 | 0.8546 |
| triton/cost/before/`gallery_gemm` | 57.59 | 0.2427 | 0.9851 |
| triton/cost/after/`gallery_gemm` | 94.414 | 0.2569 | 0.9836 |
| catlass/cost/before/`wave1_tile_before` | 54.31 | 0.8685 | 0.7524 |
| catlass/cost/after/`wave1_tile_after` | 95.68 | 0.8783 | 0.7243 |

Observed resource/numerical costs and limits:
- **triton/benefit**: no additional resource/numerical cost identified by the retained fields.
- **catlass/benefit**: no additional resource/numerical cost identified by the retained fields.
- **triton/cost**: cost recipe did not produce a device-span regression in this capture.
- **catlass/cost**: cost recipe did not produce a device-span regression in this capture.
