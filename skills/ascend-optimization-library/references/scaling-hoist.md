# Move invariant scaling across a dot product

Status: the two listed Triton gallery pairs passed A3 correctness and ten-forward
profile/identity collection. Their measurements below describe those cases only.

Consider a scalar coefficient constant across a dot's reduction dimension:
`dot(beta[row] * K[row, :], K.T)`. Distributing beta over the accumulated
output row can remove repeated input-side multiplications when the reduction
dimension exceeds the output width. The mathematical identity does not imply
identical floating-point results.

The paired [Triton example](../../../examples/optimization-library/triton/scaling_hoist.py)
uses BF16 K, FP32 beta and output, with FP32 accumulation. Before:

```python
product = tl.zeros((BT, BT), tl.float32)
for tile in reduction_tiles:
    k, rhs = load_k_pair(tile)
    scaled = (k.to(tl.float32) * beta[:, None]).to(k.dtype)
    product = tl.dot(scaled, rhs, product)
```

After:

```python
product = tl.zeros((BT, BT), tl.float32)
for tile in reduction_tiles:
    k, rhs = load_k_pair(tile)
    product = tl.dot(k, rhs, product)
product = product * beta[:, None]  # Once, after the entire K reduction.
```

These snippets show the loop structure; the runnable source expands the K/RHS
loads and masks. Both preserve a strict lower-triangular mask. Inputs are contiguous
`[groups,BT,DK]` K and `[groups,BT]` beta; this example accepts BT 32/64,
DK 16/32/128/256, 1–1024 groups, finite row-normalized K and beta in [0,1].
It implements no fallback for another domain. The independent CPU oracle
accumulates the original BF16 values in FP32, then applies row beta and the
mask. Both arms must meet rtol=atol=0.01, including near-opposite rows,
beta zero/one and very small beta. Record numerical margin; never relax the
contract to retain the candidate.

The example loads the transposed RHS directly from GM through a separate
read-only kernel formal, `KRight`; the adapter passes the same K pointer twice.
There is no input clone or additional mathematical operand. Both arms use
this convention and K tiles of at most 64 columns. In the tested Ascend
compiler, sharing a loaded K tile between vector scaling and `tl.trans(k)`
produced invalid before results. The retained post-sync-solver IR consumed an
uninitialized Cube L1 RHS buffer. A direct transposed load from the same formal
was canonicalized back to the failing dataflow. A separate formal restored
the missing GM-to-L1 load in compile-only inspection. Both revised gallery pairs
then passed their unchanged correctness contract and ten-forward captures. Keep this compiler constraint separate
from the mathematical optimization's benefit.

Recognize a coefficient loaded outside the reduction that is nevertheless
multiplied into every operand element. Confirm it is independent of the
reduction index and that no nonlinear operation or required conversion blocks
the transformation. In this example, moving the multiplication also removes
an intermediate BF16 rounding. Overflow/underflow and cancellation can differ;
quantization, saturation and coefficients varying with K invalidate a simple
move unless a new numerical proof is supplied.

Gallery regimes: BT=64/DK=256 reduces multiplications from 16,384 to 4,096
per block; BT=64/DK=32 increases them from 2,048 to 4,096. Masking and compiler
elimination can change executed work, so those counts are source arithmetic,
not measured instruction counts. Every case uses 64 independent blocks. Check each arm once,
then capture ten complete forwards without tensor inspection or warmups.

Look for changes in whole-call duration, emitted casts/vector operations and
supported vector/Cube metrics while keeping dot geometry and outputs fixed.
A small vector-active percentage does not prove the vector work is irrelevant;
dependency placement may affect the critical path. Pipe activity alone cannot
establish overlap, bandwidth, utilization against a hardware peak or distance
to roofline. A runtime effect within sample variability is inconclusive.
Unchanged emitted work or worse elapsed time falsifies the predicted benefit.

Source mechanism: [pinned KKT implementation](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/fla/chunk_scaled_dot_kkt.py:L82-L94)
and its `docs/sources/best-practices/chunk_scaled_dot_kkt.md:L75-L105`, revision
`7acb9c40a6ca827e885d902b7eaee5d45457b230`. The gallery is an independent
implementation of that mechanism; upstream source is not our execution proof.


## What the gallery establishes

Both tested reduction sizes improved device span, including DK smaller than BT.
Thus BT*DK versus BT² is not a measured crossover rule: cast placement and the
compiler's memory/dataflow costs matter. Moving the scale after the dot also
reduced the measured absolute error against this oracle. Caller-event effects
were inconclusive; these captures establish a device-work improvement, not an
application-latency improvement.

## Measured gallery

All ratios are before/after. See the [measurement ledger](gallery-evidence.md) for boundary and uncertainty definitions; raw ten-sample series are in [gallery-evidence.json](gallery-evidence.json).

| Format / planned regime | Device span before / after (µs ± SD) | Span ratio; observation | Caller events before / after (ms ± SD) | Caller observation |
| --- | --- | --- | --- | --- |
| triton / benefit | 26.396 ± 0.4543 / 11.274 ± 0.377 | 2.341×; gain | 0.224786 ± 0.07228 / 0.23505 ± 0.1124 | inconclusive |
| triton / cost | 15.942 ± 0.6692 / 11.528 ± 0.4507 | 1.383×; gain | 0.320852 ± 0.3241 / 0.228294 ± 0.05785 | inconclusive |

Selected raw-counter means below retain their exported names. Ratios are per-kernel activity fields; they are not additive or fractions of theoretical peak. Full columns, sample deviations and unavailable values remain in the JSON.

| Format / regime / arm / kernel | `aiv_vec_ratio` | `aic_mac_ratio` | `aiv_mte3_ratio` |
| --- | ---: | ---: | ---: |
| triton/benefit/before/`wave001_kkt_scale_before` | 0.0976 | 0.0391 | 0.1277 |
| triton/benefit/after/`wave001_kkt_scale_after` | 0.096 | 0.097 | 0.1102 |
| triton/cost/before/`wave001_kkt_scale_before` | 0.0704 | 0.014 | 0.1123 |
| triton/cost/after/`wave001_kkt_scale_after` | 0.0997 | 0.0207 | 0.1111 |

Observed resource/numerical costs and limits:
- **triton/benefit**: no additional resource/numerical cost identified by the retained fields.
- **triton/cost**: cost recipe did not produce a device-span regression in this capture.
