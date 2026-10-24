# E09: Reuse rotary coefficients across Q and K heads

Compute token/frequency sin and cos once per program and reuse them across Q/K heads.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/rotary_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: separate head programs each compute the same coefficients.
cos = tl.cos(position / scale * frequency).to(tl.float16).to(tl.float32)
sin = tl.sin(position / scale * frequency).to(tl.float16).to(tl.float32)
rotate_one_head(cos, sin)
# After: one token/frequency tile keeps coefficients live across heads.
cos, sin = same_rounded_coefficients(position, scale, frequency)
for head in range(HQ):
    rotate_q_head(head, cos, sin)
for head in range(HK):
    rotate_k_head(head, cos, sin)
```

## Contract and recognition

Complete split-half Q/K rotation with even D, dense contiguous FP16 ND
Q[B,T,HQ,D]/K[B,T,HK,D], int32 positions and FP32 frequencies[D/2]. Same position scale
and FP16-rounded coefficients, FP32 products and FP16 output in both arms. Preserve both Q
and K outputs and input immutability. This is not an interleaved-pair rotary layout.

Recognize identical position/frequency transcendental work in every head program. The
candidate keeps token/frequency tiles fixed but removes head parallelism through loops,
extending coefficient live ranges. Few tokens can leave too few programs; many heads
increase serial work. Compare emitted coefficient work, resources and complete span rather
than treating reuse count as speedup.

## What the gallery establishes

The many-head gallery improves about 4.81×; the single-token capture is inconclusive. A
reduced program count can offset fewer transcendental operations. Changed coefficient
rounding/layout or worse elapsed span falsifies the proposed reuse path.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e09-gallery-many-heads | 837.372 ± 1.244 / 173.932 ± 0.9173 | 4.814×; gain |
| e09-gallery-single-token | 6.08 ± 0.1147 / 5.614 ± 0.9691 | 1.083×; inconclusive |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/fused_rotary_emb.py:L51-L74).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
