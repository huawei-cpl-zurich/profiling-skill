# E07: Reuse the forward inverse RMS in input backward

Save the forward inverse RMS to remove its recomputation in the matching input-gradient pass.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/norm_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: forward returns y; backward recomputes the statistic.
r = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
# After: forward also writes its FP32 r; backward loads it.
tl.store(S + row, r)   # Forward.
r = tl.load(S + row)  # Backward.
# Both backward arms:
p = tl.sum(dy * x, 0) / D
dx = r * (dy - x * r * r * p)
```

## Contract and recognition

This pair is unweighted RMS forward plus input VJP only: no affine weight and no dW. Dense
row-major FP32 ND X and dY; same positive epsilon, row width and FP32 reduction graph.
Save the statistic from exactly the matching immutable X. Return Y and dX, retaining the
extra row statistic as internal workspace.

Recognize an inverse RMS recomputed from X in backward immediately after forward. The
statistic costs R×4 bytes of explicit workspace plus a forward store and backward load. X
must still be loaded for projection and dX; saving r does not eliminate the backward X
read. Compare both forward and backward, not backward alone.

## What the gallery establishes

The larger capture has a modest span gain; the other regresses. This is a conditional
recomputation-versus-state tradeoff. A changed epsilon/cast, stale saved state, or
whole-operation regression falsifies applying this candidate. Weighted dX/dW requires a
separate contract, as in E08.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e07-norm-gallery-benefit | 211.202 ± 8.114 / 193.986 ± 12.01 | 1.089×; gain |
| e07-norm-gallery-cost | 153.382 ± 7.726 / 169.192 ± 6.05 | 0.9066×; regression |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L82-L94),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L145-L155),
[pinned source 3](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L440-L488).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
