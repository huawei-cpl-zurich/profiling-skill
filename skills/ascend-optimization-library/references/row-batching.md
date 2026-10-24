# E01: Batch independent normalization rows

Combine short independent L2 rows in one program to amortize scheduling while retaining a
separate reduction and inverse norm for every row.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/norm_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: one row per program.
rows = tl.program_id(0) + tl.arange(0, 1)
# After: four independent rows per program; reduction remains across columns.
rows = tl.program_id(0) * 4 + tl.arange(0, 4)
x = tl.load(X + rows[:, None] * D + cols[None, :], mask, 0)
inverse = tl.rsqrt(tl.sum(x * x, 1) + eps)
tl.store(Y + rows[:, None] * D + cols[None, :], x * inverse[:, None], mask)
tl.store(S + rows, inverse, rows < R)
```

## Contract and recognition

Dense row-major FP32 ND input; FP32 arithmetic and outputs; positive epsilon. Return both
normalized rows and inverse norms. Mask both row tails and padded columns. This is L2
normalization, without a mean division or affine weight.

A one-row grid with short reductions and many small programs is the source cue. Compare
program count, compiled local resources and complete span at fixed column reduction
geometry. Four rows enlarge live tiles and can reduce available parallelism or exceed
resources for wide rows; compile and validate each admitted width. No width crossover
follows from these two captures.

## What the gallery establishes

The larger gallery improves about 5×; the small capture is inconclusive under the ledger
rule. A resource spill, unchanged work after compilation, or no repeatable span reduction
falsifies the scheduling hypothesis.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e01-norm-gallery-benefit | 468.482 ± 6.897 / 93.058 ± 2.083 | 5.034×; gain |
| e01-norm-gallery-cost | 3.284 ± 1.439 / 2.558 ± 0.2715 | 1.284×; inconclusive |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/l2_norm_fla.py:L7-L27),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/l2_norm_fla.py:L56-L86),
[pinned source 3](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/l2_norm_fla.py:L159-L196).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
