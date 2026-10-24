# E08: Accumulate private weight-gradient partials

Replace contended per-row atomic weight-gradient updates with private program partials and
a final reduction.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/gradient_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: zero DW in a timed kernel, then one program per row.
tl.atomic_add(DW + col, dy * (x * inverse), mask, sem='relaxed')
# After: each of P programs visits rows pid, pid+P, ...
partial = tl.zeros((BD,), tl.float32)
for row in assigned_rows:
    store_dx_for_row(row)
    partial += dy * (x * inverse)
tl.store(Partial + pid * D + col, partial, col < D)
# Final timed kernel, including zero partials from programs with no rows:
tl.store(DW + col, tl.sum(all_partials, 0), col < D)
```

## Contract and recognition

Complete weighted RMS VJP: return both dX and dW for dense FP32 ND X/dY[R,D] and W[D].
Compute m=dY*W and dX=r*(m-X*r*r*mean(m*X)); dW sums dY*(X*r). Same epsilon and FP32
arithmetic; changed sum order must pass the unchanged oracle tolerance. Every output and
partial must be written each call, including zero partials for idle programs.

Recognize repeated atomics to the same D weight-gradient elements. The gallery uses P=32
private accumulators, 4PD-byte explicit workspace, strided row assignment and final
reduction. This also changes grid/row scheduling; it is not an isolated atomic-instruction
replacement. Partial storage, serial per-program rows and the reduction can dominate at
low contention.

## What the gallery establishes

Both captures improve (about 4.63× and 1.15×) including baseline zeroing and candidate
final sum. Check dX as well as dW, atomic contention evidence where supported, resources
and complete span. Poor parallelism or reduction cost exceeding avoided contention
falsifies benefit.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e08-gradient-gallery-benefit | 698.486 ± 95.6 / 150.86 ± 18.2 | 4.63×; gain |
| e08-gradient-gallery-cost | 167.918 ± 11.86 / 145.802 ± 6.85 | 1.152×; gain |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L135-L191),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L295-L359),
[pinned source 3](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L440-L488).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
