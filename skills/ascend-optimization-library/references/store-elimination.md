# E03: Write ragged padding and values once

Replace a full padding store followed by valid-value overwrites with a selected value and
one bounded store.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/ragged_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before:
tl.store(destination, padding, bounds)
tl.store(destination, value, valid)
# After: value was loaded with valid mask and other=0.
result = tl.where(time[:, None] < count, value, padding)
tl.store(destination, result, bounds)
```

## Contract and recognition

Every in-bounds output element must receive either its input value or the exact padding
value. Input loads remain masked by valid length and physical bounds. Preserve partial
time/feature tiles, empty sequences and output extent. Do not remove an initialization
required by a later read or another writer.

Recognize two stores to the same output tile, with the second mask a subset of the first.
Inspect emitted stores and supported store activity rather than assuming source count
equals transaction count. Selection adds vector work; the compiler may already eliminate
overwritten stores, and launch-dominated calls may not benefit.

## What the gallery establishes

One gallery span improves about 1.50×; the short case is inconclusive. Unchanged emitted
stores, changed padding, or no complete-span gain falsifies the expected benefit. No
physical-byte saving is measured here.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e03-triton-gallery-benefit | 21.216 ± 0.3505 / 14.168 ± 0.2912 | 1.497×; gain |
| e03-triton-gallery-cost | 1.518 ± 0.07857 / 1.508 ± 0.027 | 1.007×; inconclusive |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/pack_unpack.py:L32-L51).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
