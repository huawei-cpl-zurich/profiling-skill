# E02: Precompute ragged prefix offsets

Replace repeated prefix walks in ragged consumers with one exclusive scan and cached
per-sequence offsets.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/ragged_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: repeated in every time/feature consumer tile.
start = 0
for previous in range(batch):
    start += tl.load(Lengths + previous)
# After: one prefix kernel, included in the operation, then each consumer:
count = tl.load(Lengths + lane, lane < B, 0)
exclusive = tl.cumsum(count, 0) - count
tl.store(Starts + lane, exclusive, lane < B)
# Consumer kernel:
start = tl.load(Starts + batch)
```

## Contract and recognition

Immutable nonnegative int32 lengths, valid packed input extent and offsets representable
in int32. Dense packed rows map to dense padded output; each length must fit the output
length. Preserve padding values and time/feature tail masks. The scan and consumers must
use the same lengths on an ordered stream.

Recognize a loop over all earlier sequences inside each consumer tile. Reuse grows with
time/feature tiles; the extra scan launch, dependency and B×4-byte explicit workspace can
dominate. Scan width is padded to a power of two and needs its own resource guard. Count
the scan on every call unless reuse across calls is part of the public contract.

## What the gallery establishes

Both gallery cases are slower, including the planned benefit case. The short case has
large inter-kernel-gap variability, retained in full. Fewer prefix loads alone do not
establish a useful optimization; complete span after including scan overhead is the
falsifier.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e02-triton-gallery-benefit | 260.778 ± 3.511 / 345.462 ± 18.78 | 0.7549×; regression |
| e02-triton-gallery-cost | 3.594 ± 0.897 / 137.662 ± 68.93 | 0.02611×; regression |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/pack_unpack.py:L24-L30),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/pack_unpack.py:L137-L145),
[pinned source 3](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/fla/cumsum.py:L65-L101).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
