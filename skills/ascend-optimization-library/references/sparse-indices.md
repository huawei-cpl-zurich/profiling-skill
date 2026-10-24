# E05: Visit selected attention blocks directly

Use an immutable supplied selection list to avoid dense masked key-block visits in attention.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/sparse_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: visit every key block, masking blocks absent from selected.
for block in range(number_of_key_blocks):
    active = any_selected_id_equals(block)
    state = online_update(state, block, active)
# After: skip holes without terminating the list.
for slot in range(SLOTS):
    block = tl.load(Selected + slot)
    if (block >= 0) & (block < number_of_key_blocks):
        state = online_update(state, block, True)
```

## Contract and recognition

Unique valid block IDs, arbitrary order and arbitrary -1 holes; a hole does not end the
list. Duplicates would double count on the direct path. Preserve grouped-query mapping
head//(HQ//HK), with divisible head counts, query/key tails, scale and stable online
softmax. Empty selections produce zero output with neutral all-masked state. FP16
operands/probabilities and FP32 online accumulators retain the paired numerical graph;
reordered reductions need oracle checks.

Recognize dense block traversal plus membership tests despite a supplied sparse list.
Compare visited blocks and complete attention span at identical query/key tile geometry.
Selection production is outside this pair because both arms receive the same immutable
list; include it if the requested operation produces selections. Long lists, dynamic
branches or poor locality can erase the benefit.

## What the gallery establishes

Both captures improve (about 108× and 1.91×). These compare direct visits against an
intentionally dense masked traversal, not an arbitrary tuned dense implementation. Invalid
selection semantics, changed softmax state, or list overhead exceeding skipped work
falsifies transfer to another call.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| E05-triton-gallery-benefit | 17060.7 ± 9.012 / 157.978 ± 3.237 | 108×; gain |
| E05-triton-gallery-cost | 867.072 ± 5.247 / 453.372 ± 2.276 | 1.912×; gain |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py:L85-L93),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py:L131-L166).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
