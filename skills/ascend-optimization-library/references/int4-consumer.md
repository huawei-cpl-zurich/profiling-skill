# E04: Decode packed INT4 in the GEMM consumer

Replace a separately materialized dequantized matrix with per-consumer-tile decoding,
while preserving the FP16 dot operand.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/int4_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before, separate complete dequantization kernel:
b = (((packed >> shift) & 15) - ((zeros >> shift) & 15)) * scale
store_full_fp16_matrix(b.to(tl.float16))
# GEMM loads that full matrix for each K stage.
# After, inside each GEMM K stage:
q = ((packed_tile.to(tl.uint32) >> shift) & 15).to(tl.int32)
z = ((zero_tile.to(tl.uint32) >> shift) & 15).to(tl.int32)
b = ((q - z).to(tl.float32) * scale_tile).to(tl.float16)
acc += tl.dot(a, b)
```

## Contract and recognition

Natural unsigned nibble encoding: eight consecutive N values per int32 word, low nibble
first; packed zero points use the same layout. This is not AWQ-compatible encoding. N must
be pack aligned, scales/zeros index the declared K group, and all linear-addressed tensors
require audited dense ND storage. Subtract in int32, multiply in FP32 and round to FP16
before the same FP32-accumulating dot in both arms.

Recognize a full dequantization output consumed only by GEMM. Fusion removes the explicit
2KN-byte workspace but repeats decode across M tiles and extends live operands. Inspect
lowered shift/mask/cast work and Cube-feed dependencies. Scalarized decode can overwhelm
the removed materialization; storage reduction is not evidence of lower physical traffic.

## What the gallery establishes

Both gallery captures regress (about 0.811× and 0.0188× before/after ratios). The retained
candidate illustrates costly scalarized decode, not a recommended fast path. Complete
dequant+GEMM span, unchanged rounding and compiled resources decide whether a different
decoder is worth applying.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e04-int4-gallery-benefit | 14016.5 ± 3.747 / 17283.4 ± 1.055 | 0.811×; regression |
| e04-int4-gallery-cost | 1058.37 ± 1.131 / 56231.5 ± 15.02 | 0.01882×; regression |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/awq_triton.py:L12-L105),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/awq_triton.py:L170-L208).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
