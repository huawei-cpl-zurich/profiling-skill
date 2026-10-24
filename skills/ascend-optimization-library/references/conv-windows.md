# E06: Parallelize immutable causal convolution windows

Compute each causal depthwise convolution token from immutable input/history windows
instead of advancing mutable state through sequential launches.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/conv_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: timed state initialization, then one ordered launch per token.
state[:] = initial
for token in range(T):
    output[token], state = one_token_update(x[token], weight, state)
# After: independent programs index the immutable conceptual stream.
# stream = concatenate(initial_history, x), independently per batch/channel.
output[token] = sum(stream[token + lag] * weight[lag] for lag in range(K))
# One unique writer per final-history element:
final_history[lag] = stream[T + lag]
```

## Contract and recognition

Dense contiguous FP16 ND X[B,T,D], weights[D,K] and history[B,D,K-1]; FP32
products/reduction, FP16 outputs and final state. Preserve nonzero history, all token
outputs and final history when T is shorter than K-1. No bias, activation, accepted-token
indices or padding-slot semantics are implemented. Input history remains immutable.

Recognize a Python token loop launching dependent state updates. The candidate couples
fewer launches and state writes with token parallelism; it rereads overlapping X/history
windows and weights. It does not load every input/weight once. Larger K increases
redundant reads and live tap tiles, so launch reduction alone is insufficient.

## What the gallery establishes

The supplied multi-token and single-token captures improve about 11.5× and 12.9×. Even the
single-token baseline includes state initialization; this is not proof that parallel
tokenization alone gives that ratio. Include initialization, all outputs and final state;
worse complete span or excessive window rereads falsifies the benefit.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e06-gallery-multi-token | 4599.92 ± 73.44 / 399.274 ± 6.799 | 11.52×; gain |
| e06-gallery-single-token | 171.968 ± 12.38 / 13.314 ± 0.8944 | 12.92×; gain |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/conv/causal_conv1d_update.py:L53-L120),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/conv/causal_conv1d_update.py:L125-L164).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
