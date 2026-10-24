# E11: Bound logits lifetime across linear loss and backward

Process bounded token chunks through linear projection, cross-entropy and both VJPs so
logits workspace can be reused.

## Paired code format

The structural excerpt uses schematic helpers. The complete
[paired adapter](../../../examples/optimization-library/triton/loss_lifetime_pairs.py)
contains the executable kernels, masks and full-operation integration. It is
maintained in a separate source PR; do not run this excerpt standalone.

```python
# Before: one full-token chunk, retaining full logits and dlogits.
rows = T
# After: bounded row workspace reused, with complete work inside the call.
for start in range(0, T, chunk_rows):
    logits = linear(x[start:start + chunk_rows], w)
    loss_rows, dlogits = cross_entropy(logits, target[start:])
    dx[start:start + chunk_rows] = dlogits @ w
    dw_chunk = dlogits.T @ x[start:start + chunk_rows]
    dw = dw_chunk if start == 0 else dw + dw_chunk
loss = sum(loss_rows_for_all_tokens)
```

## Contract and recognition

Summed scalar loss plus complete dX[T,D] and dW[V,D]. Dense ND FP16 X/W; FP32
logits/softmax; FP16-rounded dlogits before FP32-accumulating GEMMs. Targets must be valid
vocabulary indices. Preserve all final chunks, initialize dW on the first chunk and
accumulate later chunks. No ignored labels, smoothing or other loss modes are implemented;
changed dW reduction order needs oracle validation.

Recognize full T×V logits and gradient retained through backward. Explicit logical
workspace is 6*rows*V+4*T bytes, excluding common outputs. Both arms preallocate it before
timing: this does not measure allocator savings, physical peak memory or traffic. More
chunks launch more GEMMs/CE calls, reduce GEMM efficiency and repeatedly accumulate dW.

## What the gallery establishes

Both captures are much slower (about 0.124× and 0.240× ratios). Logical memory reduction
is the supported advantage; this candidate supplies no latency win. Count every
projection, CE, dX/dW GEMM, accumulation and final loss reduction. If the memory budget
already fits, launch/matrix costs can make chunking unsuitable.

All listed arms passed the retained paired correctness and ten-forward identity
collection. Timings are complete **profiled device-operation spans**, including
inter-kernel gaps, not unprofiled application latency. Preallocation excludes
allocation time. Ratios are before/after; hypothesis names are planned regimes,
not promised results. See [expansion evidence](expansion-gallery-evidence.md)
for sample/uncertainty boundaries and retained receipt references.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e11-gallery-memory | 1493.02 ± 32.79 / 12011.8 ± 106.1 | 0.1243×; regression |
| e11-gallery-small | 690.194 ± 75.47 / 2873.49 ± 88.54 | 0.2402×; regression |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/fused_linear_cross_enrtopy.py:L49-L69),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/fused_linear_cross_enrtopy.py:L116-L187).
The gallery is an independent implementation, not upstream execution evidence.
Physical traffic and roofline distance are **not estimated** from activity fields.
