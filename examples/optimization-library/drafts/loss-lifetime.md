# E11: bound linear-loss activation lifetime

This is an independent restricted example inspired by Triton-Ascend's
`fused_linear_cross_enrtopy.py:49–69,116–187`, whose token chunking consumes
logits and gradients before moving to the next chunk. Stable source citations:

- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/fused_linear_cross_enrtopy.py:L49-L69`
- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/fused_linear_cross_enrtopy.py:L116-L187`

No A3 correctness, compilation, timing or device peak-memory result is claimed
by this CPU-qualified draft. The source implementation supports more options;
this example does not implement ignore labels, weights, bias, smoothing,
soft-capping, z-loss, autograd state or a configurable upstream gradient.

## Exact common contract

Inputs are dense FP16 X[T,D], W[V,D], and int32 targets in [0,V). Compute
FP32 logits X@W.T and sum the per-token cross entropy losses. Compute softmax
and subtract one-hot targets in FP32, explicitly round dlogits to FP16, then
use FP32-accumulating gradient GEMMs for complete dX and dW. Return the FP32
loss scalar, complete FP32 dX[T,D], and complete FP32 dW[V,D]. Both arms use
this same rounding graph, immutable inputs, tolerances and kernels. The
FP16-rounded dlogits are part of the declared operator, not permission to
silently lower a generic FP32 gradient contract.

Before stores T×V FP32 logits and FP16 dlogits. After stores C×V of each and
consumes each chunk immediately:

```python
for start in range(0, tokens, chunk):
    logits = gemm(X[start:start+chunk], W.T)
    loss_rows[start:start+chunk], dlogits = same_ce(logits, targets[start:start+chunk])
    dX[start:start+chunk] = gemm(dlogits, W)
    dW = gemm(dlogits.T, X[start:start+chunk]) + (dW if start else 0)
loss = reduce_all_loss_rows(loss_rows)
```

The real example implements every operation in Triton, including the final
loss reduction and dW accumulation. The first dW GEMM overwrites every output
element; subsequent GEMMs add in fixed chunk order. Repeated poisoned-buffer
interpreter tests exercise that invariant. No initialization or gradient work
is performed outside the timed invocation.

## Recognition, expected benefit and tradeoffs

Consider this transformation when a linear-loss pipeline materializes a large
T×V tensor which no caller needs after its loss/gradient consumption. Its
primary expectation is lower temporary allocation, not a guaranteed speedup.
Both arms use sufficient-statistics CE; this isolates lifetime from the
separate E10 transformation that removes full log probabilities.

Logical allocated workspace excludes common outputs and includes row losses:
`6*T*V + 4*T` bytes before versus `6*C*V + 4*T` after. This counts the actual
allocated logits, dlogits and row-loss buffers; it does not establish device
peak memory, allocator behavior, physical HBM traffic or cache hit rates.

The before timeline must include three GEMMs, CE, and final loss reduction.
The after timeline must include all four chunk operations for every chunk,
plus final loss reduction: 4*ceil(T/C)+1 kernel tasks per invocation. Smaller
chunks increase launch gaps, reread weights and accumulate dW repeatedly;
they may reduce Cube efficiency and be much slower. Large logical memory
savings remain meaningful even when the complete device-operation span loses.

Inspect all task durations and inter-kernel gaps, workspace allocation and
supported memory counters. Pipeline busy ratios alone cannot establish cache
locality, useful MAC occupancy or distance to roofline. Retain complete dX/dW
correctness evidence, tail chunks and immutable input checks. Profile once
correct: exactly ten forwards, no measured output/input inspection, warmups,
reference calls or hidden reset work. Four gallery/heldout hypotheses are frozen
in the orchestration artifacts before A3 execution; none is a measured claim.
