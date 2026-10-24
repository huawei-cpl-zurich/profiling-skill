# E11: bound linear-loss activation lifetime

This is an independent restricted example inspired by Triton-Ascend's
`fused_linear_cross_enrtopy.py:49–69,116–187`, whose token chunking consumes
logits and gradients before moving to the next chunk. Stable source citations:

- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/fused_linear_cross_enrtopy.py:L49-L69`
- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/fused_linear_cross_enrtopy.py:L116-L187`

A3 correctness, compilation and captured spans are qualified for the four frozen
cases below. No device peak-memory result is claimed. The source implementation supports more options;
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
reference calls or hidden reset work. Four gallery/heldout hypotheses were frozen
in the orchestration artifacts before A3 execution; results retain the tradeoffs.

## Four-case A3 qualification

Frozen source `82e9d4` passed eight complete loss/dX/dW checks, eight captures and
80 forwards on a healthy isolated A3 target, with four cached CPU oracles.
No warmups, unprofiled forwards or runtime/compile/correctness failures occurred.
Outputs/workspaces are **preallocated in prepare**; spans cover every required
device operation and inter-kernel gap, not host allocations or application latency.

| Case (T,D,V,C) | Before span µs ± SD | After span µs ± SD | Speedup | Logical workspace bytes before→after |
| --- | ---: | ---: | ---: | ---: |
| (1024,128,4096,64) | 1493.024 ± 32.790 | 12011.816 ± 106.059 | 0.124296× | 25169920→1576960 |
| (16,64,128,4) | 690.194 ± 75.472 | 2873.488 ± 88.544 | 0.240194× | 12352→3136 |
| (769,96,2049,64) | 697.324 ± 33.305 | 9466.024 ± 106.922 | 0.073666× | 9457162→789892 |
| (9,32,257,2) | 685.826 ± 20.595 | 3694.678 ± 83.368 | 0.185625× | 13914→3120 |

Complete-span geomean 0.142145× (7.035× slower). Logical temporary storage is
smaller in every case, but physical peak memory/cache/capacity benefit remains
unmeasured. Gallery memory case: before 5 tasks→after 65 per forward; active union
1492.930→4564.950µs and internal gaps 0.094→7446.866µs. Both shorter GEMMs
and extra dispatches have costs; do not promote this entry as a speed win.

All 1,760 device tasks join same-capture PMU rows. 42 actual NPUBIN snapshots were
independently SHA-verified, all compilation identities stable across capture.
For gallery logits GEMM, exported MAC ratio falls 0.1054→0.027525, while exported
MTE2 ratio is 0.9819→0.991994. Exported cube_utilization rises 87.149→97.297%;
this does not imply more useful MAC work or greater distance to/from roofline.
Task-local ratios have different task counts/denominators and are diagnostic.

Exact target configuration, commands and retained job handles belong in the
private orchestration task record. Compact archive SHA256 `ab463b48a44771a53fae8ee5c7e831e1975d8c7c5f5047eecd9c9e7d1669348a`.
Raw ten samples, SD, active/gap/caller distributions, matched PMU rows, original
result hashes and binary snapshots are retained in orchestration evidence.
