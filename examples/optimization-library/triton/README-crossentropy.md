# E10: cross-entropy sufficient statistics

This independent example returns per-row cross-entropy loss and the gradient
of the **sum** of those losses with respect to logits. Inputs are immutable,
contiguous FP32 logits and valid INT64 targets. Both outputs are contiguous
FP32. There are no class weights, ignored labels, smoothing, softcap or z-loss.
This is a deliberately restricted contract, not a drop-in implementation of
every option in the referenced operator.

The before pipeline writes full log-probabilities to a temporary matrix, then
reads that matrix to gather target loss and derive the complete gradient.
The after pipeline writes only the maximum and shifted logsumexp per row plus the loss;
its second kernel rereads logits and forms the gradient directly:

```python
# Both forms preserve stable FP32 max/sum reduction and exp arithmetic.
maximum = max(x)
shifted_lse = log(sum(exp(x - maximum)))
loss = (maximum - x[target]) + shifted_lse
dlogits = exp((x - maximum) - shifted_lse) - (column == target)
```

Both arms deliberately use two kernels with the same row grid and block size.
Keeping the maximum and shifted logsumexp separate avoids cancellation from
large common FP32 offsets such as `[1e8, 1e8]`. The before arm likewise writes
`(x - maximum) - shifted_lse`, preserving small log-probabilities.
This isolates removal of log-probability materialization rather than counting
another launch-fusion optimization. All output and temporary allocations occur
inside the callable. Every required scratch element is overwritten on every
forward. No initialization, target gather or gradient is omitted from timing.
The retained profiler boundary is still complete device-operation span, so it
does not independently measure allocator latency before the first device task.

The temporary's logical capacity changes from `4 * rows * vocabulary` bytes to
`8 * rows` bytes (two FP32 scalars per row). Width one therefore uses more
scratch after replacement, and width two uses the same amount. This is a source/storage fact, not a measured HBM traffic
claim. Savings may matter for large vocabularies or live-memory pressure.
The after arm pays two scalar-statistic stores/loads and a second logit read;
small vocabularies may provide little advantage, and inter-kernel launch gaps
can dominate either arm. If a consumer also requires full log-probabilities,
the reduced-output contract is inapplicable. Returning fewer required outputs
would not be an optimization.

Recognize the opportunity by finding an intermediate log-probability tensor
whose only consumers are target loss and logits gradient. In the profiler,
retain both complete two-kernel sequences, ten operation spans, caller-event
timings separately, matched PMU rows and compiled identities. Compare actual
before/after binaries because both variants have the same source kernel names.
Pipeline activity ratios alone cannot establish bytes saved, useful bandwidth,
or distance to roofline. Large-vector resource limits require actual A3 compile
qualification; CPU interpreter success does not establish UB fit or speed.

The source inspiration explicitly computes LSE and target loss, derives
gradients in a second pass, and supports many optional features. It overwrites
the logits buffer; this example independently implements immutable inputs and
an explicit gradient output. Source semantics are documented by these pinned
filtered citations:

- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/cross_entropy.py:L95-L142`
- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/cross_entropy.py:L157-L206`
- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/cross_entropy.py:L210-L249`

CPU functional/interpreter qualification and four-case A3 correctness/capture
qualification passed for both retained source versions. Original `e39d04b`
used an implicitly INT64 comparison; `7649e59` applies the same exact local
index narrowing in both arms. The public labels remain INT64:

```python
# Only when the contract guarantees 0 <= target < vocabulary <= 8192.
target = tl.load(Targets + row).to(tl.int32)
gradient = tl.exp(log_probability) - (column == target).to(tl.float32)
```

The original large-case gradient was scalar-heavy (AIV scalar ratio0.976).
Its intermediate compiler artifacts extend the column mask to INT64. After
the common repair, actual compiled comparisons are INT32 and the gallery
before/after gradient task durations fall1720.660→20.986µs and
1719.602→21.220µs. This supports a compiler-sensitive integer-mask bottleneck;
it is not an independent final-instruction disassembly proof. The initial
implementation missed the pinned mask-INT32 constraint, and its complete
negative source/profile evidence remains retained.

That common repair must not be confused with a materialization speed win.
Original before/after ratios for the two benefit cases were0.99804×/0.99791×;
repaired ratios were1.02183×/0.98143×. All-case geometric means were0.98171×
and0.97074× respectively. Large-case active work rises after materialization
removal, and small-case spans are dominated by gaps and variability. No robust
speed improvement is established for this particular two-stage realization.
Logical scratch capacity is lower; physical traffic and peak-memory savings
were not measured.

Both versions together passed16 correctness checks and16captures/160profiled
forwards using the same4cached CPU oracles (zero regenerated for the repair).
Every captured forward contains exactly the two declared kernels. Original
and repaired evidence retain separate source hashes, all ten timing samples,
matched PMU rows and16verified NPUBIN snapshots per version. New source/data
remain quarantined from the first frozen-library coding study.


A third source version repairs large-common-offset cancellation in both arms.
Versions one and two combined the maximum with the logarithmic normalization
before subtracting logits, which lost small losses and distorted gradients at
offsets such as `1e8`. Their measurements remain retained as historical
evidence; their timing results do not qualify this revised source. The public
contract and INT64 labels remain unchanged, with bounded local INT32 indices.
CPU interpreter regressions cover positive/negative offsets, tied logits, and
unequal representable logits against independent FP64 expectations.
