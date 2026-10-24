# E10: cross-entropy sufficient statistics

This independent example returns per-row cross-entropy loss and the gradient
of the **sum** of those losses with respect to logits. Inputs are immutable,
contiguous FP32 logits and valid INT64 targets. Both outputs are contiguous
FP32. There are no class weights, ignored labels, smoothing, softcap or z-loss.
This is a deliberately restricted contract, not a drop-in implementation of
every option in the referenced operator.

The before pipeline writes full log-probabilities to a temporary matrix, then
reads that matrix to gather target loss and derive the complete gradient.
The after pipeline writes only one logsumexp scalar per row plus the loss;
its second kernel rereads logits and forms the gradient directly:

```python
# Both forms preserve stable FP32 max/sum reduction and exp arithmetic.
lse = max(x) + log(sum(exp(x - max(x))))
loss = lse - x[target]
dlogits = exp(x - lse) - (column == target)
```

Both arms deliberately use two kernels with the same row grid and block size.
This isolates removal of log-probability materialization rather than counting
another launch-fusion optimization. All output and temporary allocations occur
inside the callable. Every required scratch element is overwritten on every
forward. No initialization, target gather or gradient is omitted from timing.
The retained profiler boundary is still complete device-operation span, so it
does not independently measure allocator latency before the first device task.

The temporary's logical capacity changes from `4 * rows * vocabulary` bytes to
`4 * rows` bytes. This is a source/storage fact, not a measured HBM traffic
claim. Savings may matter for large vocabularies or live-memory pressure.
The after arm pays a scalar-statistic store/load and a second logit read;
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
qualification passed at source revision `e39d04b`. Neither large-vocabulary
benefit hypothesis established a speed win: before/after ratios were0.99804×
and0.99791×. All-case geometric mean was0.98171×; small-case means worsened
but variability and inter-kernel gaps preclude a strong regression claim.

The measured gradient tasks were scalar-heavy (about0.976 AIV scalar ratio).
Retained intermediate compiler artifacts show the column mask promoted to
INT64 by comparison with INT64 targets. This is a likely bottleneck requiring
further compiler evidence, not proof that materialization removal saves time.
The author's initial implementation missed the pinned mask-INT32 constraint;
a common, contract-safe target cast in both arms is a possible future repair.
The original negative version and all attempts remain retained. Scratch
capacity savings are source facts, not measured physical traffic or peak memory.
