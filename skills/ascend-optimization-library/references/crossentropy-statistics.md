# E10: Retain cross-entropy sufficient statistics

Replace full log-probability scratch with per-row maximum and shifted logsumexp
when only loss and the full logits gradient are required.

## Paired code format

This structural excerpt shows the stable arithmetic. The complete
[paired adapter](../../../examples/optimization-library/triton/crossentropy_pairs.py)
contains both kernels and integration; it is maintained in a separate source PR.

```python
maximum = tl.max(x, 0)
shifted_lse = tl.log(tl.sum(tl.exp(x - maximum), 0))
# Before: materialize every log-probability for a second kernel.
logp = (x - maximum) - shifted_lse
store_full_logp(logp)
# Second kernel: loss=-logp[target], dx=exp(logp)-one_hot(target).
# After: first kernel stores maximum and shifted_lse separately.
store_two_row_statistics(maximum, shifted_lse)
loss = (maximum - x_target) + shifted_lse
# Second kernel rereads immutable logits and the two statistics.
logp = (x - saved_maximum) - saved_shifted_lse
dx = tl.exp(logp) - (col == target.to(tl.int32)).to(tl.float32)
```

## Contract and recognition

Dense contiguous immutable FP32 ND logits[R,D] and public INT64 targets[R],
with 1<=R<=4096, 1<=D<=8192 and R*D<=8388608. Labels lie in [0,D), so the
local INT32 target cast is exact. Return per-row FP32 loss and the complete
FP32 logits VJP for the sum of losses. No class weights, ignored labels,
smoothing, softcap or z-loss. Both arms retain the same stable FP32 exp/reduction
graph, checked against an independent FP64 oracle at atol=rtol=2e-4.

Keep maximum and shifted logsumexp separate: combining them before subtracting
x loses normalization at a large common FP32 offset. Preserve label ABI and
input immutability. Retained gradient TTIR comparisons are INT32 in both large
capture arms; that is not final-disassembly proof or a guarantee for another
compiler. The full log-probabilities must not be a required public output.

Recognize scratch written solely for target loss and full-gradient consumers.
The before arm allocates 4RD logical bytes; after allocates 8R, which is larger
at D=1 and equal at D=2. Both still use two kernels with identical row grids;
the after gradient rereads logits. Allocation occurs inside the caller, but the
device span excludes allocator work before the first device task. Do not claim
allocator latency, physical peak memory or traffic savings from scratch size.

## Bounded integer predicates: a separate compiler observation

An earlier gallery repair narrowed a locally loaded target before the lane-wise
predicate, while keeping the public target tensor INT64:

```python
# Before: the loaded INT64 scalar promotes the vector comparison to INT64.
target = tl.load(Targets + row)
is_target = col == target
# After: only after proving 0 <= target < D <= 8192.
target = tl.load(Targets + row).to(tl.int32)
is_target = col == target
```

Every admitted target is in [0,8191], exactly representable in signed INT32;
column indices are also representable. Thus narrowing preserves equality and
target addressing without changing the public label dtype or values. Check the
full value domain before transferring this idea to another bounded scalar:
a public INT64 tensor alone does not require an INT64 lane predicate, but a
small observed sample alone does not justify narrowing. Preserve wider arithmetic
where sums, offsets or sentinels need it.

For the selected moderate-logit gallery fixture, retained TTIR analysis found
an 8192-lane INT64 equality before repair and 8192-lane INT32 equality in both
repaired gradient arms. This is compiler-IR evidence, not final-ISA proof or an
architecture-wide threshold. The [historical repair receipt](integer-repair-gallery.json)
retains all ten duration/scalar/vector and complete-span samples with source and
evidence hashes. The table compares v1 with v2 within each materialization arm:

| Arm | Gradient task mean v1 / v2 (µs) | Complete profiled span mean v1 / v2 (µs) |
| --- | --- | --- |
| Full log-probability scratch | 1720.660 / 20.986 | 1754.888 / 185.146 |
| Two-statistic scratch | 1719.602 / 21.220 | 1758.332 / 181.190 |

Complete span improved about 9.5–9.7× on this fixture, including inter-kernel
gaps; this is not unprofiled latency. Both historical versions shared a
normalization bug at large common FP32 offsets and are superseded by the stable
v3 math above. These captures support the selected fixture's bounded-predicate
compiler repair, not correctness of the old code over the current domain or a
materialization speedup. Apply the stable normalization and revalidate numerical
and compiler behavior together. This observation adds no gallery family or
current v3 pair; the materialization comparison remains inconclusive.

## What the gallery establishes

The corrected v3 capture establishes no robust materialization runtime win.
Large inter-kernel gaps and sample variation matter; active union is a separate
boundary, not a replacement for complete span. Inspect the emitted scratch
stores, logits rereads and task-local scalar/vector/MTE diagnostics, then compare
the complete loss-plus-gradient operation. Wrong offset normalization, a public
log-probability consumer, or no full-span gain falsifies a timing recommendation.

All arms passed correctness and ten-forward identity checks. Timings below are
complete profiled operation spans including gaps, not unprofiled latency. The
[full v3 receipt](crossentropy-gallery-v3.json) retains all span/active/gap samples,
matched PMU field names, source/binary identities and call counts. See the
[expansion ledger](expansion-gallery-evidence.md) for uncertainty assumptions.

| Gallery case | Before / after span (µs ± sample SD) | Ratio; observation |
| --- | --- | --- |
| e10-crossentropy-gallery-benefit | 190.264 ± 9.531 / 185.676 ± 17.01 | 1.025×; inconclusive |
| e10-crossentropy-gallery-cost | 180.158 ± 15.17 / 180.118 ± 15.34 | 1×; inconclusive |

Source mechanism: [pinned source 1](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/cross_entropy.py:L95-L142),
[pinned source 2](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/cross_entropy.py:L157-L206),
[pinned source 3](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/loss/cross_entropy.py:L210-L249).
The gallery is an independent restricted implementation. Physical traffic and
roofline distance are **not estimated** from task activity ratios.
