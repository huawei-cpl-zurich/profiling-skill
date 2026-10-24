# Normalization mechanisms E01 and E07

Status: independent CPU oracles, frontend checks and CPU-interpreted kernel
math pass. A3 compilation, correctness, kernel identity and profiling have not
run. These are source-backed experiment hypotheses, not measured speedups.
They are separate from the frozen teaching snapshot used by the coding study.

## E01: batch short L2-normalization rows

Before, one program loads one row and returns both normalized values and its
FP32 inverse norm. After, one program handles four independent rows, including
a masked final partial group. Both compute `x / sqrt(sum(x*x) + eps)` in FP32;
epsilon is added after the sum, without division by width.

```python
rows = program_id * 4 + arange(0, 4)
x = load_rows(rows, columns, mask=(rows < R) & (columns < D))
inverse = rsqrt(sum(x * x, axis=1) + eps)
store_rows(x * inverse[:, None], mask)
store_inverse(inverse, mask=rows < R)
```

Recognize many independent short reductions with little work per program.
Coarsening can amortize program scheduling, while increasing live vector state
and reducing the number of independently scheduled programs. Few wide rows
can therefore lose parallelism or encounter resource limits. This example's
batch factor is four; it does not copy the source's additional serial row loop.

The predeclared gallery regimes are 65,536 rows of width64 and seven rows of
width2,048. The complete-call boundary includes every row and inverse-norm
output. Inspect compiled resource use and actual device duration; program-count
arithmetic is not proof of occupancy or whole-device utilization. Equal times,
a resource rejection, or a slower batched call falsify the timing hypothesis.

Source: [row implementation](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/l2_norm_fla.py:L7-L27),
[blocked implementation](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/l2_norm_fla.py:L56-L86),
[source dispatch](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/l2_norm_fla.py:L159-L196).

## E07: retain inverse RMS across a complete forward/backward pipeline

This explicitly unweighted operation returns `y` and the input VJP `dx` for a
supplied cotangent `dy`. There is no trainable weight or omitted weight gradient.
Both arms run the complete forward and backward, with identical FP32 arithmetic
and epsilon placement:

```python
inverse = rsqrt(mean(x * x, axis=-1) + eps)
y = x * inverse
projection = mean(dy * x, axis=-1)
dx = inverse * (dy - x * inverse * inverse * projection)
```

Before, forward writes `y` and backward recomputes inverse RMS. After, forward
also writes one FP32 inverse per row and backward loads it. Both still execute
two kernels. Output buffers are preallocated in both arms; after additionally
retains `4*R` workspace bytes. The forward statistic write, backward read and
complete inter-kernel gap belong to the comparison. No cached input is supplied
for free. Backward still reads X for the projection, so the transformation does
not establish removal of a full-X read or physical traffic reduction.

Recognize an unchanged normalization statistic recalculated by backward.
Wide rows may benefit from removing repeated sum-square/mean/rsqrt work. Narrow
rows may gain nothing while paying an extra scalar write/read and retained
state lifetime. The gallery regimes are 2,048×2,048 and32×32. Compare complete
device spans, all kernel identities and scalar workspace alongside optional
supported vector counters. An isolated backward speedup with a slower full
pipeline rejects the whole-operation benefit. Pipe ratios alone cannot identify
roofline distance or bytes moved.

Source: [saved forward statistic](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L82-L94),
[backward reuse](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/norm/rms_norm.py:L145-L155).
The source has weighted and alternative casting graphs; this pair deliberately
fixes the unweighted FP32 graph and makes no claim about those other contracts.

Both examples reject unsupported dimensions/seeds/epsilon, audit dense ND input
layout and input immutability, and use fixed rtol=atol=3e-5. A failed correctness
or compile attempt is not a valid cost example. Use `norm_pairs:factory` with
`optimization_pair.py`: one correctness call per arm/case, then ten profiled
forwards without measured input/output inspection or warmups. Retain all samples,
source/binary identities and missing-counter states. No unprofiled latency or
calibrated roofline claim follows from a profile.
