# Normalization mechanisms E01 and E07

Status: CPU oracles, interpreted kernel math and eight predeclared paired A3
experiments pass. All sixteen arms have one correctness check, ten profiled
forwards, stable compiled identities and exact task-to-PMU joins. These examples
remain separate from the frozen teaching snapshot used by the coding study.

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


## A3 gallery evidence

Ascend910B4-1, Triton3.2.0, Torch-NPU2.10.0; implementation39d5e168.
The first pair completed in retained job644ef19e-daad-4ec6-8118-e9197d730eac;
the remaining seven completed in18236bc0-34a1-4ff8-9f86-eec6f7d0c66e.
Both ran on the same healthy idle physical device. The compact receipt digest is
`49966fca11c190586dd8aad3733b5185235c5741b712e0e18ffb21cf09248f9f`.
It retains all samples, original report and trace digests, compiled identities,
correctness diagnostics, PMU rows and the private holdout results.

The primary measurement is the span from the first device task's start to the
last task's completion per call. E07 includes the gap between forward and
backward. Values below are mean ± sample standard deviation across all ten
profiled calls; samples were neither warmed up nor discarded. These spans omit
first enqueue and caller/profiler overhead and are not unprofiled end-to-end
latencies. The separate caller event samples remain in the receipt.

| Mechanism and predeclared regime | Before span (µs) | After span (µs) | Before / after |
| --- | ---: | ---: | ---: |
| E01 many short rows | 468.482 ± 6.897 | 93.058 ± 2.083 | 5.034× |
| E01 few wide rows, expected cost | 3.284 ± 1.439 | 2.558 ± 0.272 | 1.284× |
| E07 wide rows | 211.202 ± 8.114 | 193.986 ± 12.013 | 1.089× |
| E07 narrow rows | 153.382 ± 7.726 | 169.192 ± 6.050 | 0.907× |

E01's gallery cost hypothesis was not supported: batching also lowered this
noisy three-microsecond mean. A separately predeclared private case did regress;
this does not license a general width-based dispatch rule. For the many-short-row
gallery, exported per-task vector time decreased138.031→55.629µs, while
its reported vector ratio rose0.2994→0.6220. These ratios describe per-task
pipeline composition; they are not measured whole-device occupancy.

E07's wide-row active union decreased76.174→71.836µs, while its inter-kernel
gap decreased135.028→122.150µs. Backward vector time decreased30.395→19.533µs,
but forward MTE3 time increased8.131→19.446µs with the saved-statistic write.
For narrow rows, active union increased6.390→6.744µs and the gap increased
146.992→162.448µs. A private wide-row span improved despite worse active union;
therefore the complete-span change cannot be attributed solely to arithmetic
reuse. Counter values retain the vendor's per-task semantics, with no inferred
physical bandwidth, occupancy or roofline percentage. No case was rerun to
obtain a preferred result.
