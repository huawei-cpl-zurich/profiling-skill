# Decode INT4 weights in their consumer tile

Status: source-backed experiment verified on A3 with CPU/interpreter checks,
eight successful device correctness checks and eighty profiled forwards across
four predeclared pairs. Consumer decode is slower in every measured case. This
is a workspace/launch tradeoff and rejection example, with no measured speed
win. It remains separate from the current demonstration's frozen guidance.

The pinned source provides both a
[standalone dequantizer](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/awq_triton.py:L12-L105)
and a
[dequantizing tile consumer](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/awq_triton.py:L170-L208).
Our independent example uses natural nibble order rather than AWQ's permutation;
it is a scoped grouped quantmatmul, not a drop-in AWQ implementation.

Before, decode the entire weight matrix to an FP16 GM workspace, then run GEMM.
After, decode only the weight tile consumed by each dot iteration. Both paths
use the same 32x64 output tile, K block 32, ascending K traversal, row-major A,
FP32 accumulator and FP16 output. Quantization groups are 32, 64 or 128 K values;
N is divisible by eight and K by the group size. M/N tile tails are masked.

Each int32 contains eight unsigned nibbles in consecutive N positions. The
same encoding is used for zero points. A consumer tile performs:

```python
shift = (column % 8) * 4
q = ((packed.to(tl.uint32) >> shift) & 15).to(tl.int32)
z = ((packed_zero.to(tl.uint32) >> shift) & 15).to(tl.int32)
weight = ((q - z).to(tl.float32) * scale.to(tl.float32)).to(tl.float16)
acc += tl.dot(a, weight)
```

The explicit FP16 weight conversion also occurs in the standalone path and in
the independent CPU oracle. Avoid unsigned subtraction before the signed cast:
negative dequantized values are valid. Preserve group boundaries, scale dtype,
input immutability and the fixed rtol=atol=1e-3 output check.

The source-level benefit is removal of an explicit FP16 workspace of 2*K*N bytes
and one launch. The before workspace is preallocated, so this does not measure
an allocation-saving benefit. Physical traffic savings remain unproved: the
compiler may stage decoded tiles in GM. A large M also repeats unpack/scaling
work across many output M tiles, where materializing weights once can amortize
its cost. Smaller packed input size alone does not establish a speedup.

Recognize a dequantization task immediately preceding GEMM, a large explicit
intermediate, and a low-M consumer that may not reuse each decoded weight much.
Compare complete-operation device spans, all expected tasks, the explicit
workspace and supported per-kernel vector/movement counters. Increased vector
activity, an unchanged/slower span, compiler scratch, or repeat-decode cost can
falsify the hypothesis. Activity ratios do not establish bytes or roofline.

Runnable adapter: `int4_pairs:factory`; source and CPU contract are
`examples/optimization-library/triton/int4_pairs.py` and `int4_contract.py`.
The predeclared matrix is held in orchestration records. Each arm/case receives
one correctness call followed by ten profiled forwards, without hidden warmups,
autotuning or measured tensor inspection. Four shared CPU oracle payloads serve
the four pairs; the same FP16 dequantization rounding precedes FP32 dot
accumulation in both arms and the oracle. All twelve compiled identities have
matching retained binary snapshots and remain unchanged during capture.

The primary result uses the span of every owned operation task on the same
device clock, including gaps between dequantization and GEMM. Each entry below
is the mean and sample standard deviation of ten profiled forwards. A span
omits first enqueue and host/profiler overhead; it is not unprofiled end-to-end
latency. Caller events collected with the profiler active are retained
separately. No first sample was discarded.

| Predeclared case | M, N, K; group size | Before span (µs) | After span (µs) | Before/after | Before explicit workspace (bytes) |
|---|---|---:|---:|---:|---:|
| Gallery benefit hypothesis | 16, 4096, 1024; 64 | 14016.484 ± 3.747 | 17283.386 ± 1.055 | 0.810980 | 8388608 |
| Gallery cost hypothesis | 2048, 512, 512; 64 | 1058.368 ± 1.131 | 56231.484 ± 15.017 | 0.018822 | 524288 |
| Held-out benefit hypothesis | 33, 3072, 1536; 128 | 16612.362 ± 1.019 | 32354.922 ± 1.984 | 0.513442 | 9437184 |
| Held-out cost hypothesis | 3073, 520, 320; 32 | 841.580 ± 0.915 | 58801.992 ± 18.602 | 0.014312 | 332800 |

Both benefit hypotheses are falsified on the measured runtime. The after arm
eliminates the adapter's full GM dequantization workspace and one launch, but
this does not establish lower physical peak memory or traffic. Larger M makes
repeated tile decoding particularly expensive in these examples.

All eight captures have identity-joined PMU rows. Decode-containing kernels
report AIV scalar ratios around 0.996–0.999 and vector ratios around 0.003–0.004.
The fused AIC MAC ratio rounds to 0.0, compared with approximately 0.096–0.258
for the plain GEMM. This supports a scalar decode bottleneck hypothesis; a
specific compiler lowering cause remains unproved. Preserve these row-level
pipe ratios and their exported precision. The exporter's separate
`cube_utilization(%)` column can read approximately 80–99% in the fused
kernels; it does not establish useful MAC throughput or roofline occupancy.
Physical traffic, compiler scratch and distance to roofline are unavailable.

Keep the negative outcomes when teaching this mechanism: source-level fusion
and a smaller explicit intermediate are reasons to investigate, while a
correct same-math implementation and complete-operation measurement determine
whether the transformation belongs in a fast path.
