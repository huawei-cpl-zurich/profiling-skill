# Decode INT4 weights in their consumer tile

Status: source-backed experiment with CPU/interpreter verification. A3 build,
correctness and profiling remain pending. Keep this entry separate from the
frozen library used by the current coding demonstration.

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
autotuning or measured tensor inspection. No runtime outcome is asserted here.
