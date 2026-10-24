# Staged complex DFT

This pedagogical pair factors a full complex DFT into two smaller complex DFTs.
The input is contiguous split real/imaginary FP32 `[batch,N]`; the output is the
unnormalized forward transform in natural frequency order. Both arms use IEEE
FP32 vector arithmetic, with fusion disabled and no dot products or TF32.

For `N=A*B`, write `n=n1+A*n2` and `k=k2+B*k1`. The first stage computes a
B-point DFT across n2 and multiplies by `exp(-2πi*n1*k2/N)`. The second stage
computes an A-point DFT across n1 and stores directly at frequency `k2+B*k1`.
Both kernels, the twiddle products, scratch writes/reads, and frequency ordering
are inside `ComplexFFT.run()`. Each scratch element is overwritten per call.
Input-independent cosine/sine tables are rounded from FP64 to FP32 during
preparation for both arms. Inputs, outputs and scratch are allocated once;
there is no cached input-dependent permutation or transform.

`examples/optimization-library/triton/staged_fft.py` provides the reusable
`ComplexFFT` helper and paired-experiment adapter. Powers of two from 32 through
512 use the staged path; other positive sizes through 512 use the same full
DFT kernel as the baseline. Zero and larger sizes are rejected on the host.
The four frozen cases are N256/B16, N32/B1, N512/B12, and N64/B1 with seeds
4201–4204. Two guards cover impulse/frequency ordering at N64 and identical
prime-N17 fallback. Guard cases collect no profiles.

The independent oracle runs CPU complex128 FFT on the represented FP32 inputs.
Every complex output must satisfy `abs(error) <= 2e-6*row_L1 + 2e-5*abs(ref)`;
shape, precision, finiteness and input immutability are also checked. This gate
is fixed before timing. The protocol compiles all twelve resources before any
forward, checks each arm once, then captures exactly ten forwards per arm in
four pairs. There are no warmups, autotuning calls, extra timing, or output
inspection inside a profiled forward.

The algorithm lead is ops-fft revision
`1878ff9c940d7bbdb592acfff008a1cc98494345`, FFT-B radix decomposition and
small DFT/twiddle stages, under the CANN Open Software License Agreement 2.0.
This independently written Triton adaptation executes device-resident work;
it does not reproduce native-wrapper host transfers or synchronization. Native
FFT-B starts at N256, while N32/N64 here are generalized pedagogical cost cases.
The cited native source supports 910B and excludes 910_93. Validation on A3
910B4-1 establishes no support for 910_93 or A5.

Filtered source citations:

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-fft#src/fft1_d/arch32/fft_b/fft1_d_b.cpp:L19-L39`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-fft#src/fft1_d/arch32/fft_b/fft1_d_b_kernel.h:L37-L250`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-fft#src/fft1_d/arch32/dft/fft1_d_dft.cpp:L1-L150`

Whole-device operation span, including the gap between stages, is the primary
comparison. Active kernel duration and gaps are reported separately. Profiled
event durations are diagnostic; they do not establish application latency,
physical traffic, bandwidth ceilings, or a roofline claim.

## Observed A3 result

All six fixtures passed the fixed error/immutability checks on 910B4-1.
Four pairs collected ten forwards per arm; the two guards collected none.
The staged implementation was slower in every case. Prospective benefit/cost
labels in the frozen matrix were hypotheses and were retained unchanged.

| N / batch | Before span µs, mean ± sample SD | After span µs, mean ± sample SD | After active union µs | After gap µs |
| --- | ---: | ---: | ---: | ---: |
| 256 / 16 | 604.078 ± 1.082 | 1890.614 ± 18.013 | 1890.590 | 0.024 |
| 32 / 1 | 8.270 ± 0.374 | 220.554 ± 162.596 | 29.810 | 190.744 |
| 512 / 12 | 1803.430 ± 1.088 | 3332.078 ± 17.872 | 3332.052 | 0.026 |
| 64 / 1 | 14.300 ± 0.336 | 175.204 ± 7.690 | 41.936 | 133.268 |

All samples, including the slow first N32 after sample, are retained in the
[compact JSON](data/optimization-w004-staged-fft.json). Small-case gaps account
for much of their span; the large cases also regress in active duration.
The first stage's exported `aiv_scalar_time(us)` means were 1684.393 and 3056.050
for N256/N512, versus `aiv_vec_time(us)` 8.491 and 12.762. These per-task fields
suggest substantial scalar/indexing work in this implementation; they do not
establish a universal limitation of staged FFTs or a whole-device utilization.
The arithmetic reduction alone did not offset the implementation's cost.

The original controller stopped after a successful N32 baseline capture because
an unsafe whole-result error substring matched event sample
`0.3014799952507019`. Both correctness checks and that capture were preserved.
The continuation verified the unchanged CPU fixture/source/compiler identities,
captured only the missing N32 after arm, then ran the remaining untouched cases.
The derived N32 report links the original partial report and capture digests;
its baseline post-capture compiler identity is explicitly reconstructed from
retained artifacts and matching compile-only identities. No correctness call or
profile was replayed. The original failed job, immutable source, reports,
NPUBIN/IR, trace/PMU exports and continuation are retained in a 193-file compact
archive, SHA256 `f08689078603dc0e3c7d4f3cbc6f70d60e882761cde01a160bdff6f8049e08cb`.
There were six scientific CPU oracles, twelve correctness forwards, eighty
profiled forwards, and no warmups or extra timing calls across both jobs.
