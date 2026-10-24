# Real FFT through half-length complex packing

This pedagogical pair computes the unnormalized forward RFFT of contiguous
FP32 real `[batch,N]` inputs, returning split FP32 real/imaginary
`[batch,N//2+1]` outputs in natural frequency order. The baseline embeds real
input in a length-N complex transform with zero imaginary input, then extracts
nonnegative frequencies. The candidate packs `z[j]=x[2j]+i*x[2j+1]`, transforms
N/2 complex values, and reconstructs the RFFT bins. Packing, extraction,
reconstruction, twiddle multiplication and scratch movement execute inside each
forward; only input-independent tables, zero imaginary input and allocations
are prepared once.

For `M=N/2`, let `Z=FFT(z)`, `A=Z[k mod M]` and
`B=conj(Z[-k mod M])`. The even- and odd-index transforms are
`E=(A+B)/2` and `O=(A-B)/(2i)`, so
`X[k]=E+exp(-2πik/N)*O`, for `0 <= k <= M`.
Both arms explicitly write zero imaginary DC and even-N Nyquist endpoints.

The complex algorithm stays fixed within each pair: N32 uses the existing
direct DFT kernel at lengths 32 and 16; N64 and larger powers of two use the
existing two-stage Cooley–Tukey kernels at both full and half length.
`real_fft.RealFFT` reuses the parent `staged_fft` kernels and compile/run
interface. Its RFFT-specific construction admits positive N through 1024;
non-power-of-two sizes use the identical full complex DFT/extraction fallback
in both arms. This does not extend the public `ComplexFFT` range beyond 512.
Both arms use IEEE FP32 arithmetic with fusion disabled.

The four frozen cases are N1024/B32, N32/B1, N512/B24 and N64/B1, seeds
4301–4304. Guards cover constant DC and alternating Nyquist rows at N64/B2,
and identical odd-N63 fallback. The matrix is byte-identical to the accepted
W03 freeze, SHA256
`8f82a5c684908b320ec5968e7d9773ed941617ca3f6090b8a19697c0c49901cd`.
The adapter accepts an individual nested matrix entry through
`optimization_pair.py --adapter real_fft:factory --case CASE.json`, or a flat
case object. Zero/oversize dimensions, conflicting parameters and mismatched
common-algorithm declarations are rejected before execution.

The CPU oracle computes complex128 RFFT of the represented FP32 values.
Every bin must satisfy `abs(error) <= 2e-6*row_L1 + 2e-5*abs(reference)`;
endpoints must have exactly zero imaginary values. Shape, dtype, finiteness,
ordering and input immutability are checked. The protocol builds six CPU
oracles once, compiles all twelve arms before the first forward, checks each
side once, then captures ten full forwards per side for four cases. Guards
collect no profiles. There are no warmups, autotuning or extra timing calls.

The source lead is ops-fft revision
`1878ff9c940d7bbdb592acfff008a1cc98494345`, under CANN Open Software License
Agreement 2.0. The bounded native host source provides the N/2 plan and A/B
coefficient formula; the reconstruction above is independently derived.
The included `fft_r2c_common.h` helper body was not retrieved, copied or
compiled. Parent complex
kernels are the project's independent Triton adaptation. Native 910B source
excludes 910_93; A3 evidence establishes no A5 or 910_93 support.

Filtered source receipts from the existing research query
`9034590d5bfd4f89aace185bc6d635d1`:

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-fft#src/rfft1_d/arch32/r2c_fft/rfft1_d_r2c_fft.cpp:L1-L240`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-fft#src/rfft1_d/arch32/r2c_fft/rfft1_d_r2c_fft.cpp:L235-L360`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-fft#src/rfft1_d/arch32/r2c_fft/rfft1_d_r2c_fft_kernel.h:L1-L260`

Prior source-path misses remain open as
`fee10c1c99ff76e0cfdb3fc0488f40a3` and
`11501d136a88a71c6d38ce1feaa9647e`; the independent derivation does not resolve
those provenance gaps.

Whole-device operation span, including all stage gaps, is the primary
comparison. Active union duration and gaps are separate diagnostics. Profiled
event time does not establish application latency, physical traffic, bandwidth
ceilings or roofline performance. All prospective benefit/cost labels remain
unchanged regardless of the measured outcome.

## Observed A3 result

All twelve correctness forwards passed on A3 910B4-1, including exact endpoint
zeros. The odd-N63 guard produced identical compiled kernel identities in
both arms. Four pairs collected exactly ten profiled forwards per side; guards
collected none. The large cases improved, while both tiny cases regressed.

| N / batch | Before span µs, mean ± sample SD | After span µs, mean ± sample SD | Before/after span | After active union µs | After gap µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 / 32 | 11062.428 ± 122.862 | 8223.178 ± 148.956 | 1.345 | 8208.208 | 14.970 |
| 32 / 1 | 153.304 ± 13.131 | 368.552 ± 27.387 | 0.416 | 10.036 | 358.516 |
| 512 / 24 | 6466.500 ± 56.500 | 3332.432 ± 77.951 | 1.940 | 3314.536 | 17.896 |
| 64 / 1 | 322.260 ± 16.265 | 502.604 ± 21.106 | 0.641 | 32.126 | 470.478 |

The large cases reduce active kernel time. For N64, active union falls from
48.098 to 32.126 µs but increased gaps make the complete operation slower.
For N32, active union slightly increases from 9.384 to 10.036 µs and gaps also
increase. These captures support the mechanism's large-case arithmetic benefit
and the tiny-case cost of the extra packing/reconstruction stage in this
implementation. They do not measure uninstrumented application latency.

All ten samples, source/NPUBIN identities, event diagnostics and exported
per-kernel PMU fields are retained in the
[compact evidence](data/optimization-w004-real-fft-packing.json). No sample was
removed. The retained compact archive contains 206 files, SHA256
`bbb5694f00c1bae9b14c1c5c1748aecc13d98510e777f4c511da268262105f47`.
The single scientific job completed without a workload or profiler failure;
there were six scientific CPU oracles, twelve correctness calls, eighty
profiled forwards, no warmups and no extra timing calls.

A final host allocation correction explicitly sets output and packed-real
storage to the FP32 input dtype, including when the caller selects an ambient
float64 default. The scientific measurement sources above remain immutable.
The final source changes two allocation calls; Triton kernels and `run()` are
unchanged. A3 passed 35 functional host tests, including an ambient-float64
constructor regression, then compile-only preparation verified all twelve arms
produce exactly the original compiler/NPUBIN identities under ambient float64.
That audit loaded retained fixtures, generated no new scientific oracle and
executed no forward or profile. The compact JSON records both source versions
and the audit receipt digest.
