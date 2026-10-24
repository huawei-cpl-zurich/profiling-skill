# Factor a complex DFT into small DFT stages

Recognize an explicit NxN complex Fourier transform whose composite N admits
smaller DFTs. Cooley-Tukey reduces arithmetic by replacing one full transform
with two smaller transforms and twiddle products. Extra kernels, scratch,
strided reads and natural-order writes can outweigh the arithmetic reduction.

Use the independently written [Triton helper](../../../examples/optimization-library/triton/staged_fft.py)
for contiguous split FP32 real/imaginary `[batch,N]`. The baseline is a full DFT;
the staged path uses two vector FP32 reductions and a full complex twiddle
multiply, with no Cube dot, TF32, real packing, hidden input permutation or
cached input-dependent transform. Natural output ordering and scratch writes
are part of every forward. Preallocation and mathematical FP32 coefficient
tables, built from FP64 cosine/sine, are shared preparation policies.

Apply only when a supported factorization and an observed full-operation benefit
justify it. Avoid an unconditional replacement on tiny transforms or when
kernel launches and scratch movement dominate. Other positive N through 512
use the same direct DFT fallback. Invalid sizes are rejected before compilation.
N32/N64 are generalized pedagogical cases outside the native FFT-B minimum 256.
This adaptation does not benchmark the native wrapper's transfers/synchronizing
work, and makes no 910_93 support claim.

The [frozen matrix](../../../examples/optimization-library/triton/staged_fft_cases.json)
fixes four cases, input bounds, seeds, the independent CPU complex128 oracle,
and the error gate before profiling. Two guards exercise impulse/frequency
ordering and the N17 identical fallback. The
[implementation and evidence notes](../../../docs/optimization-staged-complex-fft.md)
describe stage indexing and source attribution. The
[compact evidence](../../../docs/data/optimization-w004-staged-fft.json)
retains every ten-sample series and raw exported PMU field names.

All four frozen cases were unfavorable on 910B4-1: mean complete spans were
604.078→1890.614µs (N256/B16), 8.270→220.554µs (N32/B1),
1803.430→3332.078µs (N512/B12), and 14.300→175.204µs (N64/B1).
The N32 after series has substantial variation, with its first sample retained.
Large cases regress in active duration too; small cases add substantial gaps.
Choose **avoid** for this implementation and the measured cases. Evidence is
**insufficient** for a tuned implementation or an unmeasured target/shape.

Compare the complete device-operation span, including inter-stage gaps, with
active duration separately. Treat PMU fields as exported per-task evidence;
do not infer physical traffic, application latency, whole-device utilization
or roofline distance. Correct arithmetic reduction alone does not establish a
performance improvement. These four frozen cases do not establish a universal
threshold or a tuned FFT replacement.
