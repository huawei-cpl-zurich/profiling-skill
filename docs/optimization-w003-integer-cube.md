# F10 integer Cube compute format

The four frozen bounded-integer cases passed bitwise FP32 correctness on gz-a3.
Their profiled device operation spans decreased with integer arithmetic. The
separately retained instrumented event samples include host dispatch costs and
do not establish ordinary application latency.

| Case | M,N,K | Floating span mean ± sample SD (µs) | Integer span mean ± sample SD (µs) | Before/after |
|---|---|---:|---:|---:|
| F10-01 gallery benefit | 64,256,512 | 14.866 ± 0.407 | 7.698 ± 0.241 | 1.931 |
| F10-02 gallery cost | 16,64,128 | 4.070 ± 0.341 | 2.678 ± 0.330 | 1.520 |
| F10-03 heldout benefit | 96,320,2048 | 66.554 ± 1.226 | 21.626 ± 0.664 | 3.077 |
| F10-04 heldout cost | 16,128,128 | 4.284 ± 0.230 | 2.918 ± 0.193 | 1.468 |

Each arm has ten samples and exactly one attributed `integer_cube_scaled`
MIX_AIC device task per sample. The complete device span equals active time
here; the observed gap is zero because each forward uses one kernel. The two
cost hypotheses did not materialize in these instrumented device spans. No
extra timing, warmups or autotuning were performed.

The [adapter](../examples/optimization-library/triton/integer_cube.py) computes
the same prequantized integer dot with the same 16×64×128 tile, ascending K
traversal and FP32 row-scale then column-scale epilogue on both sides. Before
converts int8 operands inside each block and performs an IEEE FP32 dot. After
uses int8 operands with int32 accumulation and converts the accumulator once.
Input/scales transfer and output allocation occur before the forward; all
operand conversion, accumulator initialization, matrix work and scaling belong
to the forward. There is no external workspace or bias.

The independent cached oracle uses int64 dot and then the exact FP32 scale
order. Codes are in [-16,16], scales are seeded powers of two from 2^-5 through
2^-1, and K×256≤2^24. This is a compute-format experiment on existing integer
data; it measures no quantization error. Cases, seeds and gates were fixed by
matrix v1 SHA256
`66a1aa6d957727edd8172939a4decd74cabfe8591b3cba83f7c6e3eb9dee6890`.

The two additional guards use correctness-only mode and zero captures. The
K=133, M=17, N=65 tail uses explicit zero-masked loads. The deliberate
K=2048/code_max=127 bound violation dispatches both arms to an independent
host FP64 floating dot, then ordered FP32 scaling and H2D copy. It does not
claim NPU integer or FP32 dot execution. Non-guard out-of-bound specifications
are rejected, preventing that host reference from entering a profile pair.

There were six cached oracle constructions, twelve interface correctness
checks, eight side captures and eighty profiled forwards. Ten correctness
forwards use the NPU dot path; two execute the host reference and H2D guard
path. Those two implementation computations are separate from cached oracle
construction. All eighty captured device tasks were attributed by trace
correlation. Both arms of all six resource shapes passed compile-only probes
before any scientific forward.

The retained compiler adapter IR contains FP32 `linalg.matmul` with
`input_precision="ieee"` before, and int8 inputs/int32 accumulator after.
Same-capture MIX_AIC rows have positive `aic_mac_time(us)` on both arms,
supporting actual Cube use together with those type-specific compiler
artifacts and NPUBIN identities. For F10-03, mean exported MAC time is
2.764 µs before and 1.149 µs after. These are per-task pipe diagnostics;
no physical traffic, bandwidth or roofline conclusion follows from ratios.

[Compact data](data/optimization-w003-integer-cube.json) retains all samples,
raw PMU rows, exact kernel names, source and NPUBIN hashes, device mapping,
durable handles and guard dispatches. Full vendor reports and compiler
artifacts remain on the selected target. Scientific source-v2 emitted a
dictionary of binary hashes; readonly analysis adapted a copy to the shared
collector's singleton Triton-list schema. Original pair SHA256 values and the
lossless transformation are retained. Final source-v3 changes only this
metadata schema and its CPU test; the measurements remain bound to source-v2.
An initial staging job failed before compilation because its uploaded bundle
was outside the job directory; it executed no scientific forwards.

The implementation is independently derived from the source-qualified
scaled-mm pattern:
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/gemm/triton_scaled_mm.py`.
The source's loose tolerance and implicit masked values are not adopted.
