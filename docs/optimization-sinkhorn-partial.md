# Sinkhorn partial-gradient reducer

This study isolates MindSpeedOps `deterministic_reduce_kernel` from
`mindspeed_ops/arch32/triton/sinkhorn.py`, source revision
`4efe2b675c2d51517c3fdd14c721020b8d41e241` (Apache-2.0).
It consumes supplied FP32 partials `[N,2]` and `[N,8]` and preexisting caller
state `[3]` and `[24]`. It does not measure partial production or Sinkhorn
backward. Both arms privately copy initial state during preparation, then
reuse output storage for each complete component call. Allocation and the
initial copy are excluded from the profile.

The baseline uses the source's single-program scalar reduction order, two
scale accumulators and paired pre/post base accumulators, with exclusive
stores. The candidate loads masked next-power-of-two rows and sums each
column along axis zero. Only scale slots 0–1 and base slots 0–7 are written;
all other caller slots remain bitwise identical, including negative zero.
Both arms use the same scalar kernel for N above 1024, up to the common
admission maximum 4096. There are no atomics.

Independent CPU FP64 sums provide the reference. Each output's absolute
error must satisfy `1e-7 + (gamma_N32 + gamma_N64) * sum_abs64`, where
`gamma_Np=N*u_p/(1-N*u_p)`, `u32=2^-24`, and `u64=2^-53`.
The fixed domain admits finite partials with nonzero absolute values between
`2^-20` and 1, inclusive. The bound accounts for both accumulation and CPU
reference error and is fixed before timing. All ten errors and bounds are
retained. A fixed vector tree may round differently from serial addition;
there is no paired bitwise numerical gate. Determinism is a structural claim
about fixed reduction order, with no empirical repeat-bit guarantee.

The frozen E07 matrix has four profiled pairs and two correctness-only
guards. Each arm runs once for correctness, then exactly ten times under
profiling for each main case. The guards check phased cancellation/zeros
and the exact shared scalar fallback at N=1025. Compilation and selected
binary text/relocation comparison precede numerical execution. No warmups,
autotuning, extra correctness repeats, or unprofiled timing are part of the
study. Reported device operation spans are profiler diagnostics and do not
establish ordinary application latency, physical traffic, or whole-backward
performance.

For standalone case admission, from any working directory:

```sh
python /path/to/examples/optimization-library/triton/sinkhorn_partial_cli.py \
  --case case.json --output result --cache oracles --provenance provenance.json \
  --validate-only
```

`case.json` accepts `{"seed": 113, "num_blocks": 13}` or nested
`{"seed": 113, "parameters": {"num_blocks": 13}}`, with optional matrix
metadata such as role and expected outcome. Unsupported cases fail before
fixture-cache creation. Remove `--validate-only` to execute the paired
protocol on an appropriately configured NPU environment.

A3 validation completed all six cases. Every arm passed its single numerical
check, including all ten per-component bounds, untouched sentinel bits,
output canaries and input immutability. Six independent cached CPU payloads
were used, with twelve correctness calls and eighty profiled component calls
across eight captures. The guard cases made zero captures. The selected
binaries differ in `.text` for all vector-admitted cases, while the N=1025
guard has identical full binary, text and relocation identity.

The retained ten-sample profiler operation spans were:

| N | Before mean ± sample SD (µs) | After mean ± sample SD (µs) |
|---|---:|---:|
| 32 | 5.168 ± 0.516 | 3.600 ± 0.509 |
| 1 | 3.112 ± 0.505 | 1.142 ± 0.058 |
| 1024 | 224.090 ± 1.974 | 26.920 ± 0.344 |
| 2 | 2.380 ± 0.199 | 1.170 ± 0.022 |

These are instrumented device-operation spans with prepared output reuse;
they do not establish unprofiled API latency. The [portable structured evidence](data/optimization-wave007-E07.json) publishes
all ten raw event and operation-span samples per capture, supported per-iteration
vendor PMU fields, every component error/bound, compiler resource metadata,
selected binary hashes, six fixture bindings and the verified artifact manifest.
Original vendor reports, binaries and oracle payloads remain in the retained
compact archive identified by its SHA256; they are not duplicated in this repository.
The imported runtime reported Torch 2.10.0+cpu, torch-npu 2.10.0, Triton
3.2.0 and triton-ascend distribution 3.2.2 on Ascend910B4-1. The installed
Triton distribution metadata separately reported 3.5.0; imported compiler
identity and captured artifacts govern this result.
