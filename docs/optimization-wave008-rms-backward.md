# Provided-state RMSNorm backward row batching

This source-backed component study compares the dispatched blocked backward kernel
from `triton-ascend-ops` revision `7acb9c40a6ca827e885d902b7eaee5d45457b230`.
The baseline processes one row per iteration; the candidate processes four rows
when `D <= 1024`. Both use the driver's actual vector-core count for the grid.
`1024 < D <= 2048` retains exactly the baseline BR1 route.

The API accepts contiguous owned CPU FP32 fixture inputs `X`, `dY`, `W`, and
provided `RSTD`. Preparation copies inputs to the selected NPU and allocates
private reusable `dX`, `dW`, and per-core partial weight-gradient storage. Each
prepared API executes three device kernels: zero `dX`, blocked backward, and
final partial-`dW` reduction. The common zero and reduction are explicit portable
Triton adaptations of the original wrapper's torch operations. These results
therefore describe source-style component timing. They do not describe an
unmodified vendor backend or a complete forward/backward training step.

The independent CPU provider computes ideal reciprocal standard deviations in
FP64, rounds them to the supplied FP32 state, and evaluates the derivative of
that given state in FP64. There is no GPU forward in preparation. Gemma casting
mode 1, offset 0/1, and epsilon 1e-6 are fixed. Both kernel routes use a masked
RSTD load with `other=0` to prevent inactive BR4 tail lanes from propagating NaNs.
The odd-D guard supplies a NaN halo outside the valid RSTD view and includes
zero and cancellation rows. Tolerances are fixed before collection: dX
`atol=1e-4, rtol=1e-5`; dW `atol=2e-4, rtol=2e-5`. The changed per-program dW
accumulation grouping has no cross-arm bitwise guarantee.

`experiments/optimization-wave008/T02.json` is the frozen scientific matrix:
four main pairs and two correctness-only guards. Controlled collection compiles
all routes, screens actual emitted ELF code and resources, evaluates correctness
once per arm/case, then captures ten complete prepared APIs per main arm with
no warmup or autotuning. Profiler operation spans are diagnostic measurements,
with all samples and raw supported fields retained; they establish neither
ordinary latency nor physical traffic or occupancy.

`examples/optimization-library/triton/rms_backward_cli.py` is an exploratory
standalone interface. `--validate-only` admits case metadata without creating a
cache, loading an NPU backend, or claiming full collector qualification. The
functional tests use separate, nonfrozen shapes and seeds.

The frozen collection passed all six correctness cases and all compile/codegen
screens. The observed grid has 40 vector cores. Five candidate routes emitted
different blocked-kernel code; the wide guard retained identical complete
three-kernel artifacts. The original coverage receipt omitted scenario labels
required by the audit schema; a separate derived receipt adds those labels,
with source hashes, routes, results and layout evidence unchanged. The inventory
audit passes; this metadata correction executed no additional device calls.

| Case | Summed device work, before → after (µs) | Complete API operation span, before → after (µs) |
| --- | ---: | ---: |
| 1024 × 64 | 23.754 → 13.330 | 332.718 → 339.794 |
| 33 × 64 | 7.784 → 7.704 | 343.886 → 352.034 |
| 4096 × 128 | 62.474 → 33.828 | 340.648 → 354.894 |
| 256 × 1024 | 21.690 → 17.190 | 334.950 → 347.354 |

These are ten-sample profiled means. The complete API spans include substantial
inter-kernel gaps and show no improvement in this collection. The larger rows'
blocked-kernel work decreases, while the small-row case changes little. This
single paired study does not establish ordinary latency improvement.
`T02-results.json` retains every event/work/span sample, per-kernel duration,
correctness errors, fixture/source/binary/trace hashes and codegen receipts.
The verified compact archive binds the original raw supported PMU fields,
compiler/cache artifacts, CPU fixtures and reports. Full vendor trees remain
in the retained remote evidence; no extra timing or scientific replay was used.
