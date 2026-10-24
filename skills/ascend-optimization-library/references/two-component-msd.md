# Residual two-component int8 dot (F11)

This pair isolates dynamic activation decomposition, two integer products and
precision compensation. BEFORE dequantizes signed int8 weights with the fixed
scale and computes dense FP16 activation × FP16 weight GEMM with FP32
accumulation. AFTER computes each activation row's absolute maximum, forms
`t = X * (127 / amax)`, rounds `a1 = round(t)` and
`a2 = round(254 * (t - a1))`, then computes two int8 × int8 → int32 products.
The FP32 reconstruction evaluates `c2 * float32(1/254)`, adds FP32 c1,
multiplies by FP32 `(amax/127)`, then multiplies the weight scale and converts
to FP16. Floating fusion is disabled. Zero rows use a safe denominator and
produce zero component codes and output.

The independent Triton implementation follows the equations and ordering in
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-transformer#gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a16w8_msd.h:L713-L780`
and `:L918-L984`. It does not reproduce native grouping, scheduling or cache
policy. The source is CANN Open 2.0; the implementation is independently written.
The qualified integer Cube interface is shared with the F10 project reference.
This pair establishes no native L2 cache-hint support for F12.

`examples/optimization-library/triton/two_component_cases.json` contains the
four frozen gallery/heldout cases and two correctness-only guards. Inputs are
FP16 within [-1,1], with an explicit +1 per row; weights are int8 within [-8,8]
and each column scale is 2^-10. The fixed RNG rejects half-integer rounding
margins below 1e-5 before measurement. AFTER must match independent exact
component codes and the ordered FP32 reconstruction's numeric FP16 output
(signed zero may compare equal). BOTH arms must satisfy the dense FP64 absolute
budget `sum(abs(W))*amax*abs(scale)/(127*508) + 2^-8`. BEFORE is not required to
match the compensated approximation. There is no relative-only acceptance.

The API preallocates inputs, output and two component workspaces. AFTER
recomputes amax and both full component buffers on every call; accumulator
initialization, both products and reconstruction stay inside the complete
forward. BEFORE's weight conversion stays inside its forward. Only validation
reads component buffers; profiles perform no diagnostic I/O. Unsafe int32
bounds use an independent CPU dense FP64 fallback, admitted only as a
correctness-only guard. Guard results are not NPU performance evidence.

Use `scripts/optimization_pair.py` with `two_component:Adapter`. The adapter's
`compile_only` preflights all kernels without executing a device forward.
The protocol then checks correctness once per side and profiles exactly ten
complete forwards per side; guard cases request zero captures. Primary profile
comparisons use the whole device-operation span, including stages and gaps.
Active intervals and gaps must be reported separately. Profiled event latency
is instrumented evidence, not independent ordinary-run latency.

## A3 paired result

All four frozen cases passed both numerical gates on Ascend910B4-1, Triton
3.2.0, Torch-NPU 2.10.0. The two guards also passed, with zero captures. Each
main arm has exactly ten samples, with no warmup or discarded first sample.
[Portable evidence](two-component-msd-evidence.json) retains every sample,
original PMU column names and numeric samples, source/NPUBIN identities, correctness receipts
and retained job handles. The compact archive additionally retains compiler
artifacts, exact raw PMU rows and the original rejected compilation attempt.

| Case | Complete BEFORE span, us (mean ± sample SD) | Complete AFTER span, us (mean ± sample SD) | AFTER active / gap, us |
| --- | ---: | ---: | ---: |
| F11-01, 128×1024×1024 | 67.648 ± 1.931 | 242.878 ± 34.616 | 39.628 / 203.250 |
| F11-02, 8×64×128 | 3.346 ± 0.095 | 214.996 ± 39.111 | 5.696 / 209.300 |
| F11-03, 96×768×768 | 32.210 ± 1.051 | 230.354 ± 46.168 | 24.996 / 205.358 |
| F11-04, 16×128×256 | 4.004 ± 0.316 | 217.070 ± 44.201 | 7.066 / 210.004 |

**Avoid this implementation for these workloads.** AFTER regresses at the
complete API/device span in all four cases. It reduces the active kernel union
on the larger cases, while the additional decomposition launch introduces a
roughly 200 us measured gap in this instrumented execution. Active-only timing
would incorrectly suggest a benefit. These observations do not identify the
cause of every gap or establish unprofiled latency, physical traffic, a roofline,
or native fused-MSD performance.

The first compile-only source used `tl.div_rn`; AFTER compilation aborted in
MLIR with a null TypeRange assertion, with zero device forwards. The retained
successful source uses ordinary FP32 division. The frozen amax values are
exactly 0 or 1, and exact component/output validation passed. No code, seed,
shape, gate or tile changed after scientific timing began. The unsupported
compiler path is retained separately from the successful measurements.
