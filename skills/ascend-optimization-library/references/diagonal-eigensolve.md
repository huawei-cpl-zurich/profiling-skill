# Exact diagonal certificate (W10)

This child experiment reuses W08's five-sweep FP32 values-only Jacobi solver.
The baseline always performs all sweeps and the final stable diagonal sort.
The candidate first scans every current matrix entry on the device, requiring
finite values and exactly zero off-diagonal entries. It reads the actual device
predicate with `flag.item()` on every API call, then either sorts the original
diagonal or runs the unchanged baseline. No CPU fixture classification controls
the algorithm. Both signs of zero pass; any nonzero off-diagonal value fails.
The certificate is stricter than the source's approximate diagonal criterion.
The filtered source receipts are
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-solver#src/cheevj/kernel/cheevj_diagonal.hpp:L30-L66`
(certification), the same file at `L131-L176` (off-diagonal reduction), and
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-solver#src/cheevj/kernel/jacobi_large_impl_part1.inc:L92-L116`
(integration), pinned to `8ee0617c8f1ffddcbbd219fe4fe5a06f9c8a62cd`
under CANN Open Software License 2.0. The upstream complex squared-magnitude
threshold is approximately `1e-12`; this real FP32 adaptation requires finite
entries and exactly zero off-diagonals instead.

The frozen domain is contiguous symmetric FP32 `[1,N,N]`, five sweeps and values
only. Gallery cases are unsorted seeded diagonal N128 and bounded dense N32;
held-out cases are diagonal N96 and bounded dense N64. N128 is a qualification
of this child experiment, not a blanket extension of W08's public contract.
Guards require a `2^-20` off-diagonal pair to take the fallback and repeated
unsorted diagonal values to produce exactly sorted values. CPU FP64 eigvalsh
of the represented input supplies the independent oracle (`atol=rtol=2e-4`).
Every correctness check also requires finite sorted values and immutable input.

The mandatory one-bit device-to-host transfer and host branch are algorithmic
work inside the API. The primary diagnostic spans the first certificate kernel
to the final sort or solver kernel, including dispatch gaps. Kernel active time
and the interkernel/non-compute gap are separate: the latter is the interval
outside the attributed compute-kernel union and can include device-to-host
copies and caller dispatch. It does not establish whole-device idleness.
Profiler durations are instrumented diagnostics, not ordinary latency claims.
No measured physical traffic or roofline claim follows from these captures.

The frozen A3 gate completed on Ascend 910B4-1, physical die 6 mapped to logical
0, with fresh healthy idle isolation. All twelve compile-only preparations and
all twelve once-correctness calls passed. Guards had no captures. Each measured
side executed exactly ten instrumented calls, without warmup or autotuning;
compiled kernel/binary identities remained unchanged. CANN was 9.1.0,
Torch-NPU 2.10.0 and Triton 3.2.0. The installed `msprof op --version` reported
26.1.0-56632dc4a944410c791538451dc2156a151bd105.

The following are mean whole-operation device spans in microseconds, including
all compute kernels and intervening gaps. They are instrumented diagnostics.


| Frozen case | Baseline span (us) | Certificate span (us) | Actual candidate dispatch |
|---|---:|---:|---|
| W004-W10-01 | 286100.422 | 720.578 | diagonal-sort |
| W004-W10-02 | 70756.070 | 69947.486 | five-sweep-fallback |
| W004-W10-03 | 214126.962 | 554.860 | diagonal-sort |
| W004-W10-04 | 140827.202 | 144583.134 | five-sweep-fallback |

Diagonal candidates executed exactly two kernels per call (certificate and
sort); dense candidates executed every baseline kernel plus one certificate.
N32's small observed difference does not establish a reliable benefit on dense
inputs; N64 retains the observed overhead. The tiny-nonzero guard selected the
fallback despite its small eigenvalue effect, and the repeated-diagonal guard
selected the shortcut and sorted exactly. All input immutability checks passed.

The original `aten::item`, `aten::_local_scalar_dense`, and
`AscendCL@aclrtMemcpy` rows are retained separately for every measured candidate
call. Their nested caller durations must not be added to each other or to the
primary span. For N128, mean `aten::item` duration was 507.501 us and its nested
copy API duration was 36.583 us; the primary span already includes the
certificate-to-sort gap.

[Readable summary](diagonal-eigensolve-summary.json) points to the lossless
[evidence archive](diagonal-eigensolve-evidence.json.gz), which preserves all
ten samples per arm, all 61,680 PMU task rows, source and compiled identities,
independent CPU oracle inputs/outputs, guard results, and caller/copy rows.
The retained compact archive additionally contains the original vendor traces,
compiler cache artifacts and NPUBIN, source/harness snapshot, phase/watchdog
records, and health evidence, bound by its SHA256 in the summary.
The frozen matrix is `experiments/optimization-wave004/W10.json` and the
runnable adapter is `diagonal_eigensolve:Adapter` beside the parent adapter.
