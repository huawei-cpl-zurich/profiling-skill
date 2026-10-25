# Online reduction for prepared FP32 log-softmax

This bounded study replaces the source kernel’s separate max and exponential-sum input passes with an online, rescaled max/sum pass. The final output pass, subtraction order, FP32 casts, 2048-element blocks and contiguous row assignment to the actual 48 Vector cores remain common. This is an application of a known online reduction, with no novelty claim.

The source is Huawei/SGLang `deterministic/log_softmax_batch_invariant.py` at pin `7acb9c40a6ca827e885d902b7eaee5d45457b230`, SHA256 `1245243b76b193df6eb4b2fdd9d6efc991bd2fc36ce1bce51149db33bbb8a096`. The attributed source and derived kernel retain the MIT license and third-party notices in `logsoftmax_online_LICENSE`. Both final masked loads explicitly use `other=0`; inactive lanes are never stored.

The frozen matrix and evidence are [R08.json](../experiments/optimization-wave009/R08.json) and [R08-result.json](../experiments/optimization-wave009/R08-result.json). On native Ascend 9382 / Ascend910V1, physical NPU0/chip0, these mean **profiler-instrumented device-operation spans** were observed:

| Case | Rows × columns | Before µs | After µs | Before / after |
|---|---:|---:|---:|---:|
| 01 | 8 × 2049 | 7.350 | 6.848 | 1.073 |
| 02 | 32 × 8192 | 10.048 | 9.096 | 1.105 |
| 03 | 128 × 16385 | 45.507 | 39.511 | 1.152 |
| 04 | 256 × 32768 | 133.781 | 117.114 | 1.142 |

The geometric mean of those diagnostic ratios is 1.118. All ten samples per arm, including the first, remain in the result and compressed full profile details. This does not establish ordinary launch latency, physical memory traffic, occupancy, a roofline result or the throughput of the source’s public wrapper.

All 12 input-only compilation/resource/ELF screens preceded the six independent CPU FP64 references, each generated once and cached. All 12 single-API numerical checks passed before the first of eight captures; each capture ran exactly ten prepared API forwards. Guards were not profiled. No warmup forwards, autotuning, unprofiled timing, extra invariance calls, numerical replays or post-result tuning occurred. Compilation uses Triton’s `warmup` compilation interface without launching its prototype. Both canonical host export locks covered every capture and synchronous analysis. Actual runtime/compiler binaries, version outputs, resource metadata, selected ELF sections/relocations, fixture hashes, raw report hashes and release evidence are bound in the retained compact manifest.

The domain is finite FP32 `[rows, columns]`, rows 1–256, columns 1–32768, absolute input values at most 80, positive row-major strides with declared row padding 0–7. The six cases exercise only the frozen configurations; this bounded domain is not a claim that every possible input was tested. GA checks small uniform partial blocks. GB uses actual stride `(2056, 1)`, preserved whole backing storage and ±80 inputs. CPU FP64 stable log-sum-exp checks use fixed `atol=4e-5`, `rtol=2e-6`. CPU inspection of the sole numerical output checks each row’s exponential sum within `2e-5` and duplicate-row bit consistency. Paired bit equality is secondary because online accumulation changes rounding. Whole output, both output halos and the entire immutable input backing are checked; mismatch coordinates and values are retained on a failed gate.

The Adapter prepares private reusable output storage; allocation and input movement are excluded from the API span. First-launch compilation is cached and commits only after a linked ELF is available. `compile_probe(...).run()` always raises without compiling or launching. Changing cases or starting a repeated arm clears stale paired diagnostics. Unsupported host dtype, shape, finite-value or layout inputs reject before backend imports and cache/device work. NaN/Inf, BF16, backward, arbitrary views and source-wrapper reshape/contiguous-copy behavior are outside this interface.

The standalone `logsoftmax_online.py --case CASE --output OUT --cache CACHE --provenance PROVENANCE` command is exploratory. It accepts one case object, not the whole matrix. Its generic pair harness does not perform the controlled all-case compile and correctness ordering used for the published acquisition. The four NONFROZEN CPU/mock tests exercise strict admission, real padded layouts, prototype rejection, atomic compile failure/retry, repeated arm orders, changed global float64/float16 defaults with restoration, canaries and an external-working-directory CLI with valid/invalid JSON. They also passed on the capable A3 target before NPU initialization.

The sole acquisition completed successfully. A separate stdlib-only reader verified and transferred the retained artifacts without importing kernels or rerunning math. Local verification independently reproduced normalized traces, operation spans and physical-device PMU joins. Cleanup found no live source-path processes, a healthy idle device with empty process query and zero ECC counters, and acquired/released the die0 lease and both export locks. The original process-group ID was not serialized; process-group-wide evidence is not claimed.
