# GDN local dV: score-consumer fusion

This study compares a two-kernel materialization baseline with the source fused local-dV kernel. It measures one dense prepared component of GDN backward, not the complete gradient pipeline or public autograd API. The upstream caller consumes this local result in later state-backward work.

The complete cases are in [P10-result.json](../experiments/optimization-wave011/P10-result.json). Raw task rows and PMUs are in [P10-profile-details.json.gz](../experiments/optimization-wave011/P10-profile-details.json.gz). The frozen numerical contract is [P10.json](../experiments/optimization-wave011/P10.json); the target-selection receipt is [P10-host.json](../experiments/optimization-wave011/P10-host.json).

| Case | Prospective regime | Summed device-task time µs, before → after | Task-time speedup | Inter-task gap µs, before → after | Profiled operation span µs, before → after | Span speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W011-P10-01 | cost | 12.199775 → 12.415775 | 0.983× | 83.970100 → 0.000000 | 96.169875 → 12.415775 | 7.746× |
| W011-P10-02 | cost | 15.835700 → 12.821750 | 1.235× | 75.262325 → 0.000000 | 91.098025 → 12.821750 | 7.105× |
| W011-P10-03 | benefit | 14.619725 → 12.639775 | 1.157× | 81.908450 → 0.000000 | 96.528175 → 12.639775 | 7.637× |
| W011-P10-04 | benefit | 37.623250 → 39.763175 | 0.946× | 60.364675 → 0.000000 | 97.987925 → 39.763175 | 2.464× |

Every span mean is lower, including both benefit cases. Device-task time is mixed: benefit03 improves 1.157×, while benefit04 is about 5.7% slower. The smallest cost case is also about 1.8% slower in task sum. The per-study span geomean is 5.673× and task-sum geomean is 1.074×. These are descriptive means of the same ten retained samples, not significance claims. Large span ratios mainly remove between-kernel profiling/submission gaps; they are not ordinary application speedups. No UB-pressure, cache-traffic, occupancy or roofline cause is inferred from these timings.

## Kernel meanings and source

[The generated kernels](../examples/optimization-library/triton/gdn_local_dv/gdn_local_dv_generated.py) contain three functions:

- `gdn_local_dv_scores` is an educational source-derived baseline producer. It computes K @ Qᵀ, applies the upper-inclusive causal mask and both token-valid masks, ordinary exp(g_column − g_row), then scale, and casts scores to FP16. It writes every padded scratch cell, including masked zeros.
- `gdn_local_dv_consume` is the baseline consumer. It loads those FP16 scores, multiplies by FP16 dO with FP32 accumulation, then stores FP16 local dV. Both baseline stages execute in every repeated prepared API call.
- `gdn_local_dv_fused` is the original selected source JIT body, with only a function rename and explicit heuristic constants. It performs both steps in one kernel. Its score cast remains **before** the second dot; the gate is ordinary exp, not the forward component’s safe_exp.

The original source, public caller, helper and full supplied license remain in [vendor/](../examples/optimization-library/triton/gdn_local_dv/vendor). MindSpeed-Ops pin: `4efe2b675c2d51517c3fdd14c721020b8d41e241`; source citation `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-mindspeed-scope-4efe2b675c2d#mindspeed_ops/arch32/triton/gdn/chunk_bwd_dv_local.py`. This is an application of known fusion, with no new-mechanism claim. A standalone upstream NPU test was not established; this study provides new bounded component qualification.

Both arms use the same explicit four warps, two stages and disabled FP fusion. These are prospective extraction choices: the original wrapper does not specify warps or stages, so source-default parity is not claimed. BK=BV=128 and the source wrapper’s BT=min(64,max(16,nextpow2(T))) are preserved. Dense head-major g, device copies, private allocation and scratch preparation are common setup outside the timed calls; all score and gradient arithmetic is charged. No metadata, zeroing or materialization work is omitted from a repeated call.

## Verification and profiler evidence

All twelve input-only case/arm compile screens completed before six scientific CPU references were generated once. Actual FP16-dot/FP32-accumulator and zero-padded block loads were checked before math, including K=64 with BK=128. All twelve correctness calls then passed before the eighty profiled API calls and eight captures. Guards were not profiled. The independent CPU oracle preserves typed FP32 exponential/product/scale boundaries and the FP16 score cast. Combined tolerances are atol=2e-4, rtol=2e-3 against both the reference and paired arm; all six paired bit checks also happened to pass. This is not a universal bit-identity claim.

The retained captures account for 120 device tasks: forty AI_CORE and eighty MIX_AIC. All ten samples per arm are retained, including first samples; all raw task rows/CSV metrics remain linked. Pipe ratios remain per-task measurements, not whole-device utilization or measured HBM traffic. Code-derived avoided scratch bytes are an algorithmic hypothesis, not a measured byte counter.

The [adapter](../examples/optimization-library/triton/gdn_local_dv/gdn_local_dv.py) validates the bounded dense domain before backend work. It guards compilation, launch, validation and the entire capture—including events, synchronization and analysis—on the prepared device, with explicit PipeUtilization. Prepared close is idempotent and releases only its own references. Five functional CPU/mock tests cover the independent scalar oracle, positive gate differences, unsupported input rejection, real generic pair cleanup, device restoration, metric selection, probe errors and actual evidence-consumer binary bindings.

Target: gz-a3-2, registered py312-torch runtime, physical 0/logical 0, observed NPU9362/Ascend910V1. Runtime/compiler identities and actual core properties are recorded in the result. No BZ execution occurred. The device was released healthy, with no owned live processes and both canonical export locks free; PGID0 was explicitly not treated as process ownership evidence.

A first host-only bundle attempt omitted an existing trace fixture and failed its CPU test before real backend import, compilation, scientific reference generation or NPU calls. Its source/output and release proof remain in the compact archive. The corrected bundle added only the missing existing fixture files and ran the same frozen kernel, provider, cases and tolerances. No scientific work was replayed.

The independently verified compact archive contains 387 hash-bound files, 794 vendor-file hashes, all six actual cached reference payloads, source/compiler/device artifacts, raw traces and CSVs. Archive SHA256: `233cbc0a97d3fc476a2eb017ff91c7fa46860ed449531f02808b0d4b20c79ea1`. No warmups, autotuning, extra timing, input/output inspection during profiles, or new scientific calls were used for readout or publication.
