# F05 balanced CSR vector adaptation

All four fixed cases passed correctness, and all four became slower. These are negative results for this vector adaptation; the upstream Cube-assisted implementation was not measured. The benefit/cost labels below are prespecified hypotheses.

Before owns CSR rows; after balances 256 nonzeros per program, zeroes output, binary-searches original offsets, scans products and atomically merges row fragments. All stages are measured.

| Case | Hypothesis | Before span (us, mean ± SD) | After span (us, mean ± SD) | Before/after |
|---|---|---:|---:|---:|
| F05-01 | gallery-benefit | 518.602 ± 0.988 | 8188.996 ± 88.684 | 0.06333x |
| F05-02 | gallery-cost | 311.022 ± 0.418 | 7509.834 ± 84.952 | 0.04142x |
| F05-03 | heldout-benefit | 134.408 ± 0.629 | 3739.546 ± 60.916 | 0.03594x |
| F05-04 | heldout-cost | 612.410 ± 0.492 | 14526.794 ± 131.491 | 0.04216x |

Geometric mean before/after: **0.04465x**. Values below 1 mean a slowdown. No samples were removed.

Each pair checks each arm once against a cached FP64 oracle (atol=rtol=0.002), then profiles exactly 10 complete forwards per arm with no input/output inspection. Both correctness-only guards passed: empty CSR overwrites NaN output with zero; a 2048-element row selects the common vector fallback. Final counts:12 correctness forwards,8 captures,80 profile API forwards. Six scientific CPU fixtures were generated once and reused across source fixes.

Two earlier source versions failed AFTER compilation before any capture; each had one successful BEFORE device correctness forward. Their 4 attempted correctness calls, 2 actual device forwards and 0 captures are retained separately. The fixes expose the scan helper to the older Ascend JIT global lookup and use a fixed logarithmic search bound instead of a reduction-controlled loop. The final source then passed compile-only probes for all six shapes before execution. No source tuning followed timing.

Physical die 3 on gz-a3: Ascend910B4-1, 20 Cube/40 vector cores, Torch 2.10.0+cpu, imported Triton 3.2.0. Each forward has one device task before and two after. The per-lane binary searches, segmented scan and fragment atomics are plausible costs visible in the code; these data do not isolate one as the cause. Per-task PMU values, active unions, inter-kernel gaps, complete samples, compiler/binary hashes and correctness diagnostics are in [evidence.json](evidence.json). No whole-device utilization, physical bandwidth or roofline distance is inferred.

The primary metric is the profiled operation span on the device clock, including gaps between tasks. It excludes first enqueue/host overhead and is not unprofiled application latency. Instrumented caller event samples are retained separately.

Compact evidence archive SHA256: `ff1be1ea8712c5a93c00ee6b6b3a380a1107deb17cfa9ad346ff52e7e034082b`; 219629 bytes. Full vendor trees remain on the target. The compact archive preserves all three source versions, failed reports, final traces/PMU, oracle manifests, compile probes and matching NPU binaries.

Implementation and contract: [csr_balanced.md](../../../examples/optimization-library/triton/csr_balanced.md).
