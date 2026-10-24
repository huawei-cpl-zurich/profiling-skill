# Gated LayerNorm: fuse a gate around a row reduction

This study applies producer/consumer fusion to gated LayerNorm. It adds an
operator application, not a new optimization mechanism. It is a controlled
three-kernel baseline versus a one-kernel candidate, not a vendor-best claim.

Both arms accept contiguous FP32 `x,z[R,N]`, `gamma,beta[N]` and positive `eps`.
With `u = z * sigmoid(z)`, `gate_first=True` computes `LayerNorm(x*u)*gamma+beta`;
`False` computes `(LayerNorm(x)*gamma+beta)*u`. Both return `y`, row mean and
reciprocal standard deviation. These orders are observably different. Scope is
full-row LayerNorm, not grouped normalization or RMSNorm.

The baseline writes `u` and a second full FP32 intermediate; all gate,
normalization, affine and pointwise work executes in each forward. The candidate
keeps those intermediates local. Both use one program per row, the same centered
variance reduction, FP32 arithmetic association and launch options, with FMA
disabled. Padding is zeroed **after** centering before variance accumulation.
Inputs and output/workspace allocation are outside the measured forward.
Workspace logical bytes are `8*R*N` versus zero; both prepared arms stay alive,
so this is not an observed peak-memory reduction.

Source cues are a full-tensor gate intermediate adjacent to normalization and a
row reduction that can consume or produce the gate without changing its order.
Expected benefit is fewer launches and intermediate stores/loads. Larger live
state can increase register/UB demand or change scheduling. Inspect complete
operation spans and task counts, then per-task vector and MTE pipe composition.
Those exported PMU ratios alone do not measure physical traffic or whole-device
utilization, and do not establish a distance to roofline.

The independent CPU FP64 oracle checks all three outputs at
`abs(error) <= 5e-5 + 3e-4*abs(expected)`. Inputs remain unchanged and outputs must
be finite, FP32, correctly shaped, nonaliasing and overwrite NaN sentinels.
The constant-x/nonconstant-z guard distinguishes gate ordering; the N129 guard
checks centered-variance masking and row statistics. Frozen inputs/cases are in
[the study matrix](../../../experiments/optimization-wave005/T03.json).

Attribution: formulas and control order adapted from
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/norm/norm_gated.py:L78-L133`
and wrapper L136–L267, with semantic references in
`#tests/norm/test_norm_gated.py:L47-L103`, commit
`7acb9c40a6ca827e885d902b7eaee5d45457b230`.
The adaptation uses separate stages for the baseline and an independent
single-row layout for the candidate. Source notices for Huawei, vLLM, Tri Dao
and FLA remain in the kernel file; the source MIT permission notice is retained
in `GATED_LAYERNORM_LICENSE`. Source availability is not device-correctness proof.

The collection protocol generates six cached scientific CPU payloads once,
checks each arm once, then runs exactly ten profiled full forwards for each of
four pairs. The two guards have no captures. There are no warmups, extra timing
runs or input/output inspection in those profiled forwards. Profiled device
operation spans include interkernel gaps and are diagnostic instrumentation,
not ordinary application latency. All samples and compiler artifacts are retained.

## Retained 910B4-1 results

`gz-a3` physical5, logical0, CANN9.1.0, Torch/Torch-NPU2.10.0,
imported Triton3.2.0. Job completed without failures or retries. All12
correctness checks passed, including the two guards. The table gives mean ±
sample SD over ten **profiled complete device-operation spans**, in microseconds.
The active-union columns separate kernel work from the span's noncompute gaps.
Every pair executes three tasks per baseline forward versus one fused task.

| Case | R×N; gate first | Before span µs | After span µs | Span ratio | Before/after active union µs |
|---|---|---:|---:|---:|---:|
| W005-T03-01 | 1024×1024; False | 353.286 ± 42.936 | 44.794 ± 0.593 | 7.887 | 71.820 / 44.794 |
| W005-T03-02 | 1×64; True | 348.512 ± 37.984 | 1.988 ± 0.319 | 175.308 | 5.098 / 1.988 |
| W005-T03-03 | 768×768; True | 409.320 ± 32.560 | 36.778 ± 0.621 | 11.129 | 57.274 / 36.778 |
| W005-T03-04 | 2×128; False | 376.674 ± 25.111 | 1.810 ± 0.071 | 208.107 | 4.226 / 1.810 |

All four designated benefit/cost cases favored fusion in these instrumented spans;
the cost labels were prospective hypotheses and are not observed regressions.
The very large tiny-case ratios primarily reflect removal of interkernel/caller
dispatch gaps in this profiler-active execution. They are not established ordinary
application speedups. The single fused task's span ends with the task, while the
three-task baseline includes dispatch gaps between them. These data do not isolate
memory-traffic savings from launch aggregation or establish causality from PMU.

[Compact result data](../../../docs/data/optimization-w005-gated-layernorm.json)
retains all80 span samples, event samples, correctness diagnostics and binary/source
bindings. Its sibling `optimization-w005-gated-layernorm-rows.json.gz` preserves all
160 task-level PMU rows and operation tasks. The digest-bound private archive
retains original reports, all compiler/NPUBIN artifacts and scientific fixture
payloads. Collection handle: `remote:gz-a3:job:a2dda394-613d-4552-97f2-b010f366e0f6`.
