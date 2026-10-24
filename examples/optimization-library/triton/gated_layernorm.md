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
