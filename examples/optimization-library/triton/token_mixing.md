# Residual token mixing: add inside LayerNorm

Use this application of producer/consumer fusion when a residual sum is written
to global memory only to be read by full-row LayerNorm. The candidate removes
that sum's temporary and one launch. It still materializes the transpose on
every call, exactly as the baseline does.

The standalone interface receives contiguous FP32 `x[B,S,S]`, `gamma[S]` and
`beta[S]`, and returns `LayerNorm(x + transpose(x)) * gamma + beta` over the
last axis with epsilon `1e-5`. Inputs are immutable and the output is separate.
Both sides round the residual addition in FP32, use the same centered variance
reduction, row grid and affine order, and disable floating-point fusion.
The independent CPU oracle evaluates the mathematical expression in FP64 on
represented inputs. Its fixed gate is `atol=rtol=1e-4`.

```python
# Before: all three operations belong to the measured API.
xt = transpose_kernel(x)
z = add_kernel(x, xt)
y = layernorm_kernel(z, gamma, beta)
# After: the same transpose remains inside the API.
xt = transpose_kernel(x)
y = residual_layernorm_kernel(x, xt, gamma, beta)
```

Read `token_mixing.py` for executable formats. The baseline is a controlled
separate-stage contrast, not the best native or vendor implementation. Fusion
adds another live input tile to the normalization kernel and may increase
resource pressure. It does not remove transpose traffic or change the row
batch. A source pattern with a materialized residual is the recognition cue;
profiler evidence should show three device tasks becoming two, with the exact
transpose present in both. Inspect active union and full operation span rather
than attributing the entire change to memory traffic. Logical temporary bytes
fall by `4*B*S*S`; preallocated buffers cannot establish a peak-allocation gain.

The source is native Ascend C token-mixing code from ops-rec at
`e489c0bf6a2d5f2d05b3205fcacc01b62a908b10`, under Apache2.0. The native body adds
its two inputs in UB before normalization. This study independently expresses
that format in Triton with a y-only interface, rather than claiming native
operator performance or complete interface equivalence. Exact references:

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/mixing/token_mixing/op_kernel/token_mixing.cpp:L200-L310`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/mixing/token_mixing/docs/torch_token_mixing.md:L11-L65`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#cmake/soc_ops_config.cmake:L20-L36`

The frozen B01 matrix has four measured pairs and two correctness-only guards.
The nonsquare guard is a host rejection before fixture-cache creation. The odd
width constant/symmetric guard checks zero variance, transpose mapping and
nonzero bias. Five CPU scientific payloads are cached once; each valid side is
checked once and then executes exactly ten full API forwards under profiling.
No warmup, autotuning, extra timing or tensor inspection occurs in those ten
forwards. This is an operator application of known fusion, adding zero automatic
unique-technique claims.

The selected route is gz-a3 with observed 910B4-1 silicon; no A5/950 claim.
Profiled spans include interkernel/caller gaps and are diagnostic, not ordinary
application latency. Exported pipe percentages do not establish physical traffic,
whole-device utilization or roofline distance. The latter is not estimated.
