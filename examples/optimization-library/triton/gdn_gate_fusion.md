# Fuse the two GDN gate outputs

This is a standalone forward gate calculation, not the GDN recurrence or a
vLLM benchmark. It applies an existing fusion technique to a new operator
application; it does not add another unique fusion mechanism.

The pair accepts contiguous FP32 `a,b[R,H]`, `A_log,dt_bias[H]` and returns
separate FP32 `g,beta[1,R,H]`, preserving every input:

```python
x = a + dt_bias
s = where(x > 20, x, log(1 + exp(x)))
g = -exp(A_log) * s
beta = 1 / (1 + exp(-b))
```

Before uses four kernels: bias addition, softplus, exponent/scale, and beta
sigmoid. After computes both outputs in one kernel. Both use the same flat
256-element grid, FP32 arithmetic association, disabled FMA fusion and
per-output evaluation of `exp(A_log)`. No exponent-hoisting or scheduling
change is credited to fusion. Both required intermediate writes occur inside
the before API; every after calculation occurs inside the after API.

Look for independently launched gate stages and full intermediates with one
consumer. Fewer launches and intermediate movements may help. Coupled live
exponential/log state, UB pressure, or an already-fused backend can erase the
benefit. Pipeline percentages alone do not prove physical traffic savings.
The two eliminated FP32 intermediates occupy `8*R*H` logical bytes; preparation
keeps buffers alive for both arms, so this is not a measured peak-allocation
reduction.

Normal inputs are bounded: `A_log[-2,2]`, `a,b[-4,4]`, and `dt_bias[-1,1]`.
The correctness-only threshold guard uses finite `20 ± 1/1024` and `20`;
`where` does not imply lazy exponential evaluation or arbitrary-range safety.
The odd `R3,H17` guard checks tail masks and sigmoid symmetry. Independent CPU
FP64 outputs use represented FP32 inputs with `atol=2e-5, rtol=2e-4`.

Source: triton-ascend-kernels commit
`7acb9c40a6ca827e885d902b7eaee5d45457b230`, MIT with Huawei/vLLM notices;
[license](GDN_GATE_MIT_LICENSE). Exact native source body/interface/test:

- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/moe/fused_gdn_gating.py:L12-L155`
- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#tests/moe/test_fused_gdn_gating.py:L10-L53`

This independent pair deliberately does not transplant the upstream hardcoded
UB-size or core scheduler. Runtime eligibility is established on the recorded
910B4-1/CANN9.1 target; it is not a claim about A5 or newer source toolchains.

The frozen matrix is `experiments/optimization-wave005/T05.json`. Each of six
CPU oracle payloads is cached once. Each arm is checked once, then each main
case has exactly ten profiled complete API forwards without validation,
autotuning, warmup or extra timing. Guards have no captures. Preallocated
inputs/outputs/workspace are prepared outside the forward. The complete
profiled device-operation span includes all kernels and interkernel gaps;
it is diagnostic, not ordinary application latency. Active union and the
noncompute remainder are separate; the latter is not whole-device idle time.
Roofline distance is not estimated without supported capacity and traffic
denominators. All ten samples and original PMU fields are retained.
