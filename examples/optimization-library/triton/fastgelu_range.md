# FastGELU exponential range rewrite

The direct and sign-aware expressions are algebraically equivalent over real numbers:

```python
# before
x / (1 + exp(-a*x))
# after
(x / (1 + exp(-a*abs(x)))) * exp(half*(x-abs(x)))
```

Here `a=FP32(1.702)` and `half=FP32(0.851)` satisfy `a == 2*half`. Each arm is one independent Triton kernel with the same 256-element program geometry, explicit FP32 outputs, masked tails and disabled floating-point contraction. Inputs are contiguous, immutable, finite FP32 vectors in `[-80,80]`. This is approximate FastGELU, with no exact-GELU substitution claim. The kernel retains the specified arithmetic association and recomputes all input-dependent work on every invocation.

Consider this rewrite when the direct exponential can overflow even though its final result is finite. The sign-aware form keeps exponential arguments nonpositive, at the cost of a second exponential, an absolute value and more live arithmetic. Hardware exponential lowering may flush small values to zero, including intermediates that a later multiplication could have brought into the normal range. Avoid assuming that algebraic range safety preserves negative tails or improves speed.

The CPU FP64 oracle evaluates the direct formula using the represented FP32 coefficient and inputs. Both outputs must be finite and satisfy `atol=2e-5, rtol=2e-4`; this gate alone says little about tiny negative tails. The sole correctness execution also records tail absolute error, nonzero and subnormal counts, signed zeros, and how many reference tails would have FP32-normal magnitudes. Source exponent-range and potential-overflow counts are calculated independently on the host; they are expected arithmetic diagnostics, not instrumented device intermediate values. Signed zero is observed without a bitwise sign requirement. Input immutability and output nonaliasing are checked once.

[The immutable matrix](../../../experiments/optimization-wave005/A01.json) fixes four measured pairs and two correctness-only guards before device work. Six CPU oracle payloads are cached once. Each arm receives one correctness execution per case, followed by exactly ten profiled full API executions for each main case. Guards have zero captures. No warmups, autotuning, extra timing or profiled input/output inspection occur. A profiled device-operation span is a diagnostic boundary, not ordinary application latency. PMU pipe ratios do not establish physical traffic, bandwidth or roofline distance.

Source arithmetic is pinned to ops-nn commit `7ed342d28b44b6291aceaf747796028c48d7304c`: `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-nn#activation/fast_gelu/op_kernel/arch35/fast_gelu_dag.h:L25-L72` provides the direct expression; `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-nn#activation/fast_gelu/README.md:L15-L21` provides the sign-aware expression. The [CANN Open Software License 2.0](FASTGELU_CANN_LICENSE) and Huawei attribution are retained. This is an independent Triton adaptation. The retrieved device body targets arch35, and newer source registrations do not prove native 910B support; this study qualifies its own implementation on the observed target. It is a source-backed implementation study, with no automatic new unique-technique count.
