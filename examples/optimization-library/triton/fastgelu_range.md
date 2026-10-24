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

## Retained result

All twelve correctness checks and ten CPU functional tests passed on the configured A3 route, observed Ascend 910B4-1 physical device 5. All twelve arm/shape preparations compiled before scientific forwards, and compiled identities stayed unchanged. Four measured pairs contain eight captures, eighty profiled API forwards and eighty physical compute tasks. Both guards have zero captures. No failed attempts or measurement retries occurred.

| Case | Direct span, µs (mean ± sample SD) | Sign-aware span, µs (mean ± sample SD) | Direct/sign-aware ratio |
|---|---:|---:|---:|
| W005-A01-01 | 4.374 ± 0.967 | 4.574 ± 0.109 | 0.9563× |
| W005-A01-02 | 1.718 ± 0.164 | 1.794 ± 0.214 | 0.9576× |
| W005-A01-03 | 4.686 ± 0.233 | 5.484 ± 0.193 | 0.8545× |
| W005-A01-04 | 1.664 ± 0.122 | 1.722 ± 0.153 | 0.9663× |

The geometric mean ratio is **0.9325×**. All four means favor the direct form; small-case differences overlap the observed dispersion. The sign-aware form improves numerical tail retention while adding arithmetic; these captures do not establish a timing gain. Every sample, including the first direct-case 7.08 µs sample, is retained.

| Tail case | Reference nonzero tails | Direct nonzero / subnormal | Sign-aware nonzero / subnormal | Direct max tail absolute error | Sign-aware max tail absolute error |
|---|---:|---:|---:|---:|---:|
| W005-A01-01 | 65536 | 23565 / 0 | 52913 / 24381 | 1.53105e-37 | 9.15093e-38 |
| W005-A01-03 | 98304 | 8430 / 0 | 43508 / 29118 | 1.53164e-37 | 2.07376e-41 |
| W005-A01-GB | 8 | 2 / 0 | 5 / 2 | 1.35843e-37 | 1.41636e-41 |

The two negative-only cases include 28,531 and 14,390 reference values in the FP32-normal magnitude range. Tail preservation is incomplete: the sign-aware outputs still contain 12,623 and 54,796 negative zeros, respectively. Expected positive-overflow exponential arguments in the direct source number 41,971 and 89,874; this arithmetic count is not a device intermediate observation. The near-boundary guard retains 5 of 8 nonzero tails after versus 2 before; positive/negative zero guard observations match numerically without requiring sign-bit equality.

[Readable summary](../../../skills/ascend-optimization-library/references/fastgelu-range-summary.json) and [complete compact evidence](../../../skills/ascend-optimization-library/references/fastgelu-range-evidence.json.gz) retain every span/event sample, task PMU field, source/binary identity, range diagnostic and guard result. The private archive contains 210 verified members, 2,096,481 bytes, SHA256 `e81c36598f8685aeba42f8ab9068a5e6a57f120d320e33dac51159c289800ea9`, including all six independent oracle payloads and compiled artifacts. Full vendor reports remain in the retained remote run.
