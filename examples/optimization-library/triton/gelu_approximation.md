# Explicitly permitted GELU approximation

The pair compares the erf definition of GELU with the documented tanh/cubic
approximation. Approximation is a public permission, not an algebraic rewrite:
`allow_approx=False` makes both arms use the same erf specialization. Inputs
are contiguous finite FP32 vectors within [-8,8], with separate FP32 outputs
and immutable inputs. Unsupported ranges and malformed permission values are
rejected before launch.

```python
if allow_approx:
    z = sqrt(2 / pi) * (x + 0.044715 * x * x * x)
    y = (0.5 * x) * (1 + tanh(z))
else:
    y = (0.5 * x) * (1 + erf(x / sqrt(2)))
```

The approximation must meet a fixed absolute budget of 0.0006 against an
independent FP64 erf reference, plus a separate FP32 arithmetic allowance of
0.00002: total absolute gate 0.00062 with zero relative tolerance. The erf
path uses the tighter FP32 gate 0.00002 + 0.0002 × abs(reference). A fixed
160,001-point FP64 grid on [-8,8] checks the approximation before device
work; this empirical grid is not a proof for every possible input.

Apply this pattern when activation math is material and the public numerical
contract permits approximation. In code, look for expensive erf evaluation
with no requirement for the exact mode. In the profile and compiler artifacts,
compare the actual pointwise kernel, vector activity, resources and lowering;
the backend may already implement these functions efficiently. Tanh and cubic
arithmetic can cost more, and the approximation introduces a real error even
when it is faster. It must not silently replace an exact-erf request.

Both arms use the same 256-element tile and nonfused FP32 operation ordering.
Input/output allocation happens before the API; each forward reads the input
and writes every output. A tiny compile-only capability screen precedes the
six scientific CPU oracle payloads; all twelve case/arm preparations compile
before any full API forward. Each side then receives one correctness check
and exactly ten profiled forwards for each measured case. The exact-mode and
endpoint/near-zero guards receive correctness checks and zero captures. There
are no warmups, autotuning, extra timing runs or tensor checks during profiling.

Source formula and arithmetic evidence: ops-nn commit
`7ed342d28b44b6291aceaf747796028c48d7304c`,
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-nn#activation/gelu/README.md:L15-L24`
and `#activation/gelu/op_kernel/arch35/gelu_dag.h:L27-L78`.
The registration/example context is retained in the filtered research receipts.
This is an independent Triton formula adaptation for the observed target;
upstream arch35 device code is not compiled and is not evidence of native
910B support. The [source license](GELU_CANN_LICENSE) and attribution are
retained. Source presence does not establish runtime correctness or speed.

The primary measurement is a profiler-active complete device-operation span,
not ordinary application latency. Instrumented caller events are retained
separately. PMU percentages do not establish physical traffic, bandwidth,
theoretical utilization or a roofline distance. Missing denominators are
unavailable, not zero. All ten samples and their dispersion must be retained,
including unfavorable or noisy outcomes.
