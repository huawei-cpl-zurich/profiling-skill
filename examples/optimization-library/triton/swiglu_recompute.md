# SwiGLU activation rematerialization

This study applies the existing save/recompute tradeoff to a complete SwiGLU forward-plus-backward API. Inputs `a`, `b`, and upstream gradient `dc` are contiguous immutable FP32 matrices in `[-4,4]`. The API returns three separate FP32 matrices:

```python
s = sigmoid(a)
y = (a * s) * b
da = (dc * b) * (s + (a * s) * (1 - s))
db = (dc * a) * s
```

`before` writes `s` during its forward kernel and reads it during backward. `after` computes the same forward result without saving `s`, then recomputes sigmoid during backward. Both arms use the same two kernel boundaries, block size, FP32 arithmetic association, and disabled contraction. Required input-dependent work occurs on every API invocation. The saved sigmoid contains `4*R*D` logical bytes before and zero after. This count establishes neither peak allocated memory nor allocation latency: inputs, outputs, and workspaces are preallocated, with both arms prepared concurrently.

An agent can consider this transformation when backward retains a large activation that can be reconstructed from inputs already needed for gradients. The competing cost is repeated exponential/division work and additional live temporary state. Inspect task counts and vector/MTE pipeline fields together with the complete device-operation span; task busy ratios do not establish physical traffic or roofline distance. This is an operator application of a known optimization, not a newly discovered mechanism.

The independent CPU FP64 oracle uses the represented FP32 inputs. All three outputs must be finite and satisfy `atol=2e-5, rtol=2e-4`; input bytes and output nonaliasing are checked once. A zero/symmetric guard exercises the derivative at zero. The odd-width guard actually aliases `a` and `b` on the device, while returning the two formal partial derivatives separately. A caller differentiating through the alias must sum these partials. The harness's independent CPU clones do not preserve cross-tensor aliases, so preparation reconstructs the explicit alias contract.

The fixed matrix is [T01.json](../../../experiments/optimization-wave005/T01.json). Each arm receives one correctness execution per case; the four measured pairs then receive exactly ten profiled API executions with no input/output inspection. Both guards receive zero captures. There are no warmups, autotuning trials, or additional latency runs. Profiled spans include both kernels and their intervening gap and are instrumented diagnostics, not ordinary application latency.

Source: `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/activation/swiglu.py:L25-L157`, pinned commit `7acb9c40a6ca827e885d902b7eaee5d45457b230`, MIT repository license, LinkedIn/Huawei notices retained in the implementation. The source's in-place backward is adapted to separate immutable outputs; MLP GEMMs are outside this study. Source presence is not target compatibility evidence.

## Retained result

All twelve correctness checks passed on Ascend 910B4-1, physical device 4, using the configured A3 route. The four pairs contain eight captures and eighty profiled API forwards. Both guards made zero capture calls. All twelve arm/shape preparations compiled before correctness or profiling; compilation identities stayed unchanged during captures. Each API contains two physical compute tasks, giving 160 profiled compute tasks total.

| Case | Save span, µs (mean ± sample SD) | Recompute span, µs (mean ± sample SD) | Save/recompute ratio |
|---|---:|---:|---:|
| W005-T01-01 | 349.390 ± 15.982 | 359.266 ± 10.656 | 0.9725× |
| W005-T01-02 | 183.382 ± 18.705 | 185.094 ± 15.192 | 0.9908× |
| W005-T01-03 | 273.652 ± 20.101 | 272.898 ± 20.583 | 1.0028× |
| W005-T01-04 | 180.302 ± 12.707 | 197.190 ± 60.969 | 0.9144× |

Geometric mean of the four ratios: **0.9695×**. The saved-state removal did not establish a latency benefit in these captures. The tiny change in case 03 is well within observed dispersion; case 04 includes a retained 363.82 µs recompute sample. No samples were discarded or runs repeated. Saving sigmoid uses 16,777,216 / 512 / 9,437,184 / 2,048 logical bytes in the respective cases; recomputation saves none.

[Readable summary](../../../skills/ascend-optimization-library/references/swiglu-recompute-summary.json) and [complete compact evidence](../../../skills/ascend-optimization-library/references/swiglu-recompute-evidence.json.gz) retain ten samples per arm, per-task PMU fields, source/binary identity, mathematical diagnostics, guard alias checks, and oracle manifests. Full reports, compiled artifacts, and six CPU oracle payloads remain in the retained run. The transferred archive is SHA256 `cbb5405572eb1cec53f0075b97008d6331c8280d954a560e8f7595ed6e5df52d` (285 verified files). The nine CPU functional tests also passed on the A3 host with zero additional NPU forwards.
