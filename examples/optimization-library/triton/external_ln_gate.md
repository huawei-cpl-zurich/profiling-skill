# External-gate LayerNorm fusion

This is a new operator application of producer/consumer fusion, with zero new
unique-mechanism claim. It independently adapts the native ops-rec `ln_mul`
code format to a small Triton demonstration. It is not a claim about tuned
native operator performance.

The inputs are contiguous FP32 `x,U[A,R]` and `gamma,beta[R]`; U must have the
full input shape. The frozen fixtures use x/U in [-1,1], gamma in [.5,1.5] and
beta in [-.25,.25]. Epsilon is 1e-5. Both sides compute centered variance and
return FP32 y with the same shape. Inputs remain immutable and y is separate.

```python
centered = x - mean(x, axis=-1)
normalized = centered * rsqrt(mean(centered * centered, axis=-1) + 1e-5)
y = (normalized * gamma + beta) * U
```

The baseline writes `normalized * gamma + beta` to a global temporary and
launches a second multiplication kernel. The candidate loads U and multiplies
it after the affine operation inside the same normalization kernel. Both keep
one program per row, the same reduction/rounding and disabled FMA. U never
enters the statistics: moving it before mean/variance changes the operation.
Padded lanes are masked after subtracting the mean when computing variance.

A source cue is a full LayerNorm result whose only consumer is elementwise U
multiplication. A profiler cue is the corresponding two-task sequence and
interkernel gap, with intermediate movement as a hypothesis. Fusion removes
a launch and the logical 4*A*R-byte temporary. It can increase gate loads and
live state in the reduction kernel, which may change UB/register pressure or
occupancy. Pipe percentages do not establish physical traffic, bandwidth or
whole-device utilization. Preallocated buffers do not establish allocation or
peak-memory savings. A measured gain must include the entire operation.

The guard with zero, negative and nonuniform U and nonzero beta checks the
placement after affine; width17 checks tail masking. A mismatched-U guard is
an explicit host rejection for each arm before any device preparation. The
public fixture factory rejects it before cache writes; it is not assigned a
fictional numerical oracle. The four measured cases each have one correctness
check per side against an independent CPU FP64 oracle (atol=rtol=1e-4), followed
by exactly ten full API forwards under the profiler with no validation,
warmups, autotuning or extra timing. Both guards have zero captures.

The five numerical fixture payloads are generated once. Input/output/workspace
allocation and compilation occur outside the forward; every required LN and
gate write is inside it. Full profiled device-operation spans include all
stages and gaps. They are diagnostic and are distinct from ordinary application
latency. Active union is retained separately; the noncompute remainder is not
whole-device idle. Roofline distance is not estimated without supported traffic
and capacity denominators.

Sources are ops-rec commit `e489c0bf6a2d5f2d05b3205fcacc01b62a908b10`,
Apache-2.0. The native body and real NPU example supply code and semantic
reference; their presence alone does not establish this implementation's
correctness or speed. Native epsilon1e-5 is independently present in host tiling.

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/activation/ln_mul/op_kernel/ln_mul_kernel.h:L217-L330`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/activation/ln_mul/docs/torch_ln_mul.md:L11-L46`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/activation/ln_mul/examples/torch/test_torch_ln_mul.py:L25-L60`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/activation/ln_mul/op_host/ln_mul.cpp:L69-L74`

The adapter is `external_ln_gate:Adapter`; the immutable prospective matrix is
[wave005 B02](../../../experiments/optimization-wave005/B02.json). The companion
fixture CLI accepts nested case JSON: `external_ln_contract.py --case CASE
--output FIXTURE`. Use the repository's `scripts/optimization_pair.py` and the
configured remote/profiling workflow for device collection. Unsupported U
shapes intentionally exit with ValueError and zero calls.

## Retained results

The selected route reported Ascend910B4-1, physical device 6 and application logical 0. Ten numerical device correctness checks and two host mismatch rejections passed. Five independent FP64 scientific oracle payloads were generated once. The four measured pairs contain eight captures, eighty profiled forwards and 120 compute-task PMU rows; both guards have zero captures. Ten CPU functional tests passed locally and on the A3 runtime, including actual public CLI rejection before fixture-cache writes. No retries, failed attempts, or scientific replays occurred.

| Case | Two-stage span, µs (mean ± sample SD) | Fused span, µs (mean ± sample SD) | Before/after ratio |
|---|---:|---:|---:|
| W005-B02-01 | 151.322 ± 2.633 | 42.906 ± 0.326 | 3.5268× |
| W005-B02-02 | 156.356 ± 21.725 | 1.956 ± 0.154 | 79.9366× |
| W005-B02-03 | 160.554 ± 11.673 | 37.356 ± 0.291 | 4.2979× |
| W005-B02-04 | 140.160 ± 14.401 | 1.710 ± 0.062 | 81.9649× |

The geometric mean span ratio is **17.7523×**. All four prospective cases favor fusion; the cost labels did not yield regressions. These are profiler-active diagnostic spans. In particular, tiny-case ratios are dominated by the baseline's interkernel/caller gap and cannot be interpreted as ordinary application speedups or traffic ratios. Every sample, including the slower samples, is retained.

| Case | Baseline active union, µs | Fused active union, µs | Baseline noncompute remainder, µs |
|---|---:|---:|---:|
| W005-B02-01 | 52.288 | 42.906 | 99.034 |
| W005-B02-02 | 3.358 | 1.956 | 152.998 |
| W005-B02-03 | 41.152 | 37.356 | 119.402 |
| W005-B02-04 | 2.916 | 1.710 | 137.244 |

A publication-only correction changes the source copyright year to match the retrieved upstream 2026 notice. The original measured adapter source is retained in the private archive. The complete Python AST is identical and the fixture provider is unchanged; old/new source digests and the common AST digest are recorded. No scientific oracle, correctness run or profile was repeated for this comment change.

[Readable summary](../../../skills/ascend-optimization-library/references/external-ln-gate-summary.json) and [complete compact evidence](../../../skills/ascend-optimization-library/references/external-ln-gate-evidence.json.gz) retain all span/event samples, source/binary/fixture bindings, task PMU fields and guards. The private archive contains 233 verified payload files plus its manifest, 8,701,054bytes, SHA256 `895ea7b552ca648dc907a988fb08cffa53d0aba9baf893f2450feebc008f5d27`. The full vendor trees remain in the retained remote run.
