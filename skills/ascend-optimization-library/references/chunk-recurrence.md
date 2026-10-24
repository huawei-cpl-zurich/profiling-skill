# Chunk a gated delta recurrence

Status: all four A3 pairs passed one correctness check and ten profiled forwards
per arm after tiling value columns. The long gallery case improved its device
span; the short case regressed substantially with preparation and scan overhead.
These paired observations establish no universal crossover or roofline distance.

Read this recipe when code updates a large state for every token, there are few
independent heads/segments, and successive state updates form a long serial
path. The [upstream chunk state update](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/fla/chunk_delta_h.py:L157-L171)
motivates moving work into matrix operations. The
[operator context](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-mindspeed-scope-4efe2b675c2d#docs/zh/triton/chunk_gated_delta_rule.md:L1-L10)
covers chunk/packed/final-state semantics. Our paired kernels and algebra are
independent implementations, not a claim that these upstream kernels passed
this experiment.

The executable pair is [recurrence.py](../../../examples/optimization-library/triton/recurrence.py),
with a separate sequential CPU oracle in
[recurrence_contract.py](../../../examples/optimization-library/triton/recurrence_contract.py).
Select `recurrence:factory` with that example directory on `PYTHONPATH`.
A case specifies `t,hk,hv,dk,dv,seed,segments`; `segments` contains boundaries
starting at zero and ending at T. For example:

```json
{"t":4096,"hk":1,"hv":4,"dk":128,"dv":128,"seed":7501,"segments":[0,2048,4096]}
```

The narrow example accepts contiguous FP16 Q/K/V, FP32 beta/gates/initial state,
DK/DV 64 or 128, T ≤4096 and Hv ≤4 with Hv divisible by Hk. Value head h uses
key/query head `h // (Hv/Hk)`. K rows are normalized, beta is finite in [0,.2],
and log gates are finite in [-.01,0]. Outputs are FP16 token values and FP32
final state for every segment/head. Both outputs use rtol/atol .02 against the
same independent FP32 sequential oracle. These fixture restrictions do not
establish a general GDN dispatch envelope.

## Code pattern and transformation

Before, each token depends on a full state update:

```python
state = exp(g) * state
error = beta * (v - k @ state)
state = state + k[:, None] * error[None, :]
out = q @ state
```

After, prepare 32-token chunks independently. Let c be the cumulative log gate,
`A[i,j] = beta[i] * exp(c[i]-c[j]) * dot(k[i],k[j])` for j<i, and M=(I+A)^-1.
Prepare `u=M@(beta*v)` and `w=M@(beta*exp(c)*k)`. The state scan computes
`delta=u-w@state`, all chunk outputs, and one boundary-state update. Both arms
partition independent value columns into tiles of 32, keeping the full key
dimension in each program. DV64/128 therefore uses two/four value tiles. The strict
lower matrix is nilpotent; the example uses a finite doubling factorization.
The CPU tests verify the blocked equations against the token recurrence and
verify that value-column partitions produce the same output and final state.
The original untiled DK=DV128 token kernel exceeded A3 UB capacity
(2,115,584 bits required versus 1,572,864 available); it was rejected before
launch. The tiled source passed compile-only probes for all three kernels.
This resource result does not establish runtime correctness.

Both arms keep state in FP32. The chunk arm casts matrix operands, including
state operands for dots, to FP16 with FP32 accumulation. This changes rounding
and reduction order. Numerical margin is a real tradeoff to measure; a failed
output or final-state check rejects the candidate rather than defining a
successful adverse example.

## Paired regimes and profiler interpretation

Compare T4096, two segments, Hk1/Hv4 and DK=DV128 with a short T17 case at the
same head/state dimensions. Long segments may amortize chunk preparation and
replace many vector state operations with Cube work. Short segments pay for
preparation, inverse construction, extra launches and partial-chunk work.
The after arm also needs FP32 workspace approximately
`4 * chunks * Hv * 32 * (DK + DV + 1)` bytes, plus small index tables.

The full after call includes both `wave001_delta_prepare` and
`wave001_delta_chunk_scan`; the before call is `wave001_delta_token`. Require
same-capture attribution of both after kernels in each of ten forwards. Compare
complete-call timing and per-kernel duration. A shorter state scan can be
outweighed by preparation. Inspect actual buffer/spill diagnostics before
attributing a regression to capacity; inspect supported vector/Cube activity
before describing where execution time moved. Activity ratios alone establish
neither throughput saturation nor a distance to roofline.

An actor should preserve gate order, grouped-head mapping, segment resets,
nonzero initial state and final-state output, then guard the demonstrated
shape/layout/numerical envelope. An analyzer should test whether preparation
cost dominates, how many independent chunk tasks exist, and whether numerical
margin shrinks. A falsifier for the proposed long-sequence gain is a complete
paired capture whose preparation plus scan costs equal or exceed the token
scan beyond sample variability. Missing metrics or indistinguishable timings
remain inconclusive.
