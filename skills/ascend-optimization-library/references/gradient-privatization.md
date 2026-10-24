# Private weight-gradient accumulation (E08)

Recognize this pattern when many independent normalization rows contribute to
the same weight-gradient vector. The before implementation performs one FP32
atomic add per row and column. Privatization accumulates multiple rows inside
each program, writes a separate FP32 partial vector, and then reduces those
complete vectors. This addresses shared-vector contention; it is not another
GEMM SplitK example.

The paired operator is the complete weighted RMS backward operation:

```python
y = x * rsqrt(mean(x*x, axis=-1, keepdims=True) + eps) * w
return vjp_x(y, dy), vjp_w(y, dy)
```

Inputs `x[R,D]`, `dy[R,D]` and `w[D]` and both gradients are FP32 contiguous
physical ND. Inputs are immutable; no forward output is required. Both arms
recompute inverse RMS in the same FP32 row arithmetic. There is no free cached
statistic. The independently generated CPU oracle uses FP64 autograd. Fixed
error limits are `atol=0.002`, `rtol=0.0002`; these permit FP32 reduction-order
differences, not missing rows or missing gradients.

The accepted epsilon range is `[2**-126, 0.1]`, inclusive. The lower bound is
the smallest normal positive FP32 value (`1.1754943508222875e-38`); the
existing upper bound remains 0.1. Subnormal or underflowing values are rejected
before fixture generation because device conversion or flushing to zero can
make zero-input rows produce nonfinite gradients. Nonfinite and overflowing
values are also rejected. This is a host contract restriction; it does not
change kernel arithmetic.

The baseline includes output initialization inside every measured call:

```python
gradient_zero[grid](dw, D, 256)
weighted_vjp[(R,)](x, w, dy, dx, dw, R, D, eps, BD, R, False)
```

Each row computes the same `dX` expression and atomically adds
`dy * (x * inverse_rms)` to `dW`. The after arm instead runs:

```python
weighted_vjp[(32,)](x, w, dy, dx, partial, R, D, eps, BD, 32, True)
gradient_reduce[column_tiles](partial, dw, D, 32, 64)
```

Each program owns rows `pid, pid+32, ...`, accumulates a private FP32 vector,
and writes all its partial elements. Programs with no rows write zeros. The
last kernel sums all 32 vectors. Reusing a buffer from an earlier call never
removes this required write. Output allocation is outside capture in both
arms; initialization, arithmetic and final reduction are inside it.

Expected improvement: fewer contended updates when many rows share a modest
weight vector. Expected costs: `32*D*4` bytes of FP32 workspace, its write/read
traffic, longer per-program loops and less row parallelism. Few rows and large
width may spend more work writing mostly empty partials than performing the
original atomics. Neither arm receives an autotuned or shape-selected fallback
in this controlled pair; a real operator should dispatch only after measuring
the intended domain.

Inspect the full timeline, including `gradient_zero`, `weighted_vjp` and
`gradient_reduce`. The after arm should remove atomics from its compiled
backward specialization and add the explicit final sum. Source inspection
establishes intent; supported compiled-code evidence is needed to claim an
instruction change. Report complete device-operation span and inter-kernel
gaps separately from active kernel time. Optional atomic/stall counters may
support the contention explanation when the target provides them. Busy ratios
alone establish neither useful occupancy nor physical traffic or roofline
distance.

Four predeclared pairs passed on Ascend910B4-1, with Torch2.10.0+cpu,
Torch-NPU2.10.0 and Triton3.2.0. Each arm received one complete correctness
check followed by ten captured forwards, with no warmups or measured
input/output inspection. The two benefit hypotheses and two cost hypotheses
remain labeled as declared; no cases were substituted after measurement.

| Case | R×D | Before, µs ± SD | After, µs ± SD | Span speedup |
| --- | ---: | ---: | ---: | ---: |
| Gallery benefit | 8192×128 | 698.486 ± 95.602 | 150.860 ± 18.196 | 4.6300× |
| Gallery cost | 1×2048 | 167.918 ± 11.861 | 145.802 ± 6.850 | 1.1517× |
| Heldout benefit | 6144×96 | 673.334 ± 60.882 | 149.176 ± 21.809 | 4.5137× |
| Heldout cost | 3×4096 | 168.398 ± 14.675 | 148.558 ± 14.048 | 1.1336× |

All-case span geometric mean is 2.2855×. These are complete profiled spans,
not unprofiled application timings; SDs describe ten samples within each
capture, not independent experiment replication. Both cost cases improved in
complete span, so a full-operation regression is **not demonstrated**.

Active kernel work tells a useful additional story. The benefit cases reduce
active union from 582.904→104.238µs and 621.964→101.274µs. The cost cases
increase active union from 5.188→6.976µs (+34.5%) and 4.816→7.668µs (+59.2%).
Their inter-kernel gaps fall from 162.730→138.826µs and
163.582→140.890µs, dominating the complete span. Consequently, their small
span gains do not establish less device work or an application-level win.
The after workspace is 16,384B, 262,144B, 12,288B and 524,288B respectively.

All 160 measured tasks join same-capture PMU records. Gallery-benefit
backward scalar ratio falls 0.7220→0.3623 and vector ratio rises
0.0783→0.5874. Heldout-benefit backward MTE3 ratio falls 0.8142→0.3416.
These observations are consistent with the intended transformation, but
these counters do not directly measure contention stalls or physical bytes.
Every retained before backward TTIR contains `tt.atomic_rmw fadd`, lowered
to a ttadapter store with `atomic = <add>`; every after specialization lacks
those IR operations. This establishes an observed IR change, not a final
disassembly instruction count. All 16 distinct NPUBIN snapshots were
independently hash-verified and compiled identities stayed stable.

CPU interpreter checks additionally cover width tails, fewer/more rows than
private programs, zero-input rows, immutable inputs and poisoned workspace
reuse. The device evidence qualifies the four concrete cases, including
4096-wide input; it does not establish all resources across the full bounded
contract. Keep negative or unsupported outcomes when extending the matrix.

Source provenance: pinned Triton-Ascend snapshot `7acb9c40a6ca`,
`norm/rms_norm.py:135-191,295-359,440-488` (original broker query
`14ae662489054f7a92bdbadde7b3a21e`). Its backward kernel accumulates private
FP32 `dW`; the wrapper reduces `_dW.sum(dim=0)` and casts to `W.dtype`.
This independent example uses a fixed 32-program grid rather than the source's
runtime vector-core count, FP32 weights and outputs, recomputed inverse RMS,
and a Triton final sum. It does not claim the original casting modes or
in-place behavior. Original source provenance and notices remain owned by the
filtered reference collection; this example is independently written.

Implementation: [gradient_pairs.py](../../../examples/optimization-library/triton/gradient_pairs.py)
and [gradient_contract.py](../../../examples/optimization-library/triton/gradient_contract.py).
All raw ten-sample span/active/gap series, correctness diagnostics, compiler
identities, IR hashes and compact PMU means are in
[gradient-evidence.json](gradient-evidence.json). Measured source is
`6e00b92054f8dacc161483d95e28e11976a1a5d4`; evidence archive SHA256 is
`2c82ac0e7d08f3780c5fef6e926c00a4d6884d3fab45907d24950921f8e32c27`.
The retained evidence binds that original contract source. A subsequent API
guard narrows accepted epsilon to the normal FP32 range above; all four
measured cases use `1e-5` and have unchanged arithmetic. The original fixtures,
source hashes and captures are retained without regeneration or remeasurement.
This entry is not added to the skill router during the frozen first coding
study; its training data remains separate from that study's guidance.
