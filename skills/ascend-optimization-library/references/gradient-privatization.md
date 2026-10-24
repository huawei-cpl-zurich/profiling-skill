# Private weight-gradient accumulation (E08, qualification pending)

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

Two benefit hypotheses (8192x128, 6144x96) and two cost hypotheses (1x2048,
3x4096) are frozen before A3 acquisition, with gallery/heldout separation. CPU
interpreter checks cover width tails, fewer rows than private programs, more
rows than private programs, zero-input rows, immutable inputs and poisoned
workspace reuse. They do not prove A3 atomics support, UB fit or speed. Record
negative or unsupported outcomes without substituting friendlier shapes.

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
This draft is not in the skill router and has no A3 performance claim yet.
