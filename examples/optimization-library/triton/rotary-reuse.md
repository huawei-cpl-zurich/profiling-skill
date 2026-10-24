# Rotary coefficient reuse across heads — four A3 cases qualified

`rotary_pairs:factory` provides E09. Both paths rotate every split-half Q and K
head, preserve all inputs, and return contiguous FP16 Q/K tensors. Inputs are
contiguous Q[B,T,Hq,D], K[B,T,Hk,D], int32 positions[B,T], and FP32 invfreq[D/2].
Trusted preparation converts backend default formats to ND2 before either arm,
checking unchanged shape, dtype, strides and values against the cached CPU data.
The host scaling factor in [0.125,128] is identical under both paths. Supported D is
32, 64, 128 or 256; the bounded case contract is in `rotary_contract.py`.

Source inspiration is pinned Triton-Ascend `7acb9c40a6ca`,
`index/fused_rotary_emb.py:L51-L74`:
`ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/fused_rotary_emb.py:L51-L74`.
The filtered original retains its Huawei 2025 and DeepLink 2025 notices. This
pair independently constructs a head-parallel before path and restricts dense
layout. It explicitly promotes both inputs and the rounded coefficients to
FP32 for products/subtractions, then casts the final results to FP16. It does
not claim bitwise equivalence with every arithmetic lowering of the original.

The repeated form assigns each token tile/head combination to a program:

```python
head = tl.program_id(2)
angle = position / SCALE * inverse_frequency
c = tl.cos(angle).to(tl.float16).to(tl.float32)
s = tl.sin(angle).to(tl.float16).to(tl.float32)
# Each head program repeats identical coefficients for this token tile.
store_rotated_head(head, c, s)
```

The reused form assigns only token tiles to programs:

```python
angle = position / SCALE * inverse_frequency
c = tl.cos(angle).to(tl.float16).to(tl.float32)
s = tl.sin(angle).to(tl.float16).to(tl.float32)
for head in range(HQ):
    store_rotated_q_head(head, c, s)
for head in range(HK):
    store_rotated_k_head(head, c, s)
```

Both paths use an eight-token by D/2-frequency tile and one launch. Output
movement and arithmetic are otherwise the same. The after path removes
source-level duplicate coefficient work, while losing head-level scheduling
parallelism and keeping coefficients live through two loops. Many heads and
enough token tiles are a plausible benefit case. Very short sequences and
few heads are plausible cost cases. These were the predeclared selection
hypotheses; the actual results below retain the case that did not show a cost.
There is no extra coefficient GM tensor or preparation kernel.

Look for identical position/frequency sin/cos calculations inside independently
scheduled Q/K head work. Confirm every head actually shares those coefficients;
per-head frequencies or position conventions invalidate this transformation.
In the profile, compare the complete one-launch operation span, all ten samples,
and matched kernel identities. A supported transcendental instruction count or
compiler dump can support reuse; vector busy ratios alone cannot establish it.
Watch for fewer independently scheduled programs, longer loop bodies, spills,
and UB/register pressure. Timing can worsen even if transcendental work falls.

The independent CPU oracle computes coefficients once, rounds them to FP16,
then applies FP32 formulas to each Q and K half. Functional tests execute both
actual Triton kernels in the CPU interpreter, test asymmetric head counts and
token tails, and reject missing K outputs, wrong shapes/dtypes, nonfinite data,
input mutation and unsupported physical layouts. A3 qualification used the same
four fixed cases, one check and ten captured forwards per arm, with exact owned
kernel inventories and stable compiled identities. All checks passed; maximum
absolute error was 0.00048828125. Eight distinct NPUBIN snapshots are hash-verified.

| Case | Before µs ± sample SD | After µs ± sample SD | Speedup |
| --- | ---: | ---: | ---: |
| Gallery many heads | 837.372 ±1.244 | 173.932 ±.917 | 4.814364× |
| Gallery single token | 6.080 ±.115 | 5.614 ±.969 | 1.083007× |
| Heldout many heads | 449.614 ±.580 | 108.710 ±.359 | 4.135903× |
| Heldout few heads | 4.148 ±.115 | 4.990 ±.113 | .831263× |

These are complete profiled device-operation spans on Ascend910B4-1 with
Torch-NPU 2.10.0/Triton 3.2.0, not uninstrumented application latency. The heldout
few-head case demonstrates 20.3% more device time; the noisy gallery cost case
does not demonstrate a regression. Geomean across these four fixed cases is
2.057641×. Each forward has one task, so internal operation gaps are zero.
Raw PipeUtilization joins are available for all eight captures. The large
benefit cases show lower AIV time and a shift toward MTE3 activity, consistent
with less coefficient work and output-store pressure. Retained frontend TTIR
places sin/cos before both head loops; no backend instruction count is claimed.

The initially frozen adapter rejected Torch-NPU's default four-dimensional
format 0 before any candidate/check/profile calls. The trusted ND2 preparation
fix preserves input data and reuses the existing CPU oracle. Total acquisition
was four oracles, eight checks, eight captures and 80 forwards, no warmups.
Qualified kernel/adapter source is `e0c80b7f785641852cda5fbd8afe9b907d939d66`.
Compact archive SHA256 is
`11075b14baa8f1a61c6cbf053d9ef8f384ee04c50adc06864ad4137ee9914a37`.
Its receipt retains every sample, source/binary/trace hash and exact task join.
Resource limits outside these four points remain unqualified.

No measured physical traffic or distance to roofline is currently available.
This entry and its after code are excluded from the first coding-demonstration
guidance snapshot.
