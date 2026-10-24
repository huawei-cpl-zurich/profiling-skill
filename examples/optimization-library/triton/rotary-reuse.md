# Rotary coefficient reuse across heads — draft, CPU verified only

`rotary_pairs:factory` provides E09. Both paths rotate every split-half Q and K
head, preserve all inputs, and return contiguous FP16 Q/K tensors. Inputs are
contiguous Q[B,T,Hq,D], K[B,T,Hk,D], int32 positions[B,T], and FP32 invfreq[D/2].
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
few heads are plausible cost cases. These are hypotheses, not measured wins
or regressions. There is no extra coefficient GM tensor or preparation kernel.

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
input mutation and unsupported physical layouts. This establishes CPU semantics
only. A3 compilation, exact compiled inventory, resource evidence, one numerical
check and ten captured forwards per arm/case remain required. Retain failed
hypotheses; do not promote this entry as an A3 speed improvement before that gate.

No measured physical traffic or distance to roofline is currently available.
This draft and its after code are excluded from the first coding-demonstration
guidance snapshot.
