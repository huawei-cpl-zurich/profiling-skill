# Balance output tasks against tile movement costs

Status: source-grounded hypothesis; paired A3 evidence pending. Use family `05`
in `examples/optimization-library/triton/gemm.py` with adapter `gemm:Adapter`.

The [pinned layout-sensitive tile example](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-v130-design-550bbffa8b36#docs/contents/advanced/catlass_optimize_guidance.md:L114-L150)
shows why fewer output tasks can help while a different B layout can reverse
the result. Its historical target count is not the current target's core count.

Recognize output dimensions whose ceiling-divided tile grid narrowly exceeds
the observed Cube core count. Compare these configurations while holding dtype,
K blocking, arithmetic, mapping and matrix values fixed:

```python
# Before
BM, BN, BK = 128, 256, 32
# After
BM, BN, BK = 256, 128, 32
tasks = ceildiv(M, BM) * ceildiv(N, BN)
```

The same accumulator element count does not imply equal operand movement or
compiled resource use. A taller A tile and narrower B tile change transfers,
live operand sizes and alignment behavior. Compare both supported physical B
layouts: row-major strides [N,1] and column-major strides [1,K]. The adapter
preserves and records those strides; it does not silently transpose in the call.

Generate gallery geometry from actual C: choose positive a and odd b∈[3,31]
with a*b≤C<a*(b+1), maximize a*b and then minimize a*(b+1). Set M=256a,
N=128b, K=6144. At C=24, M=2048,N=384 gives 32 before tasks and 24 after.
This is task-count arithmetic, not an elapsed-time prediction. Do not infer
that one logical program always occupies one physical core for one wave.

Both arms use FP16 input/output and the same FP32 CPU accumulation followed
by FP16 conversion, rtol=atol=1e-3. Mask M/N/K tails independently of strides.
Compile resource evidence must cover both 128×256 and 256×128 tiles; a rejected
resource configuration is not a correct adverse performance example.

Inspect complete-call duration, observed kernel identity, actual task geometry,
and supported compiler resource or traffic evidence. Low global utilization
alone cannot distinguish too few tasks from memory stalls. The diagnosis is
falsified when the tile with fewer tasks is not faster, or layout movement
overwhelms task balance. Record these outcomes without assigning a saturated
pipe or roofline distance from activity percentages alone.

No layout-dependent benefit or cost has yet been measured for this pair.
