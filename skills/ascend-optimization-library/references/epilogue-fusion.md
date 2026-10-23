# Fuse a GEMM add epilogue

Status: source-grounded hypothesis; paired A3 evidence pending. The example is
`examples/optimization-library/triton/gemm.py`, family `02`, adapter `gemm:Adapter`.
Its shared CPU contract lives in the neighboring `gemm_contract.py` so native
examples can reuse the same oracle cache instead of computing a new baseline.

The [pinned native block-epilogue assembly](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/03_matmul_add/matmul_add.cpp:L111-L120)
motivates this independently written Triton pair. It is not evidence that our
implementation or target improves.

Recognize a GEMM result immediately consumed by an elementwise add, with no
other consumer of the intermediate. Before:

```python
intermediate = allocate_fp32(M, N)  # Allocation belongs to every before call.
gemm_fp16_inputs_fp32_store(A, B, intermediate)
add_fp32(intermediate, Z, output)
```

After, add Z while the accumulator is still FP32:

```python
acc = dot_fp16_accumulate_fp32(A, B)
acc += load_fp32(Z)
store_fp32(output, acc)
```

This removes a launch and the FP32 intermediate write/read. It also lengthens
the accumulator lifetime and adds an epilogue load to the matrix kernel. A
compute-heavy long-K GEMM may gain little while paying resource or scheduling
cost. Compare complete calls and workspace bytes, not just the matrix kernel.
The gallery tests 2048×2048×256 and 256×256×8192; those are hypotheses about
benefit and cost, not measured crossover thresholds.

Both arms use FP16 A/B, FP32 Z[M,N], and FP32 output. The unfused arm writes an
FP32 intermediate; no FP16 rounding is inserted before the add. Both must pass
the same FP32 CPU matmul-plus-add oracle at rtol=atol=1e-3. Reject fusion if
the intermediate is separately required or a layout/numerical contract differs.

In the profile, establish two expected device kernels per iteration before and
one after, ten owned iterations each, plus successful numerical checks. This
supports the launch-count change. Claim reduced physical GM traffic only when
an available counter establishes it; logical eliminated bytes are an estimate.
Missing PMU fields remain unavailable. Falsifiers include unchanged complete-call
time within variability, a slower fused call, or extra compiler-generated work.

No runtime/resource tradeoff has yet been verified for this example. Retain
negative or inconclusive outcomes rather than calling the long-K case a loss
without measurement.
