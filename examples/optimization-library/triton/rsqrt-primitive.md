# Reciprocal-root primitive study

This is an independent same-format Triton adaptation of two source expressions:

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/activation/ln_mul/op_kernel/ln_mul_kernel.h:L258-L265`, commit `e489c0bf6a2d5f2d05b3205fcacc01b62a908b10`, Apache 2.0: Sqrt then Div.
- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/norm/norm_gated.py:L103-L117`, commit `7acb9c40a6ca827e885d902b7eaee5d45457b230`, repository MIT license and Third_Party_Open_Source_Software_Notice: `tl.rsqrt`. The source attribution header is preserved in the adapter.

The native source only motivates BEFORE. Native `rsqrt_apt.cpp` includes arch35 and establishes no native 910B AFTER implementation. Upstream normalization output checks do not independently certify reciprocal-standard-deviation accuracy.

Both arms use one named kernel, block 256, the same grid, contiguous FP32 element offsets, safe masked-load value 1.0, output canaries, and identical default compiler flags. The sole compile-time branch selects `1.0 / tl.sqrt(x)` or `tl.rsqrt(x)`. There is no Newton refinement, normalization, fusion, autotuning or warmup execution. Triton `warmup` compiles without forwarding. Emitted lowering and complete binary artifacts are retained before scientific execution.

The public `Adapter.prepare` accepts immutable contiguous CPU FP32 `x[N]` in `[2^-8,2^8]`. It rejects invalid shape, layout, dtype, nonfinite, nonpositive and out-of-range values before importing the NPU backend. The fixture provider validates its configuration before generating an independent FP64 reciprocal-root oracle of represented FP32 inputs. Fixed tolerance is `atol=2e-6, rtol=2e-5`; pairwise bit equality is not required.

Use `scripts/optimization_pair.py --adapter rsqrt_primitive:factory` with this directory on `PYTHONPATH`. A case supplies `N`, `seed`, and `distribution`; loguniform cases additionally supply integer `exp_low` and `exp_high`. The frozen matrix is `experiments/optimization-wave006/N04.json`; flatten each case's `parameters` into the public case object. Guards are correctness-only. The compile audit must precede collection: proven complete device ELF equivalence means compiler no-op, zero scientific forwards and zero captures. Equal `.text` without relocation target identity is inconclusive and cannot establish equivalence.

## Frozen A3 result

The complete twelve-variant compile screen passed on Ascend910B4-1 with CANN compiler/OPP 9.1.0, imported Triton 3.2.0 and installed `triton_ascend` distribution 3.2.2. Both arms lower to AIV SIMD, with identical compiler settings and reported resource metadata. BEFORE retains `math.sqrt` plus `arith.divf`; AFTER retains `math.rsqrt`. All six full device ELFs and text sections differ: text size 804 versus 764 bytes, same entry symbol and no relocations. This proves a code difference; byte length is not an instruction-count claim. The stack note and kernel-argument-size note agree. Resource metadata does not establish peak local-memory usage.

All twelve once-per-arm correctness checks passed. Six scientific oracle caches were generated once; the four main pairs produced exactly ten profiled full API forwards per arm, eight captures and eighty compute tasks. Guards produced zero captures. Output canaries and input immutability passed. Twenty-five functional tests passed locally and on the A3 runtime, using independent nonfrozen unit seeds/shapes. Unit oracle work is separate from the six scientific cache generations. No failed scientific attempt, measurement retry or source retuning occurred.

| Frozen case | N | BEFORE mean span µs | AFTER mean span µs |
| --- | ---: | ---: | ---: |
| Gallery benefit | 65536 | 4.660 | 4.112 |
| Gallery cost | 8 | 1.776 | 1.674 |
| Heldout benefit | 49152 | 3.746 | 3.768 |
| Heldout cost | 17 | 1.356 | 1.342 |

The heldout large case is slightly slower; prospective benefit/cost designations remain unchanged. These are profiler-active operation spans, not ordinary application latency. No physical traffic, roofline, occupancy, precision benefit or native 910B Rsqrt claim follows from these observations. The independent accuracy contract passed both arms; it does not require pairwise bit equality.

All ten samples, original task PMU fields, compiler resources, code hashes, source hashes and fixture bindings are in [the compact result](../../../docs/data/optimization-w006-rsqrt-primitive.json). The verified compact archive retains 233 files, SHA256 `26397b93123d9425e0fc3671ea23c15ace1f2b1d30a207d6ae6e8ad68e8e6a1b`; full vendor trees remain on the selected target. Collection handles and retained namespace are recorded with the evidence. This study does not automatically add a unique optimization mechanism to the gallery taxonomy.
