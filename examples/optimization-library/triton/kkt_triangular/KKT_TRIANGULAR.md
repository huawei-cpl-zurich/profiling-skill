# Featurewise KKT triangular work pruning

This prepared component compares a source-derived educational full off-diagonal calculation with the pinned source's strict-lower sub-block calculation. Both return the complete chunk score matrix, including zero diagonal, upper triangle and unused tail columns.

For a chunk, the mathematical target is

```
A[i,j] = beta[i] * sum_f(K[i,f] * K[j,f] * exp(gk[i,f] - gk[j,f]))  if i > j
A[i,j] = 0                                                       otherwise
```

The featurewise source does **not** clamp gate differences. The separate scalar-gate branch has a clamp and is not the baseline in this study.

Both arms charge three kernels on every prepared call:

1. Zero the complete private output.
2. Compute off-diagonal 16×16 sub-blocks, preserving the source's centered exponential factorization and FP32 dot.
3. Run the same source diagonal-block producer, with its direct exponential differences and reduction order.

Before computes all off-diagonal blocks, then masks the upper output before storing. After uses the source predicate `i_i > i_j` to skip upper-block work. The zero and diagonal producers are identical. This is a known triangular-pruning technique applied to another source operator; no new-mechanism claim is made.

## Source and bounded domain

Source: Triton Ascend kernels revision `7acb9c40a6ca827e885d902b7eaee5d45457b230`, `attention/fla/chunk_scaled_dot_kkt_fla.py`. Original kernel, public export and actual upstream test are retained under `vendor`, with the full MIT license and third-party notices.

The upstream NPU test exercises the scalar-gate path. It does not establish featurewise correctness. The public wrapper's head-major K and beta transpose also disagree with the featurewise device indexing when H>1. This study therefore admits only dense B=H=1, where the layouts agree. It does not qualify the complete public wrapper, multiple heads, variable-length sequences or arbitrary gate values.

Inputs are independent contiguous FP32 K, feature gates and beta, with owned 128-byte halos. T is 1–1024, K is 64 or 128, and chunk sizes are 32, 64, 128 or 256. The frozen fixture uses dyadic K/beta and small nonconstant feature gates. Input validation rejects unsupported shapes/layouts/types before backend preparation. Actual device storage must be aligned dense ND format 2. Allocation and initial input copies are outside both repeated APIs; all score math, masking and per-call zeroing are inside.

Extraction removes only the source heuristic wrapper, supplies `IS_VARLEN=False`, renames the kernels, and adds the common explicit IEEE FP32 dot policy. Both arms disable FP fusion. The inter producer retains four warps and three stages; the diagonal producer retains the source's unspecified stage default, with actual compiler metadata retained. `kkt_codegen.py` documents the exact educational derivative.

## Expected benefit and tradeoffs

Large chunks have more off-diagonal blocks, so skipping the upper half can reduce dot, exponential and movement work. Both arms still launch three kernels: fewer task launches are not the hypothesis. Inspect exact inter-kernel task duration and supported AIC/AIV PMU rows separately. Ratios do not establish whole-device utilization or physical HBM traffic.

Small chunks may be dominated by the common zero/diagonal producer and launch gaps. Early-return grid programs, tail occupancy and compiler scheduling can reduce the benefit. The diagonal source's serial loop can dominate either arm. Designated cost cases are risk probes, not promises of slowdowns. If the compiler removes the educational upper calculation, retain that result instead of forcing a contrast.

## Verification and measurement

`experiments/optimization-wave011/P04.json` freezes the six cases, independent ideal and typed-boundary FP64 CPU references, tolerances and complete API before implementation. The primary numerical gate compares both references at `atol=1e-4, rtol=1e-3`; paired bit identity is secondary. Full output, halos, input immutability and private storage are checked once per arm/case.

All twelve input-only compile/code/precision/resource screens precede generation of six cached references. All twelve correctness calls precede exactly ten complete prepared calls per arm/main case: eighty profiled calls and eight captures. Guards have no profiling calls. There are no GPU warmups, autotuning calls, extra latency runs, profiling input/output inspections or discarded samples.

The capture hook guards events, launches, synchronization and export on the prepared device and requests `PipeUtilization`. The generic harness owns cleanup; the controlled collector explicitly borrows prepared items until its final cleanup. Functional CPU tests exercise domain rejection, both references, compiler precision metadata, partial-build publication, capture device scope and metric request, output validation, independent-item ownership and real harness success/error cleanup.

Measurement results and the host migration receipt are published separately when qualified. Profiled operation spans include inter-kernel submission gaps; summed device-task time, active union and gaps must be reported separately. These are not ordinary application latency, statistical-significance or roofline claims.

## Qualified GZ results

Target: actual Ascend910B4-1,20Cube/40Vector, Torch/NPU2.10, Triton-Ascend3.2.2 distribution/importedTriton3.2.0, CANN9.1.0. All12compile/type/code gates preceded6once-only cached references and12passing correctness calls. Eightcaptures account for80prepared calls and240device tasks:160AI_VECTOR_CORE and80MIX_AIC.

| Case /role (T,K,BT) | TaskSUM before→after µs | TaskSUM speedup | Span before→after µs | Span speedup |
| --- | ---: | ---: | ---: | ---: |
|cost (32,64,32)|33.736→33.002|1.022×|384.612→375.060|1.025×|
|cost (64,64,64)|36.540→35.476|1.030×|373.468→370.712|1.007×|
|benefit (512,64,128)|80.228→60.584|1.324×|373.778→375.416|0.996×|
|benefit (1024,128,256)|247.122→163.968|1.507×|440.666→412.204|1.069×|

Four-case geometric means: **1.204× taskSUM**, **1.024× span**. AlltaskSUM means improved. TheT512 benefit case's span increased0.44% as gaps grew293.550→314.832µs, despite taskSUM80.228→60.584µs. Therefore this is a mixed span result and fails the audit's all-benefit-span criterion; it is not retuned or omitted. No significance claim is made.

Maximum error against either independent reference was0.000100553036, within the frozen combined absolute/relative tolerance. Five ofsix paired-bit secondary checks were true; theT64 costcase differed at the bit level while passing both numerical reference gates and the paired tolerance. No bit-exact equivalence is claimed. Allsamples/rawPMUs, source/compiler/ELF/cachedfixture identities, code/physical audits and healthyrelease proof are in `experiments/optimization-wave011/P04-result.json`. Fullcompact SHA `638d70883014efd16579c4892ba7f3d29291ee3c48d66fcb9a98b1d5fc70c385` (3016336bytes,553verifiedfiles);790vendor files are additionally inventoried by digest.

TheoriginalBZ job was cancelled while compiling because the user restricted work to GZ. Six compilecells had finished; all12correctness counters, CPUreferences, numerical/profile calls andcaptures werezero. Theprospective migration receipt preserves source/cases/math/tolerances and requires target-specific GZ qualification. It is a routechange, not a failed numerical outcome; oldBZ artifacts are retained separately.
