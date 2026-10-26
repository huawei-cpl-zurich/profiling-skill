# KDA final dA:32-versus64 tiles

The tested source32-tile configuration was **slower in every main case** than the64-tile configuration. Both compute the same bounded final-dA operation correctly. The result is retained as a qualified performance regression; no shape, tile, flag or sample was changed after measurement.

This is a dense prepared final-dA component, not full KDA backward. The original MindSpeed source and licenses are in `examples/optimization-library/triton/kda_da/vendor/`; `kda_da_generated.py` extracts the exact two selected bodies, with their original `do_not_specialize=['T']` policy. Only the outer metadata heuristic is made explicit as `IS_VARLEN=False`. Both arms use the same bodies with BM=BN=BK64 before and originalsource32 after,2warps/1stage. The64 baseline is a documented educational configuration ablation, not a vendor-best claim.

## Operation and complete API

Inputs are CPU-owned dense F16 `A[B,T,H,BT]` and F32 `P` of the same physical shape. The first kernel casts P toF16 and computes `tmp=P_half@A.T`, storingF32. The second casts tmp toF16 and computes `-A.T@tmp_half`, then keeps the strict-lower result and storesF32. These are the actual pointer semantics; the source comment's untransposed `-A@(P@A)` is not used as the oracle. Each chunk is physically padded by zero missing rows; source masks and the valid output rows are audited. Both kernels, all valid output columns, and all real mixed-engine work are charged every API. Inputs, allocation, initial transfers and compilation are common prepared setup; the privateF32 temporary/output are reused and fully overwritten. No reset or host inspection runs between kernels.

The domain is B1..2,T1..2048,H1..8,BT64or128, with no aliases/views/autograd. A is unit lower triangular, offdiagonals integer[-2,2]/256; P is strictlower integer[-4,4]/16. The first dot's numerator is bounded by2040 at denominator4096, so itsF32 result and followingF16 cast are exact. The second numerator is at most1040400 at denominator1048576, below2^24. Thus the independentCPUFP64 oracle, both tilings and their different K reductions are exact on this fixture domain; signedzero is secondary. Unsupported inputs reject before backend/cache/movement. No broader data/layout/autograd support is claimed.

## Why compare these configurations

The source32 tile reduces the logical tile footprint and exposes more programs. The prospective small-BH benefit cases have4→16 and8→32 programs perstage; the saturated cost cases have256→1024. Larger program counts and smaller tiles can trade reuse and per-program work against useful parallelism. Here the expected benefit did not appear: the source32 configuration increased device work in all four cases. Compiler resource peaks/physical traffic/occupancy are not inferred from source dimensions. Raw per-task PMUs are retained for further analysis.

## Frozen measured results

Actual target: GZ-A3 Ascend910B4-1,20Cube/40Vector, selectedpy311 runtime, Torch/NPU2.10, importedTriton3.2/distribution3.5/Triton-Ascend3.2.2, CANN9.1.0. Twelve input-only case-arm compile/type/tile/ELF screens passed BEFORE six scientific CPU fixtures were generated once. All twelve correctness calls passed with zeroFP64 error BEFORE eighty profiled complete APIs in eight captures. Both arms charged two kernels perAPI:160 actualMIX_AIC tasks, with all vector work retained. Guards were not profiled. Five nonfrozenCPU/mock tests passed locally, from the extracted bundle at an external working directory, and on the sameA3 job before NPU initialization. No science was replayed.

| Case /role (B,T,H,BT) | TaskSUM before64→after32 µs | TaskSUM speedup | Span before64→after32 µs | Span speedup |
| --- | ---: | ---: | ---: | ---: |
| cost (1,2048,8,64) | 86.092→552.926 | 0.156× | 251.606→555.814 | 0.453× |
| cost (2,512,8,128) | 147.766→1036.292 | 0.143× | 286.652→1039.050 | 0.276× |
| benefit (1,128,1,128) | 12.668→25.164 | 0.503× | 205.092→210.200 | 0.976× |
| benefit (1,256,1,128) | 12.444→42.152 | 0.295× | 195.128→229.206 | 0.851× |

Four-case speedup geometric means: **0.240× taskSUM**, **0.568× span**. Every after mean is higher. In the cost cases, gaps decrease while much greater device work makes the complete span slower; the small benefit spans include large submission gaps that obscure the size of their task-time regressions. No statistical significance or ordinary-application-latency claim is made. All ten samples, including the first, are retained.

The public data file `experiments/optimization-wave011/P12-result.json` includes all case means/rawsamples/PMUs, originalsource/compiler/actualELF/cachedfixture bindings, code and physical audits, and healthyrelease proof. CompactarchiveSHA `eb7377af1af571d64b33d09e461935bed762eeaeddbfbb805eaa4b9643f31354` (9,982,042B,480verifiedfiles);790 additional raw vendor files are inventoried by digest and retained remotely. Device0 and both hostexport locks were released healthy/idle with no live owned workload. No actor/analyzer evaluation is claimed.

## Inspect or run

`Adapter` in `examples/optimization-library/triton/kda_da/kda_da.py` supplies the generic paired harness interface and bounded admission. Both prepare/compile/run and the entire capture (events, synchronization and analysis) use the prepareddevice, restoring the caller's device. Capture explicitly requests `PipeUtilization`; idempotent close releases only its own live buffers/state. The controlled qualification used correctness once percase/arm, then ten profiling calls without I/O checking. For functionalCPU/mock validation, run `python -m unittest discover -s tests -p test_optimization_kda_da.py -v`. Actual execution requires a compatible AscendTriton/Torch-NPU runtime and a freshly healthy isolated registeredGZ device.
