# Two residual corrections using retained Thomas factors

This SOURCE-DOCUMENTED independent FP32 adaptation follows the algorithm in
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#sparse/gtsv2/README.md:L34-L38`.
It does not copy the unseen device implementation or reproduce the native
extended-exponent arithmetic. The source's scratch layout appears in
`sparse/gtsv2/arch22/gtsv2_tiling_data.h:L33-L38`; numerical test interfaces in
`test/gtsv2/arch22/gtsv2_test.cpp:L451-L484` cover cancellation, and
`:L289-L323` distinguish backward and forward error. Retrieved upstream tests
were inspected, not executed. The source pin is
`bfc7c19d83ce246b03c65dc916ed4753e5467cf3`, under the CANN Open Software
License Agreement 2.0. Upstream device-source-path misses remain unresolved.

The before arm factors the current matrix once and solves the current RHS with
FP32 Thomas elimination. The after arm performs exactly the same work, then two
fixed residual/correction steps. Each step forms `r=b-Ax`, solves `A delta=r`
using the retained factors, and commits `x+delta` only if every component of that
RHS vector is finite. One overflow rejects the entire proposed vector. Original
matrix and RHS stay immutable; padding copies, factorization, solves, residuals
and finite checks execute inside every API call. Workspace allocation is outside
the API, while workspace contents are recomputed each call. No factors are reused
across calls. Both arms disable fusion and use one compiler stage with
multibuffering disabled. The residual association is `(d*x+dl*lower)+du*upper`.

The frozen W13 matrix contains four single-RHS systems with sizes 128, 32, 192
and 64, plus two correctness-only guards. Moderate systems use off-diagonals -1
and diagonal `2+1/64`; easy systems use seeded off-diagonals in [-.25,.25] and
diagonal 4. The FP64 dense oracle solves the represented FP32 equation. Active
output must satisfy `atol=1e-5, rtol=1e-4`; diagnostics also retain maximum and
relative L2 forward error, maximum residual, and componentwise backward error.
A small residual alone never establishes acceptance. A functional CPU test
exhibits small backward error with large forward error in an ill-conditioned
system. It is not an unsupported GPU robustness claim.

The exact diagonal guard verifies unchanged numeric results after two zero
corrections, treating signed zeros equally. The component guard starts from a
finite original vector and finite delta, forces one addition to overflow, and
checks that all components remain unchanged. It is a component test, not a dense
solve accuracy result. Neither guard is profiled. Refinement may worsen forward
error and necessarily adds work; retain negative outcomes without retuning.
