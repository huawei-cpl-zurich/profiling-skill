# Omit an unused eigenvector accumulator

This project-owned real-symmetric Triton example removes an optional algebraic
accumulator. It reuses the staged Jacobi angle and matrix-update primitives from
[jacobi_round.py](../../../examples/optimization-library/triton/jacobi_round.py); it is not a benchmark of a native public eigensolver dispatch.
The source motivation is conditional eigenvector panel updates and workspace
selection in
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-solver#src/cheevj/cheevj_kernel.cpp:L48-L125`.
The retrieved implementation is complex Hermitian with block panels; this example
uses real FP32 matrices and block-size-one round-robin pairs.

Both arms copy the original input, calculate input-dependent angles, and apply
identical left/right A updates over five complete sweeps of N-1 rounds. Sorting
uses stable pairwise ranks. The baseline initializes V=I inside every API call,
updates V=VJ every round, and discards V for values-only requests. The specialized
arm allocates and updates no V unless the caller requests vectors, which selects
the same baseline path. Pair maps alone are reusable input-independent metadata.
Batches use a second Triton grid dimension, without multiplying host launches by B.

The supported demonstration domain is even N=4..96, contiguous finite FP32
[B,N,N], diagonal 1+i/N and seeded symmetric off-diagonal entries bounded by
1/(64N); the repeated-diagonal guard is 1.25I. Five sweeps are fixed before any
measurement. This does not establish convergence for arbitrary symmetric matrices.
Values must be finite, ascending, within atol=rtol=2e-4 of independent CPU FP64
eigvalsh, and leave input unchanged. Vector guards check relative Frobenius
residual and infinity-norm orthogonality at 2e-3, with no orientation comparison.

The frozen four measured cases and two vector guards are in
[the frozen matrix](../../../experiments/optimization-wave004/W08.json). Compile-only preparation checks
representative rounds 0/1 for each required binary, never warming a device kernel.
Each side receives one correctness call, followed by exactly ten instrumented
calls for measured cases; guards receive no captures. Whole device operation
span includes every stage and intervening gap; it is separate from active task
time and application latency. Fewer launches or logical matrix accesses do not
establish measured physical bandwidth or a roofline claim.

`python -m unittest discover -s tests -p test_eigenvalues_only.py -v` exercises
the host contracts, full-sweep launch plans, fallback vector validation, and
actual nested-case CLI/cache boundaries. Device correctness and profiling require
the recorded capable A3 run; CPU tests alone do not establish device behavior.

On the retained A3 Ascend910B4-1 run, all four measured cases and both vector
guards passed. Mean instrumented device-operation spans in microseconds were:

| Batch, N | Accumulate/discard V | Omit V |
| --- | ---: | ---: |
| 4, 64 | 231478.380 | 145400.964 |
| 1, 4 | 11738.152 | 7209.810 |
| 3, 96 | 352821.088 | 226010.192 |
| 1, 8 | 26493.988 | 16562.466 |

All four cases favored removing V; the proposed small-case cost did not appear
at this instrumented boundary. These spans include host dispatch gaps across
hundreds of stages and do not predict a fused native solver. The normal vector
guard had residual 1.51e-7 and orthogonality error 1.59e-7; the repeated-diagonal
guard returned zero errors. Every compiled identity remained unchanged during
capture. No sample was dropped or source parameter adjusted after measurement.
An initial compile-only attempt overflowed UB in the N96 pairwise-rank temporary;
using one diagonal index per program made the same stable ranking feasible on
both arms before any correctness or measured call.

The [readable summary](eigenvalues-only-summary.json) retains all ten span
samples, correctness diagnostics, task counts and the compressed-evidence hash.
[Full evidence](eigenvalues-only-evidence.json.gz) preserves every normalized
device task, PMU row, iteration sample, compilation identity, source hash and
CPU oracle payload without rounding or dropping rows. Its gzip SHA256 is
`7ed6f65b3383bc9889187bc976ba2d031daf79a2056bc906de7935431d7b9884`.
The retained raw-profile/source/binary archive has SHA256
`85db0adb586448c3c405ce6eab3281a11ad54edfa68154b2b71478923047d6e3`.
