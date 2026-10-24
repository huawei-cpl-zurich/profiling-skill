F08 compares a centered two-pass population-statistics kernel with lane-local
streaming Welford statistics. Both return FP32 channel mean and population
variance for contiguous bounded FP32 NCHW input. There is no normalization,
epsilon, Bessel correction or running-stat update.

The independent oracle computes centered FP64 statistics on represented inputs.
Mean uses atol=2e-5, rtol=2e-6; variance uses atol=1e-7, rtol=2e-3. Constant
populations must return exactly zero variance. Empty populations reject before
any device allocation or division. The supported population is at most 32768,
with at most 256 channels; fixtures use uniform [-1,1], 16 plus uniform
[-2^-8,2^-8], or constant 16.

The project-owned Triton implementation follows the statistics algorithm in
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-nn#norm/batch_norm_v3/op_kernel/batch_norm_v3_welford.h:L450-L579`.
The upstream A2/A3 registration and 910b/910_93 host build evidence support the
source mechanism; actual target compilation and correctness remain separate
requirements. This example does not reproduce or resolve the upstream complete
BatchNorm `save_rstd` interface semantics, and vendors no source headers.

Each channel owns 256 lanes. The centered baseline computes mean first, then
centered squared residuals; populations up to 256 retain the loaded input
locally. Larger populations reread input in the second loop. The candidate
streams lane-local count, mean and M2 once, then merges using
`sum(count*mean)/population` and
`sum(M2+count*(mean-global_mean)^2)/population`. Individual lane counts include
tails; padded lanes have zero weight. Every forward initializes local state and
writes both outputs. Both arms have one device stage and zero GM workspace.
Logical input reads are algorithmic accounting, not measured physical traffic;
compiler temporaries, local memory and actual task counts require device evidence.

`examples/optimization-library/triton/welford-cases.json` freezes all four
profile cases and two correctness-only guards from matrix SHA256
`66a1aa6d957727edd8172939a4decd74cabfe8591b3cba83f7c6e3eb9dee6890`.
The shared `scripts/optimization_pair.py` interface accepts adapter
`welford:factory` with that directory on PYTHONPATH. Cache each CPU oracle once,
check each arm once, then capture exactly ten complete forwards per arm. No
warmup, autotune, additional timing run or output inspection occurs in capture.
Use the global remote-access skill, a healthy idle device lease and shared
capture-export slots. `scripts/optimization_profile_evidence.py` binds retained
source/compiled artifact identities, exact kernels, ten task iterations, PMU
availability and complete device-operation span with active union and gaps.

Host validation: `python -m pytest -q tests/test_optimization_welford.py`.
These functional CPU tests supplement, and do not establish, A3 correctness or
performance. No performance gain is required for this controlled comparison.
