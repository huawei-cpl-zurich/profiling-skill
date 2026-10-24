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

The frozen source passed the GZ-A3 gate on Ascend910B4-1 (20 Cube/40 Vector),
Torch/Torch-NPU 2.10.0 and Triton-Ascend distribution 3.2.2; the imported
`triton.__version__` reports 3.2.0. All four prescribed
pairs passed their frozen oracle gates, with exactly eight captures and eighty
profile forwards. Constant guards returned exact zero; both empty-population
interfaces rejected before a device call. The eighteen functional CPU tests
also passed on that target. One earlier health-only attempt used an incorrect
card/chip mapping and stopped before Torch import, with zero device calls.

| Fixed case | Regime | Centered span us, mean ± sample SD | Welford span us, mean ± sample SD | Before/after |
| --- | --- | ---: | ---: | ---: |
| F08-01 | gallery benefit | 1064.688 ± 0.847 | 557.172 ± 1.420 | 1.911 |
| F08-02 | gallery cost | 5.396 ± 0.083 | 5.546 ± 0.162 | 0.973 |
| F08-03 | heldout benefit | 241.292 ± 0.152 | 154.712 ± 0.112 | 1.560 |
| F08-04 | heldout cost | 2.604 ± 0.258 | 3.154 ± 0.619 | 0.826 |

Each row uses one capture per side, ten samples per capture and before/after
order. All observed forwards had exactly one matching kernel task; operation
active union equals the device span and inter-task gap is zero for this one-stage
pair. This does not measure host enqueue overhead or unprofiled application
latency. These measurements retain the predicted cost regimes as well as
benefits; they do not establish universal gains. The first profiled event
sample includes substantial caller overhead, so the device-task span is the
primary comparison.

[Compact evidence](../experiments/optimization-library/w003-welford/evidence.json)
retains all span/event samples, source and compiled npubin hashes, exact task
identities, oracle errors and per-task PMU exports. Those exports show dominant
scalar composition in the large streaming loops; their ratios do not prove
physical input traffic or bandwidth savings. Both algorithms use FP32 and
frozen tolerances. Raw vendor reports remain retained operationally; their
trace/CSV hashes and the compact archive digest bind this published subset.
