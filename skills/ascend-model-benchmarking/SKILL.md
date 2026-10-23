---
name: ascend-model-benchmarking
description: Compare Ascend Model profiles with one validation and ten profiled forwards, or explicitly requested warmed latency, preserving raw samples and conditions.
---

# Ascend Model benchmarking

Use this for arbitrary Model APIs, not only a particular operator. For requested
**profile comparisons**, including “profile ten”, use
`scripts/model_benchmark_profiled.py`: one compiled candidate correctness
validation, then exactly ten profiled forwards per arm. Reuse its private baseline
cache across candidate configurations. Do not route these requests through the
repeated warmed-latency controller. See [controller integration](references/controller-integration.md).

A Model class does not imply graph compilation. Establish actual candidate
fullgraph capture/execution and report submitted-kernel identity qualification.
The profile-comparison protocol profiles the trusted eager PyTorch baseline. For
explicitly requested warmed, unprofiled latency, use the separate workflow below.
Candidate execution remains fullgraph; compiled
comparisons use the same explicitly selected backend (TorchAir GE by default, or
validated `--backend npu-inductor`). An explicitly selected eager reference is
not an automatic fallback. Never switch backends automatically.

This skill covers correctness/run status, profiled event timings, explicitly
requested warm latency, correlated busy-time, dispersion, comparison conditions
and retained logs only. Report compilation and
runtime errors as run failures; do not turn them into kernel-diagnostic advice.
Pipe utilization, bottlenecks, timelines, MLIR and optimization recommendations
belong to the separate `ascend-profiling` skill/workflow and are not benchmarking
acceptance gates.

## Primary profile comparisons

Use `scripts/model_benchmark_profiled.py` for one compiled candidate
correctness validation followed by ten profiled forwards per arm, with no extra
warmup, selector discovery or unprofiled timing. Use its private baseline cache
across candidate configurations so inputs/expected outputs and the ten baseline
samples are generated once. These are **profiled device-event timings**: report
raw ten, mean and sample standard deviation; they are not the warmed unprofiled
latency statistics described below. Measured forwards assume read-only inputs;
do not add tensor I/O inspection or
extra correctness calls during/after the ten. Cache transfers and independent
input/expected restoration happen outside profiling. Trace accounting and kernel
identity have separate statuses. See [controller integration](references/controller-integration.md).

## Explicitly requested warmed latency measurements

Read [the measurement contract](references/measurement-contract.md) before
collecting or interpreting results. Use the provided controller in a trusted
environment, or the experiment's thin benchmark client when isolated. Do not
reimplement timings with ad hoc clocks or infer iteration boundaries from names.

- Compile and validate each specialization before measurement. Warm with at least
  20 calls; keep device-resident inputs and do not flush caches. Active compilation
  invalidates a batch, rather than becoming kernel execution time.
- Collect unprofiled synchronized full-forward latency separately from a warmed
  profiler replay. Report both; summed kernel durations are not elapsed latency.
  Busy-time includes device and AiCPU tasks per iteration; also report each separately.
- Run **10 full paired repetitions per configuration/case** by default with
  `scripts/model_benchmark_repeated.py`. Report mean and sample standard deviation
  (`ddof=1`) of the ten per-run latency medians and ten correlated busy-time
  medians. Preserve within-run percentiles/batches separately; those are not
  independent repetitions. Retain failed runs and mark fewer than ten successes
  incomplete, rather than silently replacing failures or rerunning until stable.
- Preserve all samples and rows. Median-threshold outliers are flags, not deletions.
  Report clock/load differences and dispersion rather than selecting a fast retry.
- Compare paired implementations on the same physical device/runtime/clock policy.
  Without evidence of comparable conditions, label results qualified and withhold
  an unqualified aggregate ranking. Across cases use geometric mean speedups.
- When transport fails, recover the same retained job. Retry only within the
  original time budget; unknown/running jobs never justify duplicate dispatch.

The generic controller and private adapter contract are documented in
[controller integration](references/controller-integration.md). Benchmark operators,
input generators and correctness tolerances belong in private adapters, not here.

## Evidence and isolation

Retain source/config identities, actual graph/backend/compiler evidence, commands,
runtime/device/clock metadata, raw latency samples, normalized correlated profiler
rows, and failures. Report cold preparation separately. Keep large raw captures
remote while retaining compact artifacts with their provenance.

Blind optimization agents receive this guidance, the operator API and a thin tool
client only. They must not receive reference implementations, input providers,
validators or controller source. Treat access to private acquisition adapters as
trusted controller access, not as an agent skill dependency.
