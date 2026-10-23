---
name: ascend-model-benchmarking
description: Measure warmed compiled Ascend Model forwards with separate end-to-end latency and correlated device busy-time, preserving raw samples and comparison conditions.
---

# Ascend Model benchmarking

Use this for arbitrary Model APIs, not only a particular operator. A Model class
does **not** imply graph compilation. Establish actual full-graph capture and
compiled execution for compiled implementations using the same explicit backend;
TorchAir GE is the default, with `--backend npu-inductor` available when selected
and validated for the workload. Never switch backends automatically; reject unsupported
graphs, implicit eager fallback and missing submitted-kernel identity evidence.
An explicitly selected eager reference is supported as described below.

This skill covers correctness/run status, warm latency, correlated busy-time,
dispersion, comparison conditions and retained logs only. Report compilation and
runtime errors as run failures; do not turn them into kernel-diagnostic advice.
Pipe utilization, bottlenecks, timelines, MLIR and optimization recommendations
belong to the separate `ascend-profiling` skill/workflow and are not benchmarking
acceptance gates.

## Obtain reproducible measurements

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

## Explicit eager reference

The controller accepts `--baseline-execution eager|compiled` (default `compiled`).
Candidates always require compiled fullgraph execution and submitted-kernel identity.
Use `eager` explicitly for a slow PyTorch reference; unsupported compiled references
fail rather than falling back. Measure eager and compiled reference comparisons in
separate evidence directories and label their execution modes in final tables.
Eager references use eval/inference mode, correctness validation, the same 20-call
warmup, three batches of 50 synchronized full-forward samples, and a separate
50-iteration correlated profile with 20 warmups. They require no graph backend.
Active compilation detection and input mutation checks still apply to both arms.
The repeated controller forwards this option unchanged, records execution modes,
and rejects mixing modes across repetitions. Ten full repeats, raw samples, failure
counts, mean and sample standard deviation are unchanged.
