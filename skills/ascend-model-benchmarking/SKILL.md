---
name: ascend-model-benchmarking
description: Measure warmed compiled Ascend Model forwards with separate end-to-end latency and correlated device busy-time, preserving raw samples and comparison conditions.
---

# Ascend Model benchmarking

Use this for arbitrary Model APIs, not only a particular operator. A Model class
does **not** imply graph compilation. Establish actual TorchAir GE full-graph
capture and compiled execution for both implementations; reject unsupported
graphs, eager fallback and missing submitted-kernel identity evidence.

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
