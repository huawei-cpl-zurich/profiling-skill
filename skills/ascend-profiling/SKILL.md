---
name: ascend-profiling
description: Profile Ascend A2/A3 and Ascend 950/A5 kernels with msprof, producing compact timing and diagnostic evidence with product-specific interpretation.
---

# Ascend kernel profiling

Identify a performance question, validate the workload, capture evidence for the
kernel actually executed, and test a controlled change. Product-specific metrics
are not portable.

## Prerequisites

- `msprof` is available on `PATH`, with `msprof op` support for the selected
  product and requested metric. Check `msprof op --help` for installed syntax.
- A compatible Ascend runtime, driver, and accessible device are configured.
  The workload's compiler/framework dependencies must already be installed.
  The bundled paired examples additionally require Torch, Torch-NPU, and
  compatible Triton-Ascend; `--manifest` works without these device packages.
- A runnable workload checks correctness and returns nonzero on failure. Know
  its inputs, kernel implementation, device selection, and timing contract.
- The included `profile_a3.py` capture helper preserves an existing
  `ASCEND_RT_VISIBLE_DEVICES` mask and sets it to `0` when absent. Set the mask
  explicitly to select another device, using the resulting application indices.
- Included helpers use Python 3.10 or later and only the standard library.
- Optional compiler attribution requires matching compiler artifacts and tools;
  simulation requires the product-compatible simulator; race/memory diagnostics
  require `mssanitizer` and supported instrumentation. These are not prerequisites
  for ordinary timing or pipe profiling.

Commands in references run from this skill folder; alternatively use absolute
paths to its scripts and references. Workload and output paths belong to the
user's working directory and should be absolute when using another directory.
No machine registry, remote service, or separate integration checkout is needed.

## Acquire evidence

Read [two-pass acquisition](references/acquisition-workflow.md). Run the
workload normally first. Discover exported names using `BasicInfo`, then use
one exact exported name for independent timing and diagnostic captures. Preserve
input, source/binary identity, arguments, device, versions, replay mode, warmup,
launch count, and capture order. Use a healthy idle device for timing.

Keep correctness, timing, and profiler availability as separate outcomes. A
profiler's zero exit status alone proves neither correctness nor a usable
selected-kernel export. Preserve full logs and reports; distinguish workload
compilation/runtime errors from profiler errors. Report all declared timing
samples with aggregation and dispersion rather than selecting a favorable subset.

## A2/A3

Read [A2/A3 msprof-op evidence](references/a2-a3-msprof-op.md) for the
`profile_a3.py` helper, replay modes, compact timing schema, and pipe inventory.
Separate `BasicInfo` captures measure device-task latency; Python wall time,
pipe-instrumented durations, sanitizer durations, and simulator timestamps are
different quantities. Kernel replay is the default; use explicit application
replay when memory snapshot requirements prevent kernel replay, recording the
choice consistently.

Read [performance reasoning](references/a3-performance-reasoning.md) when
choosing hypotheses or interpreting activity; it covers common data paths,
store routing, scalarized accesses and the next measurement to choose. Read the
[field glossary](references/a2-a3-msprof-op.md#field-glossary-and-calculations)
for row/cycle/time calculations and wait/stall diagnosis. Read
[generated synchronization](references/a3-synchronization-inspection.md) when
a buffering or handoff explanation depends on late compiler IR, and
[performance envelopes](references/a3-performance-envelope.md) to quantify a
shape-matched gap. Use the [paired examples](references/a3-paired-examples.md)
for runnable copy, gather and matrix comparisons. Normalize single-launch pipe
exports with `scripts/normalize_a3_pipe_evidence.py` and
`references/a3-pipe-activity-model.json`. Block/sub-block rows are non-temporal
activity observations, not phases or capacity evidence. High activity does not
prove saturation. Optional [timeline and correctness diagnostics](references/a3-timeline-and-correctness.md)
help resolve instruction ordering, attribution, or races; A3 simulation is not
A5 hardware timing.

## Ascend 950/A5

Read [metric semantics](references/a5-metric-semantics.md) before interpreting
PMU values and [timeline semantics](references/a5-timeline-semantics.md) before
aligning traces. Capture `PipeUtilization` for whole-task composition, adding
memory or conflict groups only for a specific hypothesis in separate replays.
`PipeTimeline` supplies simultaneous cross-pipe timing; separate `InstrTimeline`
replays supply instruction names, PCs, and durations without measured cross-replay
overlap. Sample PMU locates temporal bubbles across long repeated bursts, not
individual microsecond kernels. Never apply these A5 semantics to A2/A3.

Use `summarize_sample_pmu.py`, `summarize_timelines.py`, and `curate_profile.py`
in `scripts/` to summarize retained profiler reports and build compact archives.
These operate directly on files; retain raw evidence for later investigation.

## Capacity and compiler attribution

Read [the saturation contract](references/pipe-saturation-contract.md) before
using `scripts/analyze_pipe_saturation.py`. Capacity claims require a validated
product-specific numerator and denominator. No A2/A3 capacity model is supplied.
The A5 model covers only its documented Cube and GM-to-L1 resources; unmodeled
resources and missing capacity evidence remain `unknown`.

For version-pinned Triton compiler mappings, use
`scripts/query_triton_pipe_attribution.py` and read
[Triton attribution](references/triton-pipe-attribution.md). Distinguish direct,
inferred, and unknown mappings. Early IR and known pass sequences do not prove
the final instruction stream or make an original operation single-pipe.
