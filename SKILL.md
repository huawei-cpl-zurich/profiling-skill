---
name: ascend-profiling
description: Profile Torch or Triton control workloads on Ascend A2/A3 and Ascend 950/A5, or AscendC and Catlass kernels on A5, producing compact target-correct msprof evidence without mixing product-specific metrics.
---

# Ascend kernel profiling

Identify a performance question, obtain trustworthy evidence, connect it to the
code actually executed, make a controlled change, and check whether the result
supports the explanation. Product-specific metrics are not portable.

Resolve the observed SoC, CANN/compiler/profiler versions, execution profile,
and candidate identity. Use the user-wide `$remote-access` client for remote
operations; the selected profile owns runtime setup and device addressing.

## Interpret before optimizing

For A3, read [performance reasoning](references/a3-performance-reasoning.md)
when choosing a hypothesis or interpreting pipes. It explains architectural
paths versus actual compiler lowering, scalarized memory, masking, buffering,
wait counters and small matrix work. High activity is not proof of capacity
saturation, nor a reason to stop optimizing.

Use the benchmark's declared latency method. For A3 controls using `msprof op`,
separate exact-kernel `BasicInfo` captures provide device-task latency; Python
wall time, instrumented pipe durations, simulator timestamps and sanitizer
durations are different quantities. Keep all declared samples and report their
aggregation and dispersion; do not sample until a favorable window appears.
Keep correctness, timing and profiler availability as separate outcomes.

Compiler dumps and sanitizer checks are optional diagnostic escalations, not
mandatory gates for ordinary optimization. Read [A3 timelines and correctness
diagnostics](references/a3-timeline-and-correctness.md) when instruction order,
attribution or races are in doubt. A3 simulated timelines are not A5 hardware
timelines; do not transfer newly observed A3 mechanisms to A2 or A5 without
validation.

## Acquire live control evidence

Use a supplied target-aware measurement interface when the benchmark provides
one. Discover exported names first, then select the exact intended kernel for
timing and pipe captures. Do not replace the benchmark's timing contract.

In an ordinary checkout, `scripts/acquire_profile.py` is the supported helper
for generic Torch/Triton acquisition on `gz-a3`, `bz-a3-1`, `bz-a3-2`, or
`bz-a5`; do not reconstruct its raw `msprof op` flags. It owns the named
runtime, content-addressed workload bundle,
bounded device probe, deployed profiler syntax, persisted `cpl-remote` handle,
resume lifecycle, and compact evidence.
When using that helper, run it in place. Do not copy it into an experiment
workspace, patch it, or synthesize a replacement; pass the workload bundle and
evidence paths to the installed executable.
Read [Two-pass acquisition](references/acquisition-workflow.md), run its
`basic` pass, inspect the exported names, and then supply one relevant exact
name to its `pipe` pass. Never guess a selector from a source function name or
skip the discovery pass. The helper proves both passes used a byte-identical
complete bundle and arguments. Resume an interrupted invocation from its
persisted dispatch receipt and exact handle; never redispatch it.

## Shared saturation analysis

Use `scripts/analyze_pipe_saturation.py` only with a validated model for the
selected product. It classifies each phase as `saturated`, `unsaturated`, or
`unknown` from capacity numerator and denominator metrics. Pipe activity,
timeline occupancy, and composition are useful localization evidence but are
not capacity denominators and cannot establish saturation by themselves.

Read [Pipe saturation contract](references/pipe-saturation-contract.md) before
creating model or evidence JSON. No built-in A2/A3 capacity model is provided.
For A5, `references/a5-pipe-capacity-model.json` covers only its narrowly
reviewed Cube and GM-to-L1 resources with the documented product and metric
definitions; it is not a universal capacity model. Unmodeled resources and
missing capacity evidence remain `unknown`; never borrow another product's model.
A2/A3 `PipeUtilization` block rows are non-temporal activity observations, not
phase evidence, and must never be passed to this analyzer.

## Triton compiler attribution

For reviewed compiler mappings, query the pinned inventory with
`scripts/query_triton_pipe_attribution.py`. Read
[Triton-to-pipe attribution](references/triton-pipe-attribution.md) for its
direct, inferred, and unknown semantics. A known compiler pass sequence does
not make an original Triton operation single-pipe. Keep unsupported mappings
unknown. If the explanation depends on actual emitted operations, the optional
`mlir-triton-dump` workflow can resolve it; early IR alone may miss scheduling.

Use compiler artifacts to verify execution structure, not rank latency by static
operation counts. A dense lowered copy does not rule out a faster candidate;
generated buffering or flags do not establish realized overlap. Keep cheap
direct timing in the loop, and distinguish an observed correctness result from
one inferred from source or capture success. See the A3 reasoning guide for
the evidence boundaries and the attribution guide for the compiler stage index.

## A2/A3 Triton timing and pipe evidence

Use the repository's `$gz-a3` profile and its native `py311-torch` runtime only
for repository-managed GZ compiler validation, mutable experiment bundles, and
the workflows below. Generic BZ A3 acquisition uses the two-pass workflow
above or the supplied target-aware benchmark interface.
Select an eligible physical device through that profile; the one-shot job
exposes it to the application as logical device 0. Do not use direct SSH,
Docker, or a raw remote-agent client.

Mutable experiment candidates use `scripts/gz_a3_job_client.py`, which calls
only the neutral `$gz-a3` adapter's `stage`, `run-bundle`, and
`fetch-bundle-result` actions. It retains content-addressed upload, command,
and download receipts in a private state directory so an interrupted observer
resumes the same handle. The managed-bundle engine's `--client` and `--remote`
options remain internal integration APIs: do not invoke them directly or use
raw SCP, SSH, Docker, or a raw remote-agent client.

For reusable control acquisition, use the two-pass workflow above.
`scripts/profile_a3.py` remains an in-job primitive for an already selected A3
kernel. Run independent timing captures for repetitions and use their median
and dispersion; do not use profiled Python wall time.

Read [A2/A3 msprof-op evidence](references/a2-a3-msprof-op.md) for the command,
JSON contract, A3 pipe inventory, normalization command, and interpretation
boundaries. `PipeUtilization` ratios are activity divided by task cycles; they
are not proof that a pipe reached hardware capacity. Normalize compact exports
with `scripts/normalize_a3_pipe_evidence.py` and
`references/a3-pipe-activity-model.json`. The activity model describes export
fields and emits `unknown` saturation for every observation; it is not a
capacity model. Preserve the full transcript so compilation and runtime errors
remain actionable.

### Actionable diagnostics

Derive the callable/output contract from the supplied harness or baseline,
not another coding skill's convention. Retain exact commands, compiler/runtime
exceptions and the evidence location. Keep bulky logs and profiler trees
remote and return compact excerpts. A profiler's successful exit alone does
not establish workload correctness or a valid selected-kernel export.

Distinguish candidate compilation/runtime/correctness errors from profiler or
harness errors and transport failures. A controller timeout is not a kernel
failure: observe the existing durable handle after interruptions, never
duplicate a live job. Use the experiment's declared retry and scoring policy,
not a universal device-check budget. See the acquisition reference's failure
receipts for the existing helper classifications.

## Ascend 950/A5 diagnosis

Use this skill for A5 kernel timing, PMU diagnosis, sampled utilization, or
instruction/pipe timeline analysis. It is intentionally target-specific: do
not apply its event IDs, ceilings, formulas, `PipeTimeline`, or per-pipe
`InstrTimeline` workflow to A2/A3. Conversely, do not treat A2/A3
`TimelineDetail` flow relationships or pipe ratios as A5 evidence.

Use the configured TLA integration checkout (`TLA_ROOT`) and its checked-in
BZ-A5 wrappers for Catlass DSL and AscendC repository workflows. Read that
checkout's `AGENTS.md` and `$bz-a5` guidance before launching those workflows.
Generic BZ-A5 Torch/Triton control acquisition uses the two-pass workflow
above. Never substitute direct SSH, SCP, Docker, or an ad-hoc CANN
environment.

## Choose evidence by question

1. Run the workload normally and check its full result before profiling.
2. For a performance comparison, select a healthy idle device and preserve a
   minimally instrumented timing run. Profiled durations are diagnostic.
3. Capture `BasicInfo` first and bind later captures to the exact exported
   kernel name.
4. Use `PipeUtilization` for whole-task pipeline composition. Add `Memory`,
   `MemoryL0`, `MemoryUB`, `L2Cache`, or `ResourceConflictRatio` only for the
   suspected bottleneck; keep counter groups in separate comparable replays.
5. Use `PipeTimeline` when simultaneous cross-pipe timing matters. Use one
   `InstrTimeline` replay per pipe for instruction names, PCs, and durations.
6. Use application-level sample PMU for AIC/AIV temporal bubbles, frequency,
   and load balance across a long repeated burst. At 100 Hz it does not resolve
   a single microsecond-scale kernel.

Read [A5 metric semantics](references/a5-metric-semantics.md) before comparing
PMU values. Read [A5 timeline semantics](references/a5-timeline-semantics.md)
before aligning traces, assigning phases, or interpreting sampled data.

## Profile through supported routes

For Catlass DSL, use the mandatory adapter:

```bash
"$TLA_ROOT/execution-profiles/catlass-validation.sh" --profile bz-a5 \
  --operation codex-<task>-<metric> profile \
  --catlass-src worktrees/catlass/<task> --device <physical-device> \
  --metric PipeUtilization --kernel-name <exact-name> \
  --warm-up 0 --launch-count 1 --experiment <name> -- \
  python <workload.py> --device <physical-device>
```

For an AscendC C++ application, use the BZ diagnostic wrapper:

```bash
"$TLA_ROOT/execution-profiles/bz-a5/diagnose-kernels.sh" \
  --device <physical-device> --implementation ascendc \
  --kernel-name <exact-name> --metric PipeUtilization \
  --experiment <name> -- <executable> <arguments>
```

For `InstrTimeline`, pass one supported instruction pipe per capture. Under
`msprof`, pass the selected physical device to both the wrapper and the
application. Preserve warmup, launch count, input, binary/source identity,
frequency, device, and capture order in every comparison.

## Extract before transfer

Keep the full msprof tree on BZ-A5. Stage this skill's `scripts/` directory,
then run the relevant Python helpers against the retained tree:

- `summarize_sample_pmu.py` reads `msprof*.db` and emits compact AIC/AIV JSON,
  Markdown, and compressed normalized rows.
- `summarize_timelines.py` normalizes a simultaneous PipeTimeline and separate
  instruction traces without inventing cross-replay overlap.
- `curate_profile.py` filters exact-kernel CSVs, incorporates generated
  summaries, inventories and hashes evidence, and creates a deterministic
  `summary` or `analysis` archive.

Use `collect_profile.sh` locally to run the curator remotely and download one
verified archive through the configured wrappers. Use `--dry-run` to inspect
the route. Never recursively download the vendor profiler tree for routine
analysis; retain it remotely for later forensics.

The FlashAttention three-way and phase campaigns in the configured TLA
integration checkout are validated worked examples, not required APIs:
`scripts/profile-fa1-three-way.sh` and `scripts/profile-fa1-phases.sh`.
