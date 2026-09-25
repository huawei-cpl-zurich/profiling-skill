---
name: ascend-profiling
description: Profile Triton kernels on Ascend A2/A3 or AscendC and Catlass kernels on Ascend 950/A5 with msprof, producing compact target-correct evidence without mixing product-specific metrics.
---

# Ascend kernel profiling

Select the target before collecting or interpreting evidence. Product-specific
metrics are not portable.

## A2/A3 Triton timing

Use the repository's `$gz-a3` profile and its native `py311-torch` runtime.
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

For reproducible kernel latency, run the correctness-checked workload through
`scripts/profile_a3.py`. It performs a bounded `msprof op` `BasicInfo` capture
and emits compact JSON with device-task durations and source hashes. Prefer an
exact exported kernel name for timing; an unfiltered capture can select a
framework setup operator instead. Run independent captures for repetitions and use
their median; do not use profiled Python wall time.

Read [A2/A3 msprof-op evidence](references/a2-a3-msprof-op.md) for the command,
JSON contract, acceptance rules, and interpretation boundaries. Preserve the
full transcript so compilation and runtime errors remain actionable.

## Ascend 950/A5 diagnosis

Use this skill for A5 kernel timing, PMU diagnosis, sampled utilization, or
instruction/pipe timeline analysis. It is intentionally target-specific: do
not apply its event IDs, ceilings, or formulas to another Ascend product.

Use the configured TLA integration checkout (`TLA_ROOT`) and its checked-in
BZ-A5 wrappers. Read that checkout's `AGENTS.md` and `$bz-a5` guidance before
launching work. Never substitute direct SSH, SCP, Docker, or an ad-hoc CANN
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
