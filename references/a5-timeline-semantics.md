# Ascend 950/A5 timeline semantics

## Three different timelines

### Sample-based AI Core PMU

Plain application profiling with `--ai-core=on --aic-mode=sample-based` writes
`SAMPLE_PMU_TIMELINE` rows containing `timestampNs`, `totalCycle`, `usage`,
`freq`, `coreId`, and `coreType`. Resolve string-valued IDs through
`STRING_IDS`. Use `TASK` joined with `COMPUTE_TASK_INFO` to clip samples to the
target task window; otherwise run an application with no unrelated AI Core
work and trim inactive setup/teardown edges.

Analyze AIC and AIV separately: active windows, utilization distribution,
zero-usage gaps, frequency range, cycles per core, and the coefficient of
variation of per-core cycles. `SAMPLE_PMU_SUMMARY` is supporting evidence.
Samples have no source-stage identity and no per-sample pipe counters.

The A5 CLI sample rate is at most 100 Hz. A microsecond kernel therefore needs
a long repeated correctness-checked burst (the validated FA campaign used
20,000 launches). Sample-mode duration is diagnostic, not the retention gate.

### PipeTimeline

`PipeTimeline` is the common-clock trace for simultaneous cube, vector, MTE,
FIX, and scalar activity. Cross-pipe overlap may be measured only inside one
valid capture containing the required tracks. A scalar-only trace is not a
mixed-kernel common clock.

`scripts/summarize_timelines.py` partitions that common clock at every event
start and end. The profiler exports `ts` and `dur` as cycle-derived
floating-point microseconds; Chrome trace `displayTimeUnit` is only a viewer
hint and does not rescale those values. Boundaries are deterministically
rounded half-up to integer nanoseconds; positive events receive a minimum
one-nanosecond width and the compact evidence records the source unit and
quantization policy. Adjacent intervals retain the resulting
integer-nanosecond boundary and the set of active pipes; an empty set is an
observed no-pipe gap. These
activity phases locate overlap and bubbles but do not establish saturation.
The generated `phase-evidence.json` is accepted by
`analyze_pipe_saturation.py`.

Optional capacity evidence is joined only when it names product `a5`, target
product `Ascend950/V6`, the same raw timeline SHA-256, the common-clock domain,
and an interval whose boundaries exactly equal a generated phase. A task-wide
metric is not copied into smaller phases. A missing exact window remains
`unknown`; an explicitly truncated common timeline retains its activity
phases but rejects all capacity joins.

```bash
python3 scripts/summarize_timelines.py \
  --pipe-timeline PipeTimeline.json \
  --capacity-evidence exact-window-capacity.json \
  --output compact-timeline
python3 scripts/analyze_pipe_saturation.py \
  --product a5 --model references/a5-pipe-capacity-model.json \
  --evidence compact-timeline/phase-evidence.json
```

### InstrTimeline

Capture `InstrTimeline` separately for cube, vector, MTE1, MTE2, MTE3, and FIX.
Decoded events provide task-relative start, duration, core/sub-core, PC, and
instruction identity. Normalize with profiler task-relative timestamps; never
subtract each pipe's first event or claim that separate replays were observed
simultaneously. A combined display of separate captures is an estimated
overlay.

On the validated A5 profiler, `--instr-profiling-freq=300` may be accepted but
reported as ineffective. Do not convert it into a sampling period unless the
platform explicitly confirms interval control. Some releases truncate a pipe
after 1024 records; reject or label incomplete traces rather than extrapolate.

## Phase analysis

For binaries without source-line attribution, infer phases only from
repeatable PC motifs, pipeline ordering, documented dependencies, and
controlled source experiments. Attach a confidence level. PCs are meaningful
only within the same binary.

Phase-local traffic derived from source is a logical byte model. It is not a
measurement of Memory, MemoryL0, or MemoryUB bandwidth and those whole-task
PMU values must not be redistributed across inferred phases.

Keep ordinary task time separately because replay instrumentation perturbs the
kernel. A single instrumented 507015 failure with passing ordinary execution
and other pipe captures invalidates that capture only; repeat the exact pipe
once in a fresh profiler process before diagnosing kernel synchronization.
