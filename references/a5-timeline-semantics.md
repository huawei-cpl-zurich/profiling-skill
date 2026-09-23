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
