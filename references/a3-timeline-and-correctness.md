# A3: optional instruction and correctness diagnostics

Use these tools when ordinary correctness and pipe evidence leave an important
question unresolved. They are diagnostic options, not required steps for every
optimization. None replaces the benchmark's ordinary device timing.

For synchronization questions, native correctness, hardware timing and scoped
wait/issue counters remain usable without a simulator. Collect and retain that
evidence independently; optional simulator failures have their own availability
and error outcome, not a reason to discard hardware results. Consult the
[A3 wait-counter guidance](a3-performance-reasoning.md#synchronization-and-small-matrix-work)
before requesting instruction-level evidence.

## Compiler evidence

Use `$mlir-triton-dump` to retain compiler invocations, target/options, source
and artifact hashes, and relevant stages. TTIR and adapter IR can establish
loops, copies, scalar extraction or masks at those stages; final buffering,
flags and scheduling may appear only later. Inspect the stage that can answer
the question instead of assuming early IR is the final instruction stream.

Artifacts returned by the measured dispatch are stronger attribution than a
later reconstruction. A reconstruction needs matching compiler/toolchain,
options and target before equivalence is claimed. Matching early stages does
not prove identical final binaries or historical scheduling. PCs are only
meaningful within their bound binary. If the installed decoder cannot decode
instructions, report the gap rather than promoting source placement to fact.

Compile-only inspection need not execute an old kernel: preserve that boundary
when asked for retrospective analysis. Do not silently turn inspection into
another runtime or profiling campaign.

## A3 simulated timelines are not A5 hardware timelines

The automatic A3 `TimelineDetail` instruction route involves context/argument
capture and simulation; its documented automatic route excludes Triton. A
context-save or argument-dump failure does not prove explicit instruction
simulation is unavailable. `TimelineDetail` flow/mapping events alone do not
provide source phases or capacity evidence.

For Triton, use an explicitly supported simulator route from the installed
tool/compiler documentation. Resolve the observed SoC and its matching
simulator library through the execution profile. Keep that setup local to
the capture, not shared runtime configuration. Never borrow A5 `PipeTimeline`
or per-pipe `InstrTimeline` semantics for A3.

If a simulator application emits traces but crashes during Python teardown,
retain them as partial acquisition, not a clean success. Explicit binary/config
replay can avoid that application failure when the exact binary, argument ABI,
workspace/grid requirements and input configuration are verified. Do not copy
another kernel's hidden argument layout or guess its ABI.

Check the installed config-mode syntax: some versions reject combining
`--config` and `--soc-version` despite returning zero. Record the observed SoC
and library as provenance even when child syntax requires omitting the flag.
Require workload correctness, profiler terminal success and bound target events;
exit status alone is insufficient.

Retain capture identity, binary/config hashes, kernel name, core, lane, PC,
raw timestamps and timestamp units. A Chrome `displayTimeUnit` is a viewer
hint, not permission to rescale `ts`/`dur`; verify the exporting tool's units.
Analyze overlap only within one capture and one core's compatible clock.
Do not union unrelated cores or separate replays into a synthetic critical path.
BAR/wait/event controls do not become MTE activity because their operands name
an MTE. Pipeline labels do not establish load/compute/store source phases.

Simulator instruction order and core-local intervals can test scheduling
hypotheses. They do not establish hardware latency, utilization, bandwidth,
capacity saturation or optimization speedups. Source-phase attribution needs
debug/source evidence or a controlled, independently verified mapping.

## Race and memory diagnostics

Numerically correct results do not prove race freedom: overlapping writes can
produce equal values. If ownership, UB reuse or cross-core ordering is suspect,
consider `mssanitizer` racecheck. Synccheck is a different analysis and is not
a substitute. Memcheck bounds/register warnings are separate findings, not
automatically race diagnoses or established allocation-bound violations.

Verify the installed tool's help and instrumentation support with a small
exact-target canary before an expensive run. Native debug/sanitizer compilation
and a fresh sanitizer-enabled Triton cache are language-specific prerequisites.
For supported Triton installations, relevant settings include
`TRITON_ENABLE_SANITIZER=1`, `TRITON_ALWAYS_COMPILE=1` and
`PYTORCH_NO_NPU_MEMORY_CACHING=1`; verify emitted instrumentation rather than
trusting environment settings alone. Delimit tool flags from workload flags:

```bash
mssanitizer --tool=racecheck --log-level=info --log-file=race.log -- \
  python workload.py
```

The optional bare `--full-backtrace` flag is rejected by some versions despite
appearing in help. Usage output, a quiet log or zero exit status is not proof
of completed analysis. Inspect exact target-kernel start/finish sections;
auxiliary Torch kernels can have unrelated warnings or unsupported analyses.
Retain tool version, compile flags, commands, diagnostics and source/PC scope.

An exact completed racecheck finding supports the reported hazard, not every
dependency suggested by the source. A mutation can introduce several hazards.
A negative result is limited to the instrumented path and coverage; absent
instrumentation or incomplete analysis is inconclusive, not race-free. When
qualifying the tool itself, use preserved positive and repaired/disjoint
controls, verify the intended access survives lowering, and repeat checks.
Instrumented durations are never ordinary performance baselines.
