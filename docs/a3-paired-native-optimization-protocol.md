# Paired native wait-guidance validation

This is an end-to-end skill comparison, not an answer-recall test. Both arms
receive profiling guidance: old snapshot `b5a3949` versus only the new wait
lesson delta on that same snapshot. Do not compare against no skill or include
unrelated skill changes. Freeze exact skill trees, sources, model/settings,
neutral prompt and rubric before scored execution. Keep hardware qualification
and a real contextless interface canary separate from scored sessions.

## Task export and tool boundary

```bash
python benchmarks/blind_native_wait_tasks.py --task intercore --output <fresh-bundle>
python benchmarks/blind_native_wait_tasks.py --task intra --output <fresh-bundle>
python benchmarks/blind_native_wait_tasks.py --task phase --output <fresh-bundle>
```

Each public bundle exposes only a before kernel, identical candidate copy and
contract. Trusted host/runner are outside agent visibility. Exported starting
implementations have early waits; no after version, historical profile, timing
answer or source history is included. All in-core IDs are TPipe allocated and
released; inter-core identifiers retain matched signal/wait semantics. Tasks:

- Inter-core producer/consumer: two participants, seeded input, independent
  compute and balanced/imbalanced producer variants. Both output regions checked.
- Intra-core copies and vector work: two input regions, independent vector work,
  sum and accumulated-work outputs, one fused launch.
- Phase control: tiny and larger repeated-work variants, preserving input copy,
  runtime repetition loop and independent accumulated vector result.

Freeze 04 replaces the original constant accumulated-work loop after calibration
agents legitimately collapsed it to a constant and hoisted unused copies. Those
old results are calibration only, not accepted optimization performance. Work
now starts from independent seeded float32 inputs (different per inter-core
participant) and executes four separately rounded logistic-recurrence operations
per round. Phase inputs contain distinct payload vectors per repetition, each
mixed into the recurrent state after that round's independent work; earlier
copies cannot simply be dropped in favor of the final copy. Four UB vectors use
128KiB at the largest case. The CPU oracle requires NumPy float32; the native
host uses volatile intermediates to prevent contraction. Exact Ascend/CPU
rounding parity needs fresh hardware qualification, not relaxed tolerances.
Host task-kind configuration is captured before including candidate source to
avoid a benign macro-name collision; this is not an adversarial C++ sandbox.

Use the existing `BlindLauncher`, not a new scheduler or transport. Its opaque
domain helper is the existing `blind_optimization_tools.py` with the exported
bundle, configured A3 target and named runtime. Both arms have exactly the same
commands. `experiment-tool timing --implementation baseline|candidate` returns
five exact-kernel BasicInfo samples; `pipes` returns three PipeUtilization
captures. `--variant NAME` selects the documented task variant for measurement.
Every execution first checks four seeded cases, cycling through all supported
variants even when one is selected for measurement. Final acceptance independently
checks **all** variants regardless of what the agent chose to measure.
`events --variant tiny|long` is supported by the native tasks and returns three
warmups plus seven stream-event samples without msprof. It has a different
measurement scope and may include host-submission gaps; it is not intrinsic
kernel latency. It succeeds independently of msprof availability: its one-launch
per-iteration contract is supported by the immutable kernel-only trusted host,
not discovery of an actual exported selector or inspection of the device stream.
Event/correctness failures still fail the operation. `timing` and `pipes`
verify the exported single-launch stream through profiler selector discovery;
`check` verifies the trusted host contract without requiring the profiler.
Profiler commands disable the event loop. Preserve noisy
samples; no selective retries or required speedup. Missing optional Memory or
native compiler-IR tools are capability gaps, not zero counters or agent faults.

The domain helper copies candidate bytes before submission, compiles the source,
returns compiler diagnostics and checks outputs. Profiler operations verify one
exported compute launch; `events` labels its weaker trusted-host evidence explicitly.
Native build intermediates may contain the trusted seeded host oracle. Compact
agent evidence therefore retains compiler stdout/stderr but omits native build
IR, preprocessed sources, disassembly and metadata containing source annotations.
Native candidate/device-only IR is explicitly unavailable until the compiler
producer can establish that role; filenames alone are not sufficient provenance.
Full artifacts remain remote for coordinator-only inspection.
The same retained remote handle is observed after interruptions. No direct
SSH/client bypass or duplicate dispatch. Use at most two replacements only
after terminal infrastructure failure; infrastructure-invalid cells are unscored.
Compilation/runtime/correctness failures and agent budget exhaustion are scored.

## Frozen sessions and prompt

Three tasks × two independent repetitions × two skill arms = twelve fresh
isolated sessions, max three concurrent. Same explicit model configuration
throughout; no prior conversation or other installed skill is mounted. Use
`build_prompt(1200)` from the existing launcher unchanged, then append the exact
artifact paragraph below in both arms, with 120-second final reserve. Do not
require pipe evidence, hint at an optimization
or request a prescribed interpretation. Agent commits code, command/results,
evidence-backed reasoning, uncertainties and postmortem; never merges its branch.

Exact common artifact paragraph (as frozen for cohort 04):

> Write EXPERIMENTS.md with your hypothesis, expected and actual results, the
> commands run, failed attempts, evidence citations and remaining uncertainty.
> Commit candidate.asc, EXPERIMENTS.md and the selected compact JSON/CSV evidence
> needed to audit your conclusions on your assigned branch. Do not commit full
> profiler archives or compiler build trees. A chat-only report or a code-only
> commit does not complete this task.

The first unscored interface canary acquired baseline BasicInfo/Pipe evidence
and checked a comment-only candidate across all variants, but committed only
candidate.asc: evidence remained untracked and reasoning existed only in final
chat. Retain this artifact-protocol failure as a negative result. Freeze 02
changes only the common prompt paragraph; reuse already qualified task/helper
bytes and do not rerun hardware qualification merely for this prompt change.
Before scored cells, run a new trivial contextless artifact canary: baseline
check, comment-only candidate check, and verify its committed log and compact
evidence. Keep this canary separate from optimization and paired usefulness.

Counterbalance four waves (rep/task labels are controller-only):

| Wave | Three cells |
| --- | --- |
| 1 | intercore rep1 new; intra rep1 old; phase rep1 new |
| 2 | intercore rep1 old; intra rep1 new; phase rep1 old |
| 3 | intercore rep2 old; intra rep2 new; phase rep2 old |
| 4 | intercore rep2 new; intra rep2 old; phase rep2 new |

Freeze before/after correctness, exact selectors, source/binary identities and
compact raw reports. All before variants must compile, pass and support actual
BasicInfo/Pipe captures before scored sessions. Do not start scored work merely
because an exporter or mocked interface test passed.

## Frozen anonymous assessment

Score four dimensions 0 absent/incorrect, 1 partially sound, 2 sound:

1. Measurement/provenance: actual baseline/final evidence, selector, samples and
   correct distinction between event brackets and profiler task durations.
2. Interpretation/calibration: supported code/resource/wait explanation; no
   claim that an event bucket or zero scalar wait maps directly to source waits.
3. Controlled decisions: hypothesis, meaningful code experiment and evaluation,
   preserving work and full output correctness; unchanged well-investigated code
   is not automatically a reasoning failure.
4. Uncertainty/regressions: recognize noise, unavailable data, conflicting evidence
   and safety limitations; retain negative and inconclusive findings.

A paired win is a strictly higher total reasoning score for the new arm on the
same task/repetition. Demonstrated usefulness requires at least four of six wins,
no correctness-success reduction and no critical false hardware claims in new
arms. Ties are not wins. Report performance separately; no speedup threshold.
Critical false claims include event ID as universal source attribution, scalar
zero wait as no synchronization, cross-replay counters as a simultaneous timeline,
simulator time as hardware speedup, or noisy median ratios as established gain.
Otherwise report usefulness unproven, preserving all results. This six-pair pilot
does not establish general statistical superiority or A5 applicability.
