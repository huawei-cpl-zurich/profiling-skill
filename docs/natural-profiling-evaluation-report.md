# Natural-use profiling skill assessment

## Decision

The completed pilot **does not pass the frozen skill-usefulness gate**. Guidance
wins two of six reasoning comparisons, controls win none, and four tie. The
prospective requirement was at least four guidance wins, no reduced correctness,
and no critical unsupported hardware claims. The last two conditions hold in
the retained evidence; the reasoning-win requirement does not.

This is a negative acceptance result, not evidence that the skill is useless.
It does not authorize merging, installing, changing the rubric, forcing new
profiling, or substituting tasks. In particular, successful operation of the
harness is not a successful test of the skill's incremental diagnostic value.

## Frozen experiment and assessment method

Campaign `ascend-natural-pilot-20261016T082538Z` ran all twelve isolated agents in
four waves, with at most three agents concurrent, in approximately 71.92 minutes.
All exited normally, stayed inside their 1,200-second budgets, and committed work.
Guidance was frozen at `294f76fc602f9d121cb9826bfcdc6c9e524c1bd2`; both arms used
`gpt-5.6-sol`, low reasoning effort, identical ordinary prompts and neutral tools.
No task, guidance, budget or scoring rule was changed after launch.

See the [prospective protocol](natural-profiling-evaluation-protocol.md) and
[qualification report](natural-profiling-qualification-report.md). Qualification
established interpretable positive or negative lessons, not that finding those
lessons required pipe metrics. That distinction matters to this result.

Three fresh, contextless reviewers each assessed two task pairs from
arm-anonymous packets. Packets included public contracts, baseline/final code,
all candidate snapshots, agent reasoning and commands, complete agent-authored
notes, measurements, source identities, commit lists and final compiler outputs.
Arm labels were permuted and provenance identifying the mounted guidance was
redacted. Reviewers were instructed not to inspect original campaign directories
or look up identities. Nested notes omitted in the first export were restored
and checked by the same reviewers before completion; scores did not change.
Exact-final-source TTIR and adapter artifacts were then audited by the same
reviewers to complete the compiled-computation gate, without crediting agents
for automatically saved artifacts they had not inspected themselves.

Scores were translated back to guidance/control only after reviewers recorded
their anonymous assessments; the coordinator retained the private label map.
This is one reviewer per pair, not three independent scores per pair. Wording
and strategies may reveal clues despite provenance redaction; this is not a
guarantee of perfect blinding. The coarse 0/1/2 rubric also has a ceiling effect.
These are six qualitative comparisons, not a population-level causal estimate.

## Reasoning scores

Dimensions are measurement/provenance, interpretation/calibration, controlled
decisions, and uncertainty/regressions, each scored 0-2. A winner requires a
strictly higher total and a substantive reasoning advantage; performance alone
does not break a reasoning tie.

| Task | Workload | Guidance dimensions | Control dimensions | Result |
|---|---|---|---|---|
| 01 | Gather, four recurrence rounds | 2/2/2/2 | 2/1/2/2 | Guidance, narrowly |
| 02 | Gather, 32 recurrence rounds | 2/2/2/2 | 2/2/2/2 | Tie |
| 03 | Masked nonlinear softmax, width 1024 | 2/2/2/2 | 2/2/2/2 | Tie |
| 04 | Masked nonlinear softmax, width 4096 | 2/2/2/2 | 2/2/2/1 | Guidance, narrowly |
| 05 | Fused matmul, short K | 2/2/2/2 | 2/2/2/2 | Tie |
| 06 | Fused matmul, long K | 2/2/2/2 | 2/2/2/2 | Tie |

Task01 guidance tested a useful counterfactual: sigmoid versus the original
expression inside the selected four-row grouping. That exposed an interaction
hidden by the nearly neutral one-row expression test. The control also found a
real modest gain, but inferred identical lowering and expensive division from
aggregate timings without corresponding compiler or pipe evidence.

Task04 guidance repeated its promising two-warp candidate; the median reversed
from 22.020 to 24.580 us on the same physical device. It withdrew the claim and
restored the exact baseline. The control retained a nominal 1.93% improvement
from heavily overlapping measurements without confirming it. Its rejected
zero-clamped-maximum experiment also treated real-arithmetic softmax invariance
as sufficient numerical justification: a large negative selected score can
underflow under that rewrite. The final control code restores stable
normalization, so this is a reasoning limit, not a final shipped defect.

The four ties are substantive: controls used real measurements, controlled
changes, rejected regressions and acknowledged uncertainty too. Task06 control
found a substantially faster output geometry, but both arms made sound
evidence-supported decisions under the frozen reasoning rubric. Search breadth
or speedup cannot be substituted retrospectively for that rubric.

## Retained timing results

All values are microseconds: the median of five separate exact-selector
`msprof op` BasicInfo `Task Duration(us)` captures. They are diagnostic device-task
durations, **not uninstrumented launch latency, PyTorch speedups, or summed pipe
durations**. Each arm has its own measured starting baseline. Speedup is that
baseline divided by the selected final median, not a comparison against the
other arm's independently measured baseline.

| Task | Guidance baseline | Guidance final | Speedup | Control baseline | Control final | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 2566.129 | 2500.030 | 1.026x | 2566.889 | 2527.869 | 1.015x |
| 02 | 612.568 | 576.048 | 1.063x | 620.808 | 580.008 | 1.070x |
| 03 | 88.838 | 37.979 | 2.339x | 99.658 | 35.359 | 2.818x |
| 04 | 23.440 | unchanged | 1.000x | 24.880 | 24.400 | 1.020x* |
| 05 | 69.259 | 41.599 | 1.665x | 75.618 | 41.079 | 1.841x |
| 06 | 62.079 | 45.559 | 1.363x | 61.059 | 32.059 | 1.905x |

`*` Task04 control's small apparent gain is not established as repeatable.
Task04 guidance restored byte-identical baseline code; its baseline observation
is source-matching evidence, not an independent final timing capture. Restoring
already-measured exact bytes also occurs in other arms and is recorded as such.

Task02 control measured three baseline medians: 618.888, 620.808 and 623.308 us.
The table uses its selected 620.808-us reference, while all three are retained;
the final gain is approximately 6.28-6.95% across those references. Task05 control
also measured its final source at 40.719 us; task06 guidance at 45.619 us and
task06 control at 32.859 us. The table's call IDs and every repeat remain in the
data, rather than silently selecting or discarding individual slow samples.

Large task03/05/06 improvements have separated baseline/final ranges in retained
calls. Small launch-knob differences frequently overlap. Task05 final arms are
too close relative to dispersion to declare a performance winner. Task06
control's two final-source ranges are wholly below guidance's two ranges: a
clear descriptive advantage in these runs, separate from the reasoning tie.
No timing is invalidated merely by migration among devices on the same server.
There was no randomized interleaved final comparison, so exact percentages and
causal hardware attribution remain limited.

## Did agents use the new profiling information?

All six guidance agents read the main skill. Only task05 visibly read the linked
A3 performance-reasoning reference. Both arms had the same neutral measurement
interface; availability of `pipes` and `ir` was explicit in its help.

Across all agents there were **87 timing requests, one explicit IR request,
and zero pipe requests**. Three timing requests failed at the candidate stage;
84 successful timing requests retained 420 samples. The successful IR request
also checked correctness and discovered the kernel. There were 85 successful
native requests and 340 successful seeded checks in total. Automatically saved
compiler artifacts do not count as agent diagnostic use.

| Task | Guidance's observable information use | What remains unestablished |
|---|---|---|
| 01 | Exact timing samples drove row grouping and the expression counterfactual; noise and restored-source provenance were documented | No measured pipe bottleneck or compiler explanation of the interaction |
| 02 | Explicit IR inspection showed scalar coefficient loads and vector exponential work; this deprioritized coefficient hoisting before geometry tests | The serial-gather clue was not developed into a measured pipe-guided change; control reached a comparable grouping decision without IR |
| 03 | Timing rejected selected-load and algebra trials and selected row batching | No evidence of physical sparse transfers or a measured memory bottleneck; slower predication alone does not explain its cause |
| 04 | Contradictory repeat timing changed selection to retaining the baseline | No pipe-guided localization; good calibration is not proof of a hardware diagnosis |
| 05 | Read A3 reasoning, then tested tile/warp hypotheses and retained samples | No visible acquisition/interpretation of pipe activity; geometry changes do not isolate Cube versus transfer costs |
| 06 | Timing selected larger K tiles and repeated the exact final source | No measured scheduling/capacity explanation; control found a better output geometry without the skill |

The two narrow wins concern controlled experimentation and not overclaiming
noise. Those behaviors are consistent with the guidance, but this pilot cannot
attribute them uniquely to it: the prompt and tool help already teach evidence
retention, kernel identity, resumable work and repeated timing. Controls also
demonstrated those behaviors. Self-reported skill use is not causal evidence.

There is no observed refusal to collect pipes: agents chose other experiments,
and no pipe call reached acquisition. Consequently this pilot cannot establish
that pipe metrics were misleading, unavailable, useless, or interpreted well.
It establishes lack of voluntary pipe adoption under this task/tool/budget
combination. More budget did not produce pipe use in these runs. Even the
educational-negative tasks could be investigated meaningfully with code and
timing alone; qualification did not prove a diagnostic need for pipes.

## Correctness, failures and compiled computation

All twelve final source hashes match successful native timing evidence, with
four supplied seeded checks and one discovered compute kernel per matched call.
The gather/masked selector is `task_kernel`; matmul's exported timing selector
is `task_kernel_mix_aic`. Source and exact-final TTIR/adapter inspection preserve
the full input-dependent computation: gather plus every recurrence and both
reductions; nonlinear masked score plus stable normalization; or full K reduction
plus bias, every recurrence, SiLU and residual. No omitted stage or auxiliary
computation kernel was found. A mixed Cube/vector kernel is not two kernels.

All three intermediate failures were candidate compilation failures, retained
and recovered, not infrastructure failures:

- Task02 guidance, `e13e519348d24e26`: unsupported `rows[:, 0]` tensor indexing;
  corrected the output-row expression and reran.
- Task04 control, `14b114d94ef9491f`: `triton.language` has no `libdevice` attribute
  in this runtime; rejected the tanh variant.
- Task05 control, `6c3d79f7ee9945b7`: BiShengIR `hivm.hir.load`/block transform
  failure for the tested BK256 variant; restored and remeasured known-good code.

Checks cover the supplied shapes and four generated cases, including structured
access/masks and empty/single-selected rows. They are not a proof over arbitrary
finite normal inputs, numerical extremes, cancellation, other shapes or races.
No new runtime, independent recompilation or sanitizer run was performed for
this assessment. A default `sanitizer` request field is not a sanitizer result.
Intermediate compiler artifacts support preserved lowering, not a complete
binary-level proof or exact hardware scheduling explanation.

## Evidence and conclusion

[Assessment data](data/natural-profiling-evaluation.json) retains the full
anonymous reviews, unblinded dimension scores, all 88 native request identities,
source hashes, selectors, checks, timings and samples, final compiler hashes,
failure fingerprints, elapsed time and commits. Full commands, errors, code
snapshots, compiler text and archives remain in the retained campaign folders;
the central task record locates those artifacts. No bulky profiler trees or
machine provisioning paths are published here.

The report was checked against retained receipts: sample arrays match exported
rows, medians recompute, staged sources match request hashes, and every final
source has successful timing and retained compiler artifacts. This is an
evidence assessment, not a software behavior change requiring a new device run.

**Conclusion: keep this negative result and do not declare the skill accepted.**
Measurement discipline and two controlled-decision lessons are promising. The
new code-to-pipe knowledge itself was not exercised by optimizer agents, and
the predeclared usefulness threshold was missed. Any further skill change or
test design is a new decision, not an unreported extension of this pilot.
