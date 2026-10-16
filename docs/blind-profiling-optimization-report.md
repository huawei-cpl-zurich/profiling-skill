# Profiling guidance: blind optimization pilot

Status: benefit unproven; candidate skill remains draft. No skill installation
or merge is implied by this report.

The candidate guidance is frozen at `294f76f`. Its three independent reviews
found no actionable issues. The pilot uses six paired before-kernel tasks,
with identical model, prompt, tools and budget except for the guidance package.
See [the protocol](blind-profiling-optimization-protocol.md) for the rubric and
acceptance threshold.

## Candidate guidance

The draft adds an A3 performance-reasoning reference and an A3
timeline/correctness reference, links them from the entrypoint, and updates the
A2/A3 acquisition and Triton attribution guidance. It explains how code changes
can alter scalar/vector/memory/matrix activity without treating those counters
as calibrated capacity utilization. It separates timeline acquisition limits,
simulator evidence, synchronization interpretation and optional sanitizer/IR
diagnostics. A5 capacity semantics remain product-specific; this A3 pilot is
not new A5 hardware validation.

## Initial protocol calibration (retained, not hidden)

The original prompt did not tell agents about their externally enforced
ten-minute limit. Two agents continued additional tuning until termination;
this was not an infrastructure failure. Further scheduling was stopped to
communicate the budget explicitly and repair reviewed harness limitations.
The skill itself was not retuned against the observed answers.

Controller cancellation also exposed buffered agent logs: one cancelled cell
retained its workspace and tool receipts but lost its in-memory agent event
stream before finalization. The repaired launcher writes that stream live so
an interrupted outer controller does not erase partial observations.

| Task / arm | Outcome | Commit | Relevant observation |
| --- | --- | --- | --- |
| 01 / guidance | 600-second budget timeout | none | Correct candidates measured; further timing prevented final handoff. |
| 02 / control | 600-second budget timeout | none | Obtained scalar-heavy pipe data; continued tile-size tuning. |
| 03 / guidance | completed in 593.9 seconds | `7450caa` | Used compiler IR, pipe counters and optional memcheck; retained timing uncertainty. |
| 01 / control | controller-cancelled for protocol repair | none | Partial work retained; not scored as a candidate failure. |
| 02 / guidance | controller-cancelled for protocol repair | none | Partial work retained; not scored as a candidate failure. |
| 03 / control | controller-cancelled for protocol repair | none | Partial work retained; not scored as a candidate failure. |

The completed matrix agent changed row-wise reductions into a single IEEE
`tl.dot`. Its five-sample medians were 166.516663 and 1.959961 microseconds
(84.96x descriptive ratio). Both batches were flagged low-confidence, and the
agent correctly limited the precision of its claim. It connected early
matrix IR with runtime Cube evidence, treated MTE activity as localization
rather than capacity saturation, and did not claim sanitizer success proved
race freedom. This establishes one observed example of correct evidence use,
not an incremental benefit over a no-skill control.

The control gather agent observed approximately 95.5% scalar activity and low
MTE activity. Its interim explanation emphasized address/control overhead;
without later compiler inspection this did not resolve whether indirect
memory had scalarized. That is a useful distinction to assess in the paired
cohort, not proof that the control was incapable of discovering it.

## Protocol repairs before the scored cohort

Reviewed gaps include full computation-stream validation (an exact selected
kernel alone can omit auxiliary Torch work), the task04 volatile repetition
contract (numerical output alone cannot establish repeated accesses), clear
unsupported native compiler-inspection status, durable transfer/target resume,
and reliable launcher failure/deadline handling. Functional regressions and an
A3 canary are required before restarting fresh agents.

The new cohort reuses these tasks with unchanged skill content and a revised
execution protocol. This is disclosed protocol calibration, not an independent
held-out task-set confirmation. All original results remain retained.

A second interface failure was discovered in that cohort: public `--help`
displayed controller-only `--workspace` and `--bundle` options. The task01
control agent followed those options and could not reach the remote service;
its honest local-only result is a harness-interface failure, not a kernel or
device failure. Its paired guidance run completed, but **both task01 arms are
unscored**. Both were replaced with fresh agents after fixing the public API.
The task02 and task03 pairs completed real measurements on the same original
API version and remain scored, with their UI-recovery costs disclosed.

Scored tasks02/03 used launcher `1da87de`; tasks01/04/05/06 used `31857cf`.
Every pair has identical prompt bytes, model (`gpt-5.6-sol`, low reasoning),
tools, task bytes, 600-second budget and launcher version within that pair.
All used task bundle `ba587fa`, execution tools `7c45a75`, and, when enabled,
skill `294f76f`. This is not a uniform-version cross-task experiment. Skill
content was never changed in response to exposed task performance.

Execution used native A3 through the managed remote-access client and the named
`py311-torch` runtime. The controller freezes local inputs, not the remote
software image or model service. A full compiler-build fingerprint and image
digest are not included in the compact pilot results; reproduction requires a
compatible provisioned runtime and does not promise identical model outputs.

The later tool fix `29ed00b` persists intent before dispatch so a lost reply
cannot cause automatic resubmission. It was independently tested on A3 but
was **not** substituted into the frozen study. Transient observer retries
were handled through existing handles; they do not justify replacing results.

The final task05 control attempt exposed a separate ordinary-flow launcher bug:
local clients canceled while queued still caused obsolete requests to execute.
Repeated requests, transport retries and this stale queue consumed its budget;
the optimized source was committed but never staged before termination. All
four staged requests used the original source, even a candidate-labeled one.
No compilation/runtime failure occurred. The entire task05 pair is unscored,
including its completed guidance arm. A narrow launcher repair skips clients
already disconnected before dispatch and refuses expired-deadline dispatch;
it does not cancel work already submitted or introduce admission control.
Both arms are repeated fresh with launcher `df0e5d0`, after functional socket
and A3 tests pass. One replacement setup mistakenly retained the old read-only
launcher after a failed copy; both agents were immediately controller-cancelled
and that setup is also retained unscored. It is an orchestration mistake, not
an infrastructure failure. The actual replacement verifies launcher bytes and
unchanged prompt before launch. No failed setup is reported as a skill failure
or quietly substituted into a successful cell.

## Paired cohort results

The predeclared gate requires four of six reasoning wins, no reduced
correctness success, and no critical unsupported hardware claims. The first
three pairs tied, making that threshold unattainable even before the last
two pairs finished. All scheduled trials are nevertheless retained and scored;
the conclusion is **benefit unproven**, not acceptance or a universal claim
that profiling guidance cannot help.

Final result: **two guidance wins, four ties, no losses**; both arms completed
all six tasks with committed, measured, correct final candidates. This misses
the four-win threshold. The unsupported causal claims in task04 are a further
reason not to treat the guidance as fully validated. It remains a draft.

Dimensions below are measurement / interpretation / controlled decisions /
uncertainty, each scored 0–2. Graders did not reward raw speedup or merely
opening a skill or profiler.

| Task | Guidance dimensions | Total | Control dimensions | Total | Result |
| --- | --- | ---: | --- | ---: | --- |
| 01 runtime accumulation | 2 / 2 / 2 / 2 | 8 | 2 / 2 / 2 / 2 | 8 | tie |
| 02 indirect memory | 2 / 1 / 2 / 2 | 7 | 2 / 1 / 2 / 2 | 7 | tie |
| 03 small matrix | 2 / 2 / 2 / 2 | 8 | 2 / 2 / 2 / 2 | 8 | tie |
| 04 volatile repeated access | 2 / 1 / 2 / 2 | 7 | 2 / 1 / 2 / 1 | 6 | guidance win |
| 05 launch scheduling | 2 / 2 / 2 / 2 | 8 | 2 / 2 / 2 / 2 | 8 | tie |
| 06 native synchronization | 2 / 2 / 2 / 2 | 8 | 2 / 1 / 2 / 2 | 7 | guidance win |

These are microbenchmark **before-kernel** baselines, not PyTorch baselines or
the earlier GDN/BSA campaign. All durations are microseconds from `msprof op`
BasicInfo, using the median of all five captures. Each ratio is that arm's
before median divided by its final median, not a sum of kernel times.

| Task | Guidance before | Guidance final | Speedup | Control before | Control final | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 9.899802 | 2.219956 | 4.459x | 9.679807 | 2.239955 | 4.321x |
| 02 | 482.850342 | 470.610596 | 1.026x | 478.810425 | 470.490601 | 1.018x |
| 03 | 176.736465 | 1.759965 | 100.420x | 156.776871 | 1.899962 | 82.516x |
| 04 | 29.859404 | 28.619427 | 1.043x | 30.219395 | 29.839403 | 1.013x |
| 05 | 6.399872 | 4.079918 | 1.569x | 6.379872 | 2.459951 | 2.593x |
| 06 | 5.459891 | 3.879922 | 1.407x | 5.459891 | 3.779924 | 1.444x |

Nine of twelve final distributions are flagged low-confidence. Task05 baseline
range/median is approximately 55–57%. No noisy sample was dropped. These are
descriptive point estimates, not precise rankings of guidance arms; particularly
the small task02/04 changes do not establish durable performance gains.

The [compact dataset](data/blind-profiling-optimization-pilot.json) contains all
five samples, dispersion flags, exact before/final source, authoritative local
Git commits, call provenance, retained staged requests, token counts and full
blind grading rationales. Failed frontend attempts remain included in counts
and private raw logs. Source digests identify what was measured; they are not
an admission-control or authentication mechanism.

| Task | Guidance elapsed / tool calls | Control elapsed / tool calls |
| --- | ---: | ---: |
| 01 | 417.8 s / 4 | 412.2 s / 4 |
| 02 | 442.4 s / 8 | 430.6 s / 6 |
| 03 | 359.4 s / 6 | 403.9 s / 16 |
| 04 | 546.2 s / 7 | 409.8 s / 4 |
| 05 | 383.8 s / 3 | 552.5 s / 8 |
| 06 | 409.7 s / 3 | 450.3 s / 4 |

Tool calls include help, retries/resumes and frontend failures, not just remote
jobs. There were 73 tool invocations, 35 staged requests and 34 retained job
handles across the selected cells. Median agent elapsed time was 415.0 seconds
(range 359.4–552.5); infrastructure waiting is included, not kernel execution.

All six guidance agents read the entrypoint; only the matrix agent read a linked
A3 reasoning reference and explicitly inspected early IR. **No scored agent
acquired PipeUtilization or sanitizer evidence.** Automatically retained IR is
not credited as analysis when the agent never read it. The original calibration
did exercise pipe/IR/sanitizer acquisition, but cannot establish a paired benefit.

## What the agents actually learned from the evidence

- Runtime scalar task: both arms independently replaced the repeated
  accumulation with an algebraic expression. BasicInfo verified the reduction
  in task duration; source reasoning supplied the proposed mechanism. This
  does not demonstrate that a scalar utilization counter diagnosed it.
- Indirect-memory task: both arms changed tile size and measured small gains.
  Neither used pipe metrics or compiler IR to distinguish scalarized indirect
  memory, latency, and program overhead. Their proposed scheduling explanation
  remains a hypothesis. The blind graders gave both 7/8.
- Small matrix task: both arms used one IEEE `tl.dot` instead of row-wise
  reductions. BasicInfo changed from vector/32 blocks to cube/1 block. The
  guidance arm additionally read the A3 reasoning reference and inspected
  early TTIR; its claim stayed limited to compiler structure and the observed
  execution type, not physical traffic or capacity saturation. Both scored
  8/8. This is observed correct guidance use, but not an incremental score win.
- Volatile repeated-access task: the control tuned launch geometry; the
  guidance arm also removed masks that were redundant for the declared fixed
  shape. Trusted executed-TTIR qualification retained runtime repetitions,
  volatile loads and in-loop output stores. The guidance arm scored 7/8 versus
  6/8, chiefly through better uncertainty reporting, but both made an
  unsupported causal explanation for a rejected launch configuration. Neither
  measured the pipe activity needed to isolate that explanation.
- Launch task: both arms correctly connected changed launch grids with observed
  BasicInfo block dimensions. The guidance arm chose BLOCK256, the control
  BLOCK1024. Both scored 8/8, regardless of their different point-estimate
  speedups. The guidance notes correctly say pipe activity is not capacity,
  but that caveat does not establish that pipe evidence cannot help distinguish
  causes of a timing change. No pipe-level cause was demonstrated.
- Native task: both arms replaced 32 exact increments with a constant fill.
  The control also removed a terminal barrier, incorrectly describing a
  `V_MTE3` dependency as output-copy completion. The guidance arm retained that
  barrier and avoided the unsupported completion claim: 8/8 versus 7/8. Passing
  numerical checks alone do not prove the control's final kernel is race-free
  or establish how compiler/runtime draining works.

The volatile-task guidance notes say duplicated loop/control work "dominates";
the control says a regression was caused by lost parallelism. Timing alone
does not establish either mechanism. These were flagged by the blind reviewer,
not silently converted into demonstrated hardware conclusions. Different
physical devices on the same server are provenance, not an assumed hardware
class difference or an explanation for variation.

## Interpretation limits

Agents can discover obvious algebraic, matrix-mapping and launch changes from
code plus task timing without the skill. The native synchronization task also
allows constant-folding its repeated vector increments; that can bypass the
wait-placement lesson the original microbenchmark was intended to exercise.
Success on such a task is not evidence of understanding wait counters.

Performance covers each declared default workload, not all correctness inputs:
task01 uses work=64; task02 times contiguous runtime indices while permutation
and duplicate-hotset patterns are correctness checks; task04 times 128
repetitions with 25% selected lanes. Do not generalize these speedups across
other work counts, index distributions or repetition counts.

Reading the entrypoint is not equivalent to using linked profiling facts.
Actual evidence use must appear in the decision log. In particular, correctly
avoiding a saturation claim is useful calibration, but without a pipe capture
it does not establish a counter-driven diagnosis.

Each operational tool call includes four correctness checks, complete bounded
compute-stream discovery, and multiple profiler captures. Tool waiting is part
of the ten-minute budget. Several agents repeated baseline/check requests;
those costs are retained, not relabelled infrastructure failures or removed to
improve the skill result. This pilot therefore tests a short constrained
workflow, not the usefulness ceiling of longer profiling investigations.

The task04 validator qualifies its documented canonical loop and ordinary
bounds-mask form; it is not an adversarial equivalence verifier. Deliberately
false masks with compensating atomic stores are outside the common-code review
scope. Neither seeded numerical checks nor this compiler qualification proves
all-input correctness or race freedom.

Arm-anonymous graders saw task code, measurements, commands and reasoning but
not the arm key or supplied guidance text. Packet contents can still suggest
an arm's identity, so this is not perfect content blinding. The coarse rubric,
six pairs, reused calibration tasks, staggered runs, and noisy microsecond
captures prevent a population-wide causal claim.

The coarse rubric also has a ceiling: four tasks received equal high scores
even when one arm recorded additional evidence. Across calibration and repair
there were 24 optimization-agent starts: twelve selected cells and twelve
explicitly unscored attempts (six initial calibration, two API-affected task01,
two queue-affected task05, two canceled setup attempts). Interface canaries are
separate. This is not twelve clean first attempts or an independent held-out
confirmation after each repair.

## Reproduction and retention

See [reproduction instructions](blind-optimization-reproduction.md) and the
[exact shared prompt and rubric](blind-profiling-optimization-protocol.md).
The driver freezes six before-only bundles, a twelve-file skill allowlist,
first-party runtime helpers, model, prompt and provenance manifest. It runs
four balanced waves with at most three isolated agents; it never automatically
retries a cell or hides failed results. It uses the installed managed remote
client rather than copying or bypassing transport code.

Each agent has an independent `experiment` branch and local commits. No agent
branch was merged. Private local evidence retains their Git histories, command
logs, streamed messages, compact profiler data, source snapshots and all
calibration attempts. Archives exclude authentication/configuration state and
sockets. Large raw profiler binaries remain remote; no credential-bearing
agent state is published with this report.

The skill draft should remain unaccepted. A future independent evaluation
would need held-out tasks where different plausible code-level causes cannot
be distinguished from timing alone, and enough budget to acquire the deciding
counters. That is a recommendation, not a newly scheduled campaign or permission
to retune this skill against these exposed answers.

## Implementation validation

- Integrated source suite after the queue fix: **1,883 passed**, three subtests;
  four existing Python multiprocessing/fork warnings.
- Domain tools: **21 tests passed locally and on native A3**, including lost
  dispatch reply recovery without duplicate submission.
- Launcher: **130 combined focused local tests passed**; the three new real
  socket/deadline regressions also passed on native A3. Prior API canaries and
  the actual isolated optimization trials exercised the A3 runtime boundary.
- Reproduction driver: **four functional tests passed locally and on native
  A3**, including pair equality, concurrency, retained failures, output reuse,
  and repairing missing dependencies before reserving the output directory.
- Task qualification: original before kernels passed execution smokes; native
  task04 tests accepted preserved loops and rejected removed repetitions or
  volatility. Candidate acceptance additionally audits the complete compute
  stream, not just a selected kernel's duration.

These tests validate interfaces and retained outcomes; they do not replace the
blind usefulness gate or prove untested synchronization behavior.
