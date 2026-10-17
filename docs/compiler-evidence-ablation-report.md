# Compiler and profiling evidence diagnostic

Completed: twelve valid committed assessments in twelve fresh contexts, zero
timeouts or failed contexts, about3.5minutes elapsed and maximum concurrency3.
Three fresh assessors graded anonymous answers using a private key frozen before
the first context. Compiler evidence improved structural explanations, but did
not consistently improve performance predictions: both compiler-equipped mask
agents predicted a regression for a candidate that actually improved26.2%.

## Question and design

Does actual compiler lowering help an agent connect Triton source to a measured
mechanism, beyond source/timing or profiling evidence alone? This is not a
skill-versus-no-skill comparison and does not test autonomous evidence acquisition.

Three tasks each have four fresh, isolated assessment contexts: timing/source,
plus baseline profiling, plus compiler artifacts, or both. Historical arm labels
are `timing`, `pipes`, `compiler`, `both`; `pipes` includes baseline Memory as
well as PipeUtilization, not just pipe counters. All agents receive the same
frozen profiling skill, neutral prompt, model/reasoning settings, 300-second
wall budget and30-second finish reserve. Maximum concurrency is three.

The tasks compare runtime indexed loads with block-size/warp requests, periodic
masking with source predication/block size, and a small fp16→fp32 matrix product
with buffering/K-tile changes. Candidate timings and profiles are held out.
Compiler arms see unannotated actual artifacts for all three implementations,
not merely early source IR or an expert's selected excerpts.

All nine source workloads passed real device correctness and actual compiler
capture. Qualification used five independent BasicInfo samples and three each
PipeUtilization/Memory captures per implementation, rotating order between
implementations. Compilation/debug acquisition was separate from timing.
No new NPU work is dispatched by the offline agent runner.

## Reproducible public inputs

[Public input archive](data/compiler-evidence-ablation-inputs.tar.gz) contains
123 declared inputs plus its manifest: seven common public files per task,
thirteen baseline profiling files per task, and twenty-one compiler files per
task. Source, specialization, compiler options, target and actual artifact
identities are retained. Every diagnostic metric report is bound to exactly
one raw BasicInfo row with the same baseline kernel selector.

Only explicitly configured path prefixes were normalized. Compiler manifests
retain distinct original and served hashes; no operation text was rewritten
or answer-relevant passages highlighted. Duplicate identical raw compiler
artifacts were removed, not distinct stages. No candidate performance result,
compile duration, expert explanation, profiler binary or full capture tree is
included. Missing/unvalidated counter units are not silently converted.

- Archive SHA256: `a927866ab641d6ddd1713bfbc1f2a27d63c9d0cc49b2e672d3f5bc54a97e68d5`.
- Frozen skill: repository revision `9f4a503`, `SKILL.md` plus complete
  `references/` and `scripts/` trees:1 main file,12 references,63 scripts,
  76 files total. Do not substitute an installed skill.
- [Model configuration](data/compiler-evidence-ablation-model.json): explicit
  `gpt-5.6-sol`, low reasoning effort.
- Observed target: `Ascend910_9382`; Triton module3.2.0, distribution3.2.2.
  Exact CANN/profiler version unavailable; retained as unknown rather than
  inferred from an image label.

Extract the archive into a fresh directory, reconstruct the skill snapshot
from the named Git revision, and use the [protocol](compiler-evidence-ablation.md):

```bash
python scripts/run_compiler_evidence_ablation.py \
  --packets <extracted-public-inputs> --skill <frozen-skill-snapshot> \
  --model-json docs/data/compiler-evidence-ablation-model.json \
  --output <fresh-output-outside-workspace> \
  --forbidden <orchestration-workspace>
```

An actual-packet fake-launcher smoke completed twelve cells without invoking a
model. The coordinator additionally checked a real sandbox preflight. Public
inputs contain no original machine paths, compile-duration fields or expert
annotations. The private key and held-out dataset were separately frozen.

## Held-out measurements

These were collected before agents ran and were not agent inputs. All values
are medians of five independent BasicInfo captures, in microseconds, not sums
of launches or durations from diagnostic counter replays.

| Task | Baseline | Candidate A | Candidate B | Qualification limits |
| --- | ---: | ---: | ---: | --- |
| Indexed load |481.170380|478.850433|481.030365|A differs only0.48%; B essentially unchanged |
| Periodic mask |530.129395|391.232178|540.749207|A26.2% lower, B2.0% higher; stable samples |
| Small matmul |8.779824|8.059839|6.739865|All noisy; A overlaps baseline, B lower retained range |

Indexed-load range/median is0.66–0.94%; A's tiny shift does not establish a
robust general improvement or its cause. Matmul range/median is8.61–14.12%.
A's lower median is not a confirmed gain; B's nonoverlapping retained range is
suggestive, not a confidence interval or a universal performance guarantee.
All raw samples and exact source/options/target bindings are in the
[assessment dataset](data/compiler-evidence-ablation-assessment.json).

## What the agents concluded

Each table entry gives the next candidate chosen and predicted A/B directions.
`?` means the agent explicitly withheld a direction. A reasonable next experiment
need not be the fastest held-out candidate, and an uncertain prediction is not
automatically a failure.

| Task | Timing/source | +Profiling | +Compiler | +Both |
| --- | --- | --- | --- | --- |
| Indexed load |A; improve/?|A; improve/?|A; improve/?|A; improve/?|
| Periodic mask |A; improve/regress|A; improve/regress|A; regress/regress|B; regress/regress|
| Small matmul |B; ?/improve|B; ?/improve|B; ?/improve|A; improve/improve|

**Indexed load:** compiler-equipped agents grounded their diagnosis in the
scalar per-element gather loop, index DMA and bulk stores, instead of assuming
a vector gather from tensor syntax. Increasing BLOCK does not remove the
per-element scalar path; requesting more warps does not guarantee a hardware
effect. Profiling agents correctly observed scalar-heavy per-row activity.
Nevertheless, all four chose and predicted the same candidates. The assessor
considered their conditional A improvement plausible but its sign indeterminate
from available evidence; B uncertainty was well calibrated. More concrete
explanations did not produce a demonstrated decision advantage here.

**Periodic mask:** both compiler-equipped agents correctly saw dense input
movement, the retained scalar modulo loop and additional mask operations. This
prevented an unsupported fourfold physical-traffic explanation, but both then
predicted A would regress. A actually improved26.2%. Extra static operations and
the absence of an observed sparse transfer do not determine final latency. The
profiler-plus-compiler agent chose B first; the compiler-only agent still chose
to test A, explicitly allowing its regression hypothesis to be falsified.
Timing-only and profiling-only agents predicted the observed signs, but their
conditional traffic-elision explanation was not established. The exact reason
for A's benefit remains unresolved; a correct sign is not a correct cause.

**Small matmul:** compiler evidence distinguished a real late buffering/event
transformation for A from unchanged early adapter IR, and verified B's16→8 K
loop reduction with larger operand tiles. Profiling evidence highlighted high
MTE2 activity and low Cube activity, not achieved HBM saturation. Timing,
profiling and compiler-only agents chose B while withholding A's sign. The
combined agent preferred A, predicting a larger gain than B; retained samples
do not support that ordering. A's overlap with baseline makes its own benefit
inconclusive. Generated buffering establishes opportunity, not measured overlap.

## Categorical assessment and evidence use

All twelve next measurements were judged useful under the frozen rubric.
Mechanism attribution was supported for all four indexed-load and matmul
answers; for mask, timing/profiling-only explanations remained indeterminate,
whereas compiler-equipped structural explanations were supported. None of
this establishes an exact causal account of the observed speedups.

The mask profiling-only answer received one critical unsupported-claim finding:
it relabelled unvalidated raw `KB` counter sums as exact `KiB` physical traffic.
The combined matmul answer also inferred executed correctness from source and
capture provenance without a public recorded correctness result; the assessor
flagged this separately as a noncritical unsupported claim. Held-out qualification
did pass, but agents could not claim to have observed that omitted result.
No numeric score, winner count or post-hoc acceptance threshold was introduced.

Command audit, not citations alone, found:

- All6 profiling-equipped agents issued commands with actual metric content
  retained in their outputs; all6 compiler-equipped agents inspected actual
  TTIR/HIVM content or differences. This is not another evidence-refusal result.
- Main skill text appeared completely in retained outputs for11/12 agents;
  the A3 reasoning guide for10/12. The combined matmul agent did not open either,
  despite having the same skill mounted. Availability is not consumption.
- No agent explicitly opened the four additional compiler-consumer input files;
  their compiler inspections concentrated on the labelled TTIR/HIVM files.
  Merely listing those additional files in a manifest did not make them used.
- Inspections included bounded `sed`/`head` excerpts, filtered searches and
  diffs. Do not claim that agents read entire compiler artifacts. Retained JSONL
  aggregate output is published, but does not prove attention or that every byte
  was displayed to the model. File mentions and citations are distinct from reads.
- All12 receipts, committed answer bytes, immutable source/evidence bindings,
  per-context skill snapshots and isolation checks passed. No merge commits,
  model reruns, timeouts or failed contexts. One shell JSON-validation command
  failed and was corrected; it was not an infrastructure or kernel failure.

The dataset retains full answers, public agent summaries, command strings and
aggregate outputs/hashes, commit identities, anonymous reviews, the frozen key,
raw held-out timing samples and execution/input identities. It excludes hidden
model reasoning and credentials. The pre-agent key hash was
`757cd92cb6945df0e2e13b9aa8ffd40df661a524772794fa3b13877084ee9e39`;
it remained unchanged through assessment. A post-run resume audit validated all
twelve existing results without dispatching a replacement context.

## Implications for the skill

1. **Use lowering to verify execution structure, not rank candidates by static
   operation count.** The mask result is a concrete counterexample: correct
   structural diagnosis, wrong performance prediction. Keep cheap direct timing
   in the loop even when the IR appears unfavorable.
2. **Give compiler stages meaningful discoverable names.** All compiler agents
   used labelled TTIR/HIVM, while opaque consumer-input filenames went unopened.
   Future guidance should identify which manifest input is adapter, host LLVM or
   device LLVM and explain what each can establish, without highlighting answers.
3. **Retain the activity-versus-bandwidth and stage-versus-schedule cautions.**
   They supported calibrated gather/matmul interpretations, but did not prevent
   every unit mistake or an overoptimistic combined-evidence ranking.
4. **Unknown causes belong in the skill too.** The mask candidate's actual gain
   survives qualification, but physical-byte reduction and extra-op slowdown are
   both unsupported causal stories. Record the unresolved effect rather than
   forcing an explanation or declaring the compiler wrong.

These are recommendations, not source changes made by this diagnostic. Three
selected cases, one context per condition, imperfect blinding (citations reveal
evidence availability), shared skill exposure and different reading choices
cannot establish a general causal benefit of profiling or compiler evidence.
Anonymous assessors received the union of each task's evidence and were told
not to penalize unavailable artifacts. No automatic merge/install follows.
