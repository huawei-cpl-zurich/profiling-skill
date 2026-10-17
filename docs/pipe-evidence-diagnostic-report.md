# Does supplied pipe evidence improve agent decisions?

## Result

This diagnostic found **two local benefits from combining pipes with the skill,
but no consistent advantage meeting the prospective threshold**.

| Contrast | Substantive positive tasks | Required | Outcome |
|---|---:|---:|---|
| Pipes versus timing-only reassessment | 0/6 | 4/6 | Not demonstrated |
| Pipes plus frozen skill versus pipes | 2/6 | 4/6 | Local benefits, gate not passed |

All 18 trajectories completed both stages: 36 valid, distinct model contexts,
zero timeouts, zero failed cells and zero replacements, in 610.705 seconds
(10.18 minutes). Actual peak concurrency was three. This is a supplied-evidence
interpretation test, not a new kernel-optimization or live-acquisition test.

The results do **not** establish that pipes are useless or that the skill is
misleading. Agents correctly extracted activity information in many cases, but
often did not improve a decision already supported by source inspection. The
skill helped with calibrated uncertainty and one input-movement hypothesis;
aggregate activity still did not reliably identify the responsible code.
The earlier [negative natural-use assessment](https://github.com/huawei-cpl-zurich/profiling-skill/pull/124)
is unchanged. No skill was merged, installed or retuned.

## Frozen design and scoring

The [prospective protocol](pipe-evidence-protocol.md) and private answer key were
frozen before launch. Six existing tasks each had three independent trajectories:
timing only, timing plus pipes, and timing plus pipes plus skill revision
`294f76fc602f9d121cb9826bfcdc6c9e524c1bd2`. All used `gpt-5.6-sol`, low reasoning
effort, 240 seconds per stage and a 30-second finishing reserve. Prompts were
identical across conditions within each stage.

Before: source, shape/contract, baseline timing and two proposed source changes.
After: a fresh context received its exact committed before-answer, plus the
condition's evidence. Pipe conditions received only the three baseline batch-one
CSVs. Candidate outcomes/profiles, batch two, compiler artifacts and expert
diagnoses remained hidden. Timing-only got another opportunity to think without
new measurements. No remote execution tools were available to the agents.

Three fresh assessors each reviewed one workload family with arm labels hidden.
They committed their before-scores to separate files before inspecting after
answers. All before scores and critical-claim lists match those frozen files.
The coordinator mapped labels to conditions only after assessments completed;
the coordinator had retained the private mapping. Evidence/prose could reveal
condition clues, so this is not a guarantee of perfect blinding.

Four dimensions each scored 0-2: supported execution explanation, calibration,
discriminating next experiment and falsifiable predictions. Gold compiler data
verified actual behavior; it was not information agents were required to know.
Missing data was not itself an error. For each contrast, a substantive win
required a strictly greater before-to-after score change **and** a meaningful,
correct evidence-supported change. Positive relative deltas caused only by a
comparator deteriorating did not count as information gains. Neither contrast
had any critical unsupported claims in the assessed answers.

## Scores and observed information use

Scores below are before → after, maximum eight. They are reasoning scores,
not latency or kernel-success measurements.

| Task | Timing only | Pipes | Pipes + skill | Main observation |
|---|---:|---:|---:|---|
| 01 Gather, four rounds | 8 → 8 | 8 → 6 | 8 → 8 | Raw scalar activity encouraged an unsupported unrolling gain prediction |
| 02 Gather, 32 rounds | 8 → 8 | 8 → 8 | 6 → 6 | Skill arm corrected one hypothesis but introduced another unsupported signed gain |
| 03 Masked rows, width 1024 | 7 → 8 | 7 → 7 | 7 → 7 | Timing-only reconsideration improved calibration; raw-pipe arm did not inspect its CSVs |
| 04 Masked rows, width 4096 | 7 → 7 | 7 → 7 | 7 → 8 | Skill arm used mixed activity and uncertainty to withdraw unsupported latency signs |
| 05 Matmul, short K | 7 → 7 | 8 → 8 | 8 → 8 | Correct input-movement observations corroborated an existing choice |
| 06 Matmul, long K | 8 → 7 | 8 → 8 | 7 → 8 | Skill arm connected input activity to the smaller tile's logical RHS duplication |
| **Total** | **45 → 45** | **46 → 44** | **43 → 45** | Do not treat unequal starting totals as a randomized performance comparison |

Both pipe conditions inspected all three CSVs in five tasks; the skill condition
also inspected them in task03, giving 11/12 pipe-equipped after stages with
observed inspection. All six skill arms read the main skill; four also read the
linked A3 reasoning guide. Reading is not itself a scored benefit.

The numeric relative advantage for pipes versus timing is -2, 0, -1, 0, 0, +1
across tasks01-06. The last +1 comes only from timing-only deterioration and is
not a substantive pipe gain. Skill versus pipes is +2, 0, 0, +1, 0, +1; task01's
+2 similarly reflects avoiding deterioration, not a new score-improving change.
The predeclared substantive skill wins are task04 and task06 only.

## What this teaches about the profiler and skill

### High scalar activity does not locate the source of the cost

Gather captures reported scalar fractions around 0.969 in task01 and 0.862 in
task02, with vector fractions around 0.024 and 0.161 respectively. Agents could
correctly describe scalar-dominated activity without claiming capacity saturation.
That is useful characterization, but not a source map.

Task01 raw pipes and task02 pipes-plus-skill both promoted static recurrence
unrolling and a low-confidence improvement prediction. The private lowering
instead retains a 256-element scalarized gather loop; removing recurrence loop
control does not remove that access path. The held-out unrolling timing direction
is not robust across acquisitions. A scalar-active profile cannot distinguish
addressing, scalarized memory, recurrence control and stalls by itself.

This is a concrete diagnostic boundary: use the activity to choose a compiler or
access-path question, not to assume that the most visible source loop is the
cost. The frozen skill already discusses scalarized indirect loads in its linked
A3 guide, but neither gather skill arm opened that guide. More definitions alone
are not the demonstrated missing ingredient here; discoverability and following
the explanation through to compiler attribution remain issues.

### Masking observations helped calibrate claims, not prove sparse traffic

Task03 skill correctly read per-capture MTE2 medians of approximately
53.48/47.27/53.58 us and vector time around 28.6 us, while preserving overlap
and capacity caveats. It still retained a signed predicate improvement hypothesis
that the noisy held-out results do not establish. Baseline counters do not tell
whether a new predicate becomes sparse transfers or dense copy plus select.

Task04 skill computed the actual 48-row capture means: vector approximately
7.828/7.830/7.828 us, MTE2 approximately 3.163/4.146/3.028 us. It used mixed
overlapping activity and missing candidate/capacity evidence to withdraw signed
latency claims. This is a real calibration improvement, but neither a measured
speedup nor proof that one pipe determines the critical path. Its next selected
experiment remained the same.

Timing-only task03 also improved simply by reconsidering the source. That is
why a second-thought control was necessary: not every better after-answer is a
profiling benefit.

### Matmul input activity can support a specific code hypothesis

In task05, raw-pipe and skill agents correctly separated Cube rows from vector
groups. Cube MTE2 mean ratios were about 0.816/0.826/0.852 across captures,
while Cube arithmetic ratios were about 0.0835/0.0848/0.0746. They avoided
capacity and additive-latency claims. This added evidence corroborated the
already-selected larger-K candidate, rather than changing the decision.

Task06 skill made the clearest code-linked update: baseline block0 reported
Cube MTE2 about 36.59 us versus Cube arithmetic about 4.20 us, repeated in
another capture. It connected that input activity to the source-derived downside
of the smaller-M tile: more programs and duplicated logical RHS loads. Its
prediction changed from indeterminate to cautiously slower for that candidate,
and its proposed test would compare transfers/emitted loads against parallelism.

Held-out data supports that hypothesis: smaller-M MTE2 row medians rise from
roughly 22-24 to 44-46 us and timing worsens in both batches. BK64 lowers MTE2
medians to roughly 15-17 us and is faster, while vector arithmetic stays nearly
constant. These are corresponding observations, not an exact decomposition of
latency or a proof of physical byte counts, saturation or cache behavior.

## Limits on the conclusion

- Nine of 18 before answers already scored eight. Correct additional evidence
  could not raise those scores. Qualitative corroboration is retained separately.
- There is one trajectory per task/condition and one assessor per family. Model
  variation, unequal starting quality and regression to the mean remain; six
  tasks represent three related families, not six independent populations.
- Four tasks intentionally have bounded negative or noisy outcomes. Source
  inspection already supported strong answers on several cases. This test did
  not establish that pipe metrics were necessary for the task choices.
- Agents received baseline profiles only. Comparative candidate profiles and
  compiler dumps could answer other questions, but were deliberately withheld.
- The packets omit an explicit observed SoC/CANN/profiler-version declaration.
  The retained provenance is A3, but agents were not given those facts. Assessors
  did not penalize appropriate uncertainty or require unavailable product facts.
  Pipe field names also reveal backend context not explicit in timing-only input,
  so the contrast does not isolate numeric counters from all platform context.
- No critical forbidden claims were found. Some weak causal or signed predictions
  received partial credit; they are not evidence that the skill systematically
  misleads agents.

**Conclusion:** pipe evidence supplied useful activity descriptions and, with the
skill, two bounded diagnostic improvements. It did not consistently improve
decisions under this protocol. The unresolved practical gap is connecting
aggregate observations to the actual code path and choosing a falsifiable next
measurement—not a demonstrated lack of metric definitions. This does not change
the earlier merge gate or authorize another campaign or skill rewrite.

## Evidence, validation and reproduction

[Assessment JSON](data/pipe-evidence-diagnostic-assessment.json) includes every
answer, exact command list and output hashes, commits, stage metadata, input
identities, frozen source/model/skill/prompt hashes, all anonymous reviews and
their original before-scores, every contrast calculation and the held-out case
summaries. Full event streams, command outputs and private compiler text remain
retained under the campaign identified there; central task records locate them.

The [68.4 KiB input archive](data/pipe-evidence-diagnostic-inputs.tar.gz) contains
the exact public sources, timings, three raw baseline CSVs per task and manifest.
It contains no private gold or compiler dumps. The manifest retains a fingerprint
of the private key, but the runner does not require that private file to execute.
Fetching the frozen skill is documented in the protocol; do not install it
globally for this experiment.

Reproduction requires the runner/launcher at reachable commit `27698ae` (PR126)
and the packet/protocol code at `95dde02` (PR125). Integrate those revisions in a
development worktree; the original local integration commit is recorded as
provenance, not assumed to be a remotely fetchable commit. Verify runtime file
hashes against the assessment JSON. Extract the input archive into a fresh
directory, then run:

```sh
python scripts/run_pipe_evidence_diagnostic.py \
  --packets /path/to/extracted-inputs \
  --skill /path/to/exact-frozen-snapshot \
  --model-json docs/data/pipe-evidence-diagnostic-model.json \
  --output /path/to/fresh-results --forbidden /path/to/checkout
```

This reproduces conditions and evidence, not deterministic model answers.
Every stage is a new isolated process/repository and commits without merging.

Validation: 63 combined local functional tests passed, including real isolation
checks and deadline metadata. The packet helper passed 13 native A3 tests. Final
supported runner helper gate passed 19 native tests with three real-isolation
checks explicitly skipped because that A3 host lacks Bubblewrap. An expanded
native legacy sandbox attempt failed nine tests for that missing capability;
it remains recorded and is not claimed passing. All real isolation checks passed
locally. A separate final verifier independently audited all 36 stages, fresh
contexts, exact commits, sealed imports, unchanged evidence and concurrency.
No live acquisition or new NPU kernel behavior is validated by this diagnostic.
