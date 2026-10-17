# Supplied pipe-evidence diagnostic experiment

This prospective test separates metric information from interpretive skill
guidance. It does not replace the prior natural-adoption result, prove live
acquisition works, or authorize merging/installing the skill.

## Frozen design

Use six qualified tasks, including educational negatives, with three conditions:
timing only; timing plus pipes; timing plus pipes and unchanged skill revision
`294f76fc602f9d121cb9826bfcdc6c9e524c1bd2`. This is 18 trajectories and
36 fresh contextless agent invocations, at most three concurrent, using
`gpt-5.6-sol` with low reasoning effort. Each stage has 240 seconds.

Before: source, task contract, baseline timing samples and two candidate source
changes only. Seal the diagnosis, preferred next candidate, predicted outcome
and uncertainty. After: a new isolated process receives that sealed answer and
the same public packet; the two pipe conditions also receive the baseline's
three batch-one pipe CSVs. Only the final condition receives the frozen skill.
Prompts are identical across conditions within each stage. Timing-only receives
no additional measurements, controlling for a second opportunity to reason.

Hide all candidate timings/profiles, batch two, compiler dumps, diagnoses,
qualification reports, previous agents and repository history until both answers
are frozen. Do not require a predetermined diagnosis or reward collection alone.
Retain timeout/partial outputs and failures; never silently replace an agent.

## Packet interface and reproduction

Run the stdlib exporter with explicit local paths:

```sh
python scripts/export_pipe_evidence_packets.py \
  --qualification /path/to/natural-task-qualification.json \
  --archives /path/to/retained-archives \
  --bundles /path/to/prepared-task-bundles \
  --output /path/to/fresh-output
```

The exporter verifies all twelve recorded archive hashes, all candidate source
hashes and task shapes, successful four-seed checks, single-kernel discovery and
selectors, all five timing samples, three pipe captures and compiler hashes
before creating output. It reads archives without extracting them.

`manifest.json` lists six task IDs and hashes every output except itself.
`public/taskNN` contains exactly the six declared files: task metadata, neutral
instructions, baseline source, two candidate sources and baseline timing.
`pipes/taskNN` contains three byte-identical original pipe CSVs and a provenance
manifest. Preserve original metric names, units, per-core groups, missing values
and capture boundaries; no interpreted summaries are included.
`private/gold.json` holds both batches' outcomes, source provenance, full retained
compiler text, capture identities and predeclared bounded lessons. Mount neither
private gold nor the output root into an agent; copy only the allowed per-stage
public/pipes files, plus the permitted skill. The runner must verify these
manifest hashes before freezing its cohort.

The report's whole-kernel timings are instrumented BasicInfo task durations,
diagnostic rather than canonical minimally instrumented latency. Device migration
within a server is not an exclusion criterion.

## Prospective assessment

Freeze the private answer key before launch, grounded in both acquisition
batches and retained compiler evidence. Educational negatives are valid lessons:
noisy nulls do not prove equivalence, and early IR does not prove binary identity.
Baseline-only evidence cannot establish every hidden compiler detail; reward
appropriate uncertainty and discriminating tests rather than impossible inference.

Three arm-anonymous assessors each review one two-task family. Score each stage
0–2 on four dimensions: supported code-to-execution explanation; evidence and
uncertainty calibration; discriminating next experiment; falsifiable predictions
consistent with held-out outcomes. Do not penalize agents for unavailable
evidence or reward metric repetition. Record critical unsupported claims.

Apply the same anchors to each dimension: **0** means absent or contradicted;
**1** means generic or partly supported; **2** means specific, sound and bounded
by the available evidence. Missing measurements are not themselves an error;
explicit calibration of what remains unknown is a valid response. Private IR
and candidate captures validate the assessor's mechanism claims, not additional
agent requirements or grounds to penalize inaccessible evidence.

Predeclared bounded outcome expectations are:

| Tasks | Candidate A | Candidate B |
| --- | --- | --- |
| task01/02 | Latency indeterminate | Latency indeterminate |
| task03 | Predicate latency unclear | Warp-hint latency unclear |
| task04 | Predicate medians slower but noisy; indeterminate or qualified regression is admissible | Warp-hint latency unclear |
| task05/06 | BM16 regresses | BK64 improves |

No precise speedup or capacity/saturation claim is required. Appropriate
uncertainty can earn full credit where held-out outcomes are inconclusive.

For each task and arm, define delta as after-total minus before-total. Compare
deltas first for pipes versus timing repeat, then pipes-plus-skill versus pipes.
A contrast win requires both a strictly larger delta than the comparator and a
substantive, correct, evidence-supported change; otherwise record a tie. Do not
redefine wins after inspecting results. Report absolute after-scores secondarily
to expose ceiling effects. A positive diagnostic signal requires at least four
of six wins and a tested-arm critical-unsupported-claim after-count no greater
than the comparator's after-count; report before-counts too.
Report every task, ties, negative findings and uncertainty;
these six pairs are a qualitative pilot, not a population estimate.

Useful pipes without an incremental skill advantage suggest redundant guidance
or adoption/discoverability issues. Traceable skill-induced overclaims suggest
misleading guidance. No improvement establishes only no demonstrated benefit in
these cases, not universal metric uselessness. No claim here changes the earlier
skill merge gate or permits retrospective tuning.

## Functional checks

```sh
python -m unittest discover -s tests -p test_pipe_evidence_packets.py
```

Tests export synthetic complete cohorts through the function and CLI, preserve
raw rows/NA values, verify public/private separation, reject changed sources,
archives, shapes, selectors, timing, seeds, captures and compiler artifacts, and
refuse to overwrite existing output. A capable A3 target must additionally run
these device-less helper tests through the installed remote-access client before
PR completion. No new kernel workload is needed for this diagnostic cohort.
