# A3 profiling reverse-engineering report

> **Reproducible scaffold — campaign evidence pending.** This document defines
> the report contract and the controlled experiments. It intentionally does not
> turn historical activity ratios into capacity or saturation claims. Populate
> it with retained campaign output using `scripts/build_profile_reverse_report.py`.

## Question

Which A3 profiling observations help an agent explain and improve a Triton
kernel, which provide context only, and which invite conclusions that the
evidence cannot support?

The campaign compares every recoverable candidate from CANNBot-only,
CANNBot-plus-new-profiler, guarded-plus-new-profiler, and
guarded-plus-old-profiler treatments. Candidate identity includes source and
launch configuration. Exact duplicates remain separate lineage entries while
sharing captures only through an explicit `duplicate_of` mapping.

## Measurement protocol

For each candidate and BSA development case `47`, `46`, `49`, `44`, and `43`:

1. Compile and check correctness before profiling.
2. Discover the exact kernel name in an unprofiled run.
3. Collect BasicInfo and PipeUtilization for that exact launch.
4. Retain at least three valid samples, their median and dispersion, the source
   and configuration hashes, command, target, physical device, and durable job
   handle.
5. Count compilation, runtime, and correctness failures as candidate failures.
   Retry transport and environment failures without changing candidate identity.

The generated report computes latency and speedup per case, then takes the
geometric mean across comparable cases. Parent-relative comparisons explain a
round's code delta; starter-relative comparisons show total progress. Missing
cases are unavailable rather than silently removed from a claimed common-case
aggregate. Candidate, starter, and parent must each cover the complete case set
declared by the run contract before an aggregate latency or speedup is emitted.
Rejected samples remain in the report even when a case has no accepted timing.
Producer-schema cases require explicit `accepted_sample_indexes`: when absent,
all captured samples remain diagnostic-only and produce no median or aggregate.
Auxiliary cases remain visible but do not enter declared-suite speedups. Parent
lineage cycles are rejected rather than emitted as comparative evidence.

Activity fractions are non-temporal observations. They do not provide a
validated A3 capacity denominator or saturation threshold, so this A3 report
requires saturation to remain `unknown`.

## Candidate evidence table

No complete candidate manifest and capture set has been supplied to this
worktree yet. The generator will populate source features, code deltas,
per-case medians, starter and parent speedups, duplicates, and both candidate
and infrastructure failures without discarding any manifest entry.

For the producer schema, source features come from each execution's static
`features` analysis and code deltas preserve the occurrence's structured
`parent_feature_deltas` exactly. Free-text `hypothesis` remains a separate
field. Parent speedup is computed only for `parent_occurrence_id`, starter
speedup only for an explicitly identified starter, and an occurrence without
such evidence is marked `unlinked` rather than assigned a guessed lineage.
The canonical starter is `starter_occurrence_id`, which must agree with the
single occurrence labeled `lineage_role: starter`; `is_starter` is accepted
only as a legacy alias when the canonical field is absent.

Known historical deltas to test causally include local sequence loop bounds,
direct sparse-head rank, tile size, whole-tile skipping, scalar block-mask
decisions, fine-grained K/V load predicates, scheduling parameters, matrix-tiled
dot products, and K/V reuse across query rows. These observations motivate the
experiments; they are not causal conclusions.

## Controlled hypotheses

The machine-readable specifications live in
`references/a3-profiling-hypotheses.json`. Every experiment has before and after
kernels, a sweep, a minimum effect, and a falsification condition. The report
generator assigns exactly one outcome:

- **supported:** the median controlled effect meets the predicted direction and
  minimum magnitude;
- **refuted:** it meets that magnitude in the opposite direction;
- **inconclusive:** valid evidence exists but the effect is smaller;
- **unavailable:** no valid controlled measurement exists.

The planned comparisons are:

| Hypothesis | Before | After | Principal sweep |
|---|---|---|---|
| Local K bound | Total packed-K loop | Current sequence `sk` loop | Unrelated packed length |
| Direct head rank | O(H) prior-head scan | Direct rank | Head count |
| Key tile size | 32/64 tokens | 128 tokens; 256/512 probes | K length and boundaries |
| Empty-tile skip | Always compute | Skip the entire rejected tile | Block sparsity |
| Scalar block decision | Lane-replicated decision | One decision per block | Block count |
| Load predicate crossover | Valid-region K/V loads | Final-predicate loads | Mask irregularity |
| Schedule tuning | Default warps/stages | Best cheap candidate | Legal schedules |
| Matrix dot | Row-wise `tl.sum` | Tiled `tl.dot` | M, K, and head dimension |
| K/V reuse | One query/program | Query tile/program | Query tile height |
| Mask-mode control | Dense | Causal/streaming/sparse | Mode and sparsity |

## Multi-phase timeline experiments

Timeline capability is itself an empirical hypothesis. Single-launch kernels
will create deliberately long and unequal phases in these patterns:

- load-heavy → vector-heavy → store-heavy;
- the same work in reverse order;
- load → vector → load;
- serialized and deliberately overlapped load/compute variants;
- one-phase duration sweeps; and
- one fused launch versus equivalent separate kernels.

The analysis compares the injected order and duration with every available A3
timestamp, task/block/sub-block relation, and flow identifier. If the export
cannot recover the known order, the result is a useful negative finding: the
derived skill must not use that export for phase-local pipe diagnosis.

## Signal classification

Signals are classified only against controlled outcomes:

- **useful:** repeatedly tracks the controlled performance effect;
- **contextual:** describes changed work but does not diagnose the outcome;
- **misleading:** suggests improvement while latency or controlled work regresses;
- **unavailable:** the required signal is not exported or has no valid capture.

No signal has been accepted as useful yet. In particular, a large activity
fraction is not called saturation, a reduced movement fraction is not assumed
to improve latency, and aggregate activity is not presented as a timeline.

## Reproduction

```bash
python scripts/build_profile_reverse_report.py \
  --manifest campaign/candidate-manifest.json \
  --results campaign/candidate-results.json \
  --hypotheses campaign/hypothesis-results.json \
  --output-dir campaign/report
```

The manifest and results may be supplied directly in the producer schemas
`profiling-skill/bsa-candidate-manifest/v1` and
either legacy `profiling-skill/bsa-candidate-results/v1` or merged
`profiling-skill/bsa-candidate-merged-results/v1`. Merge contract, present and
missing shards, missing executions, and source checkpoints remain explicit.
Each occurrence remains a report
candidate; occurrences sharing an execution are linked explicitly rather than
collapsed. Unrecoverable CANNBot occurrences remain visible with no invented
starter or parent. The output directory contains `report.json`, `report.md`, and `candidates.csv`.

Structured PipeUtilization evidence retains its metric, exact selector, and
block/sub-block components. Numeric CSV strings are normalized within the same
component identity and summarized only across comparable samples; AIC and AIV
rows or distinct sub-blocks are never flattened into one metric average. The
merged producer may contain v2 samples whose latency is the maximum duration
across separately captured AIC/AIV components. Those component selectors,
evidence digests, and per-component provenance remain nested and auditable;
only the declared aggregate duration participates in latency comparisons.

Microcampaign hypothesis input must be report-ready paired evidence: top-level
`raw_cells` may retain the original cells, but every `results` item must name a
hypothesis and every valid measurement must contain before/after values and
their before/after cell IDs. Raw per-variant cells are never treated as causal
hypothesis evidence.
The JSON is the canonical joined result; Markdown and CSV are deterministic
views. Invalid lineage, unknown candidates, bad latencies, and unvalidated
saturation claims fail closed with exit status 2.

### Preliminary evidence and timeline controls

The report retains each paired measurement, its falsification condition, all
raw microbenchmark cells, the measurement campaign identity, and the separate
reporting revision. Original failure diagnostics, hashes, and remote provenance
are preserved rather than reduced to an error label. A legacy snapshot lacking
modern environment bindings must remain explicitly legacy; do not assign it a
new target contract or merge its timings with another server's measurements.

Scalar core-time measurements cannot validate phase order, duration, overlap, or
kernel boundaries. A timeline hypothesis remains `unavailable` unless retained
before/after cells have available, selector-complete semantic evidence. When
present, the campaign's separate temporal assessments determine support,
refutation, or inconclusiveness, independently of scalar timing changes. The
current preliminary captures lack that evidence; their timing pairs remain
controls, not proof that timelines recover the injected pattern.

For large preliminary corpora, `--compact-markdown` keeps verbose sample and
profile bodies in the canonical JSON appendix and bounds each displayed failure
diagnostic to 2,000 characters. CSV and JSON evidence are unchanged; the default
Markdown output remains fully expanded.

## Evidence provenance

- Query: `8dfb07b37dc34532974f00418ea65306`
- Pipe semantics:
  `ref://profiling-skill/common/profiling/repos/ascend-msprof-pipe-memory-timeline-semantics`
- Analysis semantics:
  `ref://profiling-skill/common/perf-analysis/repos/ascend-msprof-pipe-memory-timeline-analysis`
- KernelGen corpus:
  `ref://profiling-skill/triton-ascend/profiling/repos/ascendc-kernelgen-data`
- Open documentation miss: `70834351cf9a888d284f9d62e57e4b53`

The report uses only these filtered citations. It does not assert undocumented
field semantics or access the raw reference collection.
