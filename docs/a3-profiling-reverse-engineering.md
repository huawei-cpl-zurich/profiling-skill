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
aggregate.

Activity fractions are non-temporal observations. They do not provide a
validated A3 capacity denominator or saturation threshold, so this A3 report
requires saturation to remain `unknown`.

## Candidate evidence table

No complete candidate manifest and capture set has been supplied to this
worktree yet. The generator will populate source features, code deltas,
per-case medians, starter and parent speedups, duplicates, and both candidate
and infrastructure failures without discarding any manifest entry.

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
  --captures campaign/captures.json \
  --hypotheses campaign/hypothesis-results.json \
  --output-dir campaign/report
```

The output directory contains `report.json`, `report.md`, and `candidates.csv`.
The JSON is the canonical joined result; Markdown and CSV are deterministic
views. Invalid lineage, unknown candidates, bad latencies, and unvalidated
saturation claims fail closed with exit status 2.

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
