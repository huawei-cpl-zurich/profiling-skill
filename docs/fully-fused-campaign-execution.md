# Fully fused A3/A5 campaign execution

`fully_fused_campaign_execution.py` is the admission and reporting boundary
for the 4-treatment × 3-task × 2-product campaign. It composes the schema-v4
audited scheduler and production launcher; it does not provide a second path
around their hash, treatment-isolation, device-admission, or evidence checks.

## Frozen inputs

Before preparing a run, create a versioned JSON archive seal with schema
`profiling-skill/pre-campaign-archive/v1`. Each `roots` entry binds an absolute
zero-copy retained root to the deterministic tar digest, and each
`completion_files` entry binds an absolute compact terminal artifact. The seal
itself is authenticated by `seal_sha256`. Verification recomputes every value;
an old root is never copied, resumed, or modified.

The schema-v4 manifest must declare exactly A3 and A5, matmul/GDN/BSA, all four
treatments, and four rounds. A rankings JSON maps each product to all three
tasks and an ordered unique case list. Preparation binds those rankings, 24
unique `experiment/<run>/<cell>` branch names, and the expected 96 round
commits. Put `ranked_cases` and `ranked_cases_sha256` from this receipt into the
production runtime configuration and provenance. The launcher materializes
each unmerged branch when its cell is first dispatched; experiment agents do
not merge, push, switch branches, or create commits.

```text
python scripts/fully_fused_campaign_execution.py verify-archive --seal SEAL.json
python scripts/fully_fused_campaign_execution.py prepare \
  --manifest MANIFEST.json --rankings RANKINGS.json \
  --archive-seal SEAL.json --output PREPARED.json
```

## Product gates and launch

Before dispatching any candidate cell, run the pinned known-good fully fused
matmul through the same A3 and A5 controller/backend paths. Each compact gate
receipt must have three positive timing samples, a durable handle, exact
ranked-case coverage, and one logical Triton launch per case. Pass the two
receipts through the gate command. An infrastructure outcome pauses admission;
if it contains a durable handle, observe that exact handle. It is not a
candidate failure and does not consume a round.

```text
python scripts/fully_fused_campaign_execution.py gate \
  --rankings RANKINGS.json --a3-receipt A3_GATE.json \
  --a5-receipt A5_GATE.json --output GATES.json
```

Only after `GATES.json` says `passed` should the existing production campaign
launcher run. Its resource pool continuously refills every compatible healthy,
idle A3 or A5 slot, so the campaign uses maximal safe product-compatible
concurrency. Lost observers resume exact retained handles. Handleless proven
transport failures may be retried with the identical candidate; an uncertain
dispatch remains pending. Compile, runtime, correctness, and fusion failures
are counted candidate outcomes. A non-fused candidate gets no timing.

Every round commit is independently verified and retains candidate code,
manifest, structured command events, compact controller and profiler evidence,
source citations, expected result, sanitized reasoning summary, observed
result, decision, and postmortem. Large vendor reports remain remote.

## Terminal report

First generate the ordinary audited campaign report, then compact it:

```text
python scripts/fully_fused_campaign_execution.py report \
  --manifest MANIFEST.json --campaign-report REPORT.json \
  --output COMPACT.json --markdown TABLES.md
```

Compaction fails unless all 24 cells match their frozen product/task/treatment
and branch identity and each has four distinct round commits. A3 and A5 tables
show raw baseline and candidate latency, speedup, variability, fusion proof,
compact profiler evidence, bottleneck, best round, and full branch lineage.
Discarded infrastructure attempts are counted separately from candidate
failures.
