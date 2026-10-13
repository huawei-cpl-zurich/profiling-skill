# BSA candidate profiling

`scripts/bsa_candidate_profiler.py` inventories every recoverable autotune
option from the retained four-treatment BSA campaign and coordinates a
resumable A3 profiling run. It deliberately retains attempt occurrences:
retries and repeated options remain traceable even when exact duplicate
source/configuration identities share one device capture.

## Inventory

```bash
python scripts/bsa_candidate_profiler.py inventory \
  --campaign "$CAMPAIGN_ROOT" \
  --config experiments/bsa-profile-candidates.json \
  --output "$EVIDENCE_ROOT/manifest.json"
```

The manifest includes source text because it is a local runtime artifact. Do
not commit generated manifests: candidate sources and profiling results belong
under the ignored evidence directory. Hash-only options are reconciled from
another retained request when the same source hash is available. Otherwise
they remain represented as `source_unavailable`; a hash is never treated as
executable source.

Each execution also contains deterministic source features for loop bounds,
head-rank lookup, block sizes, whole-tile skipping, scalar block decisions,
K/V masking, warps/stages, reduction style, and query/grid tiling. Each
non-incumbent option links to the exact incumbent occurrence declared by its
own request and records the feature delta. A next-round incumbent is linked to
the prior winner only when the prior winner hash resolves to one unambiguous
successful occurrence. Partial or ambiguous CANNBot history stays unlinked;
the inventory never guesses lineage.

## Executor contract

The coordinator invokes an executable as:

```text
EXECUTOR --request REQUEST.json --response RESPONSE.json
```

The executor is a local adapter which may dispatch only through the global
`cpl-remote` client. Its device workload runs inside an approved,
self-contained A3 bundle; it must not open SSH or use a raw remote client.
Every request uses schema
`profiling-skill/bsa-candidate-executor-request/v1` and contains the complete
candidate source and parameters.

For `action=discover`, the executor must run all requested correctness cases,
then perform an unfiltered `msprof op` BasicInfo capture. Success is:

```json
{
  "status": "ok",
  "kernel_names": ["block_sparse_attention_kernel"],
  "basic_info": {"rows": 1},
  "evidence_sha256": "...",
  "provenance": {
    "target": "bz-a3-1", "runtime": "py311-torch",
    "physical_device": 0, "command": ["..."],
    "handle": "remote:bz-a3-1:job:..."
  }
}
```

For `action=capture`, it must execute exactly the requested case and exact
discovered selector with `PipeUtilization`, analyze the report in place, and
return only compact rows:

```json
{
  "status": "ok",
  "duration_us": 545027.2,
  "pipe_utilization": {"rows": []},
  "evidence_sha256": "...",
  "provenance": {
    "target": "bz-a3-1", "runtime": "py311-torch",
    "physical_device": 0, "command": ["..."],
    "handle": "remote:bz-a3-1:job:..."
  }
}
```

Candidate failures use `compile_error`, `runtime_error`, or
`correctness_error` and include retained diagnostics. Transport, device,
profiler-tool, missing-response, and malformed-evidence failures use
`infrastructure_error`; they do not consume a valid sample or change candidate
status permanently.

## Resume and stability

```bash
python scripts/bsa_candidate_profiler.py run \
  --manifest "$EVIDENCE_ROOT/manifest.json" \
  --config experiments/bsa-profile-candidates.json \
  --executor "$BUNDLE_EXECUTOR" \
  --output "$EVIDENCE_ROOT/results.json"
```

The result is atomically checkpointed after discovery and every capture.
Rerunning the same command skips terminal candidates and completed cases.
`--max-actions N` provides a bounded smoke or scheduled slice.

A case completes when the tightest three valid samples have relative range
`(max-min)/median <= 5%`. Samples outside that accepted subset remain in the
record. After seven valid but unstable samples, the case becomes
`measurement_unstable`, classified as infrastructure/measurement rather than
a compilation or runtime defect. Large vendor report trees remain remote.

Use `status --results RESULTS.json` for a compact terminal/retry summary.

## Parallel shards

Execution assignment is `sha256(execution_id) mod shard_count`, so it is
deterministic and does not alter candidate or occurrence identity. Run shards
in separate result paths:

```bash
python scripts/bsa_candidate_profiler.py run \
  --manifest "$EVIDENCE_ROOT/manifest.json" \
  --config experiments/bsa-profile-candidates.json \
  --executor "$BUNDLE_EXECUTOR" \
  --shard-count 8 --shard-index 0 \
  --output "$EVIDENCE_ROOT/shard-0.json"
```

Indexes are zero-based. Every invocation with the same output must retain the
same count and index. Merge any completed or checkpointed subset with:

```bash
python scripts/bsa_candidate_profiler.py merge-results \
  --manifest "$EVIDENCE_ROOT/manifest.json" \
  --config experiments/bsa-profile-candidates.json \
  --input "$EVIDENCE_ROOT/shard-0.json" \
  --input "$EVIDENCE_ROOT/shard-1.json" \
  --output "$EVIDENCE_ROOT/merged.json"
```

The merger preserves each execution state, sample, diagnostic, and provenance
object verbatim. It rejects repeated shard indexes, overlapping executions,
wrong-shard assignments, and manifest, sample-policy, or shard-count mismatch.
The merged result explicitly lists missing shard indexes and execution IDs, so
a partial merge cannot look complete.

## Remote execution policy

Before dispatch, run `cpl-remote capabilities TARGET` and
`cpl-remote preflight TARGET`. Stage the self-contained bundle with
`cpl-remote upload`, submit it with the target's named runtime, and retain the
printed handle. If observation is interrupted, observe that same handle; do
not resubmit. The executor evidence must retain the selected target, physical
device, exact command, handle, kernel selector, sample count, and compact
BasicInfo/PipeUtilization output alongside this coordinator's result.
