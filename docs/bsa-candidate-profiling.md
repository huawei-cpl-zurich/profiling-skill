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
  --starter-source "$FROZEN_STARTER/candidate.py" \
  --starter-manifest "$FROZEN_STARTER/candidate.manifest.json" \
  --output "$EVIDENCE_ROOT/manifest.json"
```

The manifest includes source text because it is a local runtime artifact. Do
not commit generated manifests: candidate sources and profiling results belong
under the ignored evidence directory. Hash-only options are reconciled from
another retained request when the same source hash is available. Otherwise
they remain represented as `source_unavailable`; a hash is never treated as
executable source.

The frozen starter is mandatory input, not inferred from the four treatments.
The manifest's top-level `starter_occurrence_id` resolves its one global
starter occurrence, which has `lineage_role: "starter"`. When its exact
source, parameters, and environment already identify a campaign execution,
the inventory reuses the first deterministic matching occurrence. Otherwise
it adds one synthetic occurrence/execution, which is profiled by the same
runner. Reports must use `starter_occurrence_id`; they must not guess a
different baseline for each treatment.

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
candidate source and parameters. It also carries an `execution_environment`
with the exact target, logical runtime, and runtime activation mode. `gz-a3`
uses `target-default`: the executor omits `--runtime`, while still recording
`py311-torch` as the logical runtime. `bz-a3-1` and `bz-a3-2` use
`named-runtime` and activate that logical runtime explicitly.

The campaign configuration also declares a nonempty `campaign_revision`. The
result contract binds that revision and the SHA-256 digest of the exact
executor executable. Resume therefore fails closed after either identity
changes; use a new output path or deliberately regenerate the checkpoint after
updating the harness.

For `action=discover`, the executor must run all requested correctness cases,
then perform an unfiltered `msprof op` BasicInfo capture. Success is:

```json
{
  "status": "ok",
  "components": [
    {"selector": "block_sparse_attention_kernel_aic", "kind": "aic"},
    {"selector": "block_sparse_attention_kernel_aiv", "kind": "aiv"}
  ],
  "basic_info": {"rows": 1},
  "evidence_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "provenance": {
    "target": "gz-a3", "runtime": "py311-torch",
    "runtime_activation": "target-default",
    "physical_device": 0, "command": ["..."],
    "handle": "remote:gz-a3:job:..."
  }
}
```

`components` is the logical candidate's exact, complete selector list in
deterministic order. A one-component adapter may instead return the legacy
`kernel_names` list when it resolves unambiguously to the declared name or a
single observed name. Multiple ungrouped legacy names remain invalid because
the coordinator cannot prove that unrelated framework tasks belong to the
candidate.

For `action=capture`, it must execute exactly the requested case and exact
discovered component selector with `PipeUtilization`, analyze the report in
place, and return only compact rows. The request repeats both `kernel_name`
and the component descriptor from discovery:

```json
{
  "status": "ok",
  "duration_us": 545027.2,
  "pipe_utilization": {"rows": []},
  "evidence_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "provenance": {
    "target": "gz-a3", "runtime": "py311-torch",
    "runtime_activation": "target-default",
    "physical_device": 0, "command": ["..."],
    "handle": "remote:gz-a3:job:..."
  }
}
```

Candidate failures use `compile_error`, `runtime_error`, or
`correctness_error` and include retained diagnostics. Transport, device,
profiler-tool, missing-response, and malformed-evidence failures use
`infrastructure_error`; they do not consume a valid sample or change candidate
status permanently.

Every successful discovery or capture must include a valid 64-hex-character
`evidence_sha256`. In production mode, provenance must exactly match the
manifest-requested target, logical runtime, and activation mode, and include a
physical device, exact command, and a handle belonging to that target.

Infrastructure responses should include their provenance handle and
`terminal`. The coordinator checkpoints the resulting retry object and passes
it verbatim on the next request. A nonterminal (or legacy terminal-unknown)
response becomes `{"mode":"observe","handle":"..."}`; the executor must
observe that durable job rather than duplicate it. A proven terminal response
becomes `{"mode":"new_attempt","prior_handle":"..."}`, authorizing a fresh
attempt while preserving its predecessor. Legacy responses which omit
`terminal` are conservatively observe-only; the adapter must eventually return
explicit terminal state or a migration/observation timeout, never redispatch.
The same rule applies when an otherwise successful terminal response contains
malformed BasicInfo, PipeUtilization, or digest evidence: it is retained as an
infrastructure failure and the next request explicitly starts a new attempt.

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

A logical sample captures every discovered component. A partial component set
is checkpointed and resumed without recapturing successful components. Its
duration is `max_component_duration`, appropriate for the concurrent AIC/AIV
parts of one logical launch; compact pipe evidence and provenance remain
separate per exact selector.

A case completes at the first chronological consecutive three-sample window
whose relative range `(max-min)/median <= 5%`. This predefined scan cannot
assemble a favorable subset from nonadjacent observations. After seven valid
but unstable logical samples, the case becomes
`measurement_unstable`, classified as infrastructure/measurement rather than
a compilation or runtime defect. Large vendor report trees remain remote.

Version-1 checkpoints predate both consecutive-window selection and
multi-component samples. Early version-2 checkpoints also predate the bound
target/runtime-activation or executor identity. The runner rejects both with
an explicit regeneration message instead of silently changing their
interpretation. A BZ
manifest or checkpoint therefore cannot be resumed or merged as a GZ
campaign, even though both use the logical `py311-torch` runtime.

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
  --executor "$BUNDLE_EXECUTOR" \
  --input "$EVIDENCE_ROOT/shard-0.json" \
  --input "$EVIDENCE_ROOT/shard-1.json" \
  --output "$EVIDENCE_ROOT/merged.json"
```

The merger preserves each execution state, sample, diagnostic, and provenance
object verbatim. It rejects repeated shard indexes, overlapping executions,
wrong-shard assignments, and manifest, sample-policy, or shard-count mismatch.
It recomputes the executor identity from `--executor` and rejects a campaign
revision or executable digest mismatch.
The merged result explicitly lists missing shard indexes and execution IDs, so
a partial merge cannot look complete.

## Remote execution policy

Before dispatch, run `cpl-remote capabilities TARGET` and
`cpl-remote preflight TARGET`. Stage the self-contained bundle with
`cpl-remote upload`. For BZ, submit with the named runtime; for GZ, use the
target-default environment and omit `--runtime`. Retain the printed handle.
If observation is interrupted, observe that same handle; do not resubmit. The
executor evidence must retain the selected target, runtime activation mode,
physical device, exact command, handle, kernel selector, sample count, and
compact BasicInfo/PipeUtilization output alongside this coordinator's result.
