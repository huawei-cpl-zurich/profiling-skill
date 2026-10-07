# Audited runtime checkpoint migration

`scripts/audited_runtime_migration.py` is the fail-closed preflight for a
one-time migration of the selector-resolution defect in the existing A3
campaign. It recognizes
exactly these checkpoints:

- `bsa-project-guarded`, experiment 3
- `gdn-cannbot`, experiment 1
- `bsa-project-cannbot`, experiment 1

The plan binds the campaign ledger, old and new runtime configurations and
closures, both controller identities, checkpoint commit, candidate and
manifest, controller-state document, durable handle, failed request, and exact
failure text. Check mode performs every validation without writing:

```console
python scripts/audited_runtime_migration.py \
  --plan migration.json --attestation preflight.json --check
```

The preflight never mutates the campaign. It writes a sealed attestation that
binds the plan, exact cell state, controller identities, and independently
revalidated runtime configuration and closure digests. Lifecycle consumption,
archival, checkpoint amendment,
controller-state advancement, and ledger retry publication are a dependent
transactional step and must consume this same validated plan contract.

Apply the transaction with the exact attestation file digest and its distinct
inner self-seal:

```console
python scripts/audited_runtime_migration_apply.py \
  --plan migration.json \
  --attestation preflight.json \
  --attestation-file-sha256 "$(sha256sum preflight.json | cut -d' ' -f1)" \
  --attestation-sha256 "$(python -c 'import json; print(json.load(open("preflight.json"))["attestation_sha256"])')" \
  --transaction /absolute/path/to/selector-fallback-v4-transaction
```

`--attestation-file-sha256` authenticates the bytes of `preflight.json`;
`--attestation-sha256` is the seal stored inside that JSON. Keep the transaction
directory outside the campaign run root. If execution is interrupted, rerun
the identical command with the same transaction directory. Never choose a new
directory for a partial transaction: its journal and archive are the recovery
authority.

Resume the production campaign with the same external trust values. The trust
is deliberately supplied on the command line rather than embedded in the
runtime configuration: the preflight attestation already binds that config's
bytes, so embedding the attestation would create a circular hash dependency.

```console
python scripts/audited_campaign_production.py \
  --manifest manifest.json \
  --runtime-config runtime-v4.json \
  --runtime-config-sha256 "$(sha256sum runtime-v4.json | cut -d' ' -f1)" \
  --admission admission.json \
  --admission-sha256 "$(sha256sum admission.json | cut -d' ' -f1)" \
  --ledger ledger.json --resume \
  --migration-attestation /absolute/path/to/preflight.json \
  --migration-attestation-file-sha256 "$(sha256sum preflight.json | cut -d' ' -f1)" \
  --migration-attestation-sha256 "$(python -c 'import json; print(json.load(open("preflight.json"))["attestation_sha256"])')"
```

All three migration arguments are required together and are accepted only with
`--resume`; they cannot authorize a fresh campaign. The launcher verifies
the outer file digest and inner seal, proves the attested old closure matches
the unchanged manifest provenance, and proves the attested new closure and
config match the active pinned runtime before it constructs an agent or
controller. Ordinary campaigns omit all three arguments and retain their
existing provenance checks.

For more than one transition, prefer repeatable ordered proof tuples. The
launcher authenticates the complete closure chain, retains it for lifecycle
resume, and forwards every cell-relevant proof to the independent verifier:

```console
python scripts/audited_campaign_production.py ... --resume \
  --migration-proof /absolute/path/to/preflight-v4.json FILE_SHA256 SEAL \
  --migration-proof /absolute/path/to/preflight-v5.json FILE_SHA256 SEAL
```

Do not mix the legacy single-attestation arguments with `--migration-proof`.
For a later preflight, add the earlier ordered trust objects to the plan's
optional `prior_migrations` array. The preflight accepts the current
controller as its old identity only after that history authenticates a chain
from the immutable seed identity to the checkpoint's latest citation. Apply
returns both the legacy latest `trusted_runtime_migration` and the complete
ordered `trusted_runtime_migrations` history.

The independent verifier receives the same trust tuple for a completed
migrated branch. Repeat `--migration-proof` in transition order when a branch
crossed more than one authenticated runtime boundary:

```console
python scripts/validate_audited_experiment.py /path/to/branch \
  --base BASE_REVISION \
  --migration-proof /absolute/path/to/preflight.json \
    ATTESTATION_FILE_SHA256 ATTESTATION_SHA256
```

Each proof must bind the branch's exact cell, seed commit, transition round,
resume parent, and old/new controller identities. Without a matching proof,
every experiment must retain the exact controller identity recorded in the
seed. The production launcher supplies a cell's authenticated proof
automatically after a migrated run completes.

The plan schema is `profiling-skill/audited-runtime-migration-plan/v1`. Its
top-level fields are `migration_id`, `run_root`, `ledger_sha256`,
`old_runtime`, `new_runtime`, and `cells`, plus optional ordered
`prior_migrations` for a later transition. Runtime bindings contain
`config_path`, `config_sha256`, and `closure_sha256`. Each cell contains:

- `cell_id`, `experiment`, and `checkpoint_commit`
- `blocked_sha256` and `controller_state_sha256`
- `candidate_sha256` and `manifest_sha256`
- `durable_handle`, `request_sha256`, `ledger_attempt_sha256`, and
  `failure_reason`
- complete `old_controller_identity` and `new_controller_identity` documents

The new controller identity must be captured after materializing the frozen
new runtime and the cell's new controller configuration. A dependent lifecycle
consumer must revalidate this sealed artifact before accepting controller
identity drift; this producer does not weaken the existing resume policy.
