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

The plan schema is `profiling-skill/audited-runtime-migration-plan/v1`. Its
top-level fields are `migration_id`, `run_root`, `ledger_sha256`,
`old_runtime`, `new_runtime`, and `cells`. Runtime bindings contain
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
