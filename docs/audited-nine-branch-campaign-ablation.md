# Audited nine-branch campaign ablation

- **Implementation branch:** `codex/audited-nine-branch-campaign`
- **Comparison base:** `main`
- **Status:** Complete
- **Next action:** Integrate the configurable-round and BZ controller commits,
  then run remote canaries.

## Required behavior

The implementation must retain the exact 3×3 matrix, four-round branches,
treatment isolation, pinned provenance, balanced deterministic order, dynamic
device admission, durable resume semantics, and result reporting. These are
independent acceptance boundaries rather than speculative extension points.

## Complexity assessment

- `ResourcePool` and `CellLauncher` are essential integration boundaries for
  the independently developed BZ adapter.
- Atomic ledgers and the distinct handleless/retained-handle recovery paths are
  essential: collapsing them would permit duplicate hour-long workloads.
- The fake pool, launcher, and CLI simulation are retained because they provide
  the requested reproducible local end-to-end example without remote hardware.
- Removed an unused skill-expansion computation and kept treatment pinning at
  the actual bundle boundary instead of introducing a second skill registry.
- Kept the feature in one PR because it has one review story: construct,
  schedule, resume, and summarize the same immutable nine-cell campaign.

## Production-wiring follow-up

Production wiring is kept in a distinct dependent commit/PR. Its resource-pool
and cell-launcher boundaries are required rather than speculative: the global
remote contract has no device-occupancy operation, so the pool consumes an
explicit pinned placement-provider snapshot and fails closed without it. The
launcher retains only the configuration needed to clone and isolate one cell,
materialize its exact skill allowlist, bind the BZ controller/backend closure,
and resume the audited runner. Tests use dependency factories solely to run the
full nine-cell path locally without Codex credentials or remote hardware.

The hardening follow-up deliberately keeps the following checks at the
launcher boundary because removing any one of them changes experimental
meaning rather than merely simplifying code:

- Manifest provenance is compared with the actual source revision, controller
  closure, benchmark baselines, and copied skill trees before any agent runs.
- Docker is the only production agent runtime, and the BZ client can reach only
  the hash-pinned global `cpl-remote`; caller-selected adapter argv was removed.
- A branch is resumed only from a valid blocked checkpoint. A completed branch
  whose ledger was lost is independently verified and reconstructed in place.
- Four-round resume accepts only the v2 checkpoint with immutable
  `round_count: 4`; accepting legacy v1 would erase the campaign's round-count
  identity at the recovery boundary.
- No lifecycle exception is relabeled as a kernel failure. Genuine candidate
  failures remain controller receipts and retain their full compact evidence;
  one `candidate_error` in any verified round makes the branch terminal
  `candidate_failed` rather than complete.
- The scheduler's infrastructure exception type is injected once when the
  launcher is constructed. Failure paths never import a mutable same-named
  module, so combined test or plugin import order cannot change exception
  identity.
- Frozen timing baselines have their own semantic schema and internal canonical
  digest, and the exact validated document is passed into the controller. This
  keeps normalization inputs distinct from benchmark correctness assets.
- The independent offline verifier is a terminal success gate, not a reporting
  convenience. This is intentionally separate from the lifecycle's own checks.
- Cell publication is an atomic rename from a marked sibling staging tree.
  Recovery deletes only an exact-owned partial staging tree; it never guesses
  that an arbitrary directory belongs to this run.
- Every resume checks the exact treatment skill directory names and re-hashes
  every copied tree before an agent can run. This closes the gap between
  initially pinned inputs and a later resumed invocation.
- Agent-turn, controller-transaction, verifier, and backend-job budgets are
  separate pins. Two explicit grace intervals preserve the nesting from the
  remote job through the backend process to the outer controller transaction.

### Completed resource-admission split

The safe vertical slice identified during review is now the prerequisite
`audited_resource_admission` module. It owns `CplRemoteResourcePool`, the exact
admission schema/parser, global-client pin checks, documentation, and focused
tests. Production imports and compatibility-re-exports the required names; it
does not retain another parser, probe loop, or copy of those tests. The runtime
and manifest bind the prerequisite module, static provider and device
allowlist, and complete global-client launcher/implementation closure.

Production adds only the campaign-owned durability boundary: every accepted
v2 admission snapshot is atomically recorded before its slots reach the
scheduler. Expiration propagates as admission infrastructure failure, leaving
queued work undispatched and retained running work observable rather than
creating a replacement dispatch.

The remaining provenance checks are not a second safe mechanical split. They
bind the same source revision, copied treatment trees, runtime closure,
baselines, Docker image, and model that the launcher publishes and later
revalidates. Extracting those checks without the launcher would create a
validator whose result can become stale before bootstrap or resume. Likewise,
atomic bootstrap, checkpoint recovery, controller receipts, and offline
verification form one fail-closed transaction and should remain together.
The resource slice is therefore removed from this PR. Do not split the
remaining recovery or provenance work mechanically by line count.

## Validation

- Focused: `pytest -q tests/test_audited_campaign.py` — 9 passed.
- Production hardening: `pytest -q tests/test_audited_campaign_production.py`
  — 33 passed and one configurable-round dependency-gated composition test
  skipped on the isolated production branch. The rebased combined focused
  branch suite passed 79 tests with that one dependency skip. In a temporary
  checkout containing #48, #50, and #51, the real-composition test ran
  unskipped and the focused integration suite passed 148 tests with the real
  runner, `CommandController`, and offline verifier.
- Broader: `PYTHONPATH=. pytest -q -k 'not test_real_bwrap_with_functional_fake_codex_runs_persistent_rounds'`
  — 739 passed, 1 skipped, and 1 deselected on the rebased production branch;
  777 passed and 1 deselected in the fully integrated checkout.
- The excluded existing test requires unprivileged Bubblewrap namespaces,
  which this local host disables before any campaign code runs.

## Final-verifier follow-up

The original batch barrier was accidental complexity: a fast cell could not
release its device until every peer in that batch finished, and one
infrastructure failure stopped unrelated queued work. The scheduler now has a
single continuous future set, refreshes admission for every assignment, and
defers only the failed cell. Raw receipts remain intact in report schema v2;
derived per-case, control, normalization, baseline, speedup, and discarded
infrastructure views are additive.

## PR 49 durable-state follow-up

Three apparent simplifications were unsafe and have been made explicit:

- A terminal controller handle proves that execution ended, not that its
  receipt is structurally complete. Malformed terminal evidence is therefore
  handleless in the ledger so resume reconstructs it instead of observing the
  same terminal job forever.
- A persisted `running` attempt with a retained handle is observable. Without
  a handle its dispatch outcome is unknown, so it is left non-retryable and
  infrastructure-pending rather than silently completed or duplicated.
- Reports validate the ledger version, run identity, manifest digest, exact
  order, and cell set before consuming any result rows.

The manifest, scheduler, and report are separable concepts but are not a safe
mid-fix PR split. The scheduler's durable ledger is keyed and ordered by the
hashed manifest, while the report must validate that exact same contract
before interpreting attempts. The review findings specifically cross both
boundaries: terminal receipt classification controls scheduler recovery and
report eligibility, and ledger identity controls both resume and reporting.
Splitting now would either duplicate those invariants or temporarily expose a
report/scheduler pair with incompatible ledger semantics. A later module-only
extraction could improve navigation but would not reduce review scope or
changed lines. The current change therefore remains one coherent workflow
despite exceeding the approximate 500-line review target.
