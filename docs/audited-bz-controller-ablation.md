# Audited BZ-A3 Controller Code Ablation

## Scope

- Source TDD plan: `.agent-state/tasks/audited-bz-controller-adapter.md` (central, uncommitted)
- Implementation branch: `codex/audited-bz-controller-adapter`
- Review base branch: `origin/main` at `5268f1f`
- Status: ready-for-pr-review

## Baseline Validation

| Command | Result | Notes |
| --- | --- | --- |
| `pytest -q tests/test_audited_bz_controller.py` | 7 passed | Controller policy, failures, resume, remeasurement, budget, CLI |

## Complexity Inventory

| Candidate | Classification | Decision |
| --- | --- | --- |
| JSON subprocess boundary | essential | Preserves dependency injection and approved BZ backend boundary |
| Durable state plus branch budget ledger | essential | Separate lifetimes: request recovery versus cross-round allowance |
| Per-operation transcript in controller state | essential | Diagnoses infrastructure retries without exposing raw profiles |
| Controller-side device discovery | speculative | Omitted; scheduler supplies eligible devices and controller proves admission |
| Raw profiler collection/copy | accidental | Omitted; retain compact remote evidence locator only |

## Ablations Performed

| Change | Reason | Validation |
| --- | --- | --- |
| Kept one batched profile request instead of a loop of capture adapters | Existing backend already produces three complete in-place repetitions | Focused suite |
| Reused the BZ client's content-addressed replay instead of adding transport code | Prevents duplicate remote submission and policy bypass | Durable-handle test |
| Shared one branch budget ledger instead of per-round counters | Implements 24-operation branch semantics with less state ambiguity | Cross-round budget test |

## Complexity Intentionally Kept

| Complexity | Reason kept |
| --- | --- |
| Explicit lifecycle stages | Required for restart safety between admission, correctness, profile, confirmation, and post-control |
| Separate observe and remeasure paths | Observation must not resubmit; measurement-only resume must not rerun correctness |

## PR Review Assessment

This is one coherent controller-adapter change: one production module, one
behavioral test module, and interface documentation. It does not modify the
audited core or remote transport. The main review risk is state-machine recovery;
focused tests cover retained-handle replay and branch-wide accounting.

## Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python -m py_compile scripts/audited_bz_controller.py` | passed | Syntax/import validation |
| `pytest -q tests/test_audited_bz_controller.py` | 7 passed | Focused adapter suite |
| `PYTHONPATH=. pytest -q` | 693 passed, 1 failed | Sole failure: host kernel denies unprivileged bubblewrap namespaces in an unrelated production-launcher test |
| `git diff --check` | passed | No whitespace errors |

## Residual Risks

- Live BZ-A3 validation is intentionally deferred until the dependent core and
  scheduler changes are integrated.
- The scheduler must provide only currently eligible devices; this adapter
  confirms admission with a known-good control but does not discover devices.

## Next Action

Commit the reviewed adapter for integration; run live BZ-A3 canaries after the
dependent core and scheduler branches are integrated.
