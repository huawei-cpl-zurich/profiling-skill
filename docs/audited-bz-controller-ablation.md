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
| Reused `CommandController._execute` for remeasurement | Preserves receipt bounds, redaction, timeout handling, and exact-handle checks without a parallel execution path | Runtime and lifecycle focused suites |
| Bound frozen baseline and calibration inputs directly in compact receipts | Makes normalized timing and speedup reproducible without retaining profiler trees | Adapter, contract, and backend tests |
| Sanctioned only `--state-dir` as mutable controller provenance | Allows production composition while binding the exact directory identity and keeping all other path arguments immutable | Runtime provenance tests |

## Complexity Intentionally Kept

| Complexity | Reason kept |
| --- | --- |
| Explicit lifecycle stages | Required for restart safety between admission, correctness, profile, confirmation, and post-control |
| Separate observe and remeasure paths | Observation must not resubmit; measurement-only resume must not rerun correctness |

## PR Review Assessment

This is one coherent controller-adapter change: one production module, its
runtime bridge, behavioral tests, and interface documentation. It does not
modify remote transport. The main review risk is state-machine recovery;
focused tests cover retained-handle replay, measurement-only resume, and
branch-wide accounting.

## Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python -m py_compile scripts/audited_bz_controller.py` | passed | Syntax/import validation |
| `pytest -q tests/test_audited_bz_controller.py` | 7 passed | Focused adapter suite |
| `PYTHONPATH=. pytest -q -k 'not test_real_bwrap_with_functional_fake_codex_runs_persistent_rounds'` | 695 passed, 1 deselected | Known host-kernel Bubblewrap restriction excluded |
| `git diff --check` | passed | No whitespace errors |
| `python -m ruff check scripts/audited_bz_controller.py scripts/audited_runtime.py tests/test_audited_bz_controller.py tests/test_audited_runtime.py` | passed | Final integrated lint cleanup |
| `python -m ruff check --select F` on changed controller, contract, backend, and tests | passed | No unused/import errors |
| Focused adapter, contract, runtime, lifecycle, and backend suite | 124 passed | Baseline/calibration evidence and mutable state provenance |
| Full applicable suite | 699 passed, 1 deselected | Known host-kernel Bubblewrap restriction excluded |

## Residual Risks

- Live BZ-A3 validation is intentionally deferred until the dependent core and
  scheduler changes are integrated.
- The scheduler must provide only currently eligible devices; this adapter
  confirms admission with a known-good control but does not discover devices.

## Next Action

Commit the reviewed adapter for integration; run live BZ-A3 canaries after the
dependent core and scheduler branches are integrated.
