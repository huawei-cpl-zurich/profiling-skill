# Audited nine-branch campaign ablation

- **Implementation branch:** `codex/audited-nine-branch-campaign`
- **Comparison base:** `main`
- **Status:** Complete
- **Next action:** Integrate the production BZ controller adapter, then run remote canaries.

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

## Validation

- Focused: `pytest -q tests/test_audited_campaign.py` — 9 passed.
- Broader: `PYTHONPATH=. pytest -q -k 'not test_real_bwrap_with_functional_fake_codex_runs_persistent_rounds'`
  — 695 passed, 1 deselected.
- The excluded existing test requires unprivileged Bubblewrap namespaces,
  which this local host disables before any campaign code runs.
