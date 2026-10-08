# Profiling Acquisition Helper Code Ablation

## Scope

- Source PRD: central task `profiling-acquisition-helper`
- Source TDD plan: central task `profiling-acquisition-helper`
- Implementation branch: `codex/profiling-acquisition-helper`
- Review base branch: `codex/triton-pipe-attribution`
- Status: ready-for-pr-review

## Branch And Diff Summary

| Item | Value |
| --- | --- |
| Main changed areas | Generic two-pass acquisition CLI, behavioral tests, skill routing |
| Suspected unrelated changes | None |

## Baseline Validation

| Command | Result | Notes |
| --- | --- | --- |
| `PYTHONPATH=. pytest -q` | 1107 passed | Full post-integration suite |
| focused tests, quick validation, Ruff, diff check | passed | No failures |

## Complexity Inventory

| ID | Candidate | Classification | Evidence | Proposed action | Status |
| --- | --- | --- | --- | --- | --- |
| C1 | Separate BasicInfo and pipe CLIs | accidental | Same transport/evidence lifecycle | One CLI with two modes | ablated |
| C2 | Kernel-name inference | speculative | Relevant operator is workload-dependent | Require agent-selected exact export | ablated |
| C3 | Embedded controller plus local controller | essential | Remote report must be analyzed in place | Keep boundary explicit | kept |
| C4 | Product runtime map | essential | A3 and A5 require different named runtimes | Keep closed map | kept |
| C5 | Bundle manifest and remote materializer | essential | Real workloads use sibling modules and data | Keep one content-addressed implementation | kept |
| C6 | Receipt and resume lifecycle | essential | Controller interruption must never redispatch a retained job | Keep one lifecycle shared by both passes | kept |

## Unsupported Capabilities

| ID | Capability | Failing case | Evidence | Current assumption | Impact | Status |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | Named rows in `PipeUtilization.csv` | A3 pipe capture has no name column | Live A3 capture | Pipe rows named their operator | Would reject valid evidence | fixed by adjacent BasicInfo binding |

## Ablations Performed

| ID | Change | Reason | Validation | Files |
| --- | --- | --- | --- | --- |
| A1 | Removed workload-specific selector inference | Arbitrary workloads may export multiple unrelated kernels | zero/nonexact selector tests | CLI and tests |
| A2 | Unified retry, receipt, log, and evidence handling across both passes/products | Avoid parallel acquisition paths | transport and exact-byte tests | CLI |
| A3 | Replaced temporary report trees with retained per-dispatch roots | Preserve remote forensic evidence | live captures | CLI |

## Fixes Performed

| ID | Change | Reason | Validation | Files |
| --- | --- | --- | --- | --- |
| F1 | Accept nonzero terminal receipts | `cpl-remote` mirrors failed remote exit | terminal-failure test | CLI |
| F2 | Use deployed flags and selector-binding BasicInfo | Match real A3/A5 exports | four successful live handles | CLI |
| F3 | Persist and flush the dispatch receipt before observation | Survive controller termination after dispatch | deliberate live A3 interruption and resume | CLI |
| F4 | Bind both passes to a complete deterministic bundle manifest | Prevent silent sibling/data drift | multi-file execution and mutation tests | CLI |
| F5 | Make discovery bounded and explicit rather than complete | Deployed msprof exposes a bounded launch count, not application completeness | late-after-20 test and live CLI help | CLI and reference |
| F6 | Preserve remote failure phase and classification | Distinguish infrastructure from workload and profiler failures | functional classification matrix | CLI |
| F7 | Resolve the approved broker client from `PATH` before the user-wide fallback | Isolated launchers mount `cpl-remote` at `/tools` | resolver and no-hidden-override subprocess tests | CLI |
| F8 | Add launcher-compatible schema, provenance, normal-run digest, and remote-content marker | Let the trusted launcher bind and consume the same compact bytes | generated-payload marker/evidence tests | CLI |

## Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `pytest -q tests/test_acquire_profile.py` | 31 passed | Includes bundle, restart/resume, late selector, failure classes, broker resolution, and launcher metadata |
| `PYTHONPATH=. pytest -q` | 1107 passed | Full suite |
| `quick_validate.py .` | passed | Skill package valid |
| Ruff and `git diff --check` | passed | Clean |
| A3 BasicInfo + PipeUtilization | passed | Three-file bundle; interrupted controller resumed same BasicInfo handle; exact selector and 24 pipe rows |
| A5 BasicInfo + PipeUtilization | passed | Same three-file bundle identity; exact selector and 24 pipe rows |

## PR Review Assessment

| Item | Assessment |
| --- | --- |
| Reviewable as one PR | Yes; one acquisition workflow and its contract tests |
| Main review risks | Embedded remote controller and deployed CSV schema variance |
| Distinct review stories | One: prevent agents from synthesizing fragile profiler commands |
| Backend/compiler/runtime/frontend mix | Transport orchestration plus remote evidence parsing are inseparable here |
| Human reviewer notes | Review same-handle recovery, complete bundle identity, bounded discovery, failure classes, and selector binding together |

## Suggested PR Split

None. Splitting the remote controller from the local lifecycle would leave
either half without a functional public acquisition interface.

## Residual Risks

| Risk | Impact | Follow-up |
| --- | --- | --- |
| Future profiler CSV renames | Fail-closed acquisition | Add a real fixture when a new deployed schema appears |
| Bounded BasicInfo capture can miss a later launch | Agent cannot select that kernel from this pass | Increase `--launch-count` within the deployed 1-5000 limit or narrow the workload; never claim the inventory is complete |

## Next Action

Commit and update PR #76 against `codex/triton-pipe-attribution`.
