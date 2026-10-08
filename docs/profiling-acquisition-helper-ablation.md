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
| `PYTHONPATH=. pytest -q` | 1091 passed | Full pre-ablation suite |
| focused tests, quick validation, Ruff, diff check | passed | No failures |

## Complexity Inventory

| ID | Candidate | Classification | Evidence | Proposed action | Status |
| --- | --- | --- | --- | --- | --- |
| C1 | Separate BasicInfo and pipe CLIs | accidental | Same transport/evidence lifecycle | One CLI with two modes | ablated |
| C2 | Kernel-name inference | speculative | Relevant operator is workload-dependent | Require agent-selected exact export | ablated |
| C3 | Embedded controller plus local controller | essential | Remote report must be analyzed in place | Keep boundary explicit | kept |
| C4 | Product runtime map | essential | A3 and A5 require different named runtimes | Keep closed map | kept |

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

## Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `pytest -q tests/test_acquire_profile.py` | 15 passed | Includes generated-payload execution |
| `PYTHONPATH=. pytest -q` | 1091 passed | Full suite |
| `quick_validate.py .` | passed | Skill package valid |
| Ruff and `git diff --check` | passed | Clean |
| A3 BasicInfo + PipeUtilization | passed | Exact selector, 24 pipe rows |
| A5 BasicInfo + PipeUtilization | passed | Exact selector, 24 pipe rows |

## PR Review Assessment

| Item | Assessment |
| --- | --- |
| Reviewable as one PR | Yes; one acquisition workflow and its contract tests |
| Main review risks | Embedded remote controller and deployed CSV schema variance |
| Distinct review stories | One: prevent agents from synthesizing fragile profiler commands |
| Backend/compiler/runtime/frontend mix | Transport orchestration plus remote evidence parsing are inseparable here |
| Human reviewer notes | Review same-handle recovery, exact workload identity, and selector binding together |

## Suggested PR Split

None. Splitting the remote controller from the local lifecycle would leave
either half without a functional public acquisition interface.

## Residual Risks

| Risk | Impact | Follow-up |
| --- | --- | --- |
| Future profiler CSV renames | Fail-closed acquisition | Add a real fixture when a new deployed schema appears |

## Next Action

Commit and open the PR against `codex/triton-pipe-attribution`.
