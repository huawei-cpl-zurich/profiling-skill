# BSA Case Cleanup Code Ablation

## Scope

- Source PRD: central task record `campaign-bsa-case-cleanup`
- Source TDD plan: central task record `campaign-bsa-case-cleanup`
- Implementation branch: `codex/campaign-bsa-case-cleanup`
- Review base branch: frozen integration commit `d499901`, targeting
  `codex/campaign-agent-adapter`
- Status: validated

## Branch And Diff Summary

| Item | Value |
| --- | --- |
| Current branch | `codex/campaign-bsa-case-cleanup` |
| Base branch | `d499901` |
| Changed files | runner, functional tests, this report |
| Diff size | 337 source/test insertions before ablation |
| Main changed areas | per-case object lifetime and observable runner behavior |
| Suspected unrelated changes | two pre-existing unused imports exposed by Ruff |

## Baseline Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python -m pytest -q tests/test_a3_benchmark_runner.py` | 8 passed | Functional lifetime, classification, cleanup-failure, cache-capability, and fusion-evidence coverage |
| `python -m pytest -q` | 647 passed | Full suite |

## Changed Areas

| Area | Files | Behavior touched | Tests |
| --- | --- | --- | --- |
| Case lifetime | `scripts/a3_benchmark_runner.py` | Releases large case-only objects after evidence retention | collection-before-next-build |
| Failure behavior | runner and test | Preserves prior evidence and candidate classification | candidate runtime failure |
| Fusion audit | runner and test | Restores hooks after retained fusion evidence | fake runtime audit |

## Complexity Inventory

| ID | Candidate | Classification | Evidence | Proposed action | Status |
| --- | --- | --- | --- | --- | --- |
| C1 | Outer `try/finally` indents the complete case pipeline | essential | one lexical boundary guarantees release on every existing return path | retain instead of introducing a one-use result protocol/helper | retained |
| C2 | Explicit NPU cache flush | essential | live canary retained 13.27 GiB after Python references were cleared and OOMed on the next 3 GiB allocation | retain after `gc.collect()`, capability-gated for runtime-compatible fakes | retained |
| C3 | Extra device synchronization | speculative | existing reference and candidate synchronizations precede evidence retention | do not add | removed |
| C4 | Audit-hook restoration in cleanup | essential | candidate exceptions otherwise leave monkeypatches installed | retain | retained |
| C5 | Separate primary-result and cleanup-error state | essential | cleanup failures must not overwrite an already classified candidate result, but a cleanup-only failure must stop the next memory-heavy case | retain | retained |

## Unsupported Capabilities

| ID | Capability | Failing case | Evidence | Current assumption | Impact | Status |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | Multi-case BSA admission | prior case remains live into next build | first fixed-reference canary still retained 13.27 GiB and OOMed on the next 3 GiB allocation | Python collection and allocator-cache release are both required | campaign unblocked | fixed canary passed |

## User Decisions

No unresolved product or API decision. Branch and base were explicitly supplied
in the task assignment.

## Ablations Performed

| ID | Change | Reason | Validation | Files |
| --- | --- | --- | --- | --- |
| A1 | Added `gc.collect()` and capability-gated `empty_cache()` but no new synchronization | first live canary proved reference assignment alone did not return enough cached memory; existing synchronizations already establish completion | focused and full local suites | runner |
| A2 | Rejected a one-use per-case result helper | it would add a parallel error/evidence protocol solely to reduce indentation | focused and full tests protect the direct boundary | runner |

## Fixes Performed

| ID | Change | Reason | Validation | Files |
| --- | --- | --- | --- | --- |
| F1 | Per-case exception-safe reference and allocator-cache release | prevent next case from overlapping prior multi-GiB tensors or cached allocations | focused functional tests | runner |

## Complexity Intentionally Kept

| ID | Complexity | Reason kept | Revisit when |
| --- | --- | --- | --- |
| K1 | Existing phase-specific exception classification | public result contract distinguishes infrastructure, compile/runtime, and correctness failures | result schema changes |

## Test Changes

| ID | Change | Behavior protected | Validation |
| --- | --- | --- | --- |
| T1 | weak-reference collection test | prior case dies before next case build | passing |
| T2 | runtime failure test | exception cleanup and prior evidence/classification | passing |
| T3 | fusion evidence test | evidence retained before audit cleanup | passing |
| T4 | missing cache-hook compatibility test | cleanup does not mask successful results in reduced runtime fakes | passing |
| T5 | throwing cache-hook candidate-failure test | cleanup failure does not replace candidate classification or prior evidence | passing |
| T6 | throwing cache-hook success-path test | cleanup-only failure is a bound infrastructure result with current evidence | passing |

## Final Validation

Focused 8 tests and full 647-test suite pass. The final changed-source
known-good BSA canary passed 50 correctness and fusion cases, with one logical
launch per case, then captured three `msprof op` samples for each of five
ranked cases on physical device 0. Retained handle:
`remote:bz-a3-2:job:20261012T175802Z-2b6ea7e61d95`.

## PR Review Assessment

The change is one cohesive source/test behavior area and remains reviewable as
one PR. The principal review risk is ensuring every phase retains compact
evidence before cleanup; functional tests cover correctness evidence, failure
classification, and fusion evidence.

## Suggested PR Split

None. The source and functional tests form one runtime-memory fix.

## Residual Risks

| Risk | Impact | Follow-up |
| --- | --- | --- |
| `empty_cache` introduces allocator work between cases | multi-case admission takes additional host time | limited to the post-evidence case boundary; no timing occurs across this cleanup |

## Next Action

Freeze the validated head for pull-request review.
