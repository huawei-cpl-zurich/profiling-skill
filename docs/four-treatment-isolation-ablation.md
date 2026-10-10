# Four-Treatment Isolation Code Ablation

## Scope

- Source PRD: approved fully fused A3/A5 four-configuration campaign plan
- Source TDD plan: central task `four-treatment-isolation`
- Implementation branch: `codex/four-treatment-isolation`
- Review base branch: `main`
- Status: ready-for-pr-review

## Branch And Diff Summary

| Item | Value |
| --- | --- |
| Current branch | `codex/four-treatment-isolation` |
| Base branch | `main` |
| Changed files | campaign schema, production materializer, direct tests, this report |
| Diff size | 448 insertions / 266 deletions across source/tests before final docs |
| Main changed areas | four treatment identities, frozen-source provenance, isolated materialization |
| Suspected unrelated changes | none |

## Baseline Validation

| Command | Result | Notes |
| --- | --- | --- |
| `pytest -q tests/test_four_treatment_isolation.py` | PASS, 8 | New behavior |
| `pytest -q tests/test_audited_campaign.py` | PASS, 39 | Campaign regression |
| `pytest -q tests/test_audited_campaign_production.py -x` | 78 pass, 1 fixture failure | Baseline before final fixture conversion |

## Complexity Inventory

| ID | Candidate | Classification | Evidence | Proposed action | Status |
| --- | --- | --- | --- | --- | --- |
| C1 | Legacy schema v1-v3 verification | essential | Retained campaigns must remain auditable | Keep read-only compatibility | kept |
| C2 | Public materialization helper plus launcher wrapper | essential | Functional tests and launcher share one boundary | Keep | kept |
| C3 | Separate test module with duplicate freeze setup | accidental | Production tests already provide full runtime fixtures | Fold into production suite | removed |
| C4 | Repeated frozen-source fixture construction | accidental | Same source contract in unit and end-to-end tests | Centralize helper | removed |

## Unsupported Capabilities

None discovered.

## Ablations Performed

| ID | Change | Reason | Validation | Files |
| --- | --- | --- | --- | --- |
| A1 | Keep treatment provenance outside the agent workspace | Avoid leaking treatment identity and alternative profiler revisions | isolation tests | production materializer |
| A2 | Fold focused tests into the production suite | Remove duplicate imports and fixture code | 124 focused tests | production tests |
| A3 | Centralize frozen source creation | Reduce repeated setup without hiding assertions | 79 production tests | production tests |

## Test Changes

| ID | Change | Behavior protected | Validation |
| --- | --- | --- | --- |
| T1 | Exact treatment/source matrix | No profile ambiguity | focused suite |
| T2 | Freeze revision failures | Old/new commits cannot be swapped | focused suite |
| T3 | Materialized allowlist and drift | No global/cross-treatment skill visibility | focused suite |

## PR Review Assessment

| Item | Assessment |
| --- | --- |
| Reviewable as one PR | Yes; cohesive size exception |
| Main review risks | Schema-v4 cutover and source-binding interpretation |
| Distinct review stories | One: define and enforce the four isolated treatment snapshots |
| Backend/compiler/runtime/frontend mix | Manifest and host-side materialization only; no device admission changes |
| Human reviewer notes | Splitting identifiers from materialization would leave an unusable intermediate schema. Most excess lines are fixture migration from 9 to 12 cells. |

## Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `PYTHONPATH=. pytest -q` | PASS, 1279 | Full local suite |
| BZ-A3 retained job `remote:bz-a3-1:job:20261010T141958Z-52d8cf42b953` | PASS, 206 | `py311-torch`; campaign, production, canary, migration suites |

## Residual Risks

| Risk | Impact | Follow-up |
| --- | --- | --- |
| Runtime configuration producers outside this repository | Must emit v3 frozen-source bindings | Document exact JSON contract |

## Next Action

Request independent and GitHub review; address only actionable common-case findings.
