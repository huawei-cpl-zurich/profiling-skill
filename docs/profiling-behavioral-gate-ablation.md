# Profiling Behavioral Gate Code Ablation

## Scope

- Source plan: `.agent-state/tasks/profiling-behavioral-gates.md` (central and uncommitted).
- Implementation branch: `codex/profiling-behavioral-gates`.
- Review base: `main` at `d6cc328144df17d09979c1d154366f52a55f5454`.
- Status: ready-for-pr-review.

## Baseline

The pre-ablation commit `bb24a40` added 519 source, 317 test, and 105 guide
lines. Its focused suite passed 18 tests and the full suite passed 1054 tests,
but those tests exercised a fake executor rather than the live trust boundary.

## Complexity inventory

| ID | Candidate | Classification | Evidence | Resolution |
| --- | --- | --- | --- | --- |
| C1 | Executor, workspace, scheduling, state, retry/resume | accidental | Duplicated audited lifecycle and was not crash-safe before its first record | Removed; separate launcher owns lifecycle |
| C2 | Runtime-input/auth handoff | speculative | Non-live core could not safely provide Codex/remote auth | Removed; separate launcher owns it |
| C3 | Arm-labelled agent paths/payload context | accidental | Could reveal treatment | Arm exists only in trusted envelope; payload schema is exact |
| C4 | Agent-controlled failure classification | accidental | Agent could claim infrastructure and discard its own failure | Classification and matching receipts exist only in trusted records |
| C5 | Subset semantic scoring | accidental | Extra/contradictory claims passed | Exact conclusions and structured claim equality |
| C6 | 317 lifecycle-oriented test lines | accidental | Missed persistent-session and real isolation boundaries | Replaced with pure contract tests |

## Unsupported capabilities

| ID | Capability | Evidence | Resolution/status |
| --- | --- | --- | --- |
| U1 | Live isolated Codex launch | Core has no audited Bubblewrap/auth integration | Deferred to `.agent-state/tasks/profiling-behavioral-launcher.md` |
| U2 | A5 live acquisition | No supported registered BZ-A5 profiling runtime/wrapper | Blocked on target provisioning |
| U3 | Durable bespoke resume | Crash before record persistence could resubmit work | Removed from core; launcher owns pre-dispatch checkpoints/observation |
| U4 | Persistent paired agents | Old runner started a subprocess per product/case | Removed; core now requires launcher-attested paired/independent sessions |

## Ablations and fixes

| ID | Change | Reason | Focused proof |
| --- | --- | --- | --- |
| A1 | Replaced mutable battery with pure manifest + trusted-record validator | Match the real trust boundary | Schema/identity tests |
| A2 | Removed executor, workspaces, runtime inputs, scheduling and state | Avoid a weaker parallel launcher | No lifecycle code remains |
| A3 | Bound A3+A5 acquisition to one attested session | Enforce persistent pairing | Pair/session test |
| A4 | Required exact four-case answer set and 24 score slots | Prevent dropped cases/cherry-picking | Aggregate test |
| A5 | Added exact launcher/model/skill/prompt/evidence identity checks and retained prompt artifact | Reject input drift and preserve the evaluated prompt bytes | Identity/prompt tests |
| A6 | Added strict bounded regular-file refs and JSON metadata/provenance | Reject missing, escaped, symlinked, changed, or malformed evidence | Artifact tests |
| A7 | Added envelope-only failure receipts and replacement counts | Preserve infra and counted failures without agent control | Failure tests |
| A8 | Added pinned trusted review for every successful acquisition and interpretation reasoning log | Keep acceptance dependent on inspected reasoning | Review tests |
| A9 | Added actual remote handle formats and direct attempt summaries | Support GZ handles and keep failure reasons analyzable | Handle/aggregate tests |

## Complexity intentionally kept

| Item | Reason |
| --- | --- |
| Separate manifest, records, and artifact tree | They encode immutable inputs, trusted launcher assertions, and bounded evidence as distinct domain boundaries |
| Exact schemas | Fail-closed evaluation is the purpose of this core |

## Test changes

The 27 focused tests cover treatment-blind agent payloads, paired persistent
sessions, trusted-vs-agent failure classification, evidence/hash/JSON/symlink/
escape checks, exact contradictory rubrics, identity drift, duplicates,
infrastructure replacements, counted failures, all-log manual review, text
artifacts, prompt retention/hash/content, GZ handles, 12/12 comparison, and atomic output. They do not assert
documentation wording.

## PR review assessment

The result is reviewable as one PR: 352 source and 287 functional-test lines
(639 source/test lines) with one pure validation/aggregation story. The live
launcher remains a separate security/runtime PR. The principal review risk is
keeping that launcher's emitted trusted-record schema exactly aligned with this
validator.

## Final validation

| Command | Result |
| --- | --- |
| `pytest -q tests/test_profile_behavioral_gate.py` | 27 passed |
| `PYTHONPATH=. pytest -q` | 1063 passed |
| `git diff --check` | passed |

## Suggested PR order

1. Behavioral core: pinned manifest and trusted-record validation/scoring.
2. Audited launcher: Bubblewrap/auth, opaque paths, persistent sessions,
   checkpoints, remote observation/retry, classification, and retention.
3. Live gate: after product heads and supported A5 profiling are available.

## Residual risks

- No live adapter yet; the core cannot execute agents.
- No supported A5 profiling route; full live acceptance cannot run.

## Next action

Commit the ablated core locally without pushing, then hand it to the primary
agent for remote A3 validation and PR/review orchestration.
