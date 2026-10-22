# BZ-A3 Global Execution Code Ablation

## Scope

- Source TDD plan: `.agent-state/tasks/audited-global-bz-execution.md`
- Implementation branch: `codex/audited-global-bz-execution`
- Review base branch: `origin/main`
- Status: ready-for-pr-review

## Branch And Diff Summary

| Item | Value |
| --- | --- |
| Changed files | Job client, focused tests, execution-contract documentation |
| Diff size before final review fixes | 3 files, 407 additions, 117 deletions |
| Main changed areas | Pinned global transport, durable dispatch, cache provenance |
| Suspected unrelated changes | None |

## Complexity Inventory

| Candidate | Classification | Evidence | Action |
| --- | --- | --- | --- |
| Global transport boundary | essential | Enforces the only approved remote interface and target set | Keep |
| Separate dispatch and observation | essential | Dispatch receipt must be fsynced before observation can fail or be interrupted | Keep |
| Execution-provenance record | essential | Prevents cached results crossing pinned-client or runtime identities | Keep |
| Rewriting a known handle on `JobError` | essential | Covers a dispatch response carrying a handle but an invalid terminal state | Keep |
| Caller-supplied adapter/remote argv | accidental | Bypassed global-client policy | Removed |

## Unsupported Capabilities

| Capability | Evidence | Resolution |
| --- | --- | --- |
| Named remote runtime in the formerly installed global client | Live default-Python run lacked Torch | Implemented as a prerequisite in the global `remote-access` client; this job client forwards only the validated `py311-torch` name |

## Fixes And Tests

| Change | Behavior protected | Validation |
| --- | --- | --- |
| Persist dispatch before observe | Crash resumes the exact handle with no second upload or run | Focused crash/interruption test |
| Bind runtime and client digest | New transport cannot reuse an old completed receipt | Focused provenance-drift test |
| Parse progress-prefixed receipts | Transfer progress does not hide the final JSON receipt | Functional fake-client tests |
| Hermetic CLI validation | Clean validation containers do not need user-wide skills | Isolated fake-home subprocess test |

## Final Validation

| Command | Result |
| --- | --- |
| `pytest -q tests/test_bz_a3_job_client.py` | 51 passed |
| Focused integration suite | 158 passed |
| Full applicable suite | 701 passed, 1 environment-inapplicable Bubblewrap test deselected |
| Ruff and diff check | Passed |

## PR Review Assessment

| Item | Assessment |
| --- | --- |
| Reviewable as one PR | Yes; one transport-hardening story and approximately the repository's 500-line review target |
| Main review risks | Durable receipt ordering and provenance completeness, both covered by functional tests |
| Suggested split | None; separating persistence from transport identity would weaken review context |

## Residual Risks

The integrated head still requires the coordinator's known-good live BZ-A3
canary after the named-runtime prerequisite is installed.

## Next Action

Push the reviewed head, rerun A3 validation, and request a fresh PR review.
