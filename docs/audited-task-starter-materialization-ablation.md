# Audited Task Starter Materialization Code Ablation

## Scope

- Source TDD plan: `.agent-state/tasks/audited-task-starter-materialization.md`
- Implementation branch: `codex/audited-task-starter-materialization`
- Review base branch: `main`
- Status: ready-for-pr-review

## Branch And Diff Summary

| Item | Value |
| --- | --- |
| Main changed areas | Starter provenance, clean materialization, seed/resume validation |
| Suspected unrelated changes | None |

## Complexity Inventory

| Candidate | Classification | Evidence | Action |
| --- | --- | --- | --- |
| Per-task candidate and manifest bindings | essential | A common revision cannot contain three different root starters | Keep |
| Seed-commit blob verification | essential | Resume must preserve agent work without trusting mutable worktree bytes | Keep |
| Seed-only crash recovery | essential | A process may die before the first blocked checkpoint exists | Keep, bounded to exact seed HEAD and two allowed files |
| Manifest v2 plus legacy v1 reader | essential | New starter provenance must not reinterpret historical manifests | Keep |
| Separate starter validation receipt | speculative | Existing canary/controller gates own executable kernel validation | Omit |
| Hash then reread frozen inputs | accidental | Creates a time-of-check/time-of-use window | Read once and hash those bytes |

## Ablations And Fixes

| Change | Reason | Validation |
| --- | --- | --- |
| Omitted a second pre-agent backend protocol | Starter manifests intentionally begin unresolved | Existing controller and canary gates remain authoritative |
| Replaced the stale matmul selector with an explicit sentinel | It named an export absent from the starter | Functional manifest-loader test |
| Hash the same starter bytes later copied | Avoid source drift between validation and materialization | Starter drift and materialization tests |
| Reject unresolved selector before controller | Prevent guaranteed profiling failure while preserving repair flow | Functional repair test proves zero premature controller requests |

## Final Validation

| Command | Result |
| --- | --- |
| Focused campaign/production/lifecycle tests | 87 passed |
| Full applicable pytest excluding real Bubblewrap | 806 passed, 1 deselected |
| Scoped Ruff and diff checks | Passed |
| BZ-A3 validation | 98 passed on retained handle `remote:bz-a3-1:job:20261006T102553Z-22309e1bde65` |

## PR Review Assessment

| Item | Assessment |
| --- | --- |
| Reviewable as one PR | Yes; one provenance-to-seed transaction |
| Main review risks | Resume verifies the seed without overwriting checkpointed candidate work |
| Distinct review stories | One: materialize and preserve task-specific starters |

## Residual Risks

The starter sentinel is deliberately not a runnable profiling selector. The
invariant prompt and existing controller/canary gates must continue requiring
the agent to publish the exact exported kernel name before profiling.

## Next Action

Open the pull request and complete the required review wave.
