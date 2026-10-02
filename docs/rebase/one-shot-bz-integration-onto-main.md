# One-shot BZ Integration Onto Main

## Scope

- Feature branch: `codex/one-shot-bz-integration`
- Reference branch: `origin/main`
- Backup branch: `backup/codex-one-shot-bz-integration-before-main-merge-20261002-102648`
- Report status: preflight

## Starting State

| Item | Value |
| --- | --- |
| Original current branch | `codex/one-shot-bz-integration` |
| Feature branch HEAD | `3fb7b26db9303a589b2d136e8b142a1326077468` |
| Reference branch HEAD | `7c1e254e682dc3709a0de21acdae74214d0af209` |
| Merge base | `caef9af127dac0d8cd8ddc17c1c52b9159766e76` |
| Working tree status | clean |
| Merge started at | pending |

## Branch Intent

Connect the four-wave, three-treatment one-shot diagnostic campaign to the
approved BZ-A3 correctness client while preserving one agent invocation per
logical cell, immutable candidate replay on infrastructure retries, and an
atomic evidence ledger.

## Feature Branch Unique Commits

| Commit | Summary | Notes |
| --- | --- | --- |
| `bf13014` | Add BZ-A3 one-shot diagnostic client | Superseded by merged PR #23 and follow-ups on main. |
| `3fb7b26` | Integrate one-shot diagnostics with BZ-A3 | Unique integration layer to preserve and update. |

`caef9af` is the old launcher base and is superseded by merged PR #22 and PR
#25 hardening on main.

## Reference Branch Changes Since Divergence

| Commit or range | Summary | Impact |
| --- | --- | --- |
| PR #22 follow-ups | Protocol-v2 in-band prompt, canonical treatments, deadlines, interruption checkpointing | Integration must accept in-band prompt requests and delegate cancellation. |
| PR #23 follow-ups | Durable BZ dispatch receipts, retained-handle observation, remote infrastructure status | Integration must preserve client statuses and never redispatch uncertain jobs. |
| PR #25 | Launcher interruption and prompt hardening | Integration wrappers must expose cancellation without weakening cleanup. |

## Changed File Overlap

| File or area | Feature branch change | Reference branch change | Risk |
| --- | --- | --- | --- |
| `scripts/bz_a3_diagnostic_client.py` | Old parent snapshot | Merged and hardened client | Mechanical overlap; keep main implementation. |
| `scripts/diagnostic_campaign.py` | Old parent snapshot | Protocol-v2 and cancellation hardening | Semantic compatibility audit required. |
| Integration files | New adapter, config, tests | None | Preserve, then adapt to final parent APIs. |

## Preflight Risk Assessment

| Risk | Evidence | Mitigation |
| --- | --- | --- |
| Retry replays an old protocol-v1 request | Main emits protocol-v2 prompt bytes | Cache and replay the opaque agent result/submission while leaving the original request untouched. |
| Wrapper hides cancellation | Main command hooks now implement `cancel()` | Add wrapper cancellation delegation and tests. |
| Retained BZ handle is redispatched | Main client owns durable receipt recovery | Call the client once per terminal attempt and preserve returned handle/status verbatim. |
| Main-target diff contains parent history | Feature predates merged parent PRs | Merge actual `origin/main`; verify final diff contains only integration/report changes. |

## Human Decisions

No semantic decision is currently unresolved. The user explicitly requested a
history-preserving merge of actual `origin/main`, not a history rewrite.

## Conflicts Encountered

Pending.

## Validation

Pending focused, full local, and authorized BZ-A3 validation.

## Final State

| Item | Value |
| --- | --- |
| Final feature branch HEAD | pending |
| Merge completed at | pending |
| Rebase or merge still in progress | no |
| Uncommitted changes | report only |

## Residual Risks

Pending validation.

## Next Action

Commit this report, merge `origin/main`, resolve parent overlap in favor of
the hardened main implementations, then update and validate the unique
integration layer.
