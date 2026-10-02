# One-shot BZ Integration Onto Main

## Scope

- Feature branch: `codex/one-shot-bz-integration`
- Reference branch: `origin/main`
- Backup branch: `backup/codex-one-shot-bz-integration-before-main-merge-20261002-102648`
- Report status: complete

## Starting State

| Item | Value |
| --- | --- |
| Original current branch | `codex/one-shot-bz-integration` |
| Feature branch HEAD | `3fb7b26db9303a589b2d136e8b142a1326077468` |
| Reference branch HEAD | `7c1e254e682dc3709a0de21acdae74214d0af209` |
| Merge base | `caef9af127dac0d8cd8ddc17c1c52b9159766e76` |
| Working tree status | clean |
| Merge started at | 2026-10-02 10:28 UTC |

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

| File | Conflict type | Resolution | Validation |
| --- | --- | --- | --- |
| `scripts/bz_a3_diagnostic_client.py` | mechanical | Kept the merged PR #23 implementation from `origin/main`. | Parent and integration tests passed. |
| `tests/test_bz_a3_diagnostic_client.py` | mechanical | Kept the merged PR #23 tests from `origin/main`. | Parent and integration tests passed. |

The compatibility audit also assigned a unique BZ receipt identity to each
primary/fallback attempt and delegated launcher cancellation through the
submission-freezing wrapper.

## Validation

| Command | Result | Notes |
| --- | --- | --- |
| `python -m pytest -q tests/test_one_shot_bz_campaign.py tests/test_diagnostic_campaign.py tests/test_bz_a3_diagnostic_client.py` | 79 passed | Covers protocol-v2 prompt bytes, cancellation, same-receipt observation, one deadline, unique namespaces, frozen assets, and BZ status mapping. |
| `python -m pytest -q` | 306 passed, 1 host failure | Only failure is the pre-existing Bubblewrap namespace test; this controller denies unprivileged namespace creation. |
| BZ-A3-1 device 2, cases 0-6 | passed | `bz-a3-1:20261002T103019Z-120-24115` |
| BZ-A3-1 device 3, cases 0-6 | passed | `bz-a3-1:20261002T103019Z-123-3759` |
| BZ-A3-2 device 12, cases 0-6 | passed | `bz-a3-2:20261002T103016Z-120-25088` |
| Final integration hook, BZ-A3-1 device 2, cases 0-6 | passed | `bz-a3-1:20261002T103938Z-118-2452` |
| Final integration hook, BZ-A3-2 device 12, cases 0-6 | passed | `bz-a3-2:20261002T103938Z-123-5071` |

## Final State

| Item | Value |
| --- | --- |
| Final validated implementation HEAD | `dcf366f` |
| Merge completed at | 2026-10-02 10:29 UTC |
| Rebase or merge still in progress | no |
| Uncommitted changes | report finalization only |

## Residual Risks

The controller host cannot execute the one real Bubblewrap namespace test;
the failure is environmental and unchanged from main. BZ validation covered
the host-owned terminal path, not a live external agent invocation.

## Next Action

Publish and review the frozen pull-request head. Do not merge until the review
wave reports clean.
