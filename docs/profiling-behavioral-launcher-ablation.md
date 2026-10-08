# Profiling Behavioral Launcher Code Ablation

## Scope

- Source PRD: user profiling-skill behavioral acceptance request
- Source TDD plan: `.agent-state/tasks/profiling-behavioral-launcher.md`
- Implementation branch: `codex/profiling-behavioral-launcher`
- Review base branch: `codex/profiling-behavioral-gates` at `b99b904286ecaf5860382689e9404ca0a573da73`
- Status: ready-for-pr-review

## Branch And Diff Summary

| Item | Value |
| --- | --- |
| Current branch | `codex/profiling-behavioral-launcher` |
| Base branch | `codex/profiling-behavioral-gates` |
| Changed files | launcher, functional tests, operator guide, this report |
| Diff size | approximately 680 source and 270 test lines before final cleanup |
| Main changed areas | Bubblewrap specialization, remote broker/checkpointing, record retention/review |
| Suspected unrelated changes | none |

## Baseline Validation

| Command | Result | Notes |
| --- | --- | --- |
| `pytest -q tests/test_profile_behavioral_launcher.py` | 13 passed | focused behavior |
| `pytest -q` | 1091 passed | full local suite |
| `git diff --check` | passed | no whitespace errors |

## Changed Areas

| Area | Files | Behavior touched | Tests |
| --- | --- | --- | --- |
| Live adapter | `scripts/profile_behavioral_launcher.py` | isolation, sessions, broker, records | focused launcher suite |
| Operator contract | `docs/profiling-behavioral-launcher.md` | launch/review/blocker workflow | exercised through APIs, not wording |

## Complexity Inventory

| ID | Candidate | Classification | Evidence | Proposed action | Status |
| --- | --- | --- | --- | --- | --- |
| C1 | Reuse `ProductionLauncher` as a base | essential | preserves audited auth, runtime, resolver, and thread parsing | keep | retained |
| C2 | Host socket broker | essential | prevents credentials/raw transports entering the sandbox | keep | retained |
| C3 | Separate artifact/review helpers | essential | core gate requires bounded hashes and external review | keep | retained |
| C4 | Generic Docker or direct-SSH backends | speculative | not requested and would duplicate/weaken existing boundary | do not add | ablated by design |
| C5 | Agent-selected infrastructure classification | accidental | would allow score manipulation | keep classification host-owned | ablated by design |

## Unsupported Capabilities

| ID | Capability | Failing case | Evidence | Current assumption | Impact | Status |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | Live BZ-A5 execution | global target lacks a registered A5 runtime/wrapper | workspace plan and current remote registry inspection | live gate needs a supported route | blocks live A5 acquisition, not adapter/unit tests | external blocker |

## User Decisions

| ID | Decision needed | Options | Recommendation | User decision | Status |
| --- | --- | --- | --- | --- | --- |
| D1 | Unsupported A5 route | weaken transport, or block honestly | block until registered | plan explicitly forbids weakening | resolved |

## Ablations Performed

| ID | Change | Reason | Validation | Files |
| --- | --- | --- | --- | --- |
| A1 | Excluded Docker/raw SSH/config mounts | one approved remote broker is smaller and safer | mount/broker tests | launcher |
| A2 | Kept external review out of the agent session | agents cannot attest their own interpretation | finalization test | launcher/tests |

## Fixes Performed

| ID | Change | Reason | Validation | Files |
| --- | --- | --- | --- | --- |
| F1 | Bind success record identity to actual Codex thread | prove persistent session, not caller label | persistent-turn test | launcher/tests |
| F2 | Check target encoded in observe/log/result handles | prevent cross-target capability escape | allowlist test | launcher/tests |
| F3 | Retain command arguments in the broker journal | preserve command evidence | broker checkpoint test | launcher |
| F4 | Pin the adapter plus inherited production boundary bytes and canonical model config | prevent trusted identity spoofing | identity-drift tests | launcher/tests |
| F5 | Bind each acquisition handle to its broker target and exact remote command | prevent cross-target or fabricated command evidence | acquisition-draft test | launcher/tests |
| F6 | Restrict observation to handles in the session journal and retain every remote result | prevent unrelated-job access and classify evidenced preflight failures | broker allowlist tests | launcher/tests |
| F7 | Fail closed on handle-less uncertain dispatches and propagate explicit observation | prevent duplicate remote jobs and stale running evidence | broker recovery tests | launcher/tests |
| F8 | Require terminal successful acquisition dispatch and confine download destinations | reject stale/failed evidence and host writes outside the sandbox | draft/download tests | launcher/tests |
| F9 | Snapshot and rehash prompt, skill, and case inputs before launch | close mutable-source timing gaps | snapshot test | launcher/tests |
| F10 | Mount inherited launcher module and execute the wrapper inside real Bubblewrap | make the isolated broker entrypoint runnable | real Bubblewrap test | launcher/tests |
| F11 | Remove host session/arm clues from agent request JSON | preserve blinded arms | exact public-payload test | launcher/tests |
| F12 | Emit null interpretation failure location and validate with the pure gate | preserve core schema compatibility | end-to-end gate test | launcher/tests |
| F13 | Preserve/normalize trusted remote failure types and full failure evidence | distinguish evidenced infrastructure from compile/runtime failures | preflight/transport/compiler-log tests | launcher/tests |
| F14 | Make explicit result update dispatch and key dispatches by key+argv+bytes | final evidence and replay identity must be exact | broker transaction tests | launcher/tests |
| F15 | Remove transfers and add request-example/finalize CLI workflows | narrow agent capability and make operation reproducible | CLI functional tests | launcher/tests/docs |

## Complexity Intentionally Kept

| ID | Complexity | Reason kept | Revisit when |
| --- | --- | --- | --- |
| K1 | Broker journal and replay | exactly-once dispatch/same-handle recovery is a core requirement | remote-access provides a native transaction API |
| K2 | Draft plus external finalization | manual reviewer must remain outside tested agent trust | gate adopts a separate signed review service |

## Test Changes

| ID | Change | Behavior protected | Validation |
| --- | --- | --- | --- |
| T1 | 13 functional tests | mounts, pinning, persistence, broker replay, classification, receipts, review | focused suite |

## Final Validation

| Command | Result | Notes |
| --- | --- | --- |
| `pytest -q tests/test_profile_behavioral_launcher.py` | 34 passed | completed-review final; includes real local Bubblewrap |
| `pytest -q` | 1112 passed in 61.84s | completed-review final full run |
| BZ-A3 focused run | 33 passed, 1 skipped in 1.45s | final handle `remote:bz-a3-1:job:20261008T154157Z-9d84d44ef2c5`; Bubblewrap unavailable remotely |

## PR Review Assessment

| Item | Assessment |
| --- | --- |
| Reviewable as one PR | yes, as one trust-boundary adapter stacked on the pure core |
| Main review risks | size and failure-envelope compatibility |
| Distinct review stories | one: run live agents and emit trusted core inputs |
| Backend/compiler/runtime/frontend mix | launcher only; product execution stays behind remote-access |
| Human reviewer notes | review allowlist, handle replay, and external-review boundary first |

## Suggested PR Split

No split recommended. Separating the broker from its isolation and record
tests would make the trust boundary harder to review and temporarily unusable.

## Residual Risks

| Risk | Impact | Follow-up |
| --- | --- | --- |
| A5 runtime unavailable | full live acceptance cannot execute | provision a supported global target runtime/wrapper |
| Real Codex/API behavior not used in local tests | live formatting may reveal integration defects | run acquisition/interpretation gates after A5 provisioning |

## Next Action

Commit the validated branch for parent inspection without pushing. Live A5
behavioral execution remains blocked on a registered global runtime/wrapper.
