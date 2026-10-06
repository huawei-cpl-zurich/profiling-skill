# Audited Resource Admission Code Ablation

## Scope

- Source plan: `.agent-state/tasks/audited-campaign-resource-admission.md`
- Implementation branch: `codex/audited-campaign-resource-admission`
- Review base: frozen campaign core `f47713b`
- Status: ready-for-pr-review

## Branch And Diff Summary

| Item | Value |
| --- | --- |
| Main changed area | Standalone pinned admission/provider module |
| Excluded areas | Bootstrap, provenance closure, controller, lifecycle, verifier, recovery |
| Suspected unrelated changes | None |

## Complexity Inventory

| Candidate | Classification | Evidence | Action |
| --- | --- | --- | --- |
| Fixed global-client resolver and hash recheck | essential | Enforces the workspace remote-access boundary across repeated admission refreshes | Keep |
| Final-line JSON receipt parser | essential | Global transfer tooling may emit progress before JSON; ambiguous receipts must fail closed | Keep |
| Injectable subprocess callable | essential | Functional tests cover exact argv and response behavior without remote mutation | Keep, constructor-only |
| Dataclass/model layer for four slot fields | speculative | Exact dictionaries already match the scheduler protocol | Omitted |
| Device discovery or occupancy guessing | unsupported | Global capabilities do not expose physical occupancy | Omitted; require pinned provider input |

## Ablations Performed

| Change | Reason | Validation |
| --- | --- | --- |
| Excluded launcher and runtime concerns from the production source | Preserve one safe vertical and a small import surface | Import and focused tests require no production launcher |
| Removed caller-supplied global-client path | Prevent alternate transport injection | Fake global client is reached through isolated authenticated-home lookup |
| Rejected schema extensions and duplicate slots | Avoid unversioned provider semantics and double scheduling | Parameterized schema tests |

## Final Validation

| Command | Result |
| --- | --- |
| Focused admission and scheduler tests | 31 passed locally; 31 passed on BZ-A3-1 |
| Full applicable suite | 717 passed, 1 environment-inapplicable test deselected |
| Ruff and diff check | Passed |

Remote validation used the source client from the named-runtime prerequisite,
the pinned temporary registry, and runtime `py311-torch`. Retained handle
`remote:bz-a3-1:job:20261006T085744Z-8b714d829046` completed with exit zero.

## PR Review Assessment

The change is one cohesive prerequisite: validate a pinned provider, intersect
it with approved global remote health, and return scheduler-compatible slots.
It is independently reviewable and should land before production integration.

## Residual Risks

The occupancy snapshot is externally produced because the global remote API
does not expose device occupancy. Its hash and schema are enforced, but its
freshness remains the campaign operator's responsibility.

## Next Action

Commit the reviewed head for coordinator integration into the production PR.
