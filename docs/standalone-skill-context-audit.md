# Standalone profiling skill context audit

This audit is repository development material, not an installed skill reference.
Original paths below refer to base revision `bc2e3f33a494f81fc855fb28bf6f3878b3be9edc`.
Key original locations: `SKILL.md:38` (campaign entrypoint), `SKILL.md:104`
(target profiles), `SKILL.md:210` (wrappers), and
`references/a2-a3-msprof-op.md:36` (BSA and frozen-job replay context).

| Original location / wording | Problem | Resolution |
| --- | --- | --- |
| `SKILL.md`: “Run agent experiment campaigns” | Campaign controller and benchmark context were presented as profiling prerequisites. | Kept campaign tooling outside the skill and removed its instructions from the package. |
| `SKILL.md`: `$gz-a3`, `$bz-a5`, `TLA_ROOT`, managed jobs and wrappers | Required private infrastructure and an unidentified integration checkout. | Removed target routing and wrappers; stated actual tool/runtime prerequisites and direct file-based workflows. |
| `references/acquisition-workflow.md`: named runtimes, dispatch receipts and remote handles | Generic acquisition depended on repository transport policy. | Replaced with direct msprof discovery, exact selection, comparison provenance, and report validation. |
| `references/a2-a3-msprof-op.md`: BSA, case 47, frozen job | Examples depended on one benchmark and its controller. | Replaced with a correctness-checking workload, explicit replay behavior, and unique capture identifiers. |
| `SKILL.md`: FlashAttention three-way/phase campaigns | Examples could not be understood without a separate checkout. | Removed; retained the reusable A5 measurement and interpretation guidance. |
| `references/a5-timeline-semantics.md`: “validated FA campaign” and “retention gate” | Historical benchmark policy looked like a profiling contract. | Explained repeated sampling in terms of sample intervals and ordinary timing. |
| `references/triton-pipe-attribution.md`: probe `--contract` | Unidentified external probe interface. | Removed; the included query tool remains the documented interface. |
| Compiler and profiler source `ref://` citations | Required a private reference broker. | Replaced with revision-qualified repository/source identifiers; source provenance is distinct from a runtime dependency. |

## Limits that remain explicit

- There is no validated A2/A3 capacity denominator; activity remains insufficient
  for saturation claims. Memory export scaling and A2/A3 FixPipe event mapping
  remain unresolved rather than guessed.
- Triton attribution is pinned to specified compiler revisions; it does not
  establish mappings for every installed compiler or product.
- A5 model coverage is narrow and its stated core counts are not universal.
- Profiler flags and optional simulator/sanitizer support vary by installation;
  installed help and product-compatible tooling determine availability.

No unresolved project-specific wording is intentionally required by the
standalone skill. Benchmark and infrastructure material remains repository-only.

## A3 interpretation follow-up

The standalone A3 material now expands former shorthand (“applicable outbound
DMA”, “wait-like metrics”, and “inspect later IR”) into explicit source-backed
routes, exported fields, calculations, diagnosis tables and late-stage search
instructions. Separate references distinguish an architectural lower bound from
a shape-matched measured envelope and show three generic paired controls.

New public field authority: Ascend/msopprof
`a52c8694979016be703006ec242901519bef31d8`, A2/A3 PipeUtilization and
ResourceConflictRatio sections. Current official compiler auto-sync and
multi-buffer guides are version-dependent descriptions; the older pinned
AscendNPU-IR operation definitions remain identified separately. The guides do
not assume that a compiler option guarantees overlap or that static wait counts
measure dynamic stall time.
