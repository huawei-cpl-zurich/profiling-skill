# Profiling discovery prompt: planning-only assessment

## Outcome

The revised prompt passes the discovery/comprehension gate: all three completed
revised-prompt agents actually read `ascend-profiling/SKILL.md`, distinguished
benchmark timing from diagnostic profiling, and proposed conditional acquisition
after correctness. None was required to collect pipe metrics. **No kernels,
benchmarks, profiles, or remote workloads were executed by these agents.**

| Completed cell | Seconds | Verified profiling skill read | Conditional profiling plan | Timing/diagnostics distinguished |
|---|---:|---|---|---|
| Previous 1 | 87.12 | Yes | Yes | Yes |
| Previous 2, unchanged retry | 68.64 | No: coding guidance only | Yes | Yes |
| Previous 3 | 71.71 | Yes | Yes | Yes |
| Revised 1, unchanged retry | 70.08 | Yes | Yes | Yes |
| Revised 2, unchanged retry | 59.65 | Yes | Yes | Yes |
| Revised 3 | 68.69 | Yes | Yes | Yes |

Thus actual skill reading was **2/3 versus 3/3**; planning some profiling was
**3/3 versus 3/3**. This tiny planning-only sample does not establish a causal
improvement, actual acquisition success, skill usefulness, or performance gains.
The common planning instruction also changes behavior relative to a coding run.

## Evidence and assessment

Human assessment inspected successful skill-read commands and returned contents
in `agent.jsonl`, not just final claims or keyword counts. Each completed cell
produced a substantive `workspace/PLAN.md`; tool journals contained only help
and capabilities calls. Candidate hashes remained identical to the initial
skeleton. The local stub has no execution backend and denies operational calls.

All revised plans proposed `experiment-tool profile --kernel NAME` only after
correctness and benchmark evidence, with discovery of the exported selector.
They distinguished activity from proven saturation, kept unsupported capacity
claims unknown, and separated A3 from A5 metric semantics. Their hypotheses
included V-tile reuse versus live-state/spill pressure, redundant normalization
versus a preprocessing launch, and memory-transfer versus compute-heavy work.
They proposed a controlled code change followed by ordinary benchmarking to
test the explanation, or skipping profiling when timing already resolves it.

The control retry proposed named-kernel profiling for resource pressure or
memory stalls without reading the profiling skill. The other two controls read
the skill and also planned target-correct acquisition. Skill availability and a
reasonable plan must not be reported as demonstrated use of live evidence.

## Reproduction and attempts

See [runner instructions](profiling-prompt-planning-smoke.md). This assessment
used source revision `6ee2ff6`, `gpt-5.6-sol` with low reasoning effort, the same
public packed-GDN API/skeleton and frozen coding/profiling guidance in both arms,
and a 300-second cap per cell. Only the profiling-discovery paragraph differed;
both complete prompts received the same planning-only override.

The previous prompt SHA256 was
`c64907425a1622d6c61397ad94000d9f93f742359cc45c7d4f27f2c816da2ae6`;
the revised prompt SHA256 was
`cedf517e2ca54b3c2100b5022d360fd65388abc69f6ae99d884d5e12a69e9cce`.
Per-cell manifests retain the effective prompt, skill and public-input hashes.

The initial six-concurrent wave completed Previous 1/3 and Revised 3. Previous 2
and Revised 1/2 timed out after 300 seconds with only thread/turn-start events:
no commands, plan, or substantive response. Those are unassessable attempts,
not evidence of refusal to read or use a skill. Their cause is unresolved.
Only those three cells were retried after terminal timeout, with unchanged
inputs/prompts/model/budgets and at most two active retries. All three retries
completed. **Nine attempts were retained to obtain six assessable responses.**

Retained run IDs are `profiling-planning-smoke-20261020-01`,
`profiling-planning-smoke-20261020-previous-retry01`, and
`profiling-planning-smoke-20261020-revised-retry01`. Initial Previous 2 maps to
the previous retry's cell 1; initial Revised 1/2 map to revised retry cells 1/2.
The orchestration task record locates the evidence archive, including failed
attempts, full transcripts, commands, manifests, isolation checks and plans.
