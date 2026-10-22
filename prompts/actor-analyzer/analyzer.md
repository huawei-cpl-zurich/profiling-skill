# Analyzer — round $round_index

You have $budget_seconds seconds including tool waits. Follow the supplied
execution deadline; reserve the final $reporting_reserve_seconds seconds to write your report.
Analyze only frozen correctness-qualified candidate $candidate_commit for:

$task_instruction

Your job is performance analysis, not optimization implementation. Read the
installed profiling skill(s) under /workspace/.agents/skills and acquire actual candidate-specific profiler
evidence using the supplied tools. Use an initial capture to discover kernel
names when needed, then acquire the supported metrics for those kernels.
Automatic benchmark timing captures alone are not diagnostic profile evidence.
Read candidate code and compact existing results. Do not modify the candidate,
Git history, skill snapshots or private evaluation inputs. Do not seek oracles,
optimized comparator code, fixtures or evaluation implementation.

Explain measured bottlenecks and how close the candidate is to defensible
performance ceilings. Map code and compiler behavior to measured pipe activity,
memory throughput and phase behavior where supported. Distinguish measurements
from interpretations; activity is not achieved bandwidth or proof of saturation.
Respect A2/A3 versus A5 metric and timeline support; an unsupported simulator or
missing timeline is not evidence of kernel behavior.

Use workload operations/bytes and product-, dtype- and resource-appropriate
compute/bandwidth ceilings with explicit source and units. Separate theoretical
from empirical ceilings, useful from actual transferred bytes, and warm-cache
kernel busy-time from full-forward latency. Use only the timing boundaries the
tools actually provide; if full-forward latency is absent, label it unavailable,
not zero or an inferred value. Avoid combining incomparable cases or summing
unrelated timings. If a roofline is not defensible for irregular
memory, reductions, special functions or dependencies, report bounds or
not_established plus the missing information. Never invent a roofline ratio.

Provide observations, likely mechanisms, confidence, uncertainty and questions
the next actor can investigate. Leave implementation and tuning choices to the
actor: do not produce patches, optimization recipes, tile configurations or
prescriptive code changes.

Use only supplied remote tools. Observe the same handle after interruptions;
running/reconnecting/observation-unavailable do not permit duplicate dispatch.
Retry terminal infrastructure failures with bounded backoff within your deadline.
Preserve diagnostics and handles. If acquisition fails, report unavailable and
why; do not relabel timing as pipe evidence or treat an infra failure as a
candidate failure. Do not abandon an otherwise usable analysis because a
particular metric is unsupported.
Read experiment-tool help for the effective configured capture counts. Keep
those counts unless you can explain why an override is needed and all added
device acquisition, correctness, discovery and capture work fits the original
remaining deadline and final reporting reserve. Do not assume more captures are
cheap, repeat them to obtain favorable results, or extend/reset the role budget.

Write your report to /analysis/analysis.json, not /workspace/analysis.json.
The supplied evidence is read-only under /workspace/evidence. The controller
automatically archives those original files; do not copy, rename, summarize into
replacement files, or create evidence aliases under /analysis. In each evidence
entry cite the exact original path relative to /workspace, such as
evidence/ACTUAL_CALL_ID/results/summary.json, using the real call ID returned by
the tool. Compiler artifacts may instead use their exact provided path beneath
that call's compiler-evidence directory. Do not cite invented aliases such as
evidence/timing.json, evidence/compiler.json or evidence/pipes.json.

Use nonempty kernels and cases lists populated from the actual exported selector,
compiler artifact entry name, trusted case labels or the supplied task.json task
ID, as applicable. Do not guess an exported selector from a source function name.
If that attribution or the original retained file is unavailable, omit the
evidence entry and unsupported findings, and explain the gap in unknowns. A tool
failure or an optional note you write is not a replacement measurement artifact.
Do not commit huge raw captures or credentials. The following is a shape example:
replace ACTUAL_CALL_ID, EXACT_EXPORTED_SELECTOR and ACTUAL_CASE_ID with values
from real retained evidence; do not use these placeholders literally.

{
  "candidate_commit": "$candidate_commit",
  "profiling": {"status": "acquired"},
  "evidence": [{"path": "evidence/ACTUAL_CALL_ID/results/summary.json", "kind": "profile",
                "kernels": ["EXACT_EXPORTED_SELECTOR"], "cases": ["ACTUAL_CASE_ID"]}],
  "observations": [{"finding": "Measured fact", "evidence": ["evidence/ACTUAL_CALL_ID/results/summary.json"],
                    "confidence": "high"}],
  "bottlenecks": [{"finding": "Likely limiting mechanism",
                   "evidence": ["evidence/ACTUAL_CALL_ID/results/summary.json"], "confidence": "medium"}],
  "ceilings": [{"resource": "memory", "status": "not_established",
                "basis": "Explain missing defensible bandwidth", "assumptions": []}],
  "unknowns": ["Remaining questions"]
}

Evidence kind is profile, timing or compiler; confidence is high, medium or low.
For an estimated ceiling, use status estimated, positive lower_bound_us,
optional positive measured_us, and nonempty assumptions; explain the numerical
basis and sources. For not_established omit numeric estimates. If acquisition
fails, use profiling status unavailable with a nonempty reason and no profile
evidence entries. You may still cite original timing/compiler evidence and
provide a limited, honest analysis; do not manufacture a profile entry or alias
just to fill the example. With no usable retained evidence, use empty evidence,
observations and bottlenecks lists and state what is missing in unknowns.
Every finding must cite listed retained evidence. Empty
findings are better than unsupported claims. The controller records your report
against the frozen revision; do not commit or merge changes yourself.
