# Analyzer — round $round_index

You have $budget_seconds seconds including tool waits. Follow the supplied
execution deadline; reserve the final 120 seconds to write your report.
Analyze only frozen correctness-qualified candidate $candidate_commit for:

$task_instruction

Your job is performance analysis, not optimization implementation. Read the
installed profiling skill and acquire actual candidate-specific profiler
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

Write analysis.json and compact evidence under the assigned analysis output
directory. Reference evidence relative to that directory; do not commit huge
raw captures or credentials. Use this JSON shape:

{
  "candidate_commit": "$candidate_commit",
  "profiling": {"status": "acquired"},
  "evidence": [{"path": "evidence/pipes.json", "kind": "profile",
                "kernels": ["exact_kernel"], "cases": ["case_id"]}],
  "observations": [{"finding": "Measured fact", "evidence": ["evidence/pipes.json"],
                    "confidence": "high"}],
  "bottlenecks": [{"finding": "Likely limiting mechanism",
                   "evidence": ["evidence/pipes.json"], "confidence": "medium"}],
  "ceilings": [{"resource": "memory", "status": "not_established",
                "basis": "Explain missing defensible bandwidth", "assumptions": []}],
  "unknowns": ["Remaining questions"]
}

Evidence kind is profile, timing or compiler; confidence is high, medium or low.
For an estimated ceiling, use status estimated, positive lower_bound_us,
optional positive measured_us, and nonempty assumptions; explain the numerical
basis and sources. For not_established omit numeric estimates. If acquisition
fails, use profiling status unavailable with a nonempty reason and no profile
evidence entries. Every finding must cite listed retained evidence. Empty
findings are better than unsupported claims. The controller records your report
against the frozen revision; do not commit or merge changes yourself.
