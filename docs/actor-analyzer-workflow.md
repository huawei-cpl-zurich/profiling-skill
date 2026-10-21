# Alternating candidate generation and performance analysis

The actor produces candidates; the analyzer explains measured performance.
This workflow is opt-in until the three paired A3 smokes pass. Existing prompts
and campaigns remain historical evidence, not migrated or overwritten.

## Contract

Use `scripts/actor_analyzer_contract.py` to render the versioned role prompts:

```python
from scripts.actor_analyzer_contract import build_prompt, validate_analysis

actor_prompt = build_prompt("actor", 1200, 1, "Implement the supplied public API")
analyzer_prompt = build_prompt("analyzer", 600, 1,
                               "Analyze the supplied public API", "a" * 40)
# Load analysis.json after the analyzer finishes, then validate its structure:
# validate_analysis(report, expected_commit)
```

Defaults are four actor → validation/freeze → analyzer rounds, independently
configurable 1,200/600-second budgets, including tool waits. The actor may use
profiling but need not. The analyzer explicitly acquires diagnostic profiling,
interprets bottlenecks and justified rooflines, and leaves optimization choices
to the actor. Compilation and numerical failures do not end the actor's round;
debugging continues until the time deadline, without an attempt-count cap.

Only correctness-qualified candidates are analyzed. If a round fails, reuse the
last verified candidate and name its actual commit, not the failed revision. If
none exists, retain diagnostics without manufacturing a performance assessment.

One unmerged branch retains candidate experiments, commands, compact results,
reasoning, expected outcomes, and postmortems. The analyzer has read-only access
to the candidate and Git metadata; only its analysis output is writable. The
trusted controller validates evidence and commits reports against the frozen
candidate. Actors receive the latest report and history on the next round.

## Evidence and ceilings

`analysis.json` identifies `candidate_commit`, profiling acquisition status,
relative evidence paths with exact kernel/case attribution, observations,
bottlenecks, ceiling assessments, and unknowns. The analyzer prompt contains the
complete minimal example. Every finding cites retained evidence and states its
confidence. Unsupported estimates use `not_established` and explain why.

`validate_analysis(report, expected_commit)` returns the same dictionary or
raises `ValueError`. It checks structure and frozen-revision attribution, not
whether a claim is true. The controller must additionally verify evidence file
containment (including symlinks), existence, provenance, and actual diagnostic
acquisition. A valid `unavailable` report preserves infrastructure failure;
it does **not** pass a profiling acquisition acceptance gate.

`roofline(compute_ops, traffic_bytes, peak_ops_per_second,
bandwidth_bytes_per_second, measured_us=None)` computes ideal compute and memory
times in microseconds and their maximum, an ideal lower bound. Inputs must use
matching product/dtype/operation and traffic-boundary conventions. An optional
`ideal_fraction = lower_bound_us / measured_us` is not capped: above one exposes
inconsistent assumptions, counting or timing boundaries. It is not evidence of
super-peak hardware. The helper does not supply hardware peaks or infer saturation
from activity. Dependency, irregular-access, reduction, or special-function
ceilings may remain unknown even when compute/memory bounds are calculable.

## Qualification

Run functional tests with:

```bash
python -m pytest -q tests/test_actor_analyzer_contract.py
```

Controller integration adds role isolation, evidence verification, independent
deadlines, failed-candidate handling and restart/no-duplicate tests. The A3
behavioral gate compares actor-only against actor/analyzer on existing natural
gather (`task01`), masked-computation (`task03`), and matmul (`task05`) before-kernels,
four rounds each. Both arms have equal actor allowance; analyzer time is extra
and reported, not an equal-total-budget speedup comparison. Acceptance requires
real diagnostic acquisition, defensible interpretations, and evidence that the
next actor evaluated the findings in all three tasks. A justified rejection of
a hypothesis is useful evidence; reading the report alone is insufficient.
Negative and infrastructure-limited results are retained separately.
