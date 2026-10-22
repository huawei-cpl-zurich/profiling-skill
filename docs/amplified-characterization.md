# Fresh paired characterization

`scripts/a3_profile_amplified_campaign.py manifest` emits nine derived pairs
(18 cells): scalar-mask-decision/blocks-32, load-masking/irregular-25, and
sparse-mode-control/dense-mode, each with repetitions 1, 32 and 128. These are
new runtime-loop/volatile workloads, never replacements for original results.
Inputs use the original deterministic generator. Numerical correctness and
retained compiler runtime-loop proof are required at discovery.

Run a pair with `pair --pair-id amplified/load-masking/128 --campaign NAME
--output DIRECTORY`. Use a new directory and campaign name for a fresh run.
All remote work uses the acquisition interface, BZ-A3-1 and its `py311-torch`
runtime. The caller owns device availability, concurrency and qualification;
this command does not start a second scheduler. Its imported
`run_pair(spec, output, acquire, campaign=NAME)` interface also accepts original
and timeline-control before/after cells. Timeline acquisition itself belongs
to the separate timeline worker and is not duplicated here.

Each side discovers exact selectors, then collects independent BasicInfo timing
and three PipeUtilization captures. Acquisitions alternate before/after while
both sides need work. Accept the first consecutive three timing samples within
5%, at most seven valid timings. Preserve every sample, including instability;
still collect pipe evidence for unstable timings. BasicInfo sums medians of
declared sequential kernels; it does not add overlapping per-core activity.

`pair-result.json` checkpoints the immutable source/arguments/campaign contract
and every capture receipt. Resume the identical command. Pending transport
observation returns exit 75, preserves the same acquisition path and does not
consume attempts; the caller waits 15 seconds before resuming it. Confirmed
infrastructure failures have two retries; deterministic profiler/evidence
failures stop after two attempts, compiler/runtime failures immediately. One
failed side never suppresses acquisition for the other. `summary --output ROOT`
exports all retained pair/cell dispositions without importing historical data.

Keep each scientific experiment on an unmerged evidence branch with its exact
source, commands, compact results, hypothesis, reasoning and postmortem. Full
vendor reports stay remote. Ratios describe captured activity, not saturation;
timelines and code-phase attribution require independent valid evidence.
