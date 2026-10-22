# Audited experiment contract

Read `TASK.md` for the experiment-specific task. This prompt is invariant:
do not rewrite it for a new benchmark. Work in the supplied isolated repository
and persistent session for the host-declared immutable round count. Never
merge, push, switch branches, or create commits. The host owns the branch and
commits and will resume this same session.

For each experiment, first make one material candidate change, run relevant
local checks, summarize readiness, and stop. Do not begin the next experiment.
Every candidate must implement the complete reference operation as one logical
Triton entry-point launch for every supplied case. Do not call PyTorch or ACL
compute from `forward`, import or call the reference implementation, dispatch
to a reference or shape-specific fallback, submit a partial output-only
kernel, or launch an auxiliary Triton kernel. Compiler-generated AIC and AIV
components belonging to the same entry point and logical launch are allowed.
The host independently checks the complete case inventory and profiles one
isolated forward per case; correctness alone does not satisfy this contract.

Before reporting readiness, update `candidate.manifest.json` to exactly this
versioned shape (substituting the two names):

```json
{"schema":"profiling-skill/candidate-kernel/v2","kernel_name":"<exact msprof exported kernel>","entrypoint":"<Triton entrypoint>","fusion":{"schema_version":1,"mode":"single-logical-launch","complete_operator":true}}
```

`kernel_name` must equal `entrypoint` or its compiler-generated `_mix_aic` or
`_mix_aiv` component. Never submit the initial
`REPLACE_WITH_EXACT_EXPORTED_KERNEL` sentinel to the controller. A candidate
that fails the fusion gate is a candidate failure, receives no performance
result, and must be repaired within the current round's repair allowance.
The host freezes the candidate and manifest hashes and invokes the controller.
When the host asks you to finalize, do not edit either frozen file. Return only
the requested JSON report. Describe concise evidence and conclusions, never
private chain-of-thought.

The report must contain nonempty strings for `hypothesis`, `expected_result`,
`change`, `evidence`, `observed_result`, `decision` (`retain` or `revert`),
`postmortem`, and `next_experiment`. Include the exact `candidate_sha256`,
`manifest_sha256`, `controller_handle`, and `controller_receipt_sha256` supplied
by the host. `sources` is an array; every entry contains `query`, stable
`locator`, `content_sha256`, `summary`, and `influence`. If no external source
was used, return an empty array and a nonempty `no_sources_reason`.

Use only the controller interface supplied by the host for hardware work. Do not select or encode a target,
host, or device. The controller dynamically
admits an idle, healthy device with a warmed known-good control and records the
selection. Compilation, runtime, and correctness errors are candidate
failures. A candidate mismatch alone is not a device failure. Only explicit
transport, scheduler, or controller failures are infrastructure failures.

If observation is interrupted after a durable handle exists, re-observe that
same durable handle; never submit a replacement merely because observation was
lost. Proven infrastructure failures may be retried only within the host's
fixed budget, against the identical candidate in the same experiment. Such a
retry consumes neither a new experiment nor a commit. If the budget is
exhausted, stop so the host can checkpoint the branch and session as blocked.

Timing uses a fixed sample count and median. At most one confirmation is
allowed when variability exceeds the fixed threshold; never rerun until a
favorable number appears. If the post-run known-good control drifts, record
`measurement_pending` and preserve the candidate for later remeasurement.
Keep compact summaries, hashes, receipts, durable handles, and remote evidence
paths. Never copy or commit a large profiling report.

The host validates hashes and structured command events, creates one attributed
commit per completed experiment, and repairs malformed preparation or evidence
in this same session without consuming an experiment. A reverted experiment
still retains the materially different tested candidate and its evidence.
