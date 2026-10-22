# Trusted controller integration

Run from the repository, with adapter modules importable through Python's normal
module search path:

```sh
python scripts/model_benchmark_repeated.py \
  --baseline reference:Model --candidate candidate:Model \
  --cases-provider private_cases:cases --validator private_cases:validate \
  --acquisition profile_capture:create_acquisition \
  --expected-kernel submitted_kernel --output results/new-run
```

The repeated controller defaults to `--repetitions 10`. Each fresh child calls
`scripts/model_benchmark.py` once with the same remaining arguments, in the same
device/runtime environment. Hold the device lease outside this command. Keep
private hook modules importable in children; construct reproducible cases within
`cases()`, not in ephemeral parent-process globals. `--repetitions` can explicitly
select a smaller development smoke, but that is not the default ten-run report.
The caller owns the original overall execution deadline and durable remote job;
ten repetitions do not grant ten new time budgets. Run this command within that
bounded job and observe the same handle after a transport interruption. Do not
restart the series merely because observation was interrupted. If the deadline
stops execution early, retain the last summary and all child artifacts and report
the series incomplete, including any final in-flight child not yet summarized.

The output directory must not already exist. Each child retains its unmodified
`run-NNNN/report.json`, raw artifacts and a sibling `run-NNNN.log`. Before launching
each child, the parent atomically updates `summary.json` with its exact command,
evidence paths and `running` state, then atomically finalizes that entry afterward.
This includes requested/attempted/completed/failed/running/unattempted counts,
failures, raw report paths, metric values, units,
arithmetic mean and sample standard deviation (`ddof=1`). Means summarize per-run
medians, never pooled call samples. Source and case specialization changes reject
the affected run. Qualified runs remain qualified; cross-repetition condition
equivalence requires separate inspection of retained telemetry. A failed child
counts as one failed repetition, not an invisible retry.
After an interruption, a retained `running` entry means its outcome is unknown,
not that it was never attempted. Observe the same outer job and inspect that
entry's artifacts; the controller does not automatically restart it. Attempted
equals completed plus failed plus running; requested equals attempted plus
unattempted. Malformed reports are retained as failed attempts and do not abort
the remaining planned repetitions.

Exit 0 means all requested acquisitions completed,
not necessarily comparable clocks or a correctness claim beyond the validator.
Exit 2 retains `summary.json`, diagnostics and any already-acquired evidence. For
a single-run diagnostic primitive, invoke `scripts/model_benchmark.py` directly;
its existing output and exit semantics remain unchanged.
`--acquisition module:factory` is required: select a validated platform adapter
explicitly. The dependent acquisition skill implementation provides
`profile_capture:create_acquisition` when installed alongside this controller; an
independent controller checkout never silently assumes that module exists. TorchAir GE
is the default backend. Explicit `--backend npu-inductor` selects NPU Inductor for
both implementations; unsupported graph/backend combinations fail without fallback.

Hooks are trusted controller code. `cases()` returns dictionaries with unique
string `id`, positional `args`, optional `kwargs`, optional Model `init_args` and
`init_kwargs`. Optional per-case `expected_kernel` is a nonempty list of exact
submitted task names and overrides CLI names for legitimate entrypoint dispatch.
Both implementations receive the same immutable device inputs.
Factories produce inference Models; constructors handle device placement.
`validate(case, arm, output)` raises on mismatch against an independent reference.
The controller snapshots nested tensor inputs and initialization arguments before
preparation. It checks structure, version counters and exact content after each
Model preparation/profile, each arm's warmup and each measured batch, always
outside timed windows. Raw-pointer writes are detected by content comparison even
when tensor versions do not change. Mutating Models fail before a valid paired
result is reported. Input snapshots consume extra device memory; inference-mode
input tensors without version tracking are explicitly unsupported. Ordinary NaNs
in unchanged inputs compare equal. Do not pass mutable inputs to this interface.

Every case retains a source-independent specialization identity: input tensor
shape/dtype/stride/device/gradient descriptors, scalar/container arguments and
Model initialization configuration. Tensor values are never serialized. Optional
`metadata` contains explicit private reproduction settings such as generator seed,
dataset/version and runtime configuration consumed by the provider; declare them
when they cannot be reconstructed from provider source alone. Configuration accepts
only finite JSON primitives, string-keyed dictionaries and lists/tuples, with
tensors represented by descriptors; known `torch.dtype`, `torch.device` and
`torch.Size` constructor options use explicit tagged descriptors. Arbitrary
objects/repr are rejected.
The descriptor and metadata are retained in controller-only
`case-NNNN/private-specialization.json`; the main report contains its digest and
relative location. Do not publish these private artifacts in blind-agent exports.
Agent-facing exporters must allowlist measurement artifacts rather than recursively
copy every JSON file. Source identities and specialization identities complement
each other; neither asserts the contents of an external unversioned dataset.

`create_adapter()` returns:

- `synchronize()`: synchronizes the selected NPU.
- `provenance()`: JSON-safe device, runtime, clock/load and acquisition metadata.
- `profile(prepared, samples, warmup, output)`: profiles the callable after warmup
  and returns normalized `iterations`, `tasks`, `capture_complete`, plus retained
  acquisition/export/correlation evidence. `capture_complete=True` is an assertion
  by this trusted adapter, not a way to bypass acquisition checks. The adapter
  restores profiler state before returning and retains original compact exports.
- `comparability(before, after)`: JSON-safe assessment with `comparable` and
  reasons. Return true only with established same-device/runtime/clock policy and
  adequate observed clock/load evidence, never simply because both runs finished.

Each normalized iteration has `iteration_id` and explicit `correlation_ids`.
Each task has unique `task_id`, `correlation_id`, exact `name`, `kind` (`device`
or `aicpu`), `start_us` and `duration_us` on one normalized clock. Independent
`expected_task_ids` per iteration may establish inventory completeness. Retain
evidence for rows excluded as warmup/host/unrelated work. The controller requires
exact submitted kernel names on every candidate iteration; repeat
`--expected-kernel` for required components. Establish names through an initial
discovery capture, not an invented selector. The adapter must validate supported
vendor schemas; unknown schemas are explicit capability errors.

This adapter interface deliberately has no automatic eager or fabricated-profile
fallback. Hardware support is accepted only after real compiled Model/Triton and
correlated profiler smokes. Do not export this controller, helpers, private hooks
or input providers into blind-agent workspaces: export only `SKILL.md`, `agents`
metadata and the relevant guidance reference, together with a thin service client.
