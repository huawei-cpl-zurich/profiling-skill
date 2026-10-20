# Trusted controller integration

Run from the repository, with adapter modules importable through Python's normal
module search path:

```sh
python scripts/model_benchmark.py \
  --baseline reference:Model --candidate candidate:Model \
  --cases-provider private_cases:cases --validator private_cases:validate \
  --acquisition private_capture:create_adapter \
  --expected-kernel submitted_kernel --output results/new-run
```

The output directory must not already exist. Exit 0 means complete acquisitions,
not necessarily comparable clocks or a correctness claim beyond the validator.
Exit 2 retains `report.json`, diagnostics and any already-acquired evidence.

Hooks are trusted controller code. `cases()` returns dictionaries with unique
string `id`, positional `args`, optional `kwargs`, optional Model `init_args` and
`init_kwargs`. Both implementations receive the same immutable device inputs.
Factories produce inference Models; constructors handle device placement.
`validate(case, arm, output)` raises on mismatch against an independent reference.

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
