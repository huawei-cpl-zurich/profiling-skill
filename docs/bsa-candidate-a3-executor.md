# BSA candidate A3 executor

`scripts/bsa_candidate_a3_executor.py` is the production adapter for the
request/response contract used by `bsa_candidate_profiler.py`. It stages each
candidate with the frozen BSA baseline, cases, and runner in a content-addressed
bundle. The bundle performs correctness and `msprof op` analysis remotely and
returns compact evidence; vendor report trees remain on the A3 host.

## Local configuration

The adapter intentionally contains no machine-specific workspace, device, or
credential values. Configure an invocation with ignored local environment:

```bash
export BSA_A3_TARGET=bz-a3-1
export BSA_A3_REMOTE_ROOT=/the/provisioned/writable/workspace
export BSA_A3_STATE_DIR=/a/durable/local/state/directory
```

`BSA_A3_TARGET` may be `bz-a3-1` or `bz-a3-2`. The named runtime is taken from
the candidate manifest's frozen A3 environment and passed to `cpl-remote run
--runtime`; it is never expanded to a runtime path. `BSA_A3_STATE_DIR` is
optional and defaults beside the response file, but campaign runs should set a
durable ignored directory explicitly.

Before a campaign, independently confirm the selected route:

```bash
~/.agents/skills/remote-access/scripts/cpl-remote capabilities "$BSA_A3_TARGET"
~/.agents/skills/remote-access/scripts/cpl-remote preflight "$BSA_A3_TARGET"
```

The coordinator then invokes the executable without adapter-specific flags:

```bash
scripts/bsa_candidate_a3_executor.py \
  --request evidence.protocol/request.json \
  --response evidence.protocol/response.json
```

## Execution and evidence

The adapter validates the request and source digest before remote work. It uses
only the installed global remote-access client to run capabilities and
preflight checks, upload the deterministic tar bundle, dispatch a file-backed
script under the named runtime, and observe the retained handle.

The remote driver probes physical devices with a bounded Torch-NPU operation.
The chosen physical ID is exposed only as provenance; the application sees
logical device 0. It first runs all requested correctness cases through the
frozen benchmark runner. Candidate compilation, execution, and comparison
failures remain `compile_error`, `runtime_error`, or `correctness_error`.
Device, transport, profiler, selector-evidence, and malformed-result failures
are `infrastructure_error`.

Discovery performs an unfiltered BasicInfo capture after correctness. It
returns only observed names matching the declared Triton entrypoint (including
the compiler's component suffix) and their compact rows. Capture reruns the
single requested case, supplies that exact discovered name to msprof with
`PipeUtilization`, verifies selector-binding BasicInfo, and explicitly launches
the candidate seven times. The distinct BasicInfo durations are cached by
candidate, case, and selector; subsequent coordinator sample requests consume
the corresponding cached duration without another remote job. Repeated
PipeUtilization rows are reduced to per-block/sub-block medians, retaining one
compact pipe table shared by those timing samples.

Every successful response records target, named runtime, selected physical
device, normalized `cpl-remote run` command, and durable handle. The evidence
digest covers the complete compact remote result.

## Interruption and recovery

Dispatch state is keyed by the canonical request plus the frozen baseline,
case data, runner, and remote-driver hashes. The durable handle is written
before observation. Repeating the same request with the same harness observes
that handle and does not upload or submit a replacement job. An observer interruption therefore
returns retryable infrastructure evidence containing the original handle; rerun
the same coordinator command after transport recovery.

Do not delete the executor state while a handle may still be running. A changed
candidate, case, selector, action, or harness revision produces a different
execution identity and therefore a separate retained job.
