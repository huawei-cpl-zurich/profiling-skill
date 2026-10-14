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
export BSA_A3_TARGET=gz-a3
export BSA_A3_REMOTE_ROOT=profiling-skill-bsa-campaign
export BSA_A3_STATE_DIR=/a/durable/local/state/directory
export BSA_A3_MIN_FREE_DEVICE_BYTES=34359738368
export BSA_A3_CAMPAIGN_REVISION=<frozen-campaign-revision>
export BSA_A3_QUALIFICATION_RECEIPT=/a/durable/local/gz-qualification.json
```

`BSA_A3_TARGET` may be `gz-a3`, `bz-a3-1`, or `bz-a3-2`. BZ targets receive
the logical runtime from the candidate manifest through `cpl-remote run
--runtime`. GZ uses its registry-provided default activation and deliberately
omits `--runtime`; responses and receipts still record `py311-torch` as the
logical runtime and `target-default` as its activation mode. `BSA_A3_STATE_DIR` is
optional and defaults beside the response file, but campaign runs should set a
durable ignored directory explicitly.

For GZ, `BSA_A3_REMOTE_ROOT` is a normalized path relative to the registry's
configured workspace. For either BZ target it remains an absolute provisioned
path. GZ uploads are retained transfers; a nonterminal upload is observed by
its original transfer handle before dispatch. The local attempt records that
handle before observation, so `retry:{mode:observe,handle}` resumes the same
transfer and continues to job dispatch without uploading the bundle again.
A failed transfer handle is included in the compact failure receipt. BZ
uploads retain their existing synchronous SSH behavior.

GZ dispatch additionally requires a passing qualification receipt bound to
the frozen campaign revision and the exact executor, remote driver, baseline,
case data, and runner hashes. Generate it before starting a campaign:

```bash
scripts/qualify_bsa_candidate_a3.py \
  --candidate benchmarks/bsa/qualification_candidate.py \
  --kernel-name block_sparse_attention_kernel \
  --campaign-revision "$BSA_A3_CAMPAIGN_REVISION" \
  --output "$BSA_A3_QUALIFICATION_RECEIPT"
```

The qualification runs the frozen fully fused candidate over cases 47, 46,
49, 44, and 43, discovers exact BasicInfo components, and collects up to seven
independent PipeUtilization captures per case. It accepts the first consecutive
three-sample logical-kernel window with relative range at most 5%. It then runs
an intentionally invalid Triton kernel and requires nonempty compiler
diagnostics classified as `compile_error`. Any failed step writes a compact
`passed:false` receipt and exits nonzero; the production GZ executor rejects a
missing, failed, or identity-drifted receipt before any remote call. A passing
receipt retains the exact ordered five-case correctness evidence, and dispatch
also rejects missing, partial, failed, or reordered correctness evidence.

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

The adapter validates the request, source digest, and exact
`execution_environment` before remote work. The contract must name the
configured target, logical `py311-torch` runtime, and the target's activation
mode, with no additional fields. It uses only the installed global
remote-access client to run capabilities and preflight checks, upload the
deterministic tar bundle, dispatch a file-backed script under the selected
activation mode, and observe the retained handle.

The remote driver takes a nonblocking `flock` lease before probing each
physical device and holds it for the entire retained job. It uses the shared
`$HOME/.cpl-device-leases/device-lease-N.lock` namespace also used by
acquisition jobs, so workflows with different remote roots cannot claim the
same device. The probe also requires the configured minimum free memory (32
GiB by default for the frozen BSA cases), reporting compact `free=N
required=M` diagnostics for rejected devices. Process exit releases the
lease, so stale lock files are harmless. The bounded scan selects
an eligible device without freezing a physical ID in source. The selected ID
is exposed only as provenance; the application sees logical device 0.

Correctness runs through the frozen benchmark runner before every profile.
The runtime audit requires exactly one logical Triton launch to produce the
output and rejects framework compute or multiple Triton launches. Compiler
AIC/AIV components of that logical kernel are permitted. Candidate
compilation, execution, and comparison failures remain `compile_error`,
`runtime_error`, or `correctness_error`, including failures raised inside the
profiled repeat. TsdOpen, LazySetDevice, device OOM/unavailability, transport,
profiler, selector-evidence, and malformed-result failures are
`infrastructure_error`.

Discovery performs an unfiltered BasicInfo capture after correctness. It
returns explicit component descriptors for observed names matching the
declared Triton entrypoint, including compiler AIC/AIV suffixes, plus compact
BasicInfo rows. The coordinator requests each component separately. Capture
reruns the single requested case, supplies the exact component selector to
msprof with `PipeUtilization`, and verifies selector-binding BasicInfo.
Each `(block_id, sub_block_id)` identity must occur exactly once across all
matching PipeUtilization CSVs. Duplicate identities fail as infrastructure
instead of synthesizing a median from ambiguous reports.

Each logical stability sample is a separate msprof process and report. The
adapter never presents multiple launches from one report as independent
samples. To avoid one remote dispatch per sample, one retained job collects a
bounded batch of three independent captures for the same case and exact
component. Later coordinator requests for those sample indexes observe the
same terminal handle and consume the corresponding compact capture. Sample
indexes outside that batch create another content-addressed retained job. For
a multi-component logical kernel the coordinator takes the maximum component
duration for the sample and retains every component's pipe rows and
provenance.

Every terminal response records product, target, logical runtime, activation mode, selected
physical device, normalized `cpl-remote run` command, durable handle, and an
explicit terminal flag. The evidence digest covers the complete compact
remote result.

## Interruption and recovery

Dispatch state is keyed by product, target, logical runtime, activation mode,
campaign revision, qualification receipt,
canonical request,
and frozen baseline, case data, runner, remote-driver, and executor hashes.
The durable handle is written before observation and its target is validated
before reuse. An observer interruption returns `terminal=false` with the
original handle; `retry:{mode:observe,handle}` observes that attempt and never
dispatches a duplicate. A terminal infrastructure response is recorded under
`attempt-000N`; only
`retry:{mode:new_attempt,prior_handle}` can retire it and dispatch the next
numbered attempt. Candidate failures are terminal and are not retried as
infrastructure.

Dispatch preparation is serialized by a local file lock. Each remote attempt
also has a deterministic execution root protected by `flock` and a durable
compact-result file. If the controller stops after submission but before
writing `dispatch.json`, the next invocation can safely reissue that attempt:
the replacement joins the same remote lock and consumes the first job's
result instead of repeating device work. Pre-dispatch failures reuse the same
attempt rather than permanently poisoning its identity.

Do not delete the executor state while a handle may still be running. A changed
candidate, case, selector, action, or harness revision produces a different
execution identity and therefore a separate retained job.
