# Audited BZ-A3 controller adapter

`scripts/audited_bz_controller.py` connects the audited experiment lifecycle to
the native BZ-A3 benchmark backend. It implements the argv expected by
`CommandController`, while leaving transfer, retained-job observation, and
remote `msprof op` analysis in the existing approved BZ job client.

## Configuration

The immutable configuration is JSON with schema
`profiling-skill/audited-bz-controller-config/v1`:

```json
{
  "schema": "profiling-skill/audited-bz-controller-config/v1",
  "benchmark": "matmul",
  "round_count": 4,
  "request_budget": 24,
  "profile_repeats": 3,
  "variability_threshold": 0.25,
  "control_drift_threshold": 0.2,
  "infrastructure_retry_budget": 3,
  "baseline": {
    "schema": "profiling-skill/baseline-timing/v1",
    "benchmark": "matmul",
    "case_medians_us": [
      {"case": 7, "median_us": 30.0},
      {"case": 8, "median_us": 33.0},
      {"case": 9, "median_us": 36.0}
    ],
    "control_median_us": 20.0,
    "sha256": "<canonical-document-sha256>"
  },
  "devices": [{"id": "target/device-evidence-id", "device": 0}],
  "development_cases": [7, 8, 9],
  "all_cases": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "backend_command": [
    "python",
    "/pinned/repository/scripts/benchmark_backend.py",
    "--benchmark",
    "matmul",
    "--job-client-json",
    "[\"python\",\"/pinned/repository/scripts/bz_a3_job_client.py\",\"...\"]"
  ]
}
```

The scheduler writes the eligible device entries after admission discovery.
Their identifiers are evidence, not constants in the agent prompt. The backend
command must ultimately use the approved remote-access-owned client; the
controller has no SSH, Docker, transfer, or device-discovery fallback.

The scheduler also supplies one frozen baseline timing document per benchmark.
Its cases must exactly match `development_cases`, and `sha256` is the canonical
JSON SHA-256 of the other four baseline fields. `control_median_us` is the
known-good calibration median retained with that baseline, not candidate
timing. A mismatched hash, benchmark, case sequence, or nonpositive value makes
the controller configuration invalid before any device work.

Run it from the isolated experiment repository so `candidate.py` and
`candidate.manifest.json` resolve there:

```sh
python /pinned/repository/scripts/audited_bz_controller.py \
  --config /campaign/controller.json \
  --state-dir /campaign/state/agent-1/controller \
  --experiment 1 \
  --candidate-sha256 "$CANDIDATE_SHA256" \
  --manifest-sha256 "$MANIFEST_SHA256"
```

Configure the outer `CommandController` timeout above the backend timeout so
the job client has time to persist a retained handle before the outer process
is interrupted.

`CommandController` treats `--state-dir` as its sole sanctioned mutable
directory argument. The directory must already exist and use an absolute path.
Reproducibility metadata binds its resolved path, filesystem device and inode,
owner, and mode; other slash-bearing controller arguments remain immutable
regular files with content hashes.

## Policy and recovery

Each round runs a warmed known-good admission control, a development correctness
check (a full check on the final round), one batched profile request with three
complete repetitions per development case, and a post-run known-good control.
Per-repetition geometric means become the receipt's three `samples_us`; compact
per-case rows and the remote evidence-file locator are retained alongside them.
Raw profiler trees stay remote.

Both known-good controls are three-sample `msprof op` captures. Successful
receipts retain their samples and medians under `calibration.before` and
`calibration.after`. The geometric mean of those two medians is the local
reference. Candidate samples are normalized by:

```text
frozen baseline control median / local bracketing-control median
```

The receipt therefore contains raw candidate samples, `baseline_median_us`,
the hash-bound baseline inputs, `normalized_samples_us`,
`normalized_median_us`, and `speedup_vs_baseline`. The shared receipt validator
recomputes every aggregate and rejects tampering. Candidate-error receipts
contain no baseline, calibration, candidate timing, normalization, or speedup
fields.

Compilation, runtime, submission, and correctness failures produce a terminal
`candidate_error`. Transport, observer, device, staging, malformed evidence,
and control failures produce nonterminal `infrastructure_error` receipts.
Successful and candidate-terminal backend operations consume the branch-wide
24-operation budget. Infrastructure attempts are recorded separately and do
not consume it.

If an infrastructure receipt contains a handle, repeat the invocation with
the same frozen hashes and:

```sh
--observe-handle bz-a3-1:retained-job
```

The adapter rejects a different handle and replays the byte-identical backend
request. The content-addressed BZ job client then observes its persisted job
instead of dispatching another workload. A handleless interruption is retried
with the same request and remains bounded by the infrastructure retry policy.

Noisy timing receives exactly one confirmation capture. Persistent noise or a
drifting post-control returns `measurement_pending`. A lifecycle remeasurement
uses:

```sh
--remeasure-handle bz-a3-1:profile-job
```

This reruns only profiling and the post-control, retaining the already accepted
candidate and correctness proof. All controller state, including the shared
operation ledger, is atomically persisted below `--state-dir`.
`CommandController.remeasure` supplies this argument, binds the returned receipt
to the exact pending handle, and applies the same size and redaction rules as a
normal submission or observation.
