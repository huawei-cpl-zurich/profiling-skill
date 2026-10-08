# Two-pass live profile acquisition

Use this workflow for a Torch/Triton workload whose success exit proves its
compilation, runtime, and correctness. It supports A3 on `bz-a3-1` or
`bz-a3-2`, and A5 on `bz-a5`. The helper selects the named runtime and a
bounded functional device probe; the workload addresses logical device 0.

Prefer a bundle so sibling imports, data, generated modules, and the selected
entrypoint are all content-addressed:

```bash
python scripts/acquire_profile.py basic \
  --product a3 --target bz-a3-1 \
  --dispatch-key '<unique-basic-key>' \
  --bundle workload-bundle --entrypoint run_case.py \
  --workload-arg=--case --workload-arg=47 \
  --evidence basic-info.json
```

`--workload workload.py` is shorthand for a one-file bundle. Symlinks are
rejected. The deterministic manifest binds every regular file's relative
path, mode, size, and SHA-256, plus the entrypoint and arguments. The pipe pass
fails locally if any bundled file, argument, or entrypoint changes.

## Select and replay one exact name

The BasicInfo pass returns `capture.observed_kernel_names`. Inspect those
names alongside the intended operation, then select one complete name without
globbing, shortening, or rewriting it:

```bash
python scripts/acquire_profile.py pipe \
  --product a3 --target bz-a3-1 \
  --dispatch-key '<unique-pipe-key>' \
  --bundle workload-bundle --entrypoint run_case.py \
  --workload-arg=--case --workload-arg=47 \
  --basic-evidence basic-info.json \
  --kernel-name '<exact observed name>' \
  --evidence pipe-utilization.json
```

The helper intentionally does not infer which operator matters. It uses the
deployed `--aic-metrics` syntax, never `--metrics` or an `msprof --device`
flag, and binds unnamed pipe rows through the adjacent exact-selector
BasicInfo export.

## Bounded discovery

Both deployed profilers support `--launch-count` from 1 through 5000 and
`--kill=off`. The helper defaults to 5000 so kernels launched after an early
setup sequence are observable. Override it with `--launch-count N` only for a
known smaller workload.

The profiler does not export the application's total launch count. Evidence
therefore reports the bound and `complete: false`; the names are exactly those
observed within the bounded capture, never a claim that every application
kernel was enumerated. If the expected kernel is absent, confirm workload
success and use a larger bound up to 5000. If it remains absent at 5000, report
the bounded miss rather than inventing a selector.

## Durable dispatch and resume

Immediately after dispatch, the helper prints and fsyncs a JSON receipt at
`EVIDENCE.dispatch.json` by default. It binds the validated handle to the
product, target, mode, runtime, dispatch key, complete bundle identity,
selector, and launch bound. Preserve both the printed handle and receipt.

After controller interruption, repeat the identical command with:

```bash
--resume-handle 'remote:TARGET:job:ID'
```

Resume validates all original acquisition metadata against the receipt and
only observes that handle. It never submits another job. Result and log
retrieval also retry the same handle after transport interruption. An existing
receipt blocks an ordinary fresh dispatch.

The full profiler report remains remote. The successful local evidence is the
exact compact JSON bytes printed remotely and verified by both the acquisition
and broker `REMOTE_CONTENT_SHA256` markers. Evidence includes its nonempty
schema name, exact product/target provenance, and the successful normal
workload exit plus compact stdout/stderr byte counts and digests. The workload
success remains the correctness authority; profiler success alone is not.

The helper uses the approved `cpl-remote` executable on `PATH` when an isolated
launcher brokers it there. On ordinary hosts it falls back to the user-wide
remote-access skill client. It fails closed if neither approved client exists;
agent prompts do not need the hidden client-path override.

The helper sends one action-first command and accepts either the user-wide
client's canonical JSON receipt or the broker's stable `REMOTE_*` text
receipt. It does not probe alternate dispatch forms after `run`. Text log
content is preserved verbatim from `REMOTE_CONTENT=` through end of output so
multiline evidence markers and base64 remain intact.

An isolated retained broker sets `CPL_REMOTE_MODE=retained-broker`; in that
explicit mode the helper binds the user-supplied dispatch key to the one
`cpl-remote run`. Ordinary user-wide clients receive their normal argv without
the broker-only option. The generated remote payload is fsynced beside the
requested evidence output, passed from that workspace-approved path, and
removed after the dispatch attempt. Run this installed helper in place—never
copy or patch it inside an experiment workspace.

## Failure receipts

Failures preserve the durable handle, remote phase, classification, and a
bounded excerpt:

- `device_unavailable` and `host_environment` identify device or host setup;
- `transport_error` identifies controller/observer transport and is resumed;
- `workload_failure` covers compilation, launch, runtime, correctness, and
  workload timeout detected by the normal pre-profile run;
- `profiler_failure` covers profiler timeout, exit, or missing success marker;
- `evidence_failure` covers selector binding, missing rows, malformed compact
  output, and digest/schema failures.

Do not discard workload, profiler, or evidence failures as infrastructure.
Only retained evidence may justify classifying host or transport failure as a
discardable attempt.

`PipeUtilization` is activity evidence. Both A3 and A5 outputs mark saturation
`unknown`; a saturated or unsaturated claim requires a separately reviewed
product-valid capacity denominator.
