# Private natural-task qualification

This is a controller-only gate, not a prompt or requirement for scored agents.
The candidate skill stays frozen at `294f76f`. The historical negative pilot
remains a separate result. Do not give interventions or qualification evidence
to optimization agents.

Prepare each task from an assembled source tree containing the natural fixtures
and the existing first-party A3 acquisition helpers:

```sh
python scripts/qualify_natural_tasks.py prepare \
  --source-root ASSEMBLED_SOURCE --task task01 --output NEW_BUNDLE
```

Preparation exports the baseline and two private scoped interventions with
their trusted seeded runner. Transfer the self-contained bundle through the
current user-wide `remote-access` client. Submit the following payload through
that same client, using registry-derived cwd and the named Torch runtime:

```sh
python qualify_natural_tasks.py run --bundle BUNDLE --output BUNDLE/batch-1
```

Repeat as an independent retained acquisition job into `batch-2`. Preserve job
handles and observe the same job after transport loss. The payload dynamically
leases an eligible physical device; applications use logical device zero.
Run at most three qualification jobs concurrently.

Each variant must pass four seeded correctness checks before profiling. An
unfiltered BasicInfo discovery must expose exactly one compute launch and an
unambiguous exported selector. Then retain five BasicInfo timing samples and
three PipeUtilization captures for that exact selector. Failed compilation,
correctness, collection, missing reports, or extra compute launches stop that
batch with a nonzero exit; diagnostics and partial results remain available.
No failed samples are silently replaced.

Download the compact `batch-1.tar.gz` / `batch-2.tar.gz` artifacts, not raw vendor
binary capture trees. These retain commands, return codes, stdout/stderr,
CSV/JSON reports, source hashes, and available compiler text. Raw captures stay
in the retained remote job workspace.

Successful acquisition is not automatic task qualification. Compare both
batches' baseline/intervention timing samples, per-pipe metrics and compiler
artifacts against actual code changes. Require reproducible resource
distinctions and intervention responses that pipe evidence helps discriminate.
Report noisy or inconclusive cases honestly; do not add overlapping pipe ratios
or infer absolute saturation from a large ratio alone. If any of the six tasks
fails this gate, stop cohort inclusion and report it, rather than filling the
cohort with trivial controls.

Only after all six qualify, freeze the private diagnoses and launch the paired
20-minute ordinary optimization cohort. Agents may use timing, autotuning,
compiler inspection or profiling voluntarily; neither collection nor pipe
citations alone earns credit.
