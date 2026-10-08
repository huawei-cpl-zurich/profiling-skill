# Two-pass live profile acquisition

Use this workflow for a self-contained Python workload or executable script
whose success exit proves compilation, runtime, and correctness. The helper
supports `a3` on `bz-a3-1` or `bz-a3-2`, and `a5` on `bz-a5`. It selects the
target's named runtime and a bounded functional device probe; the workload
always addresses logical device 0 through the environment.

## 1. Enumerate exported kernels

```bash
python scripts/acquire_profile.py basic \
  --product a3 --target bz-a3-1 \
  --dispatch-key '<unique-basic-key>' \
  --workload workload.py \
  --workload-arg=--case --workload-arg=47 \
  --evidence basic-info.json
```

The helper runs the supplied workload under the deployed
`--aic-metrics=BasicInfo` capture and returns all exact exported kernel names.
Inspect `capture.exported_kernel_names` together with the workload's intended
operation. Framework setup, layout conversion, and the target kernel may all
appear. The helper intentionally does not infer which one is relevant.

## 2. Select and replay one exact kernel

Choose one complete exported name without globbing, shortening, or rewriting
it, then run:

```bash
python scripts/acquire_profile.py pipe \
  --product a3 --target bz-a3-1 \
  --dispatch-key '<unique-pipe-key>' \
  --workload workload.py \
  --workload-arg=--case --workload-arg=47 \
  --basic-evidence basic-info.json \
  --kernel-name '<exact exported name>' \
  --evidence pipe-utilization.json
```

The pipe pass fails before dispatch if the workload bytes or arguments differ
from the BasicInfo pass, or if the selector is not an exact exported name. It
profiles with `--aic-metrics=PipeUtilization` and filters the compact result to
that selector. The helper never passes `--metrics` or an `msprof --device`
flag; physical selection is expressed only through the workload environment.

Both passes retain full report trees on the target and return exact compact
JSON bytes plus their SHA-256. Record the two durable handles. If observation
is interrupted, rerun observation of the printed handle rather than starting
another acquisition. Compilation, runtime, correctness, profiler, missing-row,
and evidence failures are counted outcomes; only proven transport or host
failures are discardable infrastructure.

`PipeUtilization` is activity evidence. Both A3 and A5 outputs therefore mark
saturation `unknown`; a saturated or unsaturated claim requires a separately
reviewed product-valid capacity denominator.
