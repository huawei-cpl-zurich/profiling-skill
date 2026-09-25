# A2/A3 `msprof op` evidence

Use `scripts/profile_a3.py` inside the managed `$gz-a3` native Python 3.11
runtime. The profile wrapper selects the physical device; the workload and
profiler see logical device 0. Do not pass a host physical ID to Triton code.

The helper runs one bounded `BasicInfo` capture with kernel replay and writes:

- `raw/`: retained vendor output;
- `msprof.log`: the complete profiler transcript;
- `evidence.json`: compact, deterministic latency evidence also printed as one
  JSON line.

Example inside the managed job:

```bash
python scripts/profile_a3.py \
  --output artifacts/profile/round-1-case-47 \
  --kernel-name '<exact exported kernel>' \
  --warm-up 3 --launch-count 1 -- \
  python benchmark_case.py --case 47
```

For performance measurements, `--kernel-name` is required: an unfiltered
capture may select the first framework setup operator rather than the Triton
kernel. An unfiltered capture is suitable only for discovering exported names.

The workload must execute one deterministic case, validate its result, and
return nonzero on compilation, runtime, or correctness failure. Keep input
construction and host-to-device transfer outside the profiled launch whenever
the benchmark interface permits it. Use separate captures for repetitions;
the experiment controller should take the median of successful captures.

## Evidence contract

Successful schema version 1 evidence has `status: success`, target family
`Ascend-A2-A3`, metric `BasicInfo`, and one `kernels` entry per matching
exported kernel. Each entry contains the raw duration samples and min, median,
p90, and max in microseconds. `sources` records each contributing CSV hash.

A failed command still writes JSON with `status: failure`, the msprof return
code, and a concise reason; the full compiler/runtime/profiler transcript stays
in `msprof.log`. Absence of the profiler terminal-success marker, absence of a
matching numeric row, timeout, and nonzero msprof exit are failures. Do not
turn them into timing samples.

`BasicInfo` task duration is device-side task latency, not end-to-end Python
latency. Multiple exported kernels are reported separately unless an exact
selector is supplied. Do not sum them unless the benchmark specification
defines that aggregate. Do not apply the A5 PMU event IDs, bandwidth
multipliers, core ceilings, or utilization formulas to A2/A3 captures.
