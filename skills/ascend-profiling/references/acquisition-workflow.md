# Two-pass msprof acquisition

Run commands from the skill folder with absolute workload/output paths. Assume
`msprof` and the workload runtime are configured as stated in `SKILL.md`.
Select the actual product and record tool versions; supported metric names and
options depend on the installed profiler. Check `msprof op --help` first.

## Discover, then select

1. Run the workload without profiling and check correctness.
2. Capture `BasicInfo` without a kernel filter. Inspect all exported `OpBasicInfo`
   rows to discover the intended kernel's exact name; a source function name is
   not necessarily its exported name.
3. Capture `BasicInfo` again with that exact selector for timing. Use separate
   captures for repetitions and retain every declared sample.
4. Capture `PipeUtilization` with the same exact selector, workload bytes,
   arguments, inputs, device, replay mode, warmup, and launch count. Use a fresh
   output directory for each capture. Diagnostic durations are not timing samples.

For A2/A3, the included helper discovers names when `--kernel-name` is omitted:

```bash
python3 scripts/profile_a3.py --output /absolute/evidence/discovery \
  --warm-up 0 --launch-count 5000 -- python3 /absolute/workload.py
python3 scripts/profile_a3.py --output /absolute/evidence/timing \
  --kernel-name '<exact exported kernel>' --warm-up 3 --launch-count 1 -- \
  python3 /absolute/workload.py
```

The discovery launch window is deliberately wider than selected-kernel captures:
unfiltered launch count 1 may capture only framework setup. Choose a bounded
window supported by the installed profiler and long enough to include the target.
If it is absent, inspect the log and workload launch order; do not invent a name
or treat discovery as a timing sample. Keep timing and pipe parameters matched
after selection; discovery itself may use different warmup and launch counts.

For direct captures, this is the usual `msprof op` form; verify spelling and
availability against the installed help before running it:

```bash
msprof op --output=/absolute/evidence/discovery --aic-metrics=BasicInfo \
  --warm-up=0 --launch-count=5000 \
  --application='python3 /absolute/workload.py'
msprof op --output=/absolute/evidence/pipe --aic-metrics=PipeUtilization \
  --kernel-name='<exact exported kernel>' --warm-up=3 --launch-count=1 \
  --application='python3 /absolute/workload.py'
```

If selecting a device explicitly, use the same device addressing convention
in profiler and application. A visibility mask can change application device
indices; do not mix physical and remapped indices. Preserve that mapping.
`profile_a3.py` preserves `ASCEND_RT_VISIBLE_DEVICES` when set and defaults it
to `0` otherwise; configure this explicitly before using another device.

For A2/A3 normalization, each pipe or memory replay must export one total
`OpBasicInfo` row matching the selector and use launch count 1. Read
[A2/A3 evidence](a2-a3-msprof-op.md). For A5 timeline groups and interpretation,
read [A5 timeline semantics](a5-timeline-semantics.md); use one supported
instruction pipe per `InstrTimeline` replay.

## Retain and check

Keep the full profiler transcript and raw report, command, exit status, selected
kernel, source/binary hashes, versions, and capture parameters. A workload
failure, missing terminal-success marker, missing selected-kernel rows, or
incomplete report is not a valid timing sample. Preserve failures separately.
Record workload correctness independently; replay success cannot establish it.
Use the included file-based analysis helpers to produce compact evidence from
retained exports. Infrastructure scheduling, transfer, and recovery are outside
this skill.
