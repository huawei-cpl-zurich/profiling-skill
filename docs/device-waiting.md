# Idle devices for retained profiling

Acquisition payloads discover chips through `npu-smi info -m`; card counts are
not physical device counts. Before allocating an NPU probe, the native status
table must report healthy hardware, zero AICore activity, no process on that
card/chip and at least 4096 MB free HBM. Missing inventory fails explicitly.

Workers share home-directory physical-device `flock` leases and recheck status
after taking the lock. Keep the returned lease open until profiling finishes.
Jobs wait indefinitely in 15-second intervals when all devices are occupied;
waiting does not consume measurement attempts. Observation loss must resume the
same remote handle, not launch another worker. Locks release automatically when
the owning process exits, allowing restart without stale lock removal. If idle
devices are actually probed but all probes fail, fail explicitly with bounded
diagnostics instead of misreporting a broken runtime as device occupancy.

`scripts/profile_device.py` provides `wait_for_device(probe)` returning
`(physical_id, environment, open_lease)`. The environment exposes only the chosen
physical chip and selects logical device zero. The supplied bounded probe takes
that environment and returns a boolean. Acquisition embeds this dependency in
the retained payload; callers need no machine-specific source paths.

BasicInfo evidence retains vendor CSV headers in `capture.rows`. The optional
`normal_run.workload_result` is the last dictionary-valued JSON line within the
last 64 KiB of workload stdout, or null; raw stdout is not copied into evidence.
Full workload and profiler transcripts remain in the retained remote directory.
