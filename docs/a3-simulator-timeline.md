# A3 simulator instruction timelines

## Acquisition under a shared device lease

`scripts/a3_timeline_probe.py` is a one-capture remote worker, not a scheduler
or transport client. Stage it with `summarize_a3_simulator.py` and a numerically
checking executable application wrapper. A supervisor must first obtain an
idle-device lease, export `ASCEND_RT_VISIBLE_DEVICES`, and hold that lease until
the worker exits. Use the selected profile's named runtime through remote-access.

```bash
python scripts/a3_timeline_probe.py --mode simulation \
  --application /retained/experiment/workload.sh --output /retained/experiment/capture-1 \
  --kernel-name EXACT_EXPORTED_NAME --soc-version OBSERVED_SUPPORTED_SOC
```

Determine the actual SoC and installed simulator support on the selected target;
there is no default simulator version. Each invocation checks the unprofiled
application before capture, retains both subprocess transcripts and commands,
and writes `result.json`, including raw file inventory and compact normalized
simulation evidence. Output directories are new per attempt and cannot be
overwritten. A failed correctness check stops before profiling.

Process failure, timeout, a reported child signal, collection failure, or missing
profiler terminal success marker prevents the clean gate even when traces exist.
Partial traces remain inspectable, not successful acquisitions. Exact kernel
binding is also mandatory. Hardware mode runs `TimelineDetail`, but its output
must pass the existing hardware acquisition harness's normalization separately;
this worker cannot promote simulator traces to hardware evidence. Select one
kernel per simulation invocation for multi-launch controls, keeping their replays
separate rather than inventing a shared clock across kernel captures.

## Retained trace interpretation

`scripts/summarize_a3_simulator.py --report RETAINED_ROOT/report --kernel-name EXACT_NAME`
analyzes retained `OPPROF_*/simulator/trace.json` files **in place** and prints
compact JSON. Run through the global remote-access client and a named runtime.
No fresh profiling workload is dispatched by this reader.

The report directory must contain one simulator capture, and its sibling
`profiler.stdout.log` must bind it to exactly one `<ProfInit>` kernel name.
Otherwise raw acquisition is reported, but semantic interpretation fails closed
(exit 2). Default acquisition status is `unknown`; use `partial` for a capture
whose application failed or crashed during teardown. Use `clean` only with
retained successful terminal acquisition evidence; the parser does not validate
application correctness or independently establish capture success.
Pass `--acquisition-failure teardown_failed` for retained teardown failures;
this sets explicit partial-capture/failure fields and cannot pass the clean gate.

The adapter recognizes observed aggregate core/pipeline string identifiers and
per-core numeric lanes with Chrome `process_name`/`thread_name` metadata. It
prefers the aggregate trace over duplicate per-core views. If no aggregate
exists, it reads the per-core files instead. Interval unions and overlaps are
computed independently per vector/cube core over all events; only the returned
example event list is capped at 256. `displayTimeUnit` is a viewer hint: Chrome
trace `ts`/`dur` retain their microsecond convention.
This was also checked against the native vendor simulator table: the four core
instruction windows span 1.00424, 1.00788, 1.00788 and 1.00000 trace units;
rounding them to the vendor table precision gives its 1.00, 1.01, 1.01 and
1.00 microseconds. Treating the viewer's `ns` hint as timestamp units would be
off by 1000. This check establishes simulator units, not hardware timing.

BAR, wait/event, and flow-control instructions remain controls, even when their
operands mention MTE2 or MTE3. Their durations do not become data-movement
activity. Operand details are retained for inspection, never used to infer the
execution pipeline or source-code phase. Pipeline activity is not automatically
a load/compute/store phase; debug/source attribution and controlled validation
are still needed.

These are **simulated instruction timestamps**, not hardware latency, hardware
utilization, capacity saturation, or device TimelineDetail. Use them to develop
instruction/pipeline hypotheses; establish performance gains separately using
paired device measurements. The hardware acquisition helper excludes simulator
paths so they cannot silently satisfy its TimelineDetail gate.

The small fixture combines exact row excerpts from a clean native Add capture
and a partial Triton `a3_phase_probe` capture, both using Ascend910B4 simulation.
They intentionally exercise lane resolution and barrier semantics, not a valid
cross-capture phase sequence or performance comparison. No source-phase claims
are accepted by this adapter yet.
