# A3 simulator instruction timelines

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
