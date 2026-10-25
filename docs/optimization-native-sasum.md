# Core-local native SASUM aggregation

The source-intended ASC version passed all twelve exact numerical gates on the
same GZ A3 target. Performance qualification remains incomplete: the first
BEFORE capture returned ten positive event samples and exported its trace, but
strict CPU/NPU correlation was unavailable. No qualified capture, paired
comparison, or speedup is reported. Public numerical workflows remain disabled.

The baseline retains the pinned ops-blas tile Abs/ReduceSum, eight-block aligned
eight partition, result zeroing, SyncAll, and per-tile atomic contribution. The
candidate accumulates the identical tile sums per core and emits one atomic per
nonempty core, preserving the source synchronization and atomic-mode operations.
Its extra local scalar buffer adds32bytes: declared UB is165216/165248bytes.
The tile is20480elements; the tiling ABI retains forty-entry arrays and332bytes.
Compiler-managed scratch and peak usage remain unknown.

The prepared interface reuses private aligned output/workspace, immutable input
backing, and tiling. Source clearing/synchronization remain charged in every API.
Allocation, input transfer, host tiling setup, and final host synchronization are
excluded. This is a source-derived prepared experiment, not unmodified vendor
BLAS host latency. Source/native C++, derivative, provider, cases, formulas, and
tolerances are unchanged across the build repair.

The original legacy CCE/O2 attempt remains byte-preserved in
`P04-legacy-result.json`, `P04-diagnostics.json.gz`, and `P04-v1.json`. Its first
BEFORE output transfer failed ACL507015 before CPU values or comparison; native
enqueue/completion and cause remain unresolved. That attempt had no profiles.

The prospectively frozen v2 repair uses actual installed FindASC/ascendc_library,
ASC language, Release `-O3`, `--npu-arch=dav-2201`, documented explicit auto-sync
with diagnostic logging and `-g`, and driver-scope saved temporaries. Both actual
Cube and Vec objects, host ABI exports, actual verbose compiler commands, SDK
module hashes, and insertion logs are retained. Cube text is empty in these ASC
images; no execution engine is inferred from that fact. Mode, optimization, and
SDK defaults changed together, so ASC numerical success does not isolate the old
fault's cause or prove an auto-sync-only explanation.

All twelve new input-only physical ABI/code/resource probes and a canonical
whole-program identity consumer test preceded loading the original six cached
CPU payloads. Payload and manifest hashes match exactly; zero new expected
payloads were generated. All twelve numerical calls passed once before profiling,
with exact nonzero FP32 bits, exact numeric zeros, unchanged input/stride holes,
intact canaries, and nonaliasing. All six paired bit comparisons also matched.
The guard's zero sign is secondary. Four main shapes and two guards are unchanged.

The first BEFORE capture attempted ten APIs and returned ten positive instrumented
event samples, mean0.273462ms. Its export contains ten generic `MIX_AIV` compute
rows with sentinel connection IDs, twenty timing-event rows, and profiler control
records. The compute rows lack HostToDevice finish flows and an exact sasum symbol
link. Their iteration ownership, kernel identity, operation spans, and PMU
attribution remain unqualified. Neither timestamps nor row order substitute for
explicit linkage. The collector stopped; AFTER and the remaining seventy profile
calls were unattempted. All twelve prepared handles were closed, and fresh release
checks confirmed no owned live processes, free die/both export locks, and a healthy
idle mapped device. No reset, replay, or further device work followed.

Official Torch-NPU v2.10 parser evidence requires both explicit flow endpoints;
no certified alternate join or installed9.1 zero-extra-execution collector recipe
was found (query `a7f05dcaee804b84a09c0e4cf5fbf0a2`, miss
`57e0325d11f8c24eda81242cf16a33bb`). The retained trace and event samples remain
unmodified. No physical traffic, occupancy, instruction-count, roofline, ordinary
latency, or performance speedup claim is made.

The default factory, make_case, prepare, validate, capture, default probe, and CLI
reject before cache access, expected generation, backend import, or input movement.
Explicit `Adapter(compile_inspection=True).compile_probe(...)` supports ASC
build/metadata inspection only; `run()` always rejects. The internal measured
lifecycle is retained for controlled CPU mocks. Eight NONFROZEN CPU-only tests pass
on A3, exercising quarantine, actual ASC metadata inspection with empty Cube text,
precision/configuration rejection, canonical identity mutation, transactional build
retry, zero-launch probes, domain/holes/dtype, CLI rejection, and real mocked
dispatcher/device-restoration/idempotent cleanup through run_pair. This gate made
zero real Ascend compiler, backend, NPU, frozen-fixture, or profiling calls.

`experiments/optimization-wave010/P04-result.json` records the current separate
numerical/performance statuses, all twelve diagnostics, original cache bindings,
382 verified archive files, source/build/publication hashes, and release proof.
`P04-v2-diagnostics.json.gz` retains sanitized raw trace/PMU data, logs, measured
host/controller/tests, and primary correlation evidence. These additions preserve
the earlier failure and distinguish legacy, measured ASC, and public quarantine
host versions. Numerical dispatch coverage passing is separate from unavailable
profile qualification.
