# Core-local native SASUM aggregation

This is an unqualified failed study. All12 input-only compile/physical ABI cells
completed before six CPU references were cached once. The first baseline API
returned to validation, where its sole output transfer failed synchronization
with ACL507015. No CPU output values or numerical comparisons were obtained.
All remaining correctness APIs, profiles and captures were unattempted. The
cause and native device completion remain unknown; no speedup is reported.

This source-derived Ascend C comparison retains the selected ops-blas SASUM tile
Abs/ReduceSum, eight-block aligned-eight partition, output zeroing and SyncAll.
The baseline issues an atomic scalar contribution for every source tile. The
candidate accumulates those same tile sums locally and issues one atomic per
nonempty core. V_S and S_MTE3 events order its scalar transfers. The extra local
buffer occupies32bytes; declared UB is165216/165248bytes before/after.
Compiler-managed scratch and peak resource usage remain unknown.

The prepared API reuses private aligned output and workspace, immutable input
backing and tiling. Source zeroing and synchronization occur inside every call.
Host allocation, input transfer, source-host allocation/tiling transfer and its
final host synchronization are excluded. This is an extracted native attempt,
not a full ops-blas build or a vendor library benchmark. Full CANN OSLA2 and
unchanged selected source/host files are retained beside the adapter.

`experiments/optimization-wave010/P04.json` fixes all cases and exact dyadic
CPU FP64 reference rules prospectively. The strided guard retains actual stride
holes in physical backing. Nonzero outputs require FP32 bit identity; zeros
require numeric identity, with zero sign secondary. The complete failure, runtime, binary and cached-fixture bindings are in
`P04-result.json`; full sanitized logs and executed wrapper/controller are in
`P04-diagnostics.json.gz`. No source, fixture or math retuning/replay occurred.

The default factory, case generation, prepare, run/capture workflow and CLI
reject before scientific expected generation, compilation or input transfer.
Explicit `Adapter(compile_inspection=True).compile_probe(...)` retains native
build/metadata inspection only; its `run()` always rejects. Inspection binds
both actual MIX AIC and AIV objects through canonical binary identity lists.
The executed pre-math collector qualified the selected AIV object; this was
incomplete coverage of the generated MIX launch. Both device objects and their
actual registrations are now retained with that limit explicit. SDK9.1 used
`dav-c220` on actual Ascend910B4-1. The hardware SyncAll overload uses FFTS
without an explicit user workspace; omitted-workspace or wrong-engine causes
are unproved. Compiler-managed peak/scratch and target compatibility remain
unresolved. No new compiler or device calls were used for that analysis.

Six NONFROZEN CPU-only host tests cover default rejection, transactional compile
inspection and zero-launch prototypes, dyadic references/partition ownership,
strict domain/holes/dtype, external-cwd CLI, SDK registration ambiguity, and the
real evidence collector/normalizer accepting both device identities and rejecting
a changed component. This host gate is separate from failed native correctness. Diagnostic profiler event durations include instrumentation;
they are not ordinary end-to-end latency. Missing diagnostics are unavailable,
not zero. No physical traffic, occupancy or roofline claim is made.
