# Expansion gallery evidence

This companion retains 24 public gallery pairs for E01–E12.
E10 uses only the corrected v3 collection, replacing the earlier capture.
The twelve original recipe files and their 33-pair evidence remain unchanged.

[JSON sample ledger](expansion-gallery-evidence.json) retains every sample,
mean, sample deviation and the relative retained summary/compact-receipt name.
These receipt names identify evidence; they are not installed source paths.
Full source/profiler receipts remain in the source streams. No private evaluation
cases, machine routing or raw machine paths are included here.

Each span covers the complete profiled device operation, from the first proven
task start to the last task end, including gaps and preparation/reduction kernels
required by that arm. All ten samples count, including the first. It is neither
unprofiled latency nor summed busy time. The initial 22-pair filtered input supplies no
caller-event or PMU samples.
The [E10 v3 receipt](crossentropy-gallery-v3.json) additionally retains active
union, internal gaps and matched task-local PMU samples with original units.

Labels use the original ledger's conservative t(9)=2.262 mean-difference rule:
abs(mean_before-mean_after) must exceed 2.262*sqrt((sd_before²+sd_after²)/10).
This assumes sample independence and describes one capture per arm; it does not
prove between-capture reproducibility or a dispatch threshold. Hypothesis names
record planned regimes. A smaller mean within uncertainty is inconclusive.

Workspace counts are adapter-explicit logical allocations. They exclude compiler
scratch and do not measure physical traffic or peak memory. The initial 22 pairs preallocate
outputs/workspace. E10 allocates inside its caller, but its device span excludes
pre-first-task allocator work. No allocator-latency advantage is established. Activity
ratios cannot yield physical-byte savings or roofline distance.
