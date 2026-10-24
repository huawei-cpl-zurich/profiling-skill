# Inverse adjacency with exclusive destination ownership

Collection is incomplete: both implementations passed the first skewed case's
correctness check, but the candidate's first capture failed with a structured
vector core timeout (507034). Its retained counter proves ten API invocations,
with zero event samples; it does not prove ten completed device forwards. The
remaining five cases, including both guards, were not executed. There was no
retry, device reset or paired speedup claim. See the
[retained result](data/optimization-w004-inverse-adjacency.json).

This experiment reduces FP32 edge vectors by int32 destination ID. The baseline
zeros its output and uses floating atomic additions. The candidate constructs
reverse adjacency from the current edge IDs, then gives each destination one
writer that gathers its edge vectors. Duplicate edges retain their multiplicity;
destinations without edges produce exact zero, including when output storage
starts with NaN.

The mechanism comes from the inverse selected-query lists and destination-owned
backward reduction in
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py:L455-L725`.
This project adaptation is a smaller reduction primitive, not full attention
backward. It replaces sorting with stable counting assembly: count each
destination's edges, prefix the counts, scan edge blocks in original order and
place current edge IDs by their cumulative rank, then gather and reduce.
The integer index is rebuilt every forward. No CPU inverse map or dense
per-worker gradient tensor is supplied.

Output and integer workspace are allocated during preparation. Counting,
prefixing, index construction and output reduction are all inside the measured
API. Every count, offset and index slot is written before use; no zeroing is
omitted from an algorithm that needs it. The baseline's output zeroing is also
inside its forward. Both arms use `num_stages=1`, `multibuffer=False` and disabled
FP fusion. The candidate preserves edge order within its inverse list, while
its block reductions can change FP32 summation order.

The frozen matrix is [W06.json](../examples/optimization-library/cases/wave004/W06.json).
It includes two skewed and two unique-destination cases, with duplicate/empty
and partial-edge-block guards. Seeds, sizes, input domain and FP64 oracle
tolerance are fixed before timing. Performance designations are hypotheses.

Use the common paired runner with the nested case JSON directly:

```bash
PYTHONPATH=examples/optimization-library/triton python scripts/optimization_pair.py \
  --adapter inverse_adjacency:Adapter \
  --case examples/optimization-library/cases/wave004/W004-W06-GA.json \
  --provenance provenance.json --cache oracle-cache --output guard-result
```

The guard role requests correctness only. The collection procedure first
compiles all cases and both arms without launching; it then checks each side
once and captures ten full forwards for each non-guard side, without warmup or
autotuning. CPU FP64 sums are compared with `atol=rtol=0.002`, and validation
also checks finite outputs, exact empty zeros and unchanged inputs. Profiled
device operation spans include stages and gaps; they are instrumented
diagnostics, not ordinary application latency or physical memory traffic.

All twelve arm specializations compiled before the first forward, and six CPU
oracle payloads were generated once. The baseline's first ten captures yielded
a mean device operation span of 362.042 us (sample SD 1.286 us), including
204.676 us active union and 157.366 us gaps. This is one-arm diagnostic evidence.
Candidate PMU and timing are unavailable. Five functional tests passed on the
A3 runtime, and eighteen shared runner regression tests passed locally.

The operational collector used an incorrect export semaphore path and lacked a
bounded capture watchdog. Its locks therefore did not guarantee coordination
with other streams. The structured timeout ended the job after approximately
550 seconds. These limitations are preserved; they do not establish the cause
of the vector timeout. Future unattempted work must use the shared canonical
export slots and a bounded watchdog. The failed source/case was not replayed.
Future operational wrappers must anchor `ENTRY` at the profile entry working
directory, use `ENTRY/optimization-study/exports/0.lock` or `1.lock` for the two
shared capture/export slots, and keep device leases under
`ENTRY/optimization-study/leases/die-N.lock`. The historical wrapper used
`leases/exportsN.lock`; it remains unchanged as part of the retained evidence.
Physical die 3 is excluded from further workloads pending an understood cause
of the timeout. This exclusion does not establish a hardware fault.
