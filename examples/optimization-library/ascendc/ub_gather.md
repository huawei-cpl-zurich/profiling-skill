# Scalar UB loads versus vector Gather

This independent native AIV pair computes `output[i] = table[indices[i]]`.
Both arms stage the complete table once per API and process 4096-index tiles.
Both compute int32 byte offsets with `Muls(index, sizeof(float))`; the scalar
arm dereferences a byte-addressed UB pointer and the candidate uses
`Gather(result, table, uint32Offsets, 0, count)`. Both retain the same
`PIPE_ALL` transitions, UB buffers, grid, GM transfers and compiler flags.
The table occupies at most 16 KiB and the two tile buffers 16 KiB each.
Compilation and the selected target's resource audit must precede execution.

The source cue is Ops-sparse revision
`bfc7c19d83ce246b03c65dc916ed4753e5467cf3`, arch22/DAV2201:
`spmv_kernel.h` GatherXWindow lines 1092–1134 and scalar UB reads lines
173–200/1169–1180. The source license is CANNOpenSoftware2.0. Its enclosing
SpMV algorithms differ and are not compared here. This is an independent
standalone fetch adaptation, not an upstream optimized SpMV implementation.
The original source uses integer multiplication, uint32 byte offsets and
retains full barriers around Gather. No new UB residency, sparse traversal,
or bank-conflict mechanism is claimed.

`N03.json` freezes four main cases and two numerical guards. The table and
indices are immutable contiguous CPU FP32/int32 inputs to preparation, with
finite table values in [-4,4], K<=4096, valid indices in [0,K), and ordered
repeats. Invalid inputs reject before device preparation; explicit indices in
the case interface also reject before fixture-cache construction. Outputs are
nonaliasing and compare exactly as int32 views, including +0/-0. Once-only
correctness checks input immutability and prefix/suffix GM canaries. The
4097-element numerical guard exercises a second tile without profiling.
Unit-test seeds/shapes are separate from frozen scientific payloads.

Use `ub_gather:factory` with the public `optimization_pair.py` interface and
`UB_GATHER_DIRECTORY` pointing to the `compile_all` output. All resources
must build and pass selected SDK kind3/vector/no-Cube checks before any
forward. Main cases retain exactly ten profiled API forwards per arm, without
warmup or extra timing launches. Preserve compiler no-effect, equal, slower,
unsupported and failed outcomes. Profiler operation spans describe instrumented
diagnostics, not ordinary latency or physical traffic.
