# A3 memory characterization

`benchmarks/a3_profile_memory.py --manifest` prints the 22 Float32 controls.
Each kernel copies exactly N indexed values to disjoint outputs. Strided
controls use compile-time address arithmetic. Irregular controls load an
int32 index array; contiguous indexed access is their baseline. The fixed
seed is 20261015. Global/local patterns are permutations, while hotset64
repeats addresses. CPU-generated reference values are checked exactly.

Run the small pilot inside a staged source directory:

```bash
python scripts/a3_memory_batch.py --pilot --output /absolute/new/pilot-folder
```

For concurrency, launch independent `--case ID` jobs through `$remote-access`
with the profile's named runtime and registry-derived cwd. Each job waits for
an idle physical device using `profile_device.py`, retaining its shared lease
through correctness, discovery, five BasicInfo captures and three separate
PipeUtilization replays. Do not select fixed device numbers or dispatch one
remote job per sample. Reobserve the same retained handle after transport loss.
Run all cases only after the three pilot cases pass.

All five durations enter the median. `(max-min)/median > 0.05` marks timing
low-confidence, never discards results. Discovery resolves the actual exported
name; subsequent captures require exactly that selector and one kernel row.
Zero process exit without terminal success or usable exports is failure.
Each invocation retains argv, stdout/stderr, exit code, vendor reports and
compact CSV rows. Failed cases remain explicit in summary.json; exit status
is nonzero if any case fails. No implicit infrastructure retry occurs in this
runner: callers may retry a terminal infrastructure failure in a new folder.

Footprints distinguish input allocation, accessed address span, distinct input
bytes, logical input bytes, output bytes and separate index bytes. None is
measured GM traffic or a bandwidth-saturation claim. Repeated accesses to 64
addresses do not count as N distinct inputs. Pipe rows are separate replays,
not additive latency components. Compiler text artifacts are retained per
invocation for later lowering inspection; presence alone proves no causal claim.

The reusable Python interface is `run_case(command, folder, leased_env,
selector=None)`. Native commands must provide their exact selector. All commands
must exit nonzero on numerical failure and emit a JSON object with
`correctness: true` after verification. The caller owns the lease and environment. This is
not a scheduler or transport client; staging and observation use remote-access.
