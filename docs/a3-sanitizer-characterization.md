# A3 sanitizer qualification

This battery qualifies detection, not performance. Instrumented durations must
never be compared with ordinary kernel timing. Numerical success is also not a
race detector: the overlapping-write controls deliberately write equal values.

## Reproduce

Select the named runtime through the global remote-access skill, upload the
self-contained sources, and run on an idle-device lease:

```bash
python scripts/a3_sanitizer_batch.py --language native --canary --output native-canary
python scripts/a3_sanitizer_batch.py --language triton --canary --output triton-canary
# Only after tool acquisition works:
python scripts/a3_sanitizer_batch.py --language native --output native-qualification
python scripts/a3_sanitizer_batch.py --language triton --output triton-qualification
```

Each language holds its device lease for the full job. The runner performs
memcheck before racecheck and labels synccheck separately. Qualification uses
three repeats per control and tool; canaries are not qualification repeats.
Each subprocess has a 300-second process-group watchdog. Missing tools, malformed
reports, timeouts and instrumentation failures stop the batch rather than
consuming the remaining repeats.

Native compilation uses the shared retained builder with `-O2 -g --sanitizer`.
The [Ascend C compiler documentation](https://asc.gitcode.com/guide/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.html)
requires debug information and disallows `-O0` for sanitizer compilation.
Triton gets separate caches and `TRITON_ENABLE_SANITIZER=1`,
`TRITON_ALWAYS_COMPILE=1`, and `PYTORCH_NO_NPU_MEMORY_CACHING=1`.
The portable invocation boundary is explicit:

```bash
mssanitizer --tool=racecheck --log-level=info --log-file=race.log -- \
  python benchmarks/a3_sanitizer_triton.py --artifacts=instrumented-artifacts
```

## Controls and interpretation

Native families are UB RAW (MTE2→V), UB WAR (MTE3→V), UB WAW
(MTE2/vector writes), inter-core overlapping versus disjoint GM writes, and
inter-core GM producer/consumer with versus without a mode-0 barrier. Repaired
controls contain explicit matching synchronization; mutants omit it without
leaving unmatched waits. Native outputs are checked numerically. The Triton
pair writes the same value from two programs into overlapping or disjoint GM
regions and retains the actual compiler artifacts.

The compact result retains exit status, complete raw reports and hashes,
diagnostic lines, commands, compiler evidence and device identity. Possible
outcomes distinguish detected hazards, no finding, other findings, unsupported
behavior, instrumentation failure, timeout and unqualified reports. Zero exit
status alone is not evidence of analysis or race freedom. Inspect exact kernel
start/finish markers and instrumentation before accepting any negative result.

No-finding mutants are not called false negatives unless compiler lowering
proves the intended hazard remains. The installed native device LLVM artifacts
are opaque and its disassembler does not decode these instructions, so native
negative coverage may remain unqualified even when source intent is clear.

## Acquisition findings so far

The observed tool is mssanitizer 26.1.0. Its help advertises bare
`--full-backtrace`, but that spelling is rejected as an invalid parameter and
prints usage while returning zero. The runner omits the optional flag and
treats usage as instrumentation failure, never a detected hazard.

The tool parses workload flags as its own options unless `--` separates its
options from the workload command. The initial Triton `--artifacts` option was
therefore rejected before Python started. The runner now inserts this separator.
Native repaired controls have run numerically correctly. Racecheck emitted
an exact-kernel start/finish with “No error detected.” Memcheck emitted a
register warning about `FFTS_BASE_ADDR` not being reset; this is an
`other-finding`, not a race or a clean memory result. Positive/negative
Triton memcheck also reports `VECTOR_MASK_0/1` reset warnings. Uninstrumented
Torch initialization kernels can report unsupported racecheck/synccheck while
the target Triton kernel is instrumented and completes analysis. Target results
therefore use exact start/finish sections, not whole-process keyword matching.

## Native results

Five native families completed three mutant and three repaired/disjoint
repeats per tool. Racecheck detected every mutant and reported no finding on
every repaired/disjoint control. Synccheck reported no finding for all of these
cases: it is not a substitute for racecheck. Memcheck emitted the register
warnings above in 27/30 checks. The GM-overlap mutant additionally produced an
out-of-bounds warning in 3/3 memcheck runs. Its allocation trace records a
65,568-byte allocation and a 16,384-byte write at that allocation's base, so
the warning does not establish an actual allocation-bound violation. Keep this
warning distinct from racecheck's well-attributed GM WAW diagnosis and audit
the tool's ownership/bounds assumptions before acting on it.

| Source change | Racecheck diagnosis | Mutant/repaired finding count |
| --- | --- | --- |
| Omit MTE2→V flag before in-place Adds | UB RAW and WAW; secondary MTE2→MTE3 RAW | 3/3 versus 0/3 |
| Omit MTE3→V flag before overwriting copied UB | UB WAR | 3/3 versus 0/3 |
| Omit MTE2→V flag before Duplicate into copied UB | UB WAW; secondary MTE2→MTE3 RAW | 3/3 versus 0/3 |
| Two AIVs write same GM range versus disjoint ranges | GM WAW between blocks 0 and 1, PIPE_MTE3 | 3/3 versus 0/3 |
| Omit cross-core barrier before reading peer GM data | GM RAW/WAR plus local UB WAW/WAR | 3/3 versus 0/3 |

For example, the first family identifies MTE2's write at PC `0x3f6c`
(`a3_sanitizer_control.asc:35`) and vector read/write at PC `0x4714`
(`:40`). The WAR family identifies MTE3 read PC `0x1770` (`:45`) and
vector write PC `0x1ec0` (`:50`). Inter-core WAW attributes both writes to
PC `0x6390` (`:65`) in different AIV blocks at the same GM address. These
PCs belong to this retained build, not a portable mapping for other kernels.
The producer/consumer mutation removes ordering for local UB reuse as well as
peer GM accesses; its multiple findings must not be attributed to a single
isolated mechanism.

This supports using racecheck to diagnose these A3 native synchronization and
ownership mistakes. It does not establish complete race coverage, absence of
false positives, or any performance effect. Triton qualification is pending.
