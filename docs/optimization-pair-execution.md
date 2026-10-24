# Paired experiment execution

`scripts/optimization_pair.py` checks both implementations before collecting
exactly ten profiled forwards per arm. Preparation must not execute warmups.
The adapter provides `make_case`, `prepare`, and `validate`; prepared arms expose
`run`, `sources`, `metadata`, and exact `expected_kernels`, with optional
`compilation_state` and `close`. The default profiling path is unchanged.

For interface guards or other checks that need no timing evidence, use:

```bash
python scripts/optimization_pair.py \
  --adapter my_adapter:Adapter --case guard.json \
  --provenance provenance.json --cache oracle-cache --output guard-result \
  --correctness-only
```

The Python API accepts `run_pair(..., correctness_only=True)`. A case whose
`role` is exactly `"guard"` automatically selects this mode, even when the
flag is omitted. Use a fresh output directory for each experiment. The fixture
cache retains its existing case/source identity and integrity checks, so the
same CPU oracle can be reused without regeneration.

A successful correctness-only report has `mode: "correctness-only"`,
`profile_status: "not-requested"`, and `timing_kind: null`. Each arm is invoked
and validated once; capture and profiled-forward counts are zero. No capture,
latency, or speedup evidence is generated. Source hashes, fixture immutability,
optional compilation-state snapshots, and cleanup diagnostics remain recorded.
The CLI returns zero only when the checks and cleanup succeed. A failed check
stops execution before subsequent arms, retains its counts and diagnostics, and
still releases all prepared arms. This mode does not convert a rejected input
into an expected successful result: adapters remain responsible for expressing
and validating interface-guard behavior.
