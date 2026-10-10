# BZ-A5 production job client

`scripts/bz_a5_job_client.py` is the A5 specialization of the audited native
Triton job boundary. It accepts the same JSON protocol as the A3 client, but
requires `product=a5`, `runtime=cann91`, and a frozen placement whose target is
`bz-a5`.

The client creates a deterministic local payload archive, hashes it, and stages
it under a content-addressed filename in the pinned writable A5 root using the
installed global remote-access client:

```text
cpl-remote --json upload --timeout SECONDS bz-a5 PAYLOAD \
  REMOTE_ROOT/payload-SHA256.tar
```

Only after that transfer completes does it submit a bounded run script:

```text
cpl-remote --json run bz-a5 --runtime cann91 --file SCRIPT --cwd REMOTE_ROOT
```

The script contains the expected digest and remote content-addressed path, not
the payload bytes. It verifies the staged archive with `sha256sum` before
creating the run work directory or extracting anything, and exits with a
digest-mismatch failure if the remote bytes differ. Thus payload size cannot
inflate the SSH command line, and a remote collision or corrupt transfer fails
closed. A completed or dispatched request receipt is reused without uploading
or dispatching again.

The admitted physical device remains trusted placement metadata. The run-file
wrapper exposes it through `ASCEND_RT_VISIBLE_DEVICES`; the benchmark process
always addresses logical device zero. A dispatch receipt is persisted before
observation. If observation is interrupted, the next identical request observes
the exact recorded `remote:bz-a5:job:...` handle and never redispatches.

Correctness and full-fusion checks are controller gates. Timing is accepted only
after those gates. Successful profile responses contain compact `msprof op`
rows, per-case samples and medians, a geometric mean, exact product/runtime/
target/device provenance, and a remote path to the compact evidence JSON. Raw
vendor report trees remain on the remote host.

Transport, observer, device, staging, profiler-tool, and evidence failures are
infrastructure failures. Candidate submission, compilation, runtime,
correctness, and fusion failures remain counted candidate outcomes.
