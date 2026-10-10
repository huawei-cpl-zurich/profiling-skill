# BZ-A5 production job client

`scripts/bz_a5_job_client.py` is the A5 specialization of the audited native
Triton job boundary. It accepts the same JSON protocol as the A3 client, but
requires `product=a5`, `runtime=cann91`, and a frozen placement whose target is
`bz-a5`.

The client does not require transfer capabilities. It creates a deterministic
local payload archive, embeds that archive in a temporary shell script, and
submits the script with the installed global remote-access client:

```text
cpl-remote --json run bz-a5 --runtime cann91 --file SCRIPT --cwd REMOTE_ROOT
```

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
