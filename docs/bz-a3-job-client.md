# Native BZ-A3 job client

`scripts/bz_a3_job_client.py` is the production JSON client for native Triton
checks and compact `msprof op` captures on BZ-A3. It accepts one JSON job on
standard input and prints one JSON result. Every job must declare
`"runtime":"py311-torch"`; missing or different runtime names are rejected
before staging.

The caller must provide:

```text
--state-dir PATH
--placements-json PATH
--remote-root ABSOLUTE_REMOTE_PATH
--cpl-remote-sha256 LOWERCASE_SHA256
[--timeout SECONDS]
```

The placements file maps logical experiment devices to an admitted target and
physical device:

```json
{"0":{"target":"bz-a3-1","device":0},"1":{"target":"bz-a3-2","device":2}}
```

Only `bz-a3-1` and `bz-a3-2` are accepted. The client resolves
`~/.agents/skills/remote-access/scripts/cpl-remote` itself and verifies its
digest before every invocation. Callers cannot supply an executable, adapter,
SSH route, or raw remote client.

Payload upload, retained dispatch, observation, terminal result lookup, and
compact stdout/stderr retrieval all use the global client's JSON interface.
Dispatch forwards the validated name as
`cpl-remote --json run TARGET --runtime py311-torch ...`, so runtime activation
is owned by the remote registry rather than encoded as a machine path in the
job client.
Once dispatch returns `remote:TARGET:job:ID`, the local state receipt records
that handle. A later invocation observes the same handle and never uploads or
dispatches a replacement. Full vendor profiling trees remain remote; the
response contains only the validated compact evidence and its remote path.
