# Audited campaign resource admission

`scripts/audited_resource_admission.py` provides the production-facing
resource pool for the audited campaign scheduler. It deliberately owns only
resource discovery; repository bootstrap, controllers, lifecycle execution,
verification, and recovery remain with the production launcher.

## Public API

- `ADMISSION_SCHEMA`: `profiling-skill/bz-a3-admission/v1`.
- `AdmissionError`: invalid pinned input or invalid remote discovery evidence.
- `file_sha256(path)`: streaming file digest helper for configuration assembly.
- `load_admission(path, expected_sha256)`: validate and return the provider
  slots.
- `CplRemoteResourcePool(...).admit()`: intersect provider slots with targets
  that pass both approved global probes.

Construction requires the admission path and SHA-256 plus the pinned global
client SHA-256:

```python
pool = CplRemoteResourcePool(
    admission_path,
    admission_sha256,
    cpl_remote_sha256=cpl_remote_sha256,
)
slots = pool.admit()
```

The client path is not configurable. It resolves the authenticated account's
global `remote-access` skill client and rechecks its digest before every probe.
The pool invokes only JSON `capabilities` and `preflight` operations against
the `bz-a3-1` and `bz-a3-2` target allowlist. A target is admitted only when
both probes pass and the capabilities include run, observe, logs, and upload.

## Provider contract

The provider file is immutable for a pool instance and must exactly match its
pinned hash and schema:

```json
{
  "schema": "profiling-skill/bz-a3-admission/v1",
  "slots": [
    {"target": "bz-a3-1", "device": 0, "healthy": true, "idle": true}
  ]
}
```

Every slot has exactly those four fields. Devices are non-negative integers,
booleans must be actual JSON booleans, and each target/device pair must be
unique. Device numbers are provider data, never protocol constants. The pool
preserves `healthy` and `idle`; the campaign scheduler selects runnable slots
from those flags.
