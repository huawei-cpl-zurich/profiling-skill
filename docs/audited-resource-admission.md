# Audited campaign resource admission

`scripts/audited_resource_admission.py` provides the production-facing
resource pool for the audited campaign scheduler. It deliberately owns only
resource discovery; repository bootstrap, controllers, lifecycle execution,
verification, and recovery remain with the production launcher.

## Public API

- `ADMISSION_SCHEMA`: `profiling-skill/bz-a3-admission/v2`.
- `AdmissionError`: invalid pinned input or invalid remote discovery evidence.
- `AdmissionReceipt` and `AdmissionSlot`: immutable receipt metadata and slot
  values, including the exact provider-file digest.
- `file_sha256(path)`: streaming file digest helper for configuration assembly.
- `slot_allowlist_sha256(slots)`: stable identity for the target/device set.
- `cpl_remote_closure_sha256(launcher)`: digest the launcher and sibling
  `cpl_remote.py` implementation.
- `load_admission(path, expected_sha256)`: read once, then hash, parse, and
  freshness-check that exact byte buffer.
- `CplRemoteResourcePool(...).admit_snapshot()`: return admitted slots with
  the exact receipt metadata; `last_receipt` retains the latest result.
- `CplRemoteResourcePool(...).admit()`: intersect provider slots with targets
  that pass both approved global probes and return scheduler-compatible
  dictionaries.

Construction requires the receipt hash, static provider and allowlist
identities, and pinned global-client closure hash:

```python
pool = CplRemoteResourcePool(
    admission_path,
    admission_sha256,
    provider_id="campaign-operator",
    allowlist_sha256=allowlist_sha256,
    cpl_remote_closure_sha256=cpl_remote_closure_sha256,
)
slots = pool.admit()
receipt = pool.last_receipt
```

The client path is not configurable. It resolves the authenticated account's
global `remote-access` skill client and rechecks the launcher plus its sibling
Python implementation before every probe.
The pool invokes only JSON `capabilities` and `preflight` operations against
the `bz-a3-1` and `bz-a3-2` target allowlist. A target is admitted only when
both probes pass and the capabilities include run, observe, logs, and upload.

## Provider contract

The provider file is immutable for a pool instance and must exactly match its
pinned hash and schema:

```json
{
  "schema": "profiling-skill/bz-a3-admission/v2",
  "provider": {
    "id": "campaign-operator",
    "allowlist_sha256": "LOWERCASE_SHA256_OF_TARGET_DEVICE_IDENTITIES"
  },
  "generated_at": "2026-10-06T09:00:00Z",
  "expires_at": "2026-10-06T09:03:00Z",
  "slots": [
    {"target": "bz-a3-1", "device": 0, "healthy": true, "idle": true}
  ]
}
```

Every slot has exactly those four fields. Devices are non-negative integers,
booleans must be actual JSON booleans, and each target/device pair must be
unique. Device numbers are provider data, never protocol constants. The pool
preserves `healthy` and `idle`; the campaign scheduler selects runnable slots
from those flags. The allowlist digest covers only the sorted target/device
identities, so occupancy flags can change while the authorized device set and
provider remain pinned.

Receipts are valid for at most five minutes and fail closed at expiration.
Production should record `last_receipt.receipt_sha256`, `generated_at`, and
`expires_at` with every scheduler admission decision. That digest always
identifies the complete provider file even though `last_receipt.slots` contains
only the targets that also passed live remote probes.

To refresh or resume after expiration, the operator writes a new complete
receipt, computes its SHA-256, and restarts/resumes the campaign with that new
receipt path/hash while retaining the same pinned provider ID and allowlist
digest. The pool never guesses occupancy or device numbers and never silently
extends an expired receipt.
