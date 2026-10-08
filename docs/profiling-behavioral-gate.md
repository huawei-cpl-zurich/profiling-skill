# Profiling behavioral gate

`scripts/profile_behavioral_gate.py` is a pure validator and aggregator for
trusted launcher records. It does not launch agents, create workspaces,
schedule or retry jobs, classify raw failures, or retain artifacts. The
separate audited launcher owns Bubblewrap/auth isolation, opaque agent-visible
paths, persistent sessions, remote observation, retry, and bounded retention.

Run the validator only on a frozen artifact tree and launcher-produced records:

```console
python scripts/profile_behavioral_gate.py \
  --manifest /frozen/manifest.json \
  --records /retained/launcher-records.json \
  --artifact-root /retained/artifacts \
  --output /retained/report.json
```

The command reads its three inputs and atomically writes only the report.

## Trust boundary

The manifest pins the prompt, both skill trees, audited launcher, model/config,
manual reviewer/config, allowed product targets, and four evidence cases (two A3 and two A5). Each case
contains a byte-pinned compact JSON evidence reference and an exact rubric:
the complete conclusion list and complete structured saturation-claim list.
Extra, missing, malformed, or contradictory claims fail that case.

The launcher record envelope—not agent output—contains the arm, session and
failure classification, launcher/model/skill identities, artifact references,
and manual-review decision. Agent-visible payloads have an exact minimal schema
and contain neither the arm nor a host workspace. The core validates this
recorded payload; the launcher gate must prove that the real sandbox exposed
the same payload and exactly one selected skill.

Every referenced artifact must be a nonempty regular nonsymlink below the
artifact root, no larger than 64 KiB, and match its SHA-256. Evidence is also
valid JSON whose schema and provenance match the envelope. Acquisition handles
must bind an allowed target, and their provenance must bind the same product,
target, and durable handle. Failure receipts are compact JSON objects whose
session, classification, and failure type exactly match their trusted record.
Both `remote:<target>:job:<id>` and the managed `gz-a3:<job-id>` handle are
accepted when they bind the declared allowed target. Command, log, reasoning,
and review-note artifacts must also decode as nonempty bounded text.

## Session and acceptance contract

Each successful acquisition record represents one persistent launcher-attested
session with both A3 and A5 outcomes. Exactly three paired successes are
required; counted failures and discarded infrastructure replacement sessions
remain in the report counts and retain their receipt references.

Each successful interpretation record represents one independent session with
all four answers. There must be exactly three terminal non-infrastructure
sessions per arm. A counted terminal failure occupies four failed score slots,
so the report always scores 24 interpretation slots. Infrastructure-discarded
sessions do not occupy a slot and may precede replacements.

Acceptance requires three acquisition pairs, candidate 12/12, candidate score
strictly above current, and a trusted positive manual review with the pinned
reviewer and nonempty notes for every successful acquisition and interpretation
reasoning log in both arms. The report lists an attempt summary plus the exact
evidence, command, log, reasoning, receipt, and manual-note references/hashes
needed for audit. Live acceptance remains blocked until the audited launcher,
product stack, and supported A5 profiling runtime are available.
