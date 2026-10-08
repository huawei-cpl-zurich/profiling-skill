# Profiling behavioral launcher

`scripts/profile_behavioral_launcher.py` is the trusted live adapter for the
pure behavioral gate. It specializes the existing production launcher's
Bubblewrap, Codex authentication, DNS, and persistent-session machinery. It
does not provide an alternate sandbox or remote transport.

The host sets `PROFILE_GATE_REQUEST` to a frozen launch-request JSON file and
`PROFILE_GATE_OUTPUT` to a new retained output path, then runs the script. The
request pins the prompt bytes, selected profiling-skill tree, model, launcher,
reviewer, allowed targets, and either the paired A3/A5 acquisition units or the
four interpretation cases. Source paths and the arm never enter the
agent-visible `request.json`; that file contains only its schema and blinded
turn payloads. Acquisition payloads expose only the assigned product, target,
and prompt hash—never the arm or host session label. Interpretation cases have
unique case IDs and exactly two A3 plus two A5 products; they carry no target,
while the product-scoped allowed-target registry requires valid, unique,
non-overlapping A3/A5 target IDs. The selected skill is mounted read-only at the
generic `.agents/skills/ascend-profiling` path. User-wide skills and Codex
plugins are absent. Prompt, selected-skill, and frozen-case bytes are copied
and rehashed before launch so later source-tree changes cannot alter a session.

Inside Bubblewrap, `cpl-remote` is a narrow socket client. The trusted host
broker invokes only the configured global remote-access client. Remote runs
must use `--file` plus a broker-only `--dispatch-key`. The broker removes the
key before invoking `cpl-remote` and snapshots the exact payload bytes. A key
reused with identical argv and bytes observes the original transaction; a key
reused after any change fails closed, while a new key permits an intentional
corrected or repeated capture. Arbitrary command payloads, raw SSH, unknown targets, and
local paths outside `/workspace` are rejected. Observe, log, and result calls
are limited to handles dispatched by the same retained broker journal. Each
acquisition turn narrows the broker to its assigned target; interpretation
turns have no remote target capability. Before dispatch, the broker atomically
records a request hash. If observation is
interrupted after a handle is known, the same request observes that handle
instead of dispatching a replacement. A handle-less uncertain dispatch is
never resubmitted and must
be reconciled externally from its retained journal evidence. Explicit
observation updates the originating dispatch, and acquisition evidence is
eligible only after that dispatch is `completed` with exit code zero. Upload
and download are not exposed to agents; `run --file` and compact stdout
evidence are the supported surface. The retained terminal output must include
`REMOTE_CONTENT_SHA256=<hex>` for the exact compact evidence file bytes, and
the launcher recomputes that digest before retaining those same bytes without
JSON reserialization. Producer-supplied evidence provenance contains exactly
the assigned product and target, not a durable handle that only becomes known
after broker dispatch. The trusted launcher separately binds the final
answer's handle and target to the selected terminal-successful dispatch, and
records that handle in the outcome beside the matching remote SHA-256. The
complete journal, bound result/log
calls, and bounded redacted Codex logs are retained on success and failure.

An acquisition agent uses one Codex thread for its A3 turn and resumed A5
turn. An interpretation agent uses one independent thread resumed across all
four cases. Each final message must be exactly one JSON object. Acquisition
answers identify the product, approved target, broker-retained handle, compact
evidence file, and reasoning. Interpretation answers identify the case,
conclusions, saturation claims, and reasoning.

Successful launcher output is deliberately `review_pending`. Its draft gate
record contains the exact agent-visible payloads and retained artifacts, but
no self-attested review result. Call `finalize_reviews` from a trusted external
review step with the manifest-pinned reviewer identity and one explicit
decision plus nonempty notes per product or case. Only that finalized record
is eligible for `profile_behavioral_gate.py`.

Infrastructure discard is fail-closed: only structured remote-access states
or failure types for transport, observer, target availability, or device busy
qualify. Agent timeout, launcher failure, compilation, runtime, profiler
command, evidence, and interpretation failures consume a scored attempt.
Failure text is classified only from the selected dispatch and its same-handle
remote observations; retained Codex prose cannot relabel that result. Within
that trusted remote text, concrete `msprof` or profiler-command diagnostics
take precedence over generic compile wording, while a missing exported kernel
selector is an evidence failure. Structured remote failure types remain
authoritative.

Live A5 acquisition remains blocked until `bz-a5` has a registered runtime or
supported wrapper in the global remote registry. Do not substitute raw SSH,
an ad-hoc environment activation, or another target.

## Operator request and review finalization

Create a self-contained interpretation request example with real byte hashes:

```console
python scripts/profile_behavioral_launcher.py request-example \
  --output-dir /tmp/profile-gate-example
```

The generated `request.json` is the executable schema example. It contains the
launch schema/version, host-only session and arm envelope, pinned prompt and
skill paths/hashes, launcher/model/reviewer identities, allowed A3/A5 targets,
and four byte-pinned case units. Before a campaign, replace the example prompt,
skill, cases, manifest identity, arm, and reviewer with the frozen campaign
values and recompute their hashes using the same canonical JSON rules.

After an external reviewer fills a decisions object keyed by product for
acquisition or case ID for interpretation, produce the exact core record:

```console
python scripts/profile_behavioral_launcher.py finalize \
  --launch-output retained/launcher-output.json \
  --decisions retained/review-decisions.json \
  --artifact-root retained/artifacts \
  --output retained/final-record.json
```

Each decision is `{"passed": true|false, "notes": "nonempty evidence-based notes"}`.
The command uses the reviewer identity pinned in launcher output and atomically
writes a record suitable for the acquisition or interpretation array consumed
by `profile_behavioral_gate.py`.
