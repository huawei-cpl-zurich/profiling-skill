# Recover a frozen characterization campaign

Use `scripts/recover_a3_profile_campaign.py` when cell records survived but shard
summary generation failed. Keep the original campaign and qualified producer
checkout intact. Recovery checks the original qualification, benchmark and
acquisition hashes, every cell run contract, and complete unique shard coverage.
The merged JSON retains the original `campaign_identity`; `reporting_revision`
identifies the recovery code independently. Normal campaign merge still requires
the current frozen source identity.

Create the report first, without contacting a remote:

```bash
python scripts/recover_a3_profile_campaign.py report \
  --campaign-dir "$original_campaign" --producer-root "$frozen_producer" \
  --qualification "$qualification" --output-dir "$report_before"
```

This writes four `shard-result.json` files, `merged.json`, and `report.md`. Missing
pipe metrics appear explicitly as `not_exported`; timing remains a reported
core-time proxy, not application latency. Report directories must be separate
from campaign directories.

Retry only exhausted infrastructure and terminal jobs lacking a failure receipt:

```bash
python scripts/recover_a3_profile_campaign.py retry \
  --campaign-dir "$original_campaign" --producer-root "$frozen_producer" \
  --qualification "$qualification" --output-dir "$retry_copy" \
  --shard-index 0 --shard-count 4 --cpl-remote "$cpl_remote"
```

Run one process per shard index (0–3) concurrently against the same retry output
directory. Each process copies and owns its shard, preserving original evidence,
attempts, contracts, and stable/inconclusive records. Only one process may own a
given shard. Each selected cell receives one additional original attempt budget
(seven acquisition attempts for the current campaign);
rerunning the command resumes that bounded recovery round, without renewing its
budget. Previous valid samples count toward the existing stability threshold.

Before starting a new attempt, recovery observes the retained failure handle
through `cpl-remote`. A running, unavailable, or completed handle prevents a new
dispatch and reports the handle to inspect. Transport interruptions during the
recovery round retain receipts and resume the same job on the next invocation.
New missing-diagnostic failures are retried as infrastructure only after an
explicit terminal failure check. Actual compiler/runtime/profiler failures stay
failures. The frozen producer still chooses `py311-torch` for GZ acquisition.

When all shards finish, rebuild the final report from the retry copy into another
report directory using the first command with `--campaign-dir "$retry_copy"`.
Retain inconclusive timing cells rather than repeatedly sampling until a desired
outcome appears. No existing sampling threshold changes during recovery.
