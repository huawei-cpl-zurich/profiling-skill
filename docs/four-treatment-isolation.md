# Four-treatment skill isolation

New audited campaigns use exactly these treatment-local skill trees:

| Treatment | Visible skills |
| --- | --- |
| `cannbot-all` | The complete frozen CANNBot skill bundle, including `ops-profiling` |
| `cannbot-new-profiler` | Frozen CANNBot coding skills without `ops-profiling`, plus the new `ascend-profiling` |
| `guarded-new-profiler` | `triton-guarded-kernel` plus the new `ascend-profiling` |
| `guarded-old-profiler` | `triton-guarded-kernel` plus the old `ascend-profiling` |

The old profiler is commit
`d6cc328144df17d09979c1d154366f52a55f5454`; the new profiler is commit
`1b9ae02303b3838f683c37a9a2fe15ce740ca56e`. The runtime configuration binds
one complete CANNBot freeze, both profiler trees, and one guarded-kernel tree
by content hash. Each profiler commit is also bound to the trusted digest of
its exported tree, so relabeling one tree as the other revision fails before
an agent starts. The CANNBot freeze carries a 40-character commit in its
`COMMIT` file. A mismatched revision, trusted tree digest, missing skill, or
extra source fails closed.

Each isolated repository contains only its declared directories under
`.agents/skills`. The agent-visible checkout contains no treatment receipt or
alternative-source metadata. Trusted treatment provenance is retained beside
the repository in the cell's `state/treatment.json`.

Campaign manifests use schema version 4 and production runtime configurations
use `profiling-skill/audited-campaign-runtime/v3`. All four cells for one task
bind the same prompt bytes, task bytes, model configuration, starter, baseline,
and benchmark inputs. Schema 4 accepts exactly the `cannbot`, `profiler-new`,
`profiler-old`, and `guarded` provenance sources. Schema versions 1–3 retain
their exact legacy provenance set for verification-only support of already
retained campaigns; new production dispatch rejects them.
