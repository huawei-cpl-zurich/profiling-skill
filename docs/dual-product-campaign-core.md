# Dual-product campaign core

Schema-v4 audited manifests model A3 and A5 as independent experiment
products. A product cell pins its product, named runtime, independently ranked
task document, baseline, starter, prompt, skill set, and branch. Targets and
physical devices remain runtime placements and never enter the frozen cell.

The product adapters are closed policy:

| Product | Named runtime | Eligible targets |
|---|---|---|
| `a3` | `py311-torch` | `bz-a3-1`, `bz-a3-2` |
| `a5` | `cann91` | `bz-a5` |

Call `build_manifest` with `products=("a3", "a5")` and
`product_task_files={product: {task: path}}`. Product provenance replaces the
legacy top-level runtime, baselines, and starters with this exact shape:

```json
{
  "products": {
    "a3": {
      "runtime": "py311-torch",
      "runtime_image_digest": "sha256:<digest>",
      "baselines": {"matmul": "<digest>", "gdn": "<digest>", "bsa": "<digest>"},
      "starters": {"matmul": {}, "gdn": {}, "bsa": {}}
    },
    "a5": {
      "runtime": "cann91",
      "runtime_image_digest": "sha256:<digest>",
      "baselines": {"matmul": "<digest>", "gdn": "<digest>", "bsa": "<digest>"},
      "starters": {"matmul": {}, "gdn": {}, "bsa": {}}
    }
  }
}
```

Each starter retains the existing pinned `candidate` and `manifest` bindings.
The remaining source, controller, model, and skills provenance is shared and
immutable. Legacy schema-v1 through schema-v3 manifests remain valid.
The CLI equivalent repeats `--product a3 --product a5` and supplies
`--a3-matmul-task`, `--a3-gdn-task`, `--a3-bsa-task`, plus the corresponding
three `--a5-...-task` arguments. Legacy generation continues to use the three
unprefixed task arguments.

## Admission and scheduling

Dual-product providers use
`profiling-skill/dual-product-admission/v3`. Every slot declares `product`,
`runtime`, `target`, `device`, `healthy`, and `idle`; the loader rejects a
target or runtime inconsistent with the product adapter. Legacy A3 admission
v2 remains supported.

The scheduler refreshes admission before every assignment, leases every
unique compatible healthy idle target/device, and skips queued cells whose
product currently has no compatible slot. Completing one future immediately
refills its slot without a wave barrier. Placement is recorded in the durable
ledger, and a retained handle is observed only with its original compatible
placement. Infrastructure attempts do not change the controller's four-round
receipt contract and do not relaunch a retained handle.

The focused mixed-product runtime proof is:

```bash
pytest -q tests/test_dual_product_campaign.py
```

It blocks an A3 and an A5 launch concurrently, verifies both leases are filled,
then checks all 18 cells finish without cross-product placement. It separately
proves that unavailable A3 capacity cannot prevent all runnable A5 cells from
finishing.
