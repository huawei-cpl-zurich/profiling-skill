# Pipe saturation analyzer contract

`scripts/analyze_pipe_saturation.py` is a product-gated, phase-local
classification interface. It does not contain product thresholds. Supply a
reviewed model whose capacity definitions match the evidence product.

```bash
python3 scripts/analyze_pipe_saturation.py \
  --product a3 \
  --model reviewed-a3-model.json \
  --evidence compact-phase-evidence.json \
  --output saturation.json
```

`--product` accepts only `a3` or `a5`. The command rejects a model or evidence
document for a different product.

## Model input

```json
{
  "schema_version": 1,
  "model_id": "reviewed-product-model-version",
  "product": "a3",
  "provenance": {
    "source": "stable-source-citation",
    "revision": "immutable-revision"
  },
  "resources": [
    {
      "resource": "pipe-name",
      "numerator_metric": "busy_metric",
      "denominator_metric": "capacity_metric",
      "saturation_threshold": 0.8,
      "capacity_validation": {
        "state": "validated-or-other-state",
        "source": "stable-capacity-citation"
      }
    }
  ]
}
```

Model provenance requires non-empty `source` and immutable `revision` string
identifiers. The example threshold is structural, not a product
recommendation. A usable
threshold must be greater than zero and at most one. Only
`capacity_validation.state == "validated"` permits classification. Other
states deliberately produce `unknown`, even when activity or composition is
high.

## Evidence input

```json
{
  "schema_version": 1,
  "product": "a3",
  "provenance": {
    "capture_id": "durable-capture-identity",
    "source_sha256": "compact-evidence-hash"
  },
  "phases": [
    {
      "phase_id": "steady-state",
      "start_ns": 100,
      "end_ns": 200,
      "metrics": {
        "busy_metric": 80,
        "capacity_metric": 100
      },
      "activity": {"pipe-name": 1.0},
      "composition": {"pipe-name": 0.9}
    }
  ]
}
```

Evidence provenance requires a non-empty `capture_id` and a 64-hex-character
`source_sha256`. Phase identifiers must be unique. Each phase has a
non-negative integer `start_ns` and a strictly greater integer `end_ns`; the
interval is preserved exactly in the output. Metric values may be numbers or
numeric strings. Missing values and the markers `NA`, `N/A`, `NaN`, `null`,
`none`, and `unknown` remain unavailable; they are never converted to zero. A
missing or non-positive denominator, or a numerator outside the normalized
range from zero through the denominator, yields `unknown` for that resource.

## Output semantics

Each resource result contains the metric names, ratio, threshold, capacity
source, state, and reason. A phase is:

- `saturated` when at least one validated resource meets its threshold;
- `unsaturated` when every modeled resource has valid capacity evidence and
  all are below threshold; or
- `unknown` otherwise.

This conservative aggregation proves the phase goal as soon as one resource
is saturated, but never calls a phase unsaturated while a modeled resource is
unresolved. The output repeats model and evidence provenance so downstream
reports remain bound to their inputs.
