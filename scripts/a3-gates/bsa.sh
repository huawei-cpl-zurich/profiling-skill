# simple-experiment-capability: managed-autotune/v1
set -euo pipefail
gate_root="$PWD/a3-gate"
mkdir -p "$gate_root"
if ! tar -xf "$TERMINAL_GATE_BUNDLE_PATH" -C "$gate_root"; then
  printf '%s\n' '{"schema":"simple-experiment-result/v1","outcome":"infrastructure_failure","diagnostics":"host gate bundle extraction failed"}'
  exit 1
fi
gate_stdout="$PWD/a3-gate.stdout"
autotune_args=()
if test -n "${AUTOTUNE_REQUEST_PATH:-}"; then
  autotune_args=(--autotune-request "$AUTOTUNE_REQUEST_PATH")
fi
set +e
python "$gate_root/scripts/a3_terminal_gate.py" --root "$gate_root" \
  --candidate "$CANDIDATE_PATH" --manifest "$CANDIDATE_MANIFEST_PATH" \
  --benchmark bsa --output "$PWD/gate-output" --round "$EXPERIMENT_ROUND" \
  "${autotune_args[@]}" \
  >"$gate_stdout"
gate_rc=$?
set -e
cat "$gate_stdout"
if ! python - "$gate_stdout" <<'PY'
import json
import sys
from pathlib import Path

lines = [line for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
value = json.loads(lines[-1]) if lines else None
if (not isinstance(value, dict)
        or value.get("schema") != "simple-experiment-result/v1"
        or value.get("outcome") not in {"success", "candidate_failure", "infrastructure_failure"}):
    raise SystemExit(1)
PY
then
  printf '%s\n' '{"schema":"simple-experiment-result/v1","outcome":"infrastructure_failure","diagnostics":"host gate entrypoint produced no result envelope"}'
  exit 1
fi
exit "$gate_rc"
