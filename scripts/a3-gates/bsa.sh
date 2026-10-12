set -euo pipefail
gate_root="$PWD/a3-gate"
mkdir -p "$gate_root"
if ! tar -xf "$TERMINAL_GATE_BUNDLE_PATH" -C "$gate_root"; then
  printf '%s\n' '{"schema":"simple-experiment-result/v1","outcome":"infrastructure_failure","diagnostics":"host gate bundle extraction failed"}'
  exit 1
fi
python "$gate_root/scripts/a3_terminal_gate.py" --root "$gate_root" \
  --candidate "$CANDIDATE_PATH" --manifest "$CANDIDATE_MANIFEST_PATH" \
  --benchmark bsa --output "$PWD/gate-output" --round "$EXPERIMENT_ROUND"
