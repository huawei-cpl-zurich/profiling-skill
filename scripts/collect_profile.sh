#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: collect_profile.sh --implementation ascendc|dsl --remote-root PATH --output DIR [--mode summary|analysis] [--kernel-name NAME ...] [--catlass-src PATH] [--operation NAME] [--dry-run]'
}

implementation= mode=analysis remote_root= output= catlass_src=
operation=codex-ascend-profile-collect dry_run=0
kernels=()
while (($#)); do
  case "$1" in
    --implementation) implementation=$2; shift 2;;
    --remote-root) remote_root=$2; shift 2;;
    --output) output=$2; shift 2;;
    --mode) mode=$2; shift 2;;
    --kernel-name) kernels+=("$2"); shift 2;;
    --catlass-src) catlass_src=$2; shift 2;;
    --operation) operation=$2; shift 2;;
    --dry-run) dry_run=1; shift;;
    -h|--help) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
done

: "${TLA_ROOT:?set TLA_ROOT to the configured TLA integration checkout}"
[[ $implementation == ascendc || $implementation == dsl ]] || { echo 'invalid --implementation' >&2; exit 2; }
[[ $mode == summary || $mode == analysis ]] || { echo 'invalid --mode' >&2; exit 2; }
[[ -n $remote_root && -n $output ]] || { usage >&2; exit 2; }
[[ $implementation != dsl || -n $catlass_src ]] || { echo '--catlass-src is required for dsl' >&2; exit 2; }

skill_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stage="artifacts/ascend-profiling/${operation}"
remote_out="${stage}/compact"
mkdir -p "$output"

upload=("$TLA_ROOT/execution-profiles/bz-a5/scp.sh" --recursive upload "$skill_root/scripts" "$stage")
curate=(python3 "$stage/scripts/curate_profile.py" --input "$remote_root" --output "$remote_out" --mode "$mode")
for kernel in "${kernels[@]}"; do curate+=(--kernel-name "$kernel"); done

if [[ $implementation == dsl ]]; then
  run=("$TLA_ROOT/execution-profiles/catlass-validation.sh" --profile bz-a5 --operation "$operation" run --catlass-src "$catlass_src" -- "${curate[@]}")
else
  run=("$TLA_ROOT/execution-profiles/bz-a5/run.sh" -- "${curate[@]}")
fi
download=("$TLA_ROOT/execution-profiles/bz-a5/scp.sh" download "$remote_out/ascend-profile-$mode.tar.gz" "$output/ascend-profile-$mode.tar.gz")
checksum=("$TLA_ROOT/execution-profiles/bz-a5/scp.sh" download "$remote_out/ascend-profile-$mode.sha256" "$output/ascend-profile-$mode.sha256")

if ((dry_run)); then
  printf 'UPLOAD='; printf '%q ' "${upload[@]}"; printf '\nRUN='; printf '%q ' "${run[@]}"
  printf '\nDOWNLOAD='; printf '%q ' "${download[@]}"; printf '\nCHECKSUM='; printf '%q ' "${checksum[@]}"; printf '\n'
  exit 0
fi

"${upload[@]}"
"${run[@]}"
"${download[@]}"
"${checksum[@]}"
(cd "$output" && sha256sum -c "ascend-profile-$mode.sha256")
printf 'REMOTE_RAW=%s\nLOCAL_ARCHIVE=%s\n' "$remote_root" "$output/ascend-profile-$mode.tar.gz"
