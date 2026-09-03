#!/usr/bin/env bash
# Point weights/ at one export in the run store (a sibling of this repo, not
# tracked in git -- see README).
#   ./use-weights.sh              # show what is available and what is active
#   ./use-weights.sh R1_e8_ema    # switch
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
store="$root/../weights"
live="$root/weights"

if [[ $# -eq 0 ]]; then
    echo "available:"
    for d in "$store"/*/; do [[ -d "$d" ]] && echo "  $(basename "$d")"; done
    echo "active:"
    for f in model.onnx model.int8.onnx manifest.json; do
        [[ -e "$live/$f" ]] || continue
        target="$(readlink "$live/$f" || true)"
        echo "  $f -> ${target:-<real file, not a symlink>}"
    done
    exit 0
fi

run="$1"
[[ -f "$store/$run/model.onnx" ]] || { echo "no $store/$run/model.onnx" >&2; exit 1; }

mkdir -p "$live"
# Drop whatever is there now: stale symlinks, and stale int8 from another run.
rm -f "$live/model.onnx" "$live/model.int8.onnx" "$live/manifest.json"
for f in model.onnx model.int8.onnx manifest.json; do
    [[ -e "$store/$run/$f" ]] && ln -s "../../weights/$run/$f" "$live/$f"
done

echo "active: $run"
ls -l "$live" | grep -E '\->' | sed 's/^/  /'
echo
echo "submission.zip is now stale -- rebuild with: make zip"
