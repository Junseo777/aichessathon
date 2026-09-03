#!/usr/bin/env bash
set -uo pipefail

RAW="${RAW_DIR:-/workspace/data/raw}"
mkdir -p "$RAW"
cd "$RAW"

ELITE_MONTHS="${ELITE_MONTHS:-2025-06 2025-07 2025-08 2025-09 2025-10 2025-11}"
MONTHLIES="${MONTHLIES:-2026-07 2026-06}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

fetch() {  # url dest
  local url="$1" dest="$2" remote local_sz
  remote=$(curl -sIL --max-time 60 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  if [[ -f "$dest" ]]; then
    local_sz=$(stat -c%s "$dest")
    if [[ -n "$remote" && "$local_sz" == "$remote" ]]; then
      log "OK    $dest ($(numfmt --to=iec $local_sz))"
      return 0
    fi
    log "RESUME $dest at $(numfmt --to=iec ${local_sz:-0}) / $(numfmt --to=iec ${remote:-0})"
  else
    log "START $dest ($(numfmt --to=iec ${remote:-0}))"
  fi
  curl -L -C - --retry 8 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 -o "$dest" "$url" || { log "FAIL  $dest"; return 1; }
  local_sz=$(stat -c%s "$dest")
  if [[ -n "$remote" && "$local_sz" != "$remote" ]]; then
    log "SHORT $dest: $local_sz != $remote"; return 1
  fi
  log "DONE  $dest ($(numfmt --to=iec $local_sz))"
}

log "=== ELITE (2400+ backbone, no evals) ==="
for m in $ELITE_MONTHS; do
  fetch "https://database.nikonoel.fr/lichess_elite_${m}.zip" "lichess_elite_${m}.zip"
done

log "=== MONTHLIES (eval coverage + 6c cross-check) ==="
for m in $MONTHLIES; do
  fetch "https://database.lichess.org/standard/lichess_db_standard_rated_${m}.pgn.zst" \
        "lichess_db_standard_rated_${m}.pgn.zst"
done

log "=== TOTAL ==="
du -sh "$RAW"
ls -la "$RAW"
