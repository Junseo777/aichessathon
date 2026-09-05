#!/usr/bin/env bash
# One lane of a feature arena: A (with) vs B (without), competition clock, openings from
# sparring/openings.tsv with both colours, a fixed number of games.
# usage: arena.sh <out-dir> <dirA> <dirB> <games> [first-colour-offset 0|1]
set -uo pipefail
OUT=$1; A=$2; BDIR=$3; GAMES=$4; OFFSET=${5:-0}
ROOT=/Users/junseo/chess-master; REPO=$ROOT/aichessathon; PY=$REPO/.venv/bin/python
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"; mkdir "$OUT/.lock" 2>/dev/null || { echo "$OUT already running" >&2; exit 1; }
trap 'rmdir "$OUT/.lock" 2>/dev/null' EXIT
OPENINGS=(); while IFS= read -r line; do [ -n "$line" ] && OPENINGS+=("$line"); done < "$ROOT/sparring/openings.tsv"
cd "$REPO"; START=$(date +%s)
echo "game,opening,white,black,result,termination,score_with,seconds" > "$OUT/results.csv"
for ((n = 1; n <= GAMES; n++)); do
  g=$((n + OFFSET))
  idx=$(( ((g - 1) / 2) % ${#OPENINGS[@]} )); IFS=$'\t' read -r oname fen <<< "${OPENINGS[$idx]}"
  if [ $((g % 2)) -eq 1 ]; then white=$A; black=$BDIR; wtag=with; btag=without; else white=$BDIR; black=$A; wtag=without; btag=with; fi
  tag=$(printf "g%03d_%s_%sw" "$n" "$oname" "$wtag")
  echo "=== [$(date +%H:%M:%S)] game $n/$GAMES $oname white=$wtag ===" | tee -a "$OUT/run.log"
  t0=$(date +%s)
  "$PY" -m harness.play --white "$white" --black "$black" --fen "$fen" --pgn "$OUT/$tag.pgn" > "$OUT/$tag.log" 2>&1 < /dev/null
  t1=$(date +%s)
  line=$(grep -m1 " vs .*: " "$OUT/$tag.log" || echo ""); result=$(echo "$line" | sed -E 's/.*: ([a-z]+) by .*/\1/'); term=$(echo "$line" | sed -E 's/.* by (.*)$/\1/')
  case "$result:$wtag" in white:with|black:without) s=1 ;; white:without|black:with) s=0 ;; draw:*|void:*) s=0.5 ;; *) s=NA ;; esac
  echo "$n,$oname,$wtag,$btag,$result,$term,$s,$((t1 - t0))" >> "$OUT/results.csv"
  echo "    -> $result by $term (with scores $s) in $((t1 - t0))s" | tee -a "$OUT/run.log"
done
echo "DONE after $(( $(date +%s) - START ))s" | tee -a "$OUT/run.log"
