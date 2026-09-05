#!/usr/bin/env bash
# Wait for the box per Junseo's schedule of 2026-09-05 16:45 UTC.
# The box is ours only when all three hold, checked every 60 s, three clear
# minutes in a row:
#   - /workspace/bracket/DONE_p exists (or ABORT_p)
#   - no queue_s.sh running, or DONE_s / ABORT_s exists
#   - no harness/runner.py, bracket.py --lane or ladder_lane.py --lane
# Never touches another session's tmux.
set -uo pipefail
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes -p 23259 -i $HOME/.ssh/id_ed25519 root@213.173.107.199"
CLEAR=0
for i in $(seq 1 720); do
  status=$($SSH '
    p=no;  [ -e /workspace/bracket/DONE_p ] || [ -e /workspace/bracket/ABORT_p ] && p=yes
    s=no
    if [ -e /workspace/bracket/DONE_s ] || [ -e /workspace/bracket/ABORT_s ]; then s=yes
    elif ! pgrep -f "queue_s.sh" >/dev/null 2>&1; then s=yes; fi
    n=$(pgrep -cf "harness/runner.py|bracket.py --lane|ladder_lane.py --lane" 2>/dev/null || echo 0)
    echo "$p $s $n"
  ' 2>/dev/null)
  set -- ${status:-no no 99}
  if [ "$1" = yes ] && [ "$2" = yes ] && [ "$3" = 0 ]; then
    CLEAR=$((CLEAR + 1))
  else
    CLEAR=0
  fi
  echo "[$(date -u +%H:%M:%S)] DONE_p=$1 s_clear=$2 lanes=$3 clear_streak=${CLEAR}/3"
  if [ "$CLEAR" -ge 3 ]; then
    echo "=== BOX IS YOURS at $(date -u +%H:%M:%SZ) ==="
    $SSH 'echo "nr_throttled BEFORE: $(grep nr_throttled /sys/fs/cgroup/cpu.stat)"; tmux ls 2>&1; uptime'
    exit 0
  fi
  sleep 60
done
echo "gate never cleared in 12 h"
exit 1
