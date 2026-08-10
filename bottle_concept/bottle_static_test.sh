#!/usr/bin/env bash
# Static-restart bottle validation: for each grid pose, restart bottle_concept fresh (clean voxel
# bank — no cross-pose accumulation), let it converge on the now-static bottle, log fit-vs-GT, stop.
# Real scenario: the object is static until grasped. Requires the full stack + kinova-bridge (15010).
#
#   ./bottle_static_test.sh [gridN]      (default 3 -> 9 poses; must match Eval.MoveGridN)
set -u
cd "$(dirname "$0")"

N="${1:-3}"
RUN_SECS=16          # converge time per pose (fit settles in ~10-12s)
GAP=5                # gap after clean SIGTERM exit (DDS participant removal propagates)
LOG=etc/bottle_static_eval.csv
total=$((N * N))

pkill -TERM -x bottle_concept 2>/dev/null; sleep 4; pkill -9 -x bottle_concept 2>/dev/null
# Keep etc/bottle_home_pose.txt if present (it is the fixed grid origin = the bottle's true table
# spot). Delete it only to re-capture home, and only with the bottle actually at its home position,
# else the grid origin drifts. Here we KEEP it.
rm -f "$LOG" /tmp/bstatic_*.log
[ -f etc/bottle_home_pose.txt ] && echo "using fixed home: $(cat etc/bottle_home_pose.txt)" || echo "no home file — pose 0 will capture current bottle pose as home"
echo "static-restart test: $total poses, ${RUN_SECS}s/pose, grid ${N}x${N}"

for K in $(seq 0 $((total - 1))); do
    echo "=== pose $K/$total ==="
    BOTTLE_TEST_POSE=$K setsid ./bin/bottle_concept etc/config.toml > /tmp/bstatic_$K.log 2>&1 < /dev/null &
    pid=$!
    sleep "$RUN_SECS"
    kill -TERM "$pid" 2>/dev/null
    for i in $(seq 1 10); do pgrep -x bottle_concept >/dev/null || break; sleep 1; done
    pkill -9 -x bottle_concept 2>/dev/null     # safety net
    sleep "$GAP"
    grep -q "already an agent connected" /tmp/bstatic_$K.log 2>/dev/null && \
        echo "  WARN: id-collision on pose $K (increase GAP)"
done

echo "DONE -> $LOG ($(wc -l < "$LOG" 2>/dev/null || echo 0) rows)"
