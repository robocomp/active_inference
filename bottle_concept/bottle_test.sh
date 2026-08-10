#!/usr/bin/env bash
# Stress-test bottle_concept start/stop robustness against the shutdown/teardown crashes.
# Launches the agent, lets it reach operation, SIGTERMs it, and checks for a clean exit.
# Repeats N times. Requires the rest of the stack (Webots + bridges + robot/room/voxelizer) up.
#
#   ./bottle_test.sh [cycles]      (default 20)
set -u
cd "$(dirname "$0")"

CYCLES="${1:-20}"
RUN_SECONDS=8        # how long to let bottle operate before stopping it
GAP_SECONDS=5        # pause between a stop and the next start
LOG=/tmp/bottle_test_run.log
SUMMARY=/tmp/bottle_test_summary.txt

: > "$SUMMARY"
pass=0
fail=0

echo "bottle_test: $CYCLES cycles, ${RUN_SECONDS}s run / ${GAP_SECONDS}s gap" | tee -a "$SUMMARY"

for i in $(seq 1 "$CYCLES"); do
    pkill -9 -x bottle_concept 2>/dev/null
    sleep 1
    : > "$LOG"

    # Run, then stop with SIGTERM (clean-shutdown path) after it has had time to operate.
    timeout --signal=TERM "$RUN_SECONDS" ./bin/bottle_concept etc/config.toml > "$LOG" 2>&1
    rc=$?

    crash=$(grep -ciE 'ThreadSyscallException|terminate called|Abortado|Segmentation|core. generado' "$LOG")
    op=$(grep -cE "Operating|created node 'bottle|created instance" "$LOG")

    # rc: 0 = clean _Exit; 124 = timeout killed it after run window (also acceptable = stayed alive);
    #     143 = exited on SIGTERM; 134/139 = abort/segfault = FAIL.
    if [ "$crash" -gt 0 ] || [ "$rc" = "134" ] || [ "$rc" = "139" ]; then
        status="FAIL (rc=$rc crash_lines=$crash)"
        fail=$((fail+1))
    elif [ "$op" -eq 0 ]; then
        status="WARN (rc=$rc never reached operation — peers down?)"
        fail=$((fail+1))
    else
        status="ok   (rc=$rc op=$op)"
        pass=$((pass+1))
    fi
    line=$(printf "cycle %2d/%d: %s" "$i" "$CYCLES" "$status")
    echo "$line" | tee -a "$SUMMARY"

    # Show the crash tail immediately if this cycle failed.
    if echo "$status" | grep -q FAIL; then
        echo "    --- tail ---" | tee -a "$SUMMARY"
        tail -6 "$LOG" | sed 's/^/    /' | tee -a "$SUMMARY"
    fi

    sleep "$GAP_SECONDS"
done

pkill -9 -x bottle_concept 2>/dev/null
echo "-----------------------------------------" | tee -a "$SUMMARY"
echo "bottle_test DONE: $pass passed, $fail failed (of $CYCLES)" | tee -a "$SUMMARY"
