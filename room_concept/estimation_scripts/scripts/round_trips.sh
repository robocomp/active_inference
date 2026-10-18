#!/bin/bash
# Door-crossing reliability test: fresh fleet, 3 round trips (6 crossings) through the main door, then stop.
# usage: round_trips.sh <run_name> [main_door]   (e.g. try43; main_door defaults to door_1 = DOOR_1, the remembered
#        name in door_concept/etc/door_identities.csv). Logs in crossing/<run_name>/logs. Webots tworooms-piso AND
#        robot_concept must be running (run_crossing.sh does not start robot_concept; without it every agent aborts).
# ★ The first leg targets the main door BY NAME. It used to take the first door born, which was the main door only
#   because it was seen first from the start pose; try46 (10-17) birthed DOOR_0 (shut, no room behind) first.
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
echo stop > $B/controller/remote_cmd.txt  # DISARMED until the main door is born: the start pose faces it, and an armed
                                          # controller explores furniture meanwhile (try52: wedged on the table, door never seen).
                                          # Each leg arms it with `run` + `cross` below.
python3 scripts/door.py close; python3 scripts/door.py close DOOR_0; rm -f ../etc/last_robot_pose.txt $B/ltsm_agent/etc/ltsm_memory.json
NORUN=1 bash scripts/run_crossing.sh crossing/$1 -5.0 6.8 0 >/dev/null 2>&1
MAIN=${2:-door_1}
for i in $(seq 1 150); do grep -a -q "created instance for node '$MAIN'" $L/door_concept.log 2>/dev/null && break; sleep 2; done
if ! grep -a -q "created instance for node '$MAIN'" $L/door_concept.log; then
  echo "ABORT: '$MAIN' never born in 300 s; doors born: $(grep -a -o "created instance for node 'door_[0-9]*'" $L/door_concept.log | grep -o "door_[0-9]*" | tr '\n' ' ')"
  bash scripts/run_crossing.sh crossing/$1 collect >/dev/null 2>&1; exit 1
fi; sleep 5
done_n() { grep -a -c "while crossing — passage DONE" $L/controller.log; }
cur=$MAIN
ok=0
for leg in 1 2 3 4 5 6; do
  echo run > $B/controller/remote_cmd.txt; sleep 2; n0=$(done_n); t0=$(date +%s)
  echo "cross $cur" > $B/controller/remote_cmd.txt
  # ★ A door can be briefly absent from the graph (try46 leg 2: door_1 deleted and re-born seconds after the
  #   crossing); the controller then REFUSES the mission ("not in the graph") and nothing ever re-sends it.
  #   Re-send on every refusal for as long as the leg's window is open.
  refused_n() { grep -a -c "mission 'cross $cur': door '$cur' is not in the graph" $L/controller.log; }
  r0=$(refused_n)
  for i in $(seq 1 100); do
    [ $(done_n) -gt $n0 ] && break
    if [ $(refused_n) -gt $r0 ]; then r0=$(refused_n); echo "  leg $leg: '$cur' not in the graph — re-sending"
      echo run > $B/controller/remote_cmd.txt; sleep 2; echo "cross $cur" > $B/controller/remote_cmd.txt; fi
    sleep 3
  done
  if [ $(done_n) -gt $n0 ]; then ok=$((ok+1)); r="OK in $(( $(date +%s)-t0 )) s"; else r="FAILED"; fi
  room=$(grep -a "current room changed" $L/controller.log | tail -1 | awk '{print $NF}')
  echo "leg $leg through $cur: $r  (now in $room)"
  [ "$r" = "FAILED" ] && break
  sleep 15; cur=$(grep -a -o "entry mirror door_[0-9]* is now" $L/door_concept.log | tail -1 | awk '{print $3}')
  sleep 25
done
C=$L/controller.log; echo "=== $ok/6 crossings; lock-on locked=$(grep -a -c 'lock-on LOCKED' $C) giveup=$(grep -a -c 'lock-on GIVE_UP' $C); cross unreachable=$(grep -a -c 'cross ended unreachable' $C); open/cross back-steps=$(grep -a -c 'back from' $C)"
grep -a -c "WITHDRAWN" $L/door_concept.log | sed 's/^/door_concept WITHDRAWN lines: /'
f=$(ls -t $B/room_concept/tmp/agent_events_*.jsonl | head -1); echo "room births/re-entries:"; grep -a -o '"msg":"\[room\]\[\(recall\)\][^"]*RE-ENTERED[^"]\{0,20\}\|"msg":"\[room\]\[startup-proto\][^"]*BORN through[^"]\{0,10\}' $f | cut -c8-80
grep -a -E "\[promotion\] DONE" $L/ltsm_agent.log | tail -2 | grep -o "passage '[^']*' (crossings=[0-9]*)"
bash scripts/run_crossing.sh crossing/$1 collect >/dev/null 2>&1; echo "fleet stopped"
echo "--- recreations"; grep -a -E "recreating '|recreated the door|NOT recreating" /home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/$1/logs/door_concept.log | cut -c1-150
