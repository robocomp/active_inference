#!/bin/bash
# Door-crossing reliability test: fresh fleet, 3 round trips (6 crossings) through the main door, then stop.
# usage: round_trips.sh <run_name>   (e.g. try43). Logs in crossing/<run_name>/logs. Webots tworooms-piso must be running.
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts; cd $E; L=crossing/$1/logs
echo run > $B/controller/remote_cmd.txt   # a leftover "stop" from the previous run would park the new fleet
python3 scripts/door.py close; python3 scripts/door.py close DOOR_0; rm -f ../etc/last_robot_pose.txt $B/ltsm_agent/etc/ltsm_memory.json
NORUN=1 bash scripts/run_crossing.sh crossing/$1 -5.0 6.8 0 >/dev/null 2>&1
until grep -a -q "created instance for node 'door_" $L/door_concept.log 2>/dev/null; do sleep 2; done; sleep 5
done_n() { grep -a -c "while crossing — passage DONE" $L/controller.log; }
cur=$(grep -a -o "created instance for node 'door_[0-9]*'" $L/door_concept.log | head -1 | grep -o "door_[0-9]*")
ok=0
for leg in 1 2 3 4 5 6; do
  echo run > $B/controller/remote_cmd.txt; sleep 2; n0=$(done_n); t0=$(date +%s)
  echo "cross $cur" > $B/controller/remote_cmd.txt
  for i in $(seq 1 100); do [ $(done_n) -gt $n0 ] && break; sleep 3; done
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
