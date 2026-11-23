#!/bin/bash
# Leg-1 diagnosis (2026-11-23): does the controller slow down when its threads land on E-cores (16-31 on the i9-14900KF)?
# Cycles the controller's affinity (all threads) default -> P (0-15) -> E (16-31) every 60 s, logging the controller.log
# line count at each switch so [stage] lines can be binned by phase, plus the CPU each controller thread ran on every 5 s.
# usage: affinity_toggle.sh <run_dir>
RUN=$1; OUT=$RUN/affinity.log; : > $OUT
while ! pgrep -x controller >/dev/null; do sleep 2; done
sleep 20
phases=(default P E); i=0
while pid=$(pgrep -x controller); do
  ph=${phases[$((i % 3))]}
  case $ph in default) cpus=0-31;; P) cpus=0-15;; E) cpus=16-31;; esac
  for t in /proc/$pid/task/*; do taskset -pc $cpus ${t##*/} >/dev/null 2>&1; done
  echo "$(date +%s%3N) PHASE $ph line $(wc -l < $RUN/logs/controller.log 2>/dev/null)" >> $OUT
  for k in $(seq 12); do
    echo "$(date +%s%3N) PSR main $(ps -o psr= -p $pid | tr -d ' ') threads $(ps -L -o psr= -p $pid | awk '{if($1<16)p++;else e++}END{print "P"p+0" E"e+0}')" >> $OUT
    sleep 5; pgrep -x controller >/dev/null || break
  done
  i=$((i+1))
done
