#!/bin/bash
# Leg-1 validation of the cortex remote-delete fix (2026-11-23): 10 tours of leg 1 only (LEGS=room_2), every object agent from
# main (as the failing runs had them), robot_concept restarted on the fixed libdsr just before. Per run: leg verdict, dead edges
# in the graph at tour start (snap_edge with no target node), controller stage medians, rate holds, load.
S=/tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-room-concept/1c86b8d6-47c4-4a7b-8f4b-04a583e967d4/scratchpad
C=/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing
CA="chair_concept table_concept bottle_concept hood_concept refrigerator_concept nightstand_concept bed_concept toilet_concept"
for i in $(seq -w 1 10); do
  r=l1fix_$i
  [ -f $C/STOP_SERIES ] && break
  mkdir -p $C/$r
  ( while true; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg) | $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,5p | tr -s ' ' | tr '\n' ';')"; sleep 10; done ) > $C/$r/load.log 2>&1 &
  LP=$!
  LEGS=room_2 RUN=$r CONCEPT_AGENTS="$CA" bash $S/c3a_run.sh > $S/$r.out 2>&1
  kill $LP 2>/dev/null
  dead=$(awk -F, '$2=="snap_edge" && $8==""' $C/$r/graph_trace.csv | wc -l)
  all=$(awk -F, '$2=="snap_edge"' $C/$r/graph_trace.csv | wc -l)
  st=$(python3 - $C/$r <<'E'
import re,sys,statistics as s
S={}
for l in open(sys.argv[1]+'/logs/controller.log',errors='replace'):
    if l.startswith('[stage]'):
        for m in re.finditer(r'([\w.]+) mean ([\d.]+) ms',l): S.setdefault(m.group(1),[]).append(float(m.group(2)))
g=lambda k: s.median(S[k]) if k in S else float('nan')
print('total %.1f refresh %.1f build %.1f exec %.1f'%(g('compute.total'),g('ctl.refresh_graph_state'),g('compute.build_planning_step'),g('compute.execute_plan')))
E
)
  echo "$r $(date +%T) $(sed -n 2p $C/$r/legs.tsv | cut -f3) | dead edges $dead of $all | $st | holds $(grep -ac 'below the minimum' $C/$r/logs/controller.log)"
done
