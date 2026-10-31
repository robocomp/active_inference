#!/bin/bash
# Each Vac* invariant must be VIOLATED (the state it forbids is reachable). Usage: ./RoomTransitionExit_vacuity.sh <cfg>
TLA=${TLA:-/home/pbustos/.vscode/extensions/tlaplus.vscode-ide-2026.10.280155/tools/tla2tools.jar}
CFG=${1:-RoomTransitionExit_pingpong.cfg}; T=$(mktemp -d)
for v in VacTourEnds VacAccepted VacRejected VacDegraded VacPhantom VacLeg7 VacArchived; do
  sed -e '/^INVARIANT/d' -e '/^PROPERTY/d' "$CFG" > $T/v.cfg; echo "INVARIANT $v" >> $T/v.cfg
  r=$(java -Xmx3g -XX:+UseParallelGC -cp $TLA tlc2.TLC -workers 4 -noGenerateSpecTE -metadir $T/m -config $T/v.cfg RoomTransitionExit.tla 2>&1 | grep -c "is violated")
  echo "$v: $([ $r -gt 0 ] && echo reached || echo NOT REACHED)"
done
