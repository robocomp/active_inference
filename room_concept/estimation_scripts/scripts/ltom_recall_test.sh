#!/bin/bash
# LTOM RECALL TEST (owner 2026-11-13): does long-term object memory come back as a PRIOR on returning to a known room?
#   pass 1 "store":  ltom_agent starts with an EMPTY repertory (the old one is archived, never deleted); a warm-start tour
#                    (stored room layouts, so every room keeps its identity) looks round each room with every concept agent
#                    up, so ltom SEES the objects and stores them. Stopped gracefully ⇒ ltom saves its repertory.
#   pass 2 "recall": a FRESH fleet + agents + ltom_agent with pass 1's repertory; the same tour. On each known room the
#                    agents' tracker births should RESURRECT pass-1 objects (same uid) instead of issuing new ones.
#   graded by ltom_recall_report.py (per class: stored, re-seen, resurrected with a pass-1 uid, new, first-birth latency).
# usage: ltom_recall_test.sh <name> [fixture_dir] [look_s=45]      env CONCEPT_AGENTS / WT (concept_agents.sh), PASSES ("1 2")
# Webots is SHARED: claim it from the other sessions before running, release it after.
NAME=${1:?usage: ltom_recall_test.sh <name> [fixture_dir] [look_s]}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B=/home/pbustos/robocomp/components/active_inference; E=$B/room_concept/estimation_scripts
FIX=$(readlink -f "${2:-$E/crossing/fixtures/three_rooms_door0}")
export LOOK_S=${3:-45}
REP=$B/ltom_agent/etc/ltom_repertory.json
# ★A STALE ltom_agent DESTROYS THE OWNER'S MEMORY (2026-11-13): main's bin/ltom_agent predated the human model, and on its
# graceful stop it rewrote the repertory WITHOUT the "humans" section — the owner's name and chair were lost. Refuse a binary
# older than any of its sources; rebuild with `make -C ltom_agent/build -j8` (the ltom owner: test main's binary).
newest=$(find $B/ltom_agent/src $B/ltom_agent/CMakeLists.txt -newer $B/ltom_agent/bin/ltom_agent -type f 2>/dev/null | head -1)
[ -n "$newest" ] && { echo "ltom_agent binary is OLDER than $newest — rebuild it first"; exit 1; }
cd $E; mkdir -p crossing/$NAME
# The memory as it was before the test: archived, kept in the run dir, and PUT BACK at the end (the test's objects do not stay).
[ -f $REP ] && cp $REP crossing/$NAME/repertory_before.json
mkdir -p crossing/ltom_archive; [ -f $REP ] && cp $REP crossing/ltom_archive/ltom_repertory_before_${NAME}_$(date +%Y%m%d_%H%M%S).json
trap '[ -f crossing/$NAME/repertory_before.json ] && cp crossing/$NAME/repertory_before.json $REP && echo "ltom repertory restored to its pre-test state"' EXIT
for pass in ${PASSES:-1 2}; do
  RUN=${NAME}_p$pass
  if [ $pass = 1 ]; then
    # EMPTY of OBJECTS, but the "humans" array is KEPT (the ltom owner: the owner's name and owned things must survive).
    python3 -I - "$REP" crossing/$NAME/repertory_before.json <<'PY'
import json, os, sys
before = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
json.dump({"entries": [], "humans": before.get("humans", []), "version": before.get("version", 1)}, open(sys.argv[1], "w"))
PY
    rm -f $B/ltom_agent/etc/ltom_repertory.csv
    echo "pass 1 (store): ltom repertory EMPTY of objects, humans kept"
  else
    [ -f crossing/$NAME/repertory_after_p1.json ] || { echo "pass $pass: no pass-1 repertory — run pass 1 first"; exit 1; }
    cp crossing/$NAME/repertory_after_p1.json $REP
    echo "pass $pass (recall): ltom repertory = pass 1's ($(python3 -I -c "import json;print(len(json.load(open('$REP'))['entries']))") entries)"
  fi
  ON_FLEET_UP="LTOM=1 bash $S/concept_agents.sh start $E/crossing/$RUN" bash $S/warm_start_tour.sh $RUN $FIX 2>&1 | tee crossing/$NAME/tour_p$pass.txt
  LTOM=1 bash $S/concept_agents.sh stop
  cp $REP crossing/$NAME/repertory_after_p$pass.json 2>/dev/null
done
python3 -I $S/ltom_recall_report.py crossing/$NAME | tee crossing/$NAME/report.txt
