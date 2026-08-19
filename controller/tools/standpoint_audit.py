#!/usr/bin/env python3
"""Read controller/standpoint_audit.jsonl and answer ONE question: which target did the robot arrive at?

Every row carries the three poses side by side — `wire` is what the PRODUCER published, `tgt` is what
the controller is driving to after its own repair, `rob` is where the robot is. An arrival is only an
arrival if d_wire is small too; a row with d_tgt ~ 0 and d_wire ~ metres is a completion reported for
a cell the robot never visited, which is the failure this file exists to name.

usage: standpoint_audit.py [path] [--tail N]
"""
import sys, json, statistics as st

path = "controller/standpoint_audit.jsonl"
tail = 0
args = sys.argv[1:]
skip = -1
for i, a in enumerate(args):
    if a == "--tail": tail = int(args[i + 1]); skip = i + 1
    elif not a.startswith("-") and i != skip: path = a

rows = []
for line in open(path):
    line = line.strip()
    if not line: continue
    try: rows.append(json.loads(line))
    except json.JSONDecodeError: pass      # torn tail line: skipped, never misread

facts = [r for r in rows if r["event"].startswith("fact-")]
rep = [r for r in rows if r["event"] == "repair"]
arr = [r for r in rows if r["event"] == "arrival"]
print(f"{len(rows)} rows: {len(rep)} repair, {len(arr)} arrival")

if rep:
    moved = [r for r in rep if r["repair_m"] > 0.05]
    print(f"\nREPAIR: {len(moved)}/{len(rep)} moved the standpoint >5 cm; "
          f"repair_m p50={st.median([r['repair_m'] for r in rep]):.2f} max={max(r['repair_m'] for r in rep):.2f} m")
    by_branch = {}
    for r in rep: by_branch.setdefault(r["branch"], []).append(r["repair_m"])
    for b, v in by_branch.items():
        print(f"  {b:<18} n={len(v):<5} p50={st.median(v):.2f} max={max(v):.2f}")
    onto_robot = [r for r in rep if r["d_tgt"] < 0.35 and r["d_wire"] > 1.0]
    if onto_robot:
        print(f"  ★{len(onto_robot)} repairs pulled a standpoint {st.median([r['d_wire'] for r in onto_robot]):.2f} m away "
              f"to within {max(r['d_tgt'] for r in onto_robot):.2f} m of the ROBOT — a repair that reaches the robot "
              f"converts 'unreachable' into 'arrived'.  reasons: {set(r['detail'] for r in onto_robot)}")

if facts:
    kinds = {}
    for r in facts: kinds[r["event"]] = kinds.get(r["event"], 0) + 1
    print(f"\nFACTS REPORTED TO THE PRODUCER: {kinds}")
    print(f"  the standpoints they refer to were a median "
          f"{st.median([r['d_wire'] for r in facts]):.2f} m from the robot — "
          f"reported, not driven to, and NOT counted as observed")
    for why in sorted(set(r["detail"] for r in facts)):
        print(f"    · {why}")

if arr:
    real = [r for r in arr if r["arrival_real"] == 1]
    print(f"\nARRIVAL: {len(real)}/{len(arr)} passed the arrival guard")
    phantom = [r for r in real if r["d_wire"] > 1.0]
    print(f"  d_tgt  p50={st.median([r['d_tgt'] for r in arr]):.2f} m   (distance to OUR target)")
    print(f"  d_wire p50={st.median([r['d_wire'] for r in arr]):.2f} m   (distance to the PRODUCER's cell)")
    if phantom:
        print(f"  ★★★{len(phantom)} accepted arrivals were {st.median([r['d_wire'] for r in phantom]):.2f} m "
              f"(median) from the published cell — the producer is being told it was visited.")
    noplan = [r for r in arr if r["branch"] == "no-path"]
    if noplan:
        print(f"  ★{len(noplan)} arrivals were reported with NO ACTIVE PATH — compute() returns "
              f"goal_reached=true on an empty path; that is not an arrival.")

for r in rows[-tail:] if tail else []:
    print(f"  {r['t_ms']} {r['event']:<8} wire=({r['wire_x']:6.2f},{r['wire_y']:6.2f}) "
          f"tgt=({r['tgt_x']:6.2f},{r['tgt_y']:6.2f}) rob=({r['rob_x']:6.2f},{r['rob_y']:6.2f}) "
          f"d_wire={r['d_wire']:5.2f} d_tgt={r['d_tgt']:5.2f} rep={r['repair_m']:5.2f} "
          f"{r['branch']}/{r['detail']} new={r['target_new']} real={r['arrival_real']}")
