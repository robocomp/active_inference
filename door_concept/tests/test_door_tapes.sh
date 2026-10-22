#!/usr/bin/env bash
# test_door_tapes.sh — Task 5b tape regressions: the DoorCore replayed over the recorded tours tourR1 / tourR2
# (tools/door_replay), checked for the tour-blocking door defects.
#   DECISIONS  every decision the core takes (BIRTH, SUPPRESS, MERGE, REACQUIRE, REMOVE) equals the recorded log's, except the
#       ones listed in ALLOWED below, each with its reason. A decision that disappears or appears anywhere else FAILS.
#   D1  tourR2: door_2 (the real DOOR_0, born at cycle 3601) is not removed (it was, at 3629, after one frame stretched its
#       width 1.01 → 2.51 m), AND its silhouette reads it OCCUPIED: on every cycle the tracker hands it a full DOOR_0 mask
#       (≥ 20000 px — a smaller slice is the detector boxing part of the leaf) occ > free, and the median occ/free ≥ 2.
#   D2  both tapes: no mask update moves the aperture by more than 3 σ of what it predicted before the update — the first
#       update against the birth seed (Door.PriorAlongWallStd 0.60 m), every later one against the previous row's σ.
#   D4  the image-geometry existence channel (leaf contour + depth + doorway frame) is pooled by view novelty
#       (door_view_pool.h): with the model's own prior no door on these tapes — every one is a real door after D1/D2 —
#       reaches the −4 floor (it walked tourR1 door_4 there for 45 cycles), nor with Door.RecessPriorStd 0.1, 0.2, 0.3;
#       and the DECISIONS do not move for 0.03, 0.1, 0.2, 0.3. (At 0.03 the prior pins the leaf 4 σ from where it is —
#       a mis-specified model — and tourR2's door_2 still reaches the floor, without a removal: printed, not asserted.)
#   SMOKE  the replay also runs the live-only belief consumers (the NBV planner, the DOF table, the reported Σ) on every
#       door every cycle, built WITHOUT -DNDEBUG: an index error aborts it, as it aborted the agent (rtD1g). Asserted:
#       each replay prints its "live-only belief consumers … ran N times, clean" line with N > 0.
#   D-C  rtD1i (to cycle 715) and tour3A (to cycle 3768): the windows in which the replay is FAITHFUL to the live run — the
#       affordance/anchor slice the core does not rebuild first acts at 715 (rtD1i: aff_door_1_approach EXECUTING) and 3769
#       (tour3A: the first actuator OPEN). Every decision in them must equal the live log's: in particular rtD1i's real
#       DOOR_0 (door_5) must not be REMOVED at cycle 614 while YOLO detects it (Task D-C review C1).
#   D-C round 2  tour3C (to cycle 870): its faithful window — aff_door_2_approach first EXECUTES at 871. Every decision in it
#       must equal the live log's, and door_2 (the real DOOR_0, born at 426) must not be REMOVED. ⚠ The tour-blocking
#       removal of tour3C (the RESTORED DOOR_0, cycle 3917) lies far outside the window: restores happen in SpecificWorker
#       and the replay does not rebuild them (nor the anchor/affordance slices that diverge from 871 on). Its mechanism —
#       a restored aperture with the birth seed's σ_s taking the open stretch past its wall's end for its opening — is
#       covered by tests/test_door_along_wall_extent.cpp and test_door_core_existence §12b.
#   D3  NOT A DEFECT, asserted so it stays documented: tourR2's births carry the names the identity table loaded for those
#       apertures (door_4 IS DOOR_1's remembered name in the 4-row table .bak_tourP9; door_2 is DOOR_0's).
#   W   (task W, ab7_A1 / ab7_C1 / ab7_A2 / ab7_C2; SKIPPED if those tapes are absent) the CROSSED door seen from the far room: the
#       replay's --mirror re-creates the entry mirror the live agent made at the room switch (SpecificWorker, not rebuilt by
#       the core) and the door's WIDTH, true 1.00 m, must stay within 1.5 prior σ (Door.PriorWidthStd 0.06 ⇒ 0.09 m) of it on
#       every cycle after the switch (MIRROR_PHI=1.553: the leaf the crossing left open, 89 deg — the live log's
#       "remembers its leaf from the crossing" — which the tape cannot carry: the actuator slice is not rebuilt). Before the fix the first small far mask (116 px against a 200 px predicted hull,
#       midpoints 124 px = 14 σ apart) took it to 0.75 / 0.75 / 0.47 / 0.83: the sides-midpoint factor had no free coordinate
#       left but w. (The approach offer's P(complete) falls from ~1 to 0 between w 0.85 and 0.75 m (the body 0.70 + margin, at the pose sigma) and follows the aperture on the way back, see
#       test_door_approach_pose §W, so a width that holds is an approach that stays offered.)
# The tapes are 1.5 GB each and live outside the repo: DOOR_TAPES (default: the main checkout's crossing dir). Missing
# tapes ⇒ SKIPPED (exit 0). Each replay takes ~2-4 min; all fifteen run in parallel (~15 cores).
# Run (from anywhere):  door_concept/tests/test_door_tapes.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAPES="${DOOR_TAPES:-/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing}"
for t in tourR1 tourR2 rtD1i tour3A tour3C; do
    if [[ ! -f "$TAPES/$t/door_inputs.bin" || ! -f "$TAPES/$t/door_decisions.bin" ]]; then
        echo "SKIPPED: $TAPES/$t tapes not found (set DOOR_TAPES)"; exit 0
    fi
done
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
cd "$HERE" || exit 2
g++ -std=c++23 -O2 -w -I/usr/include/eigen3 -I/usr/include/opencv4 tools/door_replay.cpp src/core/door_core.cpp \
    src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp \
    src/epistemic_planner.cpp \
    src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
    ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc -o "$OUT/door_replay" || { echo "BUILD FAIL"; exit 2; }
for t in tourR1 tourR2; do
    mkdir -p "$OUT/$t"
    "$OUT/door_replay" "$TAPES/$t/door_inputs.bin" "$TAPES/$t/door_decisions.bin" --csv "$OUT/$t" --trace '*' > "$OUT/$t.txt" 2>&1 &
    for sd in 0.03 0.1 0.2 0.3; do
        "$OUT/door_replay" "$TAPES/$t/door_inputs.bin" "$TAPES/$t/door_decisions.bin" --recess-prior-std $sd --trace '*' \
            > "$OUT/$t@$sd.txt" 2>&1 &
    done
done
for tu in rtD1i:715 tour3A:3768 tour3C:870; do
    t="${tu%%:*}"; u="${tu##*:}"
    "$OUT/door_replay" "$TAPES/$t/door_inputs.bin" "$TAPES/$t/door_decisions.bin" --until "$u" --trace '*' > "$OUT/$t.txt" 2>&1 &
done
W_TAPES=(ab7_A1:2500 ab7_C1:2500 ab7_A2:3400 ab7_C2:5700)
for tu in "${W_TAPES[@]}"; do
    t="${tu%%:*}"; u="${tu##*:}"
    [[ -f "$TAPES/$t/door_inputs.bin" && -f "$TAPES/$t/door_decisions.bin" ]] || continue
    MIRROR_PHI=1.553 "$OUT/door_replay" "$TAPES/$t/door_inputs.bin" "$TAPES/$t/door_decisions.bin" --mirror --until "$u" --trace '*' > "$OUT/W_$t.txt" 2>&1 &
done
# Task E: ab7_A1 with the Task-W midpoint term ABLATED (the collapse probe) — the per-cycle prices of the door's epistemic look
# and of its approach (--look), on one scale (nats).
if [[ -f "$TAPES/ab7_A1/door_inputs.bin" && -f "$TAPES/ab7_A1/door_decisions.bin" ]]; then
    MIRROR_PHI=1.553 "$OUT/door_replay" "$TAPES/ab7_A1/door_inputs.bin" "$TAPES/ab7_A1/door_decisions.bin" --mirror --until 2500 \
        --ablate-hull-midpoint --look > "$OUT/E_ab7_A1.txt" 2>&1 &
fi
wait
python3 -I - "$OUT" <<'EOF'
import csv, math, os, re, statistics, sys
out = sys.argv[1]
PRIOR_S_STD = 0.60          # Door.PriorAlongWallStd: the first update's prior σ along the wall
# (tape, kind, cycle, name) of a RECORDED decision the fixed core no longer takes, and why. Nothing may appear.
ALLOWED = {
    ("tourR2", "REMOVE", 3629, "door_2"): "D1 — the real DOOR_0, removed for a silhouette its stretched width (2.51 m) predicted",
    ("tourR1", "REMOVE", 3613, "door_4"): "D2 — door_4 had jumped 3.66 m off DOOR_0 on its first update; it now stays on the doorway",
    ("tourR1", "BIRTH", 3448, "door_5"): "D2 — DOOR_0's detection now associates to door_4, which holds the doorway, so no second door is born",
}
# Task Q — RICOH door slices now reach the tracker, so a birth/suppression the recorded (ZED-only) run took may be taken a few
# cycles EARLIER (the ricoh's evidence accrues to the same candidate) and at a slightly different position (the ricoh's
# wall-fused range is blended in). That is a PAIR (a recorded decision gone + its replacement new), never an unexplained one:
# same kind (and, for SUPPRESS, same note up to its numbers), the replay at most 3 cycles later and Q_CYCLES earlier than the reference,
# within Q_DIST metres. Anything unpaired still FAILS.
Q_CYCLES = {"BIRTH": 15, "SUPPRESS": 15}
Q_DIST = {"BIRTH": 0.25, "SUPPRESS": 0.12}   # a ricoh-seeded birth sits up to ~one bearing-lever (sigma_b r ~ 0.1 m, 2 sigma) off the ZED one; the fit then corrects it
# Explicit, reasoned Task-Q changes beyond the pairing rule (the shut DOOR_0 is born from the ricoh long before the ZED birth).
Q_GONE = {("tourR2", "BIRTH", 3601, "door_2"): "Q - the real DOOR_0 is now born earlier from the ricoh (see Q_NEW)"}
Q_NEW = {("tourR2", "BIRTH", 3489, "door_2"): "Q - the shut DOOR_0 born 112 cycles before the recorded ZED birth (3601), 0.11 m from it"}
def q_pair(gone, new):
    """Pair each gone decision with a new one satisfying the Task-Q rule; returns (unpaired gone, unpaired new, pairs)."""
    pairs, new = [], list(new)
    rest = []
    for g in gone:
        kind = g[2]
        ok = None
        if kind in Q_CYCLES:
            mg = re.search(r" a=(\S+) b=(\S+)", g[0]); ng = re.search(r"note='([^']*)'", g[0])
            for n in new:
                if n[2] != kind: continue
                if not (-3 <= g[1] - n[1] <= Q_CYCLES[kind]): continue
                mn = re.search(r" a=(\S+) b=(\S+)", n[0]); nn = re.search(r"note='([^']*)'", n[0])
                if kind == "SUPPRESS" and re.sub(r"[0-9.]+", "#", ng.group(1) if ng else "") != re.sub(r"[0-9.]+", "#", nn.group(1) if nn else ""): continue
                d = math.hypot(float(mg.group(1)) - float(mn.group(1)), float(mg.group(2)) - float(mn.group(2)))
                if d <= Q_DIST[kind]: ok = n; break
        if ok is not None: new.remove(ok); pairs.append((g, ok))
        else: rest.append(g)
    # SUPPRESS is a log-only diagnostic (it creates and removes nothing): an unpaired one is reported, not failed.
    for lst in (rest, new):
        for k in [k for k in lst if k[2] == "SUPPRESS"]:
            print(f"Q: unpaired SUPPRESS (log only, allowed): {k[0][:100]}")
            lst.remove(k)
    return rest, new, pairs
fails = []
DEC = re.compile(r"(?:EQUAL|ref|replay)\s+(cycle (\d+) t \d+ (BIRTH|SUPPRESS|MERGE|REMOVE|REACQUIRE|RELEASE|RESTORE) id \d+ '([^']*)'.*)$")

def decisions(txt):
    """(ref, replay) lists of the decision strings the replay printed (EQUAL lines count for both)."""
    ref, rep = [], []
    for line in open(txt):
        s = line.strip()
        m = DEC.search(s)
        if not m: continue
        key = (m.group(1), int(m.group(2)), m.group(3), m.group(4))
        if s.startswith("EQUAL"): ref.append(key); rep.append(key)
        elif s.startswith("ref"): ref.append(key)
        elif s.startswith("replay"): rep.append(key)
    return ref, rep

def floor_check(tape, txt, assert_floor=True):
    """D4: the lowest L every door reached; a real door at the -4 floor FAILS."""
    low = {}
    for line in open(txt):
        m = re.match(r"TRACE (\S+) cycle (\d+) L (\S+)", line)
        if m: low[m.group(1)] = min(low.get(m.group(1), 9.0), float(m.group(3)))
    print(f"{tape} D4 lowest L per door: " + ", ".join(f"{d} {v:+.2f}" for d, v in sorted(low.items())))
    if not low: fails.append(f"D4 {tape}: no TRACE rows (did --trace run?)")
    for d, v in low.items():
        if v <= -3.99 and assert_floor: fails.append(f"D4 {tape}: {d}, a real door, reached the L floor ({v:+.2f})")

def decision_check(tape, txt, label):
    ref, rep = decisions(txt)
    if not ref: fails.append(f"{label}: the replay printed no decisions (did it run?)")
    gone = [k for k in ref if k not in rep]
    new = [k for k in rep if k not in ref]
    gone, new, qpairs = q_pair(gone, new)
    for g, n in qpairs:
        print(f"{label} Q-paired: {g[2]} {g[3] or ''} ref cycle {g[1]} -> replay cycle {n[1]} (the ricoh evidence; within the Task-Q rule)")
    for k in gone:
        why = ALLOWED.get((tape, k[2], k[1], k[3])) or Q_GONE.get((tape, k[2], k[1], k[3]))
        print(f"{label} decision no longer taken: {k[0][:110]}  — {why or 'NOT ALLOWED'}")
        if not why: fails.append(f"{label}: decision changed: {k[0][:110]}")
    for k in new:
        if Q_NEW.get((tape, k[2], k[1], k[3])):
            print(f"{label} NEW decision (allowed): {k[0][:100]} - {Q_NEW[(tape, k[2], k[1], k[3])]}")
            continue
        print(f"{label} NEW decision: {k[0][:110]}")
        fails.append(f"{label}: new decision: {k[0][:110]}")
    for (t, kind, cyc, name), why in ALLOWED.items():
        if t == tape and not any(k[2] == kind and k[1] == cyc and k[3] == name for k in gone):
            fails.append(f"{label}: the allowed change {kind} {name} @{cyc} did not happen ({why[:40]}…)")
    return ref, rep

import glob
for txt in sorted(glob.glob(f"{out}/*.txt")):
    m = [re.search(r"live-only belief consumers .* ran (\d+) times, clean", l) for l in open(txt)]
    m = [x for x in m if x]
    if not m or int(m[0].group(1)) == 0: fails.append(f"SMOKE {txt.split('/')[-1]}: the live-only belief consumers did not run clean (crash?)")
    else: print(f"SMOKE {txt.split('/')[-1]}: live-only belief consumers ran {m[0].group(1)} times, clean")
for tape in ("rtD1i", "tour3A", "tour3C"):   # D-C: the faithful windows — decisions equal, no real door at the floor
    ref, rep = decision_check(tape, f"{out}/{tape}.txt", f"{tape} (faithful window)")
    if tape == "tour3C":   # D-C round 2: the real DOOR_0 is door_2 on this tape
        gone = [k for k in rep if k[2] == "REMOVE" and k[3] == "door_2"]
        if gone: fails.append(f"tour3C: door_2 (the real DOOR_0) REMOVED in the faithful window: {gone[0][0][:100]}")
        if not any(k[2] == "BIRTH" and k[3] == "door_2" for k in rep): fails.append("tour3C: door_2 never born (did the replay run?)")
for tape in ("tourR1", "tourR2"):
    for sd in ("0.03", "0.1", "0.2", "0.3"):
        decision_check(tape, f"{out}/{tape}@{sd}.txt", f"{tape} (RecessPriorStd {sd})")
        floor_check(f"{tape} (RecessPriorStd {sd})", f"{out}/{tape}@{sd}.txt", assert_floor=(sd != "0.03"))
    txt = f"{out}/{tape}.txt"
    floor_check(tape, txt)
    ref, rep = decision_check(tape, txt, tape)

    # D2: per-update jumps against the prior predictive
    bpos = {}
    for line in open(txt):
        m = re.search(r"cycle (\d+) t \d+ BIRTH id \d+ '([^']+)' a=(\S+) b=(\S+)", line)
        if m and m.group(2) not in bpos: bpos[m.group(2)] = (float(m.group(3)), float(m.group(4)))
    prev, worst = {}, {}
    for r in csv.DictReader(open(f"{out}/{tape}/ai2_log.csv")):
        d = r["node"]; s, w = float(r["s"]), float(r["w"])
        if d not in prev:
            if d in bpos:
                z = math.hypot(float(r["cx"]) - bpos[d][0], float(r["cy"]) - bpos[d][1]) / PRIOR_S_STD
                worst[d] = max(worst.get(d, (0, "")), (z, f"first update {bpos[d][0]:.2f},{bpos[d][1]:.2f} -> {float(r['cx']):.2f},{float(r['cy']):.2f}"))
        else:
            ps, pw, sds, sdw = prev[d]
            zs, zw = abs(s - ps) / max(sds, 1e-6), abs(w - pw) / max(sdw, 1e-6)
            worst[d] = max(worst.get(d, (0, "")), (zs, f"row {r['cycle']} s {ps:.3f}->{s:.3f}"), (zw, f"row {r['cycle']} w {pw:.3f}->{w:.3f}"))
        prev[d] = (s, w, float(r["std_s"]), float(r["std_w"]))
    for d, (z, what) in sorted(worst.items()):
        print(f"{tape} {d}: worst update {z:.1f} sigma ({what})")
        if z > 3.0: fails.append(f"D2 {tape} {d}: an update moved the aperture {z:.1f} sigma ({what})")

    if tape == "tourR2":
        # D1: the silhouette on door_2's full-mask cycles
        rows = []
        for line in open(txt):
            m = re.search(r"^TRACE door_2 cycle (\d+) .*since_det 0 mask \S+ px (\d+) \| sil occ (\S+) free (\S+)", line)
            if m and int(m.group(2)) >= 20000: rows.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
        print(f"tourR2 D1: door_2 full-mask cycles (occ/free): " + " ".join(f"{c}:{o:.0f}/{f:.0f}" for c, o, f in rows))
        if len(rows) < 5: fails.append(f"D1 tourR2: only {len(rows)} full-mask cycles for door_2 (did it live?)")
        bad = [(c, o, f) for c, o, f in rows if c > 3601 and not o > f]
        if bad: fails.append(f"D1 tourR2: door_2's silhouette read free >= occ on a full mask: {bad}")
        ratios = [o / max(f, 1.0) for c, o, f in rows if c > 3601]
        if ratios and statistics.median(ratios) < 2.0: fails.append(f"D1 tourR2: median occ/free {statistics.median(ratios):.2f} < 2")
        for name, (x, y) in {"door_4": (-0.5436, 4.6060), "door_2": (0.3516, 2.5360)}.items():
            ok = name in bpos and math.hypot(bpos[name][0] - x, bpos[name][1] - y) < 0.5
            print(f"tourR2 D3: '{name}' born {'at %.2f,%.2f' % bpos[name] if name in bpos else 'NEVER'} — the table's row for ({x},{y})")
            if not ok: fails.append(f"D3 tourR2: no birth named '{name}' at its remembered aperture ({x},{y})")
# W: the crossed door's width after the room switch (see the header)
PRIOR_W_STD = 0.06          # Door.PriorWidthStd
for tape in ("ab7_A1", "ab7_C1", "ab7_A2", "ab7_C2"):
    txt = f"{out}/W_{tape}.txt"
    if not os.path.exists(txt):
        print(f"W {tape}: SKIPPED (tape not found)"); continue
    mirrors, low = [], {}
    for line in open(txt):
        m = re.match(r"MIRROR cycle (\d+) '(\S+)'", line)
        if m: mirrors.append((int(m.group(1)), m.group(2))); continue
        m = re.match(r"TRACE (\S+) cycle (\d+) L \S+ \| s \S+ w (\S+)", line)
        if m:
            for mc, mn in mirrors[-1:]:
                if m.group(1) == mn and int(m.group(2)) > mc: low[(mc, mn)] = min(low.get((mc, mn), 9.0), float(m.group(3)))
    if not mirrors: fails.append(f"W {tape}: the replay made no entry mirror (did it run?)"); continue
    for key, v in sorted(low.items()):
        print(f"W {tape} mirror '{key[1]}' @{key[0]}: lowest width {v:.3f} m (true 1.00, tolerance {1.5 * PRIOR_W_STD:.2f})")
        if v < 1.0 - 1.5 * PRIOR_W_STD: fails.append(f"W {tape} '{key[1]}' @{key[0]}: width fell to {v:.3f} m")
    if not low: fails.append(f"W {tape}: no trace rows for the mirror")
# ── Task E: a poor door estimate makes the EPISTEMIC look beat the goal-directed approach (ab7_A1, Task-W term ablated) ──
# Each LOOK row: the NBV gain on the belief's REPORTED sigma (nats), the approach's value (ValueApproach x P(complete under the
# reported width sigma), nats), the look viewpoint's angle off the door normal and distance. After the entry mirror's first mask
# update the reported sigma_w is > 0.1 m (the mask disagrees with the hull): from there the look must out-price the approach,
# the approach must stay OFFERED (value > 0, never withdrawn into a dead end), and every look viewpoint is frontal.
_e = f"{out}/E_ab7_A1.txt"
if os.path.exists(_e):
    rows = []
    for ln in open(_e):
        m = re.match(r"LOOK (\S+) cycle (\d+) w ([\d.]+) sd_w_post ([\d.]+) sd_w_rep ([\d.]+) \| look ([\d.]+) nats at ([\d.]+) m ([\d.]+) deg off the normal \(valid (\d)\) \| approach ([\d.]+)", ln)
        if m: rows.append((int(m[2]), float(m[3]), float(m[5]), float(m[6]), float(m[7]), float(m[8]), int(m[9]), float(m[10])))
    mir = re.search(r"MIRROR cycle (\d+)", open(_e).read())
    if not rows or not mir: fails.append("E ab7_A1: the replay printed no LOOK rows / no mirror")
    else:
        post = [r for r in rows if r[0] >= int(mir[1])]
        honest = [r for r in post if r[2] > 0.10]
        if not honest: fails.append("E ab7_A1: the reported sigma_w never exceeded 0.10 m after the mirror (honest uncertainty missing)")
        else:
            wins = sum(1 for r in honest if r[3] > r[7])
            print(f"E ab7_A1: {len(honest)} cycles with reported sigma_w > 0.10 m (first @{honest[0][0]}): look > approach on {wins}; "
                  f"lowest approach value {min(r[7] for r in post):.3f}; look viewpoint {min(r[4] for r in post if r[6]):.2f}-{max(r[4] for r in post if r[6]):.2f} m, "
                  f"<= {max(r[5] for r in post if r[6]):.1f} deg off the normal")
            if honest[0][3] <= honest[0][7]: fails.append(f"E ab7_A1: at the first honest-sigma cycle @{honest[0][0]} the look ({honest[0][3]:.3f}) does not out-price the approach ({honest[0][7]:.3f})")
            if wins < 0.95 * len(honest): fails.append(f"E ab7_A1: the look out-prices the approach on only {wins}/{len(honest)} honest-sigma cycles")
        if min(r[7] for r in post) <= 0.0: fails.append("E ab7_A1: the approach was WITHDRAWN (value 0) — a poor estimate must only lower it")
        if any(r[6] and r[5] > 15.0 for r in post): fails.append("E ab7_A1: a look viewpoint more than 15 deg off the door normal")
for f in fails: print("FAIL:", f)
print("tape regressions:", "PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
EOF
