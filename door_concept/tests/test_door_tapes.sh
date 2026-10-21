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
#   D3  NOT A DEFECT, asserted so it stays documented: tourR2's births carry the names the identity table loaded for those
#       apertures (door_4 IS DOOR_1's remembered name in the 4-row table .bak_tourP9; door_2 is DOOR_0's).
# The tapes are 1.5 GB each and live outside the repo: DOOR_TAPES (default: the main checkout's crossing dir). Missing
# tapes ⇒ SKIPPED (exit 0). Each replay takes ~2-4 min; all ten run in parallel (~10 cores).
# Run (from anywhere):  door_concept/tests/test_door_tapes.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAPES="${DOOR_TAPES:-/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing}"
for t in tourR1 tourR2; do
    if [[ ! -f "$TAPES/$t/door_inputs.bin" || ! -f "$TAPES/$t/door_decisions.bin" ]]; then
        echo "SKIPPED: $TAPES/$t tapes not found (set DOOR_TAPES)"; exit 0
    fi
done
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
cd "$HERE" || exit 2
g++ -std=c++23 -O2 -w -I/usr/include/eigen3 -I/usr/include/opencv4 tools/door_replay.cpp src/core/door_core.cpp \
    src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp \
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
wait
python3 -I - "$OUT" <<'EOF'
import csv, math, re, statistics, sys
out = sys.argv[1]
PRIOR_S_STD = 0.60          # Door.PriorAlongWallStd: the first update's prior σ along the wall
# (tape, kind, cycle, name) of a RECORDED decision the fixed core no longer takes, and why. Nothing may appear.
ALLOWED = {
    ("tourR2", "REMOVE", 3629, "door_2"): "D1 — the real DOOR_0, removed for a silhouette its stretched width (2.51 m) predicted",
    ("tourR1", "REMOVE", 3613, "door_4"): "D2 — door_4 had jumped 3.66 m off DOOR_0 on its first update; it now stays on the doorway",
    ("tourR1", "BIRTH", 3448, "door_5"): "D2 — DOOR_0's detection now associates to door_4, which holds the doorway, so no second door is born",
}
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
    for k in gone:
        why = ALLOWED.get((tape, k[2], k[1], k[3]))
        print(f"{label} decision no longer taken: {k[0][:110]}  — {why or 'NOT ALLOWED'}")
        if not why: fails.append(f"{label}: decision changed: {k[0][:110]}")
    for k in new:
        print(f"{label} NEW decision: {k[0][:110]}")
        fails.append(f"{label}: new decision: {k[0][:110]}")
    for (t, kind, cyc, name), why in ALLOWED.items():
        if t == tape and not any(k[2] == kind and k[1] == cyc and k[3] == name for k in gone):
            fails.append(f"{label}: the allowed change {kind} {name} @{cyc} did not happen ({why[:40]}…)")
    return ref, rep

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
for f in fails: print("FAIL:", f)
print("tape regressions:", "PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
EOF
