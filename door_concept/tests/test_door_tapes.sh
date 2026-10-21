#!/usr/bin/env bash
# test_door_tapes.sh — Task 5b tape regressions: the DoorCore replayed over the recorded tours tourR1 / tourR2
# (tools/door_replay), checked for the three tour-blocking door defects.
#   D1  tourR2: door_2 (the real DOOR_0, born at cycle 3601) must NOT be removed. It was, at 3629 (L −4.00), because
#       one frame stretched its width 1.01 → 2.51 m and the silhouette then predicted a 2.5 m door over a 1 m doorway.
#   D2  both tapes: no mask update may move the aperture by more than 3 σ of what it predicted before the update —
#       the first update against the birth seed (Door.PriorAlongWallStd 0.60 m), every later one against the previous
#       row's posterior σ. tourR1 door_4's first update jumped 3.66 m along the wall (6.1 σ); tourR2 door_2's 7th
#       stretched w by 1.49 m (36 σ).
#   D3  NOT A DEFECT, asserted so it stays documented: tourR2's births carry the names the identity table loaded for
#       those apertures (door_4 IS DOOR_1's remembered name in the 4-row table .bak_tourP9; door_2 is DOOR_0's).
# The tapes are 1.5 GB each and live outside the repo: DOOR_TAPES (default: the main checkout's crossing dir). Missing
# tapes ⇒ SKIPPED (exit 0). Each replay takes ~2-3 min; both run in parallel.
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
    "$OUT/door_replay" "$TAPES/$t/door_inputs.bin" "$TAPES/$t/door_decisions.bin" --csv "$OUT/$t" > "$OUT/$t.txt" 2>&1 &
done
wait
python3 -I - "$OUT" <<'EOF'
import csv, math, re, sys
out = sys.argv[1]
PRIOR_S_STD = 0.60          # Door.PriorAlongWallStd: the first update's prior σ along the wall
fails = []

def births(txt):
    """BIRTH decisions the REPLAY made: name -> (cycle, x, y)."""
    b = {}
    for line in open(txt):
        m = re.search(r"(?:EQUAL|replay)\s+cycle (\d+) t \d+ BIRTH id \d+ '([^']+)' a=(\S+) b=(\S+)", line)
        if m: b.setdefault(m.group(2), (int(m.group(1)), float(m.group(3)), float(m.group(4))))
    return b

def removes(txt):
    return [l.strip() for l in open(txt) if re.search(r"(?:EQUAL|replay)\s+cycle \d+ t \d+ REMOVE", l)]

for tape in ("tourR1", "tourR2"):
    txt = f"{out}/{tape}.txt"
    B = births(txt)
    rows = list(csv.DictReader(open(f"{out}/{tape}/ai2_log.csv")))
    prev, worst = {}, {}
    for r in rows:
        d = r["node"]; s, w = float(r["s"]), float(r["w"])
        if d not in prev:
            # first mask update: the fitted aperture centre against the birth seed, in units of the prior σ_s
            if d in B:
                z = math.hypot(float(r["cx"]) - B[d][1], float(r["cy"]) - B[d][2]) / PRIOR_S_STD
                worst[d] = max(worst.get(d, (0, "")), (z, f"first update {B[d][1]:.2f},{B[d][2]:.2f} -> {float(r['cx']):.2f},{float(r['cy']):.2f}"))
        else:
            ps, pw, sds, sdw = prev[d]
            zs, zw = abs(s - ps) / max(sds, 1e-6), abs(w - pw) / max(sdw, 1e-6)
            worst[d] = max(worst.get(d, (0, "")), (zs, f"row {r['cycle']} s {ps:.3f}->{s:.3f}"), (zw, f"row {r['cycle']} w {pw:.3f}->{w:.3f}"))
        prev[d] = (s, w, float(r["std_s"]), float(r["std_w"]))
    for d, (z, what) in sorted(worst.items()):
        print(f"{tape} {d}: worst update {z:.1f} sigma ({what})")
        if z > 3.0: fails.append(f"D2 {tape} {d}: an update moved the aperture {z:.1f} sigma ({what})")
    for l in removes(txt): print(f"{tape} REMOVE: {l}")
    if tape == "tourR2":
        if any("'door_2'" in l for l in removes(txt)): fails.append("D1 tourR2: door_2 (the real DOOR_0) was removed")
        for name, (x, y) in {"door_4": (-0.5436, 4.6060), "door_2": (0.3516, 2.5360)}.items():
            ok = name in B and math.hypot(B[name][1] - x, B[name][2] - y) < 0.5
            print(f"tourR2 D3: '{name}' born {'at %.2f,%.2f' % B[name][1:] if name in B else 'NEVER'} — the table's row for ({x},{y})")
            if not ok: fails.append(f"D3 tourR2: no birth named '{name}' at its remembered aperture ({x},{y})")
for f in fails: print("FAIL:", f)
print("tape regressions:", "PASS" if not fails else f"{len(fails)} FAIL")
sys.exit(1 if fails else 0)
EOF
