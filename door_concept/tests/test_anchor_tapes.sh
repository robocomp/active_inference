#!/usr/bin/env bash
# test_anchor_tapes.sh — Task D-B tape regressions for the PASSAGE ANCHOR, on rtD1h (tools/anchor_replay: the anchor re-run
# from door_inputs.bin + door_decisions.bin; the door's geometry and leaf are RECONSTRUCTED from the run's log — width 1.00,
# thickness 0.10, the leaf of each leg's "carries its leaf" line — see anchor_replay.cpp). Scored against the Webots truth.
#   I5  leg 5's hang (cycle 5164, live t 445.28 s): the anchor's FIRST update must not accept the alias. Before the fix the
#       same reconstruction jumped 23–90° on it (thickness 0.04–0.20; 0.10: yaw −42°, live: +84°); now the update's own
#       fit-vs-prior test (alias_log_ratio) keeps the prior: P(fit) < 0.05 and the pose within 0.2 m / 5° of the truth.
#   LOST  legs 1, 3, 5 (each had its tracker captured somewhere in the reconstruction) and leg 2 (a correct anchor): the
#       PUBLISHED pose (the lost check's two modes combined) has median error ≤ 0.10 m / 3° and worst yaw ≤ 20°; leg 2's
#       P(lost) never reaches 0.5 (a correct anchor is not re-initialised). Reported per leg: the FRACTION of updates with
#       P(lost) > 0.5 — the time the published pose is mostly the room initialisation's (σ ≈ 0.3 m): the fallback's cost.
#   ⚠ The replay's room initialisation is TRUTH-CARRIED: the hang's room init fixed in the world and brought into each later
#     robot frame by the Webots pose (anchor_replay.cpp). It holds the live room-frame error at the hang and NO later
#     localisation noise or relocalisation jump — live, the room channel is noisier than here.
# Missing tapes ⇒ SKIPPED (exit 0). ~3–6 min (the legs run in parallel).
# Run (from anywhere):  door_concept/tests/test_anchor_tapes.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${ANCHOR_TAPES:-/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/rtD1h}"
for f in door_inputs.bin door_decisions.bin gt_error.csv; do
    [[ -f "$D/$f" ]] || { echo "SKIPPED: $D/$f not found (set ANCHOR_TAPES)"; exit 0; }
done
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
cd "$HERE" || exit 2
Q=/usr/include/x86_64-linux-gnu/qt5
g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore tools/anchor_replay.cpp src/core/door_cycle_io.cpp \
    -o "$OUT/anchor_replay" -lzstd -lfastdds -lfastcdr -lQt5Core || { echo "BUILD FAIL"; exit 2; }
LEAF=("0.42,-1,1,92,9.6" "0.42,-1,1,95,0.5" "-0.45,1,-1,90,0.5" "0.42,-1,1,73,10.4" "-0.42,1,-1,90,0.5")
R="$OUT/anchor_replay $D/door_inputs.bin $D/door_decisions.bin --gt $D/gt_error.csv --thickness 0.10"
$R --hang 4 --frames 1 --no-lost --leaf="${LEAF[4]}" > "$OUT/i5.txt" 2>&1 &
for h in 0 1 2 4; do $R --hang $h --leaf="${LEAF[$h]}" > "$OUT/leg$h.txt" 2>&1 & done
wait
python3 -I - "$OUT" <<'EOF'
import re, sys
out = sys.argv[1]
fails = []
def check(ok, what):
    print(("  [PASS] " if ok else "  [FAIL] ") + what)
    if not ok: fails.append(what)
first = [l for l in open(f"{out}/i5.txt") if l.startswith("FIRST")]
if not first: check(False, "I5: leg 5's first update replayed")
else:
    l = first[0]; print("  " + l.strip())
    pf = float(re.search(r"P\(fit\) ([\d.]+)", l).group(1))
    m = re.search(r"GT err lat ([-+\d.]+) yaw ([-+\d.]+)", l)
    lat, yaw = float(m.group(1)), float(m.group(2))
    check(pf < 0.05, f"I5: the alias fit is not accepted (P(fit) {pf:.3f} < 0.05)")
    check(abs(lat) < 0.2 and abs(yaw) < 5.0, f"I5: the anchor stays on the doorway after its first update (lat {lat:+.2f} m, yaw {yaw:+.1f} deg)")
for h, name in [(0, "leg 1"), (1, "leg 2"), (2, "leg 3"), (4, "leg 5")]:
    txt = open(f"{out}/leg{h}.txt").read()
    m = re.search(r"SUMMARY .* published lat_med ([\d.]+) lat_max ([\d.]+) yaw_med ([\d.]+) yaw_max ([\d.]+)", txt)
    if not m: check(False, f"{name}: replayed"); continue
    lm, lx, ym, yx = map(float, m.groups())
    pl_all = [float(x) for x in re.findall(r"P\(lost\) ([\d.]+)", txt)]
    frac = sum(1 for p in pl_all if p > 0.5) / max(1, len(pl_all))
    print(f"  {name}: published lat med {lm:.3f} max {lx:.3f} m, yaw med {ym:.1f} max {yx:.1f} deg; P(lost) > 0.5 on {100 * frac:.0f} % of {len(pl_all)} updates (room-init fallback)")
    check(lm <= 0.10 and ym <= 3.0 and yx <= 20.0, f"{name}: published median ≤ 0.10 m / 3 deg, worst yaw ≤ 20 deg")
    if h == 1:
        pl = [float(x) for x in re.findall(r"P\(lost\) ([\d.]+)", txt)]
        check(pl and max(pl) < 0.5, f"leg 2 (a correct anchor): P(lost) never reaches 0.5 (max {max(pl) if pl else -1:.3f})")
print("anchor tape regressions: " + ("PASS" if not fails else f"{len(fails)} FAILED"))
sys.exit(1 if fails else 0)
EOF
