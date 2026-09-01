#!/usr/bin/env bash
# Arm 6 — does episode LENGTH buy information, and where does linearisation break?
# Spec and pre-registered endpoints: EXPERIMENT.md, "Arm 6".
#
#   tools/arm6_setup.sh base|long     arm one leg (base = 0.20 rad, long = 1.00 rad)
#   tools/arm6_setup.sh check         is the RUNNING fleet running the armed config?
#   tools/arm6_setup.sh ratio         MUST read ~0.99 — this arm is UNINJECTED
#   tools/arm6_setup.sh save <leg>    file a finished leg, with the guards the earlier arms earned
#
# It never starts or stops anything: the user owns the lifecycle.
set -euo pipefail
BR=/home/pbustos/robocomp/components/webots-bridge/etc
RC=/home/pbustos/robocomp/components/active_inference/room_concept/etc
SRC=/home/pbustos/robocomp/components/active_inference/room_concept/tmp/sdf_localizer/gt_error.csv
ROT_BASE=0.20
ROT_LONG=1.00
die() { echo "arm6: $*" >&2; exit 1; }
leg_rot() { case "$1" in base) echo $ROT_BASE;; long) echo $ROT_LONG;; *) die "unknown leg '$1'";; esac; }

setkey() {
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
key, val, path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n, c = re.subn(r'^%s\s*=\s*\S+' % re.escape(key), '%s = %s' % (key, val), s, count=1, flags=re.M)
if c != 1: sys.exit("arm6: %s appears %d times in %s, expected once" % (key, c, path))
open(path, 'w').write(n)
PY
}

if [ "${1:-}" = check ]; then
  now=$(date +%s); ok=1
  probe() {
    local pid age start
    pid=$(pgrep -x "$1" | head -1) || true
    [ -z "$pid" ] && { printf "  %-18s NOT RUNNING\n" "$1"; ok=0; return; }
    age=$(ps -o etimes= -p "$pid" | tr -d " "); start=$((now - age))
    printf "  %-18s pid %-8s up %dh%02dm\n" "$1" "$pid" "$((age/3600))" "$(((age%3600)/60))"
    if [ "$(stat -c %Y "$2")" -gt "$start" ]; then
      printf "    ⚠ STALE — %s was written AFTER this process started; it is NOT using it.\n" "$2"; ok=0
    else printf "    config predates the process ✓\n"; fi
  }
  probe Webots2Robocomp "$BR/config.toml"
  probe room_concept    "$RC/config.toml"
  echo "  ── on disk (WheelScaleV MUST be 0.0 — this arm is UNINJECTED) ──"
  grep -H "^WheelScaleV" "$BR/config.toml" | sed "s|^|    |"
  grep -H "^MotionCalibEpisodeMinRot\|^MotionCalibEpisodeMinTrans\|^MotionCalibApply \|^MapMode\|^StableSdfMseMax" "$RC/config.toml" | sed "s|^|    |"
  # ★ the binary must actually READ the key: it was a header default until 2026-09-01.
  # ⚠ NOT `| grep -q`: grep exits on the first match, strings takes SIGPIPE, and `set -o pipefail`
  # turns that into a pipeline failure — a FALSE NEGATIVE on a safety check, which is the worst
  # kind. Count instead, and let the count be the answer.
  nkey=$(strings /home/pbustos/robocomp/components/active_inference/room_concept/bin/room_concept \
         | grep -c MotionCalibEpisodeMinRot || true)
  if [ "${nkey:-0}" -eq 0 ]; then
    echo "  ✗ THE BINARY DOES NOT CONTAIN MotionCalibEpisodeMinRot — it predates the key. Rebuild."; ok=0
  else echo "  binary reads MotionCalibEpisodeMinRot ✓"; fi
  [ "$ok" = 1 ] && echo "  agreement OK — necessary, not sufficient." || echo "  ✗ DO NOT DRIVE."
  exit 0
fi

if [ "${1:-}" = ratio ]; then
  python3 - "$SRC" <<'PY'
import csv, math, sys
gt=[]; odo=0.0
for r in csv.DictReader(open(sys.argv[1])):
    try:
        gt.append((float(r['gt_x']), float(r['gt_y'])))
        odo += math.hypot(float(r['dx_local']), float(r['dy_local']))
    except (TypeError, ValueError, KeyError): pass
if len(gt) < 2: sys.exit("  no usable rows yet")
g=sum(math.dist(a,b) for a,b in zip(gt,gt[1:]))
print("  %d rows | GT %.2f m | odometry %.2f m" % (len(gt), g, odo))
if len(set(p[0] for p in gt)) == 1: sys.exit("  GROUND TRUTH CONSTANT — parked, or no robot_gt_*.")
if g < 5.0: sys.exit("  only %.2f m — does not resolve below ~10 m." % g)
r=odo/g
print("  ratio odo/GT = %.4f  ->  %s" % (r,
      "CLEAN ✓ uninjected" if r < 1.03 else
      "⚠ STOP: reads INJECTED. The bridge is on an arm-4 config." if r > 1.06 else "AMBIGUOUS"))
PY
  exit 0
fi

if [ "${1:-}" = save ]; then
  leg="${2:-}"; case "$leg" in base|long) ;; *) die "usage: $0 save {base|long}";; esac
  dst="$RC/runs/arm6_$leg.csv"; mkdir -p "$RC/runs"
  [ -f "$SRC" ] || die "no $SRC"
  want=$(leg_rot "$leg")
  have=$(sed -n "s/^MotionCalibEpisodeMinRot *= *\([0-9.]*\).*/\1/p" "$RC/config.toml" | head -1)
  [ "$have" = "$want" ] || die "leg $leg needs MotionCalibEpisodeMinRot=$want, config says $have."
  inj=$(sed -n "s/^WheelScaleV *= *\([0-9.]*\).*/\1/p" "$BR/config.toml" | head -1)
  [ "$inj" = "0.0" ] || die "this arm is UNINJECTED but the bridge says WheelScaleV=$inj."
  other="$RC/runs/arm6_$([ "$leg" = base ] && echo long || echo base).csv"
  [ -f "$other" ] && cmp -s "$SRC" "$other" && die "byte-identical to the other leg — same run twice."
  [ -f "$dst" ] && [ ! "$SRC" -nt "$dst" ] && die "$SRC is not newer than $dst. Did the leg run?"
  pgrep -x room_concept >/dev/null && die "room_concept is RUNNING — it is still appending."
  cp -v "$SRC" "$dst"
  cp -v "$RC/motion_calib_state.csv" "$RC/runs/arm6_${leg}_episodes.csv" 2>/dev/null || \
    echo "  ⚠ no motion_calib_state.csv — the EPISODE log is this arm's primary evidence!"
  python3 - "$dst" <<'PY'
import csv, math, sys
xs=[]
for r in csv.DictReader(open(sys.argv[1])):
    try: xs.append((float(r['gt_x']), float(r['gt_y'])))
    except (KeyError, TypeError, ValueError): pass
d=sum(math.dist(a,b) for a,b in zip(xs,xs[1:]))
print("  %d rows, ground-truth path %.1f m%s" % (len(xs), d,
      "   ⚠ SHORT — target ~220 m to match arm 3R." if d < 150 else ""))
PY
  exit 0
fi

leg="${1:-}"; case "$leg" in base|long) ;; *) die "usage: $0 {base|long|check|ratio|save <leg>}";; esac
pgrep -x room_concept >/dev/null && die \
  "room_concept is RUNNING. Stop it first — it rewrites motion_calib_state.csv from its warm window."

if cmp -s "$BR/config.toml.pre-arm2" "$BR/config.toml"; then
  echo "bridge already uninjected — untouched, no bridge restart needed"
else
  pgrep -x Webots2Robocomp >/dev/null && die \
    "the bridge carries an injection and must be restored; stop Webots2Robocomp first."
  cp -v "$BR/config.toml.pre-arm2" "$BR/config.toml"
  echo "  ⚠ bridge config CHANGED to uninjected — RESTART IT before driving"
fi

# ★ ONLY the rotation trigger varies. Everything else is pinned to arm 3R's conditions so the two
# arms are comparable, and so a difference cannot be attributed to anything but the trigger.
setkey MotionCalibEpisodeMinRot   "$(leg_rot "$leg")" "$RC/config.toml"
setkey MotionCalibEpisodeMinTrans 0.25      "$RC/config.toml"
setkey MotionCalibApply           true      "$RC/config.toml"
setkey MotionCalibApplyMask       1         "$RC/config.toml"
setkey StableSdfMseMax            0.076     "$RC/config.toml"
setkey MapMode                    '"given"' "$RC/config.toml"
rm -fv "$RC/motion_calib_state.csv" "$RC/camera_calib_Shadow_ricoh.txt" \
       "$RC/camera_calib_Shadow_zed.txt" "$RC/image_edge_mount.csv" 2>/dev/null || true

echo
echo "── arm 6 leg '$leg' armed: uninjected, MotionCalibEpisodeMinRot = $(leg_rot "$leg") rad ──"
grep -n "^WheelScaleV" "$BR/config.toml"
grep -n "^MotionCalibEpisodeMinRot\|^MotionCalibEpisodeMinTrans\|^MotionCalibApply \|^StableSdfMseMax\|^MapMode" "$RC/config.toml"
echo
echo "Start, then: arm6_setup.sh check; drive ~10 m; arm6_setup.sh ratio (MUST read ~0.99)."
echo "★ Drive the SAME route both legs — this arm compares information per radian TURNED, so the"
echo "  rotation content of the two legs has to match or the comparison is against the route."
