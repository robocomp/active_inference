#!/usr/bin/env bash
# Arm 3R — re-run ARM 3's conditions (NO injection, the native robot) with the instrument as it
# now stands, to find out whether arm 3's 4.4x firing drop corresponds to any real WORK.
#
# Arm 3 measured firing % only, its raw CSVs are gone, and firing % is an indicator over a
# threshold that says nothing about how hard a solve was. Arm 4 measured the work directly and it
# did not move. This arm settles which reading arm 3 supports.
#
#   tools/arm3r_setup.sh off|on     arm one leg
#   tools/arm3r_setup.sh check      is the RUNNING fleet running the armed config?
#   tools/arm3r_setup.sh ratio      MUST read ~0.99 here — this arm is UNINJECTED
#   tools/arm3r_setup.sh save <leg> file a finished leg, with the arm-4 guards
#
# It never starts or stops anything: the user owns the lifecycle.
set -euo pipefail
BR=/home/pbustos/robocomp/components/webots-bridge/etc
RC=/home/pbustos/robocomp/components/active_inference/room_concept/etc
SRC=/home/pbustos/robocomp/components/active_inference/room_concept/tmp/sdf_localizer/gt_error.csv
die() { echo "arm3r: $*" >&2; exit 1; }

setkey() {
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
key, val, path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n, c = re.subn(r'^%s\s*=\s*\S+' % re.escape(key), '%s = %s' % (key, val), s, count=1, flags=re.M)
if c != 1: sys.exit("arm3r: %s appears %d times in %s, expected once" % (key, c, path))
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
  grep -H "^MotionCalibApply \|^MotionCalibApplyMask\|^MapMode\|^StableSdfMseMax" "$RC/config.toml" | sed "s|^|    |"
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
if len(set(p[0] for p in gt)) == 1:
    sys.exit("  GROUND TRUTH CONSTANT — robot parked, or the supervisor is not publishing.")
if g < 5.0: sys.exit("  only %.2f m — does not resolve below ~10 m." % g)
r=odo/g
print("  ratio odo/GT = %.4f  ->  %s" % (r,
      "CLEAN ✓ this arm is uninjected" if r < 1.03 else
      "⚠ STOP: reads INJECTED (~1.10). The bridge is still on the arm-4 config." if r > 1.06 else
      "AMBIGUOUS — drive further."))
PY
  exit 0
fi

if [ "${1:-}" = save ]; then
  leg="${2:-}"; case "$leg" in off|on) ;; *) die "usage: $0 save {off|on}";; esac
  dst="$RC/runs/arm3r_$leg.csv"; mkdir -p "$RC/runs"
  [ -f "$SRC" ] || die "no $SRC"
  want=$([ "$leg" = off ] && echo false || echo true)
  have=$(sed -n "s/^MotionCalibApply *= *\([a-z]*\).*/\1/p" "$RC/config.toml" | head -1)
  [ "$have" = "$want" ] || die "leg $leg needs MotionCalibApply=$want but the config says $have."
  gate=$(sed -n "s/^StableSdfMseMax *= *\([0-9.]*\).*/\1/p" "$RC/config.toml" | head -1)
  [ "$gate" = "0.076" ] || die "arm 3's gate is 0.076 but the config says $gate."
  inj=$(sed -n "s/^WheelScaleV *= *\([0-9.]*\).*/\1/p" "$BR/config.toml" | head -1)
  [ "$inj" = "0.0" ] || die "this arm is UNINJECTED but the bridge config says WheelScaleV=$inj."
  other="$RC/runs/arm3r_$([ "$leg" = off ] && echo on || echo off).csv"
  [ -f "$other" ] && cmp -s "$SRC" "$other" && die "byte-identical to the other leg — same run twice."
  [ -f "$dst" ] && [ ! "$SRC" -nt "$dst" ] && die "$SRC is not newer than $dst. Did the leg run?"
  pgrep -x room_concept >/dev/null && die "room_concept is RUNNING — it is still appending."
  cp -v "$SRC" "$dst"
  python3 - "$dst" <<'PY'
import csv, math, sys
xs=[]
for r in csv.DictReader(open(sys.argv[1])):
    try: xs.append((float(r['gt_x']), float(r['gt_y'])))
    except (KeyError, TypeError, ValueError): pass
d=sum(math.dist(a,b) for a,b in zip(xs,xs[1:]))
print("  %d rows, ground-truth path %.1f m%s" % (len(xs), d,
      "   ⚠ SHORT — target ~200 m to match arm 4." if d < 150 else ""))
PY
  exit 0
fi

leg="${1:-}"; case "$leg" in off|on) ;; *) die "usage: $0 {off|on|check|ratio|save <leg>}";; esac
pgrep -x room_concept >/dev/null && die \
  "room_concept is RUNNING. Stop it first — it rewrites motion_calib_state.csv from its warm window."

if cmp -s "$BR/config.toml.pre-arm2" "$BR/config.toml"; then
  echo "bridge already uninjected — untouched, no bridge restart needed"
else
  pgrep -x Webots2Robocomp >/dev/null && die \
    "the bridge still carries the arm-4 injection and must be restored; stop Webots2Robocomp first
      (Webots itself can keep running). This arm is the NATIVE robot: WheelScaleV must be 0.0."
  cp -v "$BR/config.toml.pre-arm2" "$BR/config.toml"
  echo "  ⚠ bridge config CHANGED to uninjected — RESTART IT before driving"
fi

setkey MotionCalibApply     "$([ "$leg" = off ] && echo false || echo true)" "$RC/config.toml"
setkey StableSdfMseMax      0.076     "$RC/config.toml"
setkey MotionCalibApplyMask 1         "$RC/config.toml"
setkey MapMode              '"given"' "$RC/config.toml"
rm -fv "$RC/motion_calib_state.csv" "$RC/camera_calib_Shadow_ricoh.txt" \
       "$RC/camera_calib_Shadow_zed.txt" "$RC/image_edge_mount.csv" 2>/dev/null || true
echo
echo "── arm 3R leg '$leg' armed: uninjected, calibration $([ "$leg" = off ] && echo false || echo true), gate 0.076 ──"
grep -n "^WheelScaleV" "$BR/config.toml"
grep -n "^MotionCalibApply \|^MotionCalibApplyMask\|^StableSdfMseMax\|^MapMode" "$RC/config.toml"
echo
echo "Start, then: arm3r_setup.sh check; drive ~10 m; arm3r_setup.sh ratio (MUST read ~0.99)."
