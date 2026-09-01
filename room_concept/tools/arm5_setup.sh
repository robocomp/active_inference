#!/usr/bin/env bash
# Arm 5 — 2x2: {calibration ON/OFF} x {corrector abundant/starved}, injection held at 10%.
# Spec and pre-registered endpoints: EXPERIMENT.md, "Arm 5".
#
#   tools/arm5_setup.sh A|B|C|D    arm one leg   (A off/abundant  B on/abundant
#                                                 C off/starved   D on/starved)
#   tools/arm5_setup.sh check      is the RUNNING fleet running the armed config?
#   tools/arm5_setup.sh ratio      is the injection ACTING? drive ~10 m first
#   tools/arm5_setup.sh save <leg> file a finished leg, with the guards that arm 4 earned
#   tools/arm5_setup.sh restore    back to the uninjected baseline
#
# It never starts or stops anything: the user owns the lifecycle.
set -euo pipefail

BR=/home/pbustos/robocomp/components/webots-bridge/etc
RC=/home/pbustos/robocomp/components/active_inference/room_concept/etc
SRC=/home/pbustos/robocomp/components/active_inference/room_concept/tmp/sdf_localizer/gt_error.csv
GATE_ABUNDANT=0.076
GATE_STARVED=0.16      # ~17x fewer solves; chosen from the arm-4 sdf_mse distribution, not guessed

die() { echo "arm5: $*" >&2; exit 1; }

# leg -> (apply, gate). The 2x2 lives in ONE place so a leg cannot be half-armed.
leg_apply() { case "$1" in A|C) echo false;; B|D) echo true;;  *) die "unknown leg '$1'";; esac; }
leg_gate()  { case "$1" in A|B) echo $GATE_ABUNDANT;; C|D) echo $GATE_STARVED;; *) die "unknown leg '$1'";; esac; }

setkey() {  # $1 = key, $2 = value, $3 = file
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
key, val, path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n, c = re.subn(r'^%s\s*=\s*\S+' % re.escape(key), '%s = %s' % (key, val), s, count=1, flags=re.M)
if c != 1:
    sys.exit("arm5: %s appears %d times in %s, expected once" % (key, c, path))
open(path, 'w').write(n)
PY
}

# ── check / ratio / save run while the fleet is UP or DOWN as appropriate ────────────────────
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
  echo "  ── on disk ──"
  grep -H "^WheelScaleV" "$BR/config.toml" | sed "s|^|    |"
  grep -H "^MotionCalibApply \|^MotionCalibApplyMask\|^MapMode\|^StableSdfMseMax" "$RC/config.toml" | sed "s|^|    |"
  [ "$ok" = 1 ] && echo "  agreement OK — necessary, not sufficient. Confirm the ratio before driving." \
                || echo "  ✗ DO NOT DRIVE until the lines above are clean."
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
    sys.exit("  GROUND TRUTH CONSTANT — robot parked, or the supervisor is not publishing robot_gt_*.")
if g < 5.0: sys.exit("  only %.2f m — the ratio does not resolve below ~10 m." % g)
r=odo/g
print("  ratio odo/GT = %.4f  ->  %s" % (r,
      "INJECTION ACTING (~1.10)" if r > 1.06 else
      "CLEAN / NO INJECTION (~0.99)" if r < 1.03 else
      "AMBIGUOUS — drive further, or calibration is already correcting it (legs B and D)"))
PY
  exit 0
fi

if [ "${1:-}" = save ]; then
  leg="${2:-}"; case "$leg" in A|B|C|D) ;; *) die "usage: $0 save {A|B|C|D}";; esac
  dst="$RC/runs/arm5_$leg.csv"; mkdir -p "$RC/runs"
  [ -f "$SRC" ] || die "no $SRC"
  # A leg is its treatment: refuse to file one the live config contradicts.
  want_apply=$(leg_apply "$leg"); want_gate=$(leg_gate "$leg")
  have_apply=$(sed -n "s/^MotionCalibApply *= *\([a-z]*\).*/\1/p" "$RC/config.toml" | head -1)
  have_gate=$(sed -n "s/^StableSdfMseMax *= *\([0-9.]*\).*/\1/p" "$RC/config.toml" | head -1)
  [ "$have_apply" = "$want_apply" ] || die \
    "leg $leg needs MotionCalibApply=$want_apply but the config says $have_apply."
  [ "$have_gate" = "$want_gate" ] || die \
    "leg $leg needs StableSdfMseMax=$want_gate but the config says $have_gate."
  # ...and never the same bytes as ANY other leg. This is the arm-4 duplicate bug.
  for other in A B C D; do
    [ "$other" = "$leg" ] && continue
    f="$RC/runs/arm5_$other.csv"
    [ -f "$f" ] && cmp -s "$SRC" "$f" && die \
      "source is BYTE-IDENTICAL to leg $other — the same run saved twice, not two legs."
  done
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
      "   ⚠ SHORT — the other legs are ~200 m; this one is not matched to them." if d < 150 else ""))
PY
  exit 0
fi

if [ "${1:-}" = restore ]; then
  pgrep -x room_concept    >/dev/null && die "stop room_concept first."
  pgrep -x Webots2Robocomp >/dev/null && die "restore changes the bridge config; stop the bridge first."
  cp -v "$BR/config.toml.pre-arm2" "$BR/config.toml"
  setkey MotionCalibApply true "$RC/config.toml"
  setkey StableSdfMseMax "$GATE_ABUNDANT" "$RC/config.toml"
  echo "restored: uninjected, calibration on, gate $GATE_ABUNDANT"
  exit 0
fi

# ── arm one leg ──────────────────────────────────────────────────────────────────────────────
leg="${1:-}"; case "$leg" in A|B|C|D) ;; *) die "usage: $0 {A|B|C|D|check|ratio|save <leg>|restore}";; esac
pgrep -x room_concept >/dev/null && die \
  "room_concept is RUNNING. Stop it first — it rewrites motion_calib_state.csv from its warm
      window, so wiping the evidence under a live process leaves this leg inheriting the last one."

[ -f "$BR/config.toml.arm4-inj10" ] || die "missing $BR/config.toml.arm4-inj10"
if cmp -s "$BR/config.toml.arm4-inj10" "$BR/config.toml"; then
  echo "bridge already carries the 10% injection — untouched, no bridge restart needed"
else
  pgrep -x Webots2Robocomp >/dev/null && die \
    "the bridge config must change; stop Webots2Robocomp first (Webots itself can keep running)."
  cp -v "$BR/config.toml.arm4-inj10" "$BR/config.toml"
  echo "  ⚠ bridge config CHANGED — restart it before driving"
fi

setkey MotionCalibApply     "$(leg_apply "$leg")" "$RC/config.toml"
setkey StableSdfMseMax      "$(leg_gate "$leg")"  "$RC/config.toml"
setkey MotionCalibApplyMask 1                     "$RC/config.toml"
setkey MapMode              '"given"'             "$RC/config.toml"

# Evidence persists across restarts and the agent rewrites it every window.
rm -fv "$RC/motion_calib_state.csv" "$RC/camera_calib_Shadow_ricoh.txt" \
       "$RC/camera_calib_Shadow_zed.txt" "$RC/image_edge_mount.csv" 2>/dev/null || true

echo
echo "── leg $leg armed: calibration $(leg_apply "$leg"), gate $(leg_gate "$leg") ──"
grep -n "^WheelScaleV" "$BR/config.toml"
grep -n "^MotionCalibApply \|^MotionCalibApplyMask\|^StableSdfMseMax\|^MapMode" "$RC/config.toml"
echo
echo "Start room_concept, then: arm5_setup.sh check, drive ~10 m, arm5_setup.sh ratio."
echo "★ Report the REALISED firing rate for this leg — the 0.29%% predicted for the starved gate is"
echo "  an underestimate, because starving raises sdf_mse which pushes cycles back over the gate."
