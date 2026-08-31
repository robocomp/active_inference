#!/usr/bin/env bash
# Arm 4 — dose response at a LARGE initial odometry error (WheelScaleV = 0.10).
# Sets the config for one leg. It does NOT start or stop anything: the user owns the lifecycle.
#
#   tools/arm4_setup.sh off      # injected, calibration NOT applied
#   tools/arm4_setup.sh on       # injected, calibration applied (k_v only)
#   tools/arm4_setup.sh restore  # back to the uninjected baseline, calibration on
#
# After each call: STOP the bridge + room_concept, delete the evidence (this script does it),
# then START them again. A component reads its config ONCE at start-up, so a leg driven without
# a restart is the previous leg wearing this one's name.
set -euo pipefail

BR=/home/pbustos/robocomp/components/webots-bridge/etc
RC=/home/pbustos/robocomp/components/active_inference/room_concept/etc

die() { echo "arm4: $*" >&2; exit 1; }

# ── refuse to touch anything while the agents are up ─────────────────────────────────────────
if pgrep -x room_concept >/dev/null || pgrep -x Webots2Robocomp >/dev/null; then
  echo "arm4: room_concept and/or Webots2Robocomp are RUNNING." >&2
  echo "      Stop them first (SIGTERM / Ctrl-C, never kill -9) — a live agent rewrites" >&2
  echo "      motion_calib_state.csv from its warm in-memory window, so deleting it now" >&2
  echo "      restores the warm one and the leg inherits the previous leg's evidence." >&2
  exit 1
fi

# ★★★ MapMode MUST be "given" for arm 4. The live file currently carries an UNCOMMITTED
# MapMode = "estimate" (the wall-SLAM layout experiment). Measured on the running agent
# 2026-08-31, 155 s: sdf_mse median 0.55 against StableSdfMseMax = 0.076, so the stability gate
# CANNOT pass, the optimiser fires on 100% of cycles at 46-94 ms, and the solver alone eats 81%
# of one core. Arm 3's endpoint is optimiser firing %; under "estimate" that endpoint is pinned
# at 100 and measures the layout estimator, not the calibration. Both legs are set to "given".
set_mapmode() {  # $1 = given|estimate
  python3 - "$1" <<'PY'
import re, sys
p='/home/pbustos/robocomp/components/active_inference/room_concept/etc/config.toml'
s=open(p).read()
n,c=re.subn(r'^MapMode\s*=\s*"(given|estimate)"', 'MapMode              = "%s"' % sys.argv[1], s, count=1, flags=re.M)
assert c==1, 'MapMode not found exactly once'
open(p,'w').write(n)
PY
}

set_apply() {  # $1 = true|false
  python3 - "$1" <<'PY'
import re, sys
p='/home/pbustos/robocomp/components/active_inference/room_concept/etc/config.toml'
s=open(p).read()
n,c=re.subn(r'^MotionCalibApply\s*=\s*(true|false)', 'MotionCalibApply    = %s' % sys.argv[1], s, count=1, flags=re.M)
assert c==1, 'MotionCalibApply not found exactly once'
open(p,'w').write(n)
PY
}

wipe_evidence() {
  # Persists across restarts; an arm inheriting a warm window is not the arm it claims to be.
  rm -fv "$RC/motion_calib_state.csv" \
         "$RC/camera_calib_Shadow_ricoh.txt" \
         "$RC/camera_calib_Shadow_zed.txt" \
         "$RC/image_edge_mount.csv" 2>/dev/null || true
}

case "${1:-}" in
  off|on)
    [ -f "$BR/config.toml.arm4-inj10" ] || die "missing $BR/config.toml.arm4-inj10"
    cp -v "$BR/config.toml.arm4-inj10" "$BR/config.toml"
    if [ "$1" = off ]; then set_apply false; else set_apply true; fi
    set_mapmode given
    ;;
  restore)
    [ -f "$BR/config.toml.pre-arm2" ] || die "missing $BR/config.toml.pre-arm2"
    cp -v "$BR/config.toml.pre-arm2" "$BR/config.toml"
    set_apply true
    # NOT touched: MapMode. Whoever set "estimate" owns it — restore means "undo arm 4",
    # not "undo somebody else's experiment".
    ;;
  *) die "usage: $0 {off|on|restore}" ;;
esac

wipe_evidence

echo
echo "── leg '${1}' armed ─────────────────────────────────────────────────────────────"
grep -n '^WheelScaleV'        "$BR/config.toml"
grep -n '^MotionCalibApply '  "$RC/config.toml" || grep -n '^MotionCalibApply' "$RC/config.toml"
grep -n '^MotionCalibApplyMask' "$RC/config.toml"
    grep -n '^MapMode' "$RC/config.toml"
echo
echo "NOW: start the bridge and room_concept, then VERIFY FROM BEHAVIOUR, not from the banner —"
echo "  odometry/GT distance ratio ~1.10 injected, ~0.99 clean; it resolves within ~10 m."
echo "  tools/gt_analyze.py on the run's gt_error.csv, or the [calib] line's odom-vs-gt figure."
