#!/usr/bin/env bash
# Arm 4 — dose response at a LARGE initial odometry error (WheelScaleV = 0.10).
# Sets the config for one leg. It does NOT start or stop anything: the user owns the lifecycle.
#
#   tools/arm4_setup.sh off        # injected, calibration NOT applied
#   tools/arm4_setup.sh on         # injected, calibration applied (k_v only)
#   tools/arm4_setup.sh check      # is the RUNNING fleet actually running the armed config?
#   tools/arm4_setup.sh ratio      # is the injection ACTING? drive ~10 m first, then run this
#   tools/arm4_setup.sh save <leg> # copy the finished leg's gt_error.csv aside, with checks
#   tools/arm4_setup.sh restore    # back to the uninjected baseline, calibration on
#
# After each call: STOP the bridge + room_concept, delete the evidence (this script does it),
# then START them again. A component reads its config ONCE at start-up, so a leg driven without
# a restart is the previous leg wearing this one's name.
set -euo pipefail

BR=/home/pbustos/robocomp/components/webots-bridge/etc
RC=/home/pbustos/robocomp/components/active_inference/room_concept/etc

die() { echo "arm4: $*" >&2; exit 1; }

# ── ratio: the behavioural signature of the injection ────────────────────────────────────────
# ★★★ THE ONLY PROOF THAT THE VALUE IS ACTING. `check` proves the file was read; this proves the
# number is doing something. Odometry path over ground-truth path: ~1.10 injected, ~0.99 clean,
# and it resolves within about 10 m of driving. Runs live, as often as you like.
# dx_local/dy_local are the per-cycle odometry increments in the robot frame. On the OFF leg they
# are raw; on the ON leg the calibrator is correcting them, so the ratio there is EXPECTED to fall
# back toward 1.0 as k_v converges — on that leg a ratio near 1.0 is the treatment working, not a
# missing injection. Read this on the OFF leg to prove the dose, and on the ON leg to watch it go.
if [ "${1:-}" = ratio ]; then
  python3 - "$RC/../tmp/sdf_localizer/gt_error.csv" <<'PY'
import csv, math, sys
gt=[]; odo=0.0; ts=[]
for r in csv.DictReader(open(sys.argv[1])):
    try:
        gt.append((float(r['gt_x']), float(r['gt_y']))); ts.append(float(r['ts_ms']))
        odo += math.hypot(float(r['dx_local']), float(r['dy_local']))
    except (TypeError, ValueError, KeyError): pass
if len(gt) < 2: sys.exit("  no usable rows yet")
g = sum(math.dist(a,b) for a,b in zip(gt,gt[1:]))
print("  %d rows over %.0f s" % (len(gt), (max(ts)-min(ts))/1000))
print("  ground-truth path %.2f m | odometry %.2f m" % (g, odo))
if len(set(p[0] for p in gt)) == 1:
    sys.exit("  GROUND TRUTH IS CONSTANT — either the robot is parked, or the supervisor is not\n"
             "  publishing robot_gt_*. Check est_* moves before blaming the supervisor.")
if g < 5.0:
    sys.exit("  only %.2f m driven — the ratio does not resolve below ~10 m. Drive more." % g)
r = odo/g
verdict = ("INJECTION IS ACTING (~1.10 expected)" if r > 1.06 else
           "CLEAN / NO INJECTION (~0.99 expected)" if r < 1.03 else
           "AMBIGUOUS — drive further, or the calibrator is already correcting it")
print("  ratio odo/GT = %.4f  ->  %s" % (r, verdict))
PY
  exit 0
fi

# ── check: is what is RUNNING what is on disk? ───────────────────────────────────────────────
# ★★★ THE TEST THAT CATCHES THE FAILURE THAT KEEPS HAPPENING. Twice now a leg has been driven
# against a config the running process had never read: on 2026-08-30 because the value went into
# a file the bridge does not read, and on 2026-08-31 because the bridge was never restarted. Both
# are the same defect and both are caught by one comparison — **if the process started BEFORE the
# config file was last written, the process is not running that config.** No banner can tell you
# this, because a banner reports what was loaded at a start-up you may be reasoning about the
# wrong instance of. This is the only subcommand that runs while the fleet is UP; that is its job.
if [ "${1:-}" = check ]; then
  now=$(date +%s); rc_ok=1
  probe() {  # $1 = process name, $2 = config path
    local pid start age
    pid=$(pgrep -x "$1" | head -1) || true
    if [ -z "$pid" ]; then printf "  %-18s NOT RUNNING\n" "$1"; rc_ok=0; return; fi
    age=$(ps -o etimes= -p "$pid" | tr -d " ")
    start=$((now - age))
    printf "  %-18s pid %-8s up %sh%02dm\n" "$1" "$pid" "$((age/3600))" "$(((age%3600)/60))"
    if [ "$(stat -c %Y "$2")" -gt "$start" ]; then
      printf "    ⚠ STALE — %s was written AFTER this process started.\n" "$2"
      printf "      The running process is NOT using it. Restart before driving.\n"
      printf "      (A COMMENT-only edit trips this too — it compares mtimes, not values. That is\n"
      printf "       the safe direction: it cannot miss a real change, it can only over-warn.)\n"
      rc_ok=0
    else
      printf "    config %s predates the process ✓\n" "$2"
    fi
  }
  probe Webots2Robocomp /home/pbustos/robocomp/components/webots-bridge/etc/config.toml
  probe room_concept    "$RC/config.toml"
  echo "  ── values ON DISK (which is only what is RUNNING if both lines above are ✓) ──"
  grep -H "^WheelScaleV" /home/pbustos/robocomp/components/webots-bridge/etc/config.toml | sed "s|^|    |"
  grep -H "^MotionCalibApply \|^MotionCalibApplyMask\|^MapMode" "$RC/config.toml" | sed "s|^|    |"
  echo
  if [ "$rc_ok" = 1 ]; then
    echo "  Process/config agreement OK. This is NECESSARY, NOT SUFFICIENT — it proves the file"
    echo "  was read, not that the value is acting. Still confirm the odometry/GT ratio reads"
    echo "  ~1.10 within the first 10 m. A signature says what is happening; a file does not."
  else
    echo "  ✗ DO NOT DRIVE THIS LEG until the lines above are clean."
  fi
  exit 0
fi

# ── save a finished leg ──────────────────────────────────────────────────────────────────────
# ★★★ A STEPPER THAT INFERS "a run happened" FROM PROCESS STATE RACES. Two different bugs saved
# the same stale CSV under two arm names on 2026-08-30, and the analysis then compared a run with
# itself. So this refuses to guess: it checks the file's mtime is newer than the destination it
# would overwrite, prints the distance the file actually covers, and never runs while the agent
# is up (the localiser is still appending).
if [ "${1:-}" = save ]; then
  leg="${2:-}"
  case "$leg" in off|on) ;; *) die "usage: $0 save {off|on}" ;; esac
  if pgrep -x room_concept >/dev/null; then
    die "room_concept is RUNNING — stop it first, it is still appending to gt_error.csv."
  fi
  src=/home/pbustos/robocomp/components/active_inference/room_concept/tmp/sdf_localizer/gt_error.csv
  dst="$RC/runs/arm4_$leg.csv"
  [ -f "$src" ] || die "no $src"
  mkdir -p "$RC/runs"
  if [ -f "$dst" ] && [ ! "$src" -nt "$dst" ]; then
    die "$src is NOT newer than $dst — this is the stale-file bug. Did the leg actually run?"
  fi
  cp -v "$src" "$dst"
  python3 - "$dst" <<'PY'
import csv, math, sys
xs=[]
for r in csv.DictReader(open(sys.argv[1])):
    try: xs.append((float(r['gt_x']), float(r['gt_y'])))
    except (KeyError, TypeError, ValueError): pass
d=sum(math.dist(a,b) for a,b in zip(xs, xs[1:]))
print("  %d rows, ground-truth path %.1f m" % (len(xs), d))
print("  ⚠ SHORT — arm 3's legs were 203-215 m. A leg this short is not matched to them."
      if d < 150 else "  length is in arm 3's range (203-215 m).")
PY
  exit 0
fi

# ── who has to be DOWN, and who does not ─────────────────────────────────────────────────────
# room_concept ALWAYS: it rewrites motion_calib_state.csv from its warm in-memory window, so a
# delete under a live process just restores the warm one and the leg inherits the previous leg's
# evidence. The calibrator must start this leg from its priors.
#
# ★ The BRIDGE only when its config actually changes. Going from the OFF leg to the ON leg the
# injection is IDENTICAL — the treatment is entirely `MotionCalibApply` on the room_concept side
# — so the bridge should stay up across the pair. That is not merely convenient: the standing
# rule is that arms are back to back inside one session or they are not comparable, and leaving
# the producer running keeps both legs on one continuous stream from one simulator session.
# Rewriting an identical file would also bump its mtime and make `check` cry STALE for no reason.
if pgrep -x room_concept >/dev/null; then
  echo "arm4: room_concept is RUNNING. Stop it first (SIGTERM / Ctrl-C, never kill -9) — it" >&2
  echo "      rewrites motion_calib_state.csv from its warm window, so wiping the evidence" >&2
  echo "      under a live process leaves the leg inheriting the previous leg's calibration." >&2
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
    if cmp -s "$BR/config.toml.arm4-inj10" "$BR/config.toml"; then
      echo "bridge config already carries the injection — left untouched, no restart needed"
    else
      pgrep -x Webots2Robocomp >/dev/null && die \
        "the bridge config must CHANGE for this leg, so Webots2Robocomp must be stopped first
      (it reads its config once at start-up). Webots itself can keep running."
      cp -v "$BR/config.toml.arm4-inj10" "$BR/config.toml"
      echo "  ⚠ bridge config CHANGED — it must be restarted before this leg is driven"
    fi
    if [ "$1" = off ]; then set_apply false; else set_apply true; fi
    set_mapmode given
    ;;
  restore)
    [ -f "$BR/config.toml.pre-arm2" ] || die "missing $BR/config.toml.pre-arm2"
    pgrep -x Webots2Robocomp >/dev/null && die \
      "restore changes the bridge config; stop Webots2Robocomp first."
    cp -v "$BR/config.toml.pre-arm2" "$BR/config.toml"
    set_apply true
    # NOT touched: MapMode. Whoever set "estimate" owns it — restore means "undo arm 4",
    # not "undo somebody else's experiment".
    ;;
  *) die "usage: $0 {off|on|check|ratio|save <leg>|restore}" ;;
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
