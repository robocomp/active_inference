#!/usr/bin/env bash
# Arm 7 — apply ALL calibration parameters online (MotionCalibApplyMask = -1) vs forward scale only (1).
# Pre-registration: EXPERIMENT.md §13. Sets the config for one leg; it does NOT start or stop anything.
#
#   tools/arm7_setup.sh native-k     # no injection,          mask 1  (today's default)
#   tools/arm7_setup.sh native-all   # no injection,          mask -1
#   tools/arm7_setup.sh gyro-k       # GyroBias = +0.002 rad/s, mask 1
#   tools/arm7_setup.sh gyro-all     # GyroBias = +0.002 rad/s, mask -1
#   tools/arm7_setup.sh check        # is the RUNNING fleet running what is on disk?
#   tools/arm7_setup.sh save <leg>   # file the finished leg's heading log (room_concept stopped)
#   tools/arm7_setup.sh report       # grade every saved leg
#   tools/arm7_setup.sh restore      # GyroBias 0, mask 1
#
# Each leg: stop room_concept (Ctrl-C / SIGTERM, never kill -9) -> run this -> restart room_concept
# (and the bridge, if this script says its config changed) -> controller mission "calib turns" ->
# when it finishes, stop room_concept -> `save <leg>`. Legs back to back in ONE session.
set -euo pipefail

RC=/home/pbustos/robocomp/components/active_inference/room_concept
BR=/home/pbustos/robocomp/components/webots-bridge/etc
DOSE=0.002
RUNS="$RC/etc/runs"
die() { echo "arm7: $*" >&2; exit 1; }

set_key() {  # $1 file, $2 regex key, $3 replacement line
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
p, key, line = sys.argv[1:]
s = open(p).read()
n, c = re.subn(r'^' + key + r'\s*=\s*[^\s#]+', line, s, count=1, flags=re.M)
assert c == 1, f'{key} not found exactly once in {p}'
if n != s: open(p, 'w').write(n)
PY
}
get_key() { sed -n "s/^$2 *= *\([^ #]*\).*/\1/p" "$1" | head -1; }

case "${1:-}" in
  check)
    now=$(date +%s)
    for pc in "Webots2Robocomp" "room_concept"; do
      pid=$(pgrep -x "$pc" | head -1) || true
      [ -z "$pid" ] && { printf "  %-16s NOT RUNNING\n" "$pc"; continue; }
      start=$((now - $(ps -o etimes= -p "$pid" | tr -d ' ')))
      for f in $(tr '\0' ' ' < /proc/$pid/cmdline); do
        case "$f" in *etc/config*) cfg="$f"; [ -f "$cfg" ] || cfg="$(readlink -f /proc/$pid/cwd)/$f"
          if [ "$(stat -c %Y "$cfg")" -gt "$start" ]; then echo "  ⚠ $pc: $cfg written AFTER start — RESTART it"
          else echo "  ✓ $pc reads $cfg (predates the process)"; fi;; esac
      done
    done
    echo "  on disk: mask=$(get_key "$RC/etc/config.toml" MotionCalibApplyMask)" \
         " apply=$(get_key "$RC/etc/config.toml" MotionCalibApply)" \
         " GyroBias(config.toml)=$(get_key "$BR/config.toml" GyroBias)" \
         " GyroBias(config)=$(get_key "$BR/config" SensorNoise.GyroBias)"
    exit 0;;
  save)
    leg="${2:-}"; case "$leg" in native-k|native-all|gyro-k|gyro-all) ;; *) die "usage: save <leg>";; esac
    pgrep -x room_concept >/dev/null && die "room_concept is RUNNING — stop it first (it is still appending)."
    want_mask=$([[ "$leg" == *-all ]] && echo -1 || echo 1)
    [ "$(get_key "$RC/etc/config.toml" MotionCalibApplyMask)" = "$want_mask" ] || \
      die "config mask is not $want_mask — this is not the '$leg' leg."
    src=$(ls -t "$RC"/tmp/heading/heading_*.csv | head -1)
    mkdir -p "$RUNS"
    for o in "$RUNS"/arm7_*.csv; do
      [ -f "$o" ] && cmp -s "$src" "$o" && die "$src is byte-identical to $o — the leg was not re-driven."
    done
    cp -v "$src" "$RUNS/arm7_$leg.csv"
    exit 0;;
  report)
    exec python3 "$RC/tools/arm7_report.py" "$RUNS";;
  native-k|native-all|gyro-k|gyro-all|restore) ;;
  *) die "usage: $0 {native-k|native-all|gyro-k|gyro-all|check|save <leg>|report|restore}";;
esac

pgrep -x room_concept >/dev/null && die "room_concept is RUNNING. Stop it first (Ctrl-C / SIGTERM, never kill -9):
      it rewrites motion_calib_state.csv from its warm window, so a wipe under it does nothing."

leg="$1"
mask=$([[ "$leg" == *-all ]] && echo -1 || echo 1)
bias=$([[ "$leg" == gyro-* ]] && echo "$DOSE" || echo 0.0)

# The bridge reads ONE of two files depending on its command line; both carry the knob, so set both.
old_bias=$(get_key "$BR/config.toml" GyroBias)
if [ "$(python3 -c "print(float('$old_bias') != float('$bias'))")" = True ]; then
  pgrep -x Webots2Robocomp >/dev/null && die "GyroBias changes ($old_bias -> $bias): stop Webots2Robocomp first
      (it reads its config once). Webots itself can keep running."
  set_key "$BR/config.toml" GyroBias "GyroBias = $bias"
  set_key "$BR/config" 'SensorNoise\.GyroBias' "SensorNoise.GyroBias = $bias"
  echo "  ⚠ bridge GyroBias CHANGED to $bias — restart Webots2Robocomp before this leg"
else
  echo "  bridge GyroBias already $bias — leave the bridge running"
fi
set_key "$RC/etc/config.toml" MotionCalibApplyMask "MotionCalibApplyMask = $mask"
set_key "$RC/etc/config.toml" MotionCalibApply "MotionCalibApply    = true"

if [ "$leg" = restore ]; then
  # Put back the warm calibration this robot had BEFORE arm 7 (saved by the first leg's wipe).
  for f in motion_calib_state.csv camera_calib_Shadow_ricoh.txt camera_calib_Shadow_zed.txt image_edge_mount.csv; do
    [ -f "$RC/etc/pre-arm7/$f" ] && cp -v "$RC/etc/pre-arm7/$f" "$RC/etc/$f"
  done
else
  # Keep the pre-arm-7 warm state ONCE, so `restore` can give the robot its calibration back.
  if [ ! -d "$RC/etc/pre-arm7" ]; then
    mkdir -p "$RC/etc/pre-arm7"
    for f in motion_calib_state.csv camera_calib_Shadow_ricoh.txt camera_calib_Shadow_zed.txt image_edge_mount.csv; do
      [ -f "$RC/etc/$f" ] && cp -v "$RC/etc/$f" "$RC/etc/pre-arm7/$f"
    done
  fi
  # A leg starts COLD: an inherited warm window is not the leg it claims to be.
  rm -fv "$RC/etc/motion_calib_state.csv" "$RC/etc/camera_calib_Shadow_ricoh.txt" \
         "$RC/etc/camera_calib_Shadow_zed.txt" "$RC/etc/image_edge_mount.csv" 2>/dev/null || true
fi
echo
echo "── arm 7 leg '$leg': MotionCalibApplyMask = $mask, GyroBias = $bias ──"
echo "NOW: start room_concept (and the bridge if it changed), run tools/arm7_setup.sh check,"
echo "     then the controller mission \"calib turns\". When it ends: stop room_concept, save $leg."
