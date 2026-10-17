#!/usr/bin/env bash
# Refactor oracle (docs/refactor_proposal_2026-10-13.md §4, step 0): every selftest and bench mode that
# runs WITHOUT the simulator, recorded once on a known-good tree and diffed after each refactor step.
#
#   tools/refactor_oracle.sh                 build, run all, diff against tools/oracle/ (exact)
#   tools/refactor_oracle.sh --rtol 1e-4     same, numbers compared within a relative tolerance
#   tools/refactor_oracle.sh --record        (re)write tools/oracle/ from the current tree
#   tools/refactor_oracle.sh --no-build      skip the build (binaries already current)
#
# ★ The oracle records what the tree DOES, not what it should do. Some cases fail their own checks on
#   the recorded tree (see tools/oracle/README); a refactor step must reproduce those failures too.
# ★ Every case runs with ASLR OFF (setarch -R): the long estimator benches are address-dependent — with
#   ASLR on, 1 run in 4 of wss_default diverged (IoU .973 vs .958), consistent with alignment-dependent
#   vectorised reductions cascading through a chaotic 1100-frame run. Those CHAOTIC cases are compared
#   by their PASS/FAIL verdicts only (numeric drift is printed, not failed): any real code change moves
#   their floats. Every other case is compared exactly (or within --rtol/--atol).
# ★ All cases are single-threaded by construction (wall_slam_selftest re-execs itself with
#   OMP_NUM_THREADS=1), so they run in parallel here; the slowest (reloc_selftest) takes ~2.5 min.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
BIN=$ROOT/../bin                     # where the CMake selftests land (RUNTIME_OUTPUT_DIRECTORY)
ORACLE=$ROOT/tools/oracle
WORK=$ROOT/build/oracle
mkdir -p "$WORK"

record=0; build=1; tol=()
while (( $# )); do
    case $1 in
        --record)   record=1 ;;
        --no-build) build=0 ;;
        --rtol)     tol+=(--rtol "$2"); shift ;;
        --atol)     tol+=(--atol "$2"); shift ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done

# ── standalone one-file selftests (header-only units under src/) ───────────────────────────────
declare -A STANDALONE_EXTRA=(
    [layout_explorer_selftest]="src/layout_explorer.cpp src/wall_map.cpp src/wall_segmenter.cpp"
)
STANDALONE=(free_space_polygon_selftest layout_explorer_selftest image_edge_frame_selftest stride_span_selftest
            motion_noise_vc_selftest surprise_selftest joint_calib_selftest joint_mounts_selftest
            lidar_mount_selftest motion_calib_lever_selftest motion_noise_innov_selftest mount_factors_selftest
            mount_info_selftest pose_field_bias_selftest rest_density_selftest wheel_noise_selftest
            calib_estimator_selftest calib_intake_selftest)

# ── cases: name | env | command ─────────────────────────────────────────────────────────────────
CASES=(
    "gn_selftest||$BIN/gn_selftest"
    "reloc_selftest||$BIN/reloc_selftest"
    "corner_gather_test||$BIN/corner_gather_test"
    "motion_calib||$BIN/motion_calib --selftest"
    "mount_replay||$BIN/mount_replay --selftest"
    "wss_default||$BIN/wall_slam_selftest"
    "wss_tour|WS_TOUR=1|$BIN/wall_slam_selftest"
    # Box estimator, single rooms of the WS_BOXES_RANDOM sweep (the 12-room sweep takes 17 min): its
    # two worst (idx 4 = IoU .812, idx 2 = .864 on the recorded tree) and one healthy room (idx 0).
    "wss_box_room0|WS_BOXES=1 WS_BOXES_ROOM_IDX=0|$BIN/wall_slam_selftest"
    "wss_box_room2|WS_BOXES=1 WS_BOXES_ROOM_IDX=2|$BIN/wall_slam_selftest"
    "wss_box_room4|WS_BOXES=1 WS_BOXES_ROOM_IDX=4|$BIN/wall_slam_selftest"
)
for t in "${STANDALONE[@]}"; do CASES+=("$t||$WORK/$t"); done
CHAOTIC=" reloc_selftest wss_default wss_tour wss_box_room0 wss_box_room2 wss_box_room4 "

if (( build )); then
    if (( $(pgrep -c cc1plus) > 0 )); then echo "another build is running (pgrep cc1plus) — wait for it"; exit 2; fi
    echo "building CMake selftests (-j8) ..."
    make -C build -j8 gn_selftest reloc_selftest wall_slam_selftest corner_gather_test motion_calib mount_replay \
        > "$WORK/build.log" 2>&1 || { echo "BUILD FAILED — see $WORK/build.log"; exit 1; }
fi
# Standalone selftests are always compiled (seconds): --no-build skips only the CMake targets, and a
# stale or missing standalone binary would silently record "file not found" as the oracle.
echo "building standalone selftests ..."
for t in "${STANDALONE[@]}"; do
    g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/$t.cpp ${STANDALONE_EXTRA[$t]:-} -o "$WORK/$t" \
        2> "$WORK/$t.cerr" || { echo "COMPILE FAILED: $t — see $WORK/$t.cerr"; exit 1; }
done
for c in "${CASES[@]}"; do
    IFS='|' read -r name _ cmd <<< "$c"
    [[ -x ${cmd%% *} ]] || { echo "MISSING BINARY for $name: ${cmd%% *}"; exit 1; }
done

# ★ PINNED INPUT STATE. reloc_selftest restores the motion calibrator from etc/motion_calib_state.csv, a file
#   the LIVE agent rewrites on every run (try46, 2026-10-17: 80 -> 226 measurements), so the oracle drifted
#   with no code change. The recorded copy lives beside the outputs; it is swapped in for the run and the
#   live file is put back on exit, whatever happens.
PINNED=(etc/motion_calib_state.csv)
for f in "${PINNED[@]}"; do
    snap=$ORACLE/pinned_$(basename "$f")
    if (( record )); then cp "$f" "$snap"; fi
    [[ -f $snap ]] || { echo "no pinned copy $snap — run with --record"; exit 1; }
    cp "$f" "$WORK/live_$(basename "$f")"
    cp "$snap" "$f"
done
restore_pinned() { for f in "${PINNED[@]}"; do cp "$WORK/live_$(basename "$f")" "$f"; done; }
trap restore_pinned EXIT

echo "running ${#CASES[@]} cases ..."
for c in "${CASES[@]}"; do
    IFS='|' read -r name envs cmd <<< "$c"
    ( env $envs setarch -R timeout 900 $cmd > "$WORK/$name.out" 2>&1; echo $? > "$WORK/$name.rc" ) &
done
wait

if (( record )); then
    mkdir -p "$ORACLE"
    for c in "${CASES[@]}"; do
        IFS='|' read -r name _ _ <<< "$c"
        { cat "$WORK/$name.out"; echo "[exit $(cat "$WORK/$name.rc")]"; } > "$ORACLE/$name.txt"
    done
    echo "recorded ${#CASES[@]} cases into $ORACLE (tree $(git rev-parse --short HEAD)$(git diff --quiet -- src tools || echo '+dirty'))"
    exit 0
fi

bad=0
for c in "${CASES[@]}"; do
    IFS='|' read -r name _ _ <<< "$c"
    { cat "$WORK/$name.out"; echo "[exit $(cat "$WORK/$name.rc")]"; } > "$WORK/$name.txt"
    mode=("${tol[@]}"); [[ $CHAOTIC == *" $name "* ]] && mode=(--verdicts)
    if out=$(python3 -I tools/oracle_compare.py "$ORACLE/$name.txt" "$WORK/$name.txt" "${mode[@]}"); then
        printf '  same  %s%s\n' "$name" "${out:+  $out}"
    else
        printf '  DIFF  %s\n%s\n' "$name" "$out"; bad=1
    fi
done
(( bad == 0 )) && echo "oracle: all ${#CASES[@]} cases match" || echo "oracle: DIFFERENCES (outputs in $WORK)"
exit $bad
