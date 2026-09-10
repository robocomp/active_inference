#!/usr/bin/env bash
# Hunt the bottle-join DATA RACE in retina with ThreadSanitizer.
#
# PREREQ: build the instrumented binaries:
#   cd ~/robocomp/components/cortex     && cmake -B build-tsan -DTSAN=ON       && cmake --build build-tsan -j
#   cd ~/robocomp/components/active_inference/retina
#   CT=~/robocomp/components/cortex/build-tsan; LIBRARY_PATH="$CT/gui:$CT/api:$CT/core" \
#       cmake -B build-tsan -DENABLE_TSAN=ON && cmake --build build-tsan -j   # -> bin/retina
#
# Then run this; in another terminal cycle bottle_concept start/stop to trigger the race.
# Race reports -> /tmp/vox_tsan.txt  (also each report duplicated as /tmp/vox_tsan.report.<pid>.<n>).
set -u
cd "$(dirname "$0")"

LIBTSAN="$(gcc -print-file-name=libtsan.so)"
[ -e "$LIBTSAN" ] || LIBTSAN="$(ls -1 /usr/lib/x86_64-linux-gnu/libtsan.so* 2>/dev/null | head -1)"
echo "Using libtsan: $LIBTSAN"

# Like the ASan harness: -ltsan was linked directly (no -fsanitize driver) so there is NO
# .preinit_array __tsan_init that would trip the Ice/IceStorm FactoryTable static-init SEGV. Strip
# the libtsan DT_NEEDED so the loader keeps the normal (Ice-safe) init order; libtsan is provided
# first via LD_PRELOAD below (TSan still needs to initialise before everything else).
if command -v patchelf >/dev/null && readelf -d ./bin/retina 2>/dev/null | grep -qi 'NEEDED.*libtsan'; then
    echo "Stripping libtsan DT_NEEDED from ./bin/retina"
    patchelf --remove-needed libtsan.so.2 ./bin/retina 2>/dev/null \
        || patchelf --remove-needed "$(basename "$LIBTSAN")" ./bin/retina 2>/dev/null || true
fi

# Load the TSan-instrumented cortex libs (so the DDS-sync and compute accesses inside libdsr are
# both instrumented -> TSan can see the race between them).
CT="$HOME/robocomp/components/cortex/build-tsan"
TSAN_LIB_DIRS="$(find "$CT" -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)"
echo "Instrumented cortex libs: $TSAN_LIB_DIRS"
export LD_LIBRARY_PATH="$TSAN_LIB_DIRS:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$LIBTSAN"

# halt_on_error=0  : keep running and collect EVERY race (don't die on the first)
# history_size=7   : deepest per-thread access history (find the other side of the race)
# second_deadlock_stack=1 + report stacks; ignore_noninstrumented_modules=1 cuts FastDDS/Qt/ONNX noise
# exitcode=0       : a race must not crash the process during the hunt
export TSAN_OPTIONS="halt_on_error=0:exitcode=0:history_size=7:second_deadlock_stack=1:ignore_noninstrumented_modules=1:report_thread_leaks=0:report_signal_unsafe=0:log_path=/tmp/vox_tsan.report"

echo "Running ./bin/retina etc/config_tsan.toml (CPU YOLO) under TSan; reports -> /tmp/vox_tsan.txt"
exec ./bin/retina etc/config_tsan.toml 2>&1 | tee /tmp/vox_tsan.txt
