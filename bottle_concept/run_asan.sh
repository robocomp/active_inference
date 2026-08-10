#!/usr/bin/env bash
# Catch the heap corruption that crashes paintAndFlush, WITHOUT relinking bottle_concept.
#
# Linking -fsanitize=address reorders DT_NEEDED and trips an Ice/IceStorm static-init-order
# fiasco (SEGV in FactoryTable during _dl_init, before main). Instead we LD_PRELOAD libasan
# onto the NORMAL binary: Ice's library init order is preserved, but libasan still intercepts
# every malloc/free/new/delete globally — enough to pinpoint invalid-free / use-after-free /
# double-free anywhere in the process, including inside libdsr_gui.
#
# PREREQ: build the INSTRUMENTED binary so ./bin/bottle_concept has bottle_concept's OWN code under ASan
# (catches the corrupting WRITE, not just the later free):
#   cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# build-asan links -lasan DIRECTLY (no -fsanitize driver) so there is NO .preinit_array __asan_init
# entry. We then strip the libasan DT_NEEDED (below) and provide libasan via LD_PRELOAD instead, so
# ASan initialises first WITHOUT perturbing the Ice/IceStorm static-init order (which, with the
# driver's preinit __asan_init, SEGVs in IceInternal::FactoryTable before main).
# Then run this; start room/robot + bottle_concept in another terminal. Report -> /tmp/bottle_asan.txt
set -u
cd "$(dirname "$0")"

LIBASAN="$(gcc -print-file-name=libasan.so)"
if [ ! -e "$LIBASAN" ]; then
    LIBASAN="$(ls -1 /usr/lib/x86_64-linux-gnu/libasan.so* 2>/dev/null | head -1)"
fi
echo "Using libasan: $LIBASAN"

# Strip the libasan DT_NEEDED that -lasan added (idempotent). Without this the loader maps libasan
# as a normal dependency; with the preload below that is harmless, but stripping keeps the binary's
# init order identical to the normal build. __asan_* stay UND and resolve from the preloaded libasan.
if command -v patchelf >/dev/null && readelf -d ./bin/bottle_concept 2>/dev/null | grep -qi 'NEEDED.*libasan'; then
    echo "Stripping libasan DT_NEEDED from ./bin/bottle_concept"
    patchelf --remove-needed libasan.so.8 ./bin/bottle_concept 2>/dev/null \
        || patchelf --remove-needed "$(basename "$LIBASAN")" ./bin/bottle_concept
fi

# verify_asan_link_order=0 : required — libasan is preloaded, not linked first
# protect_shadow_gap=0     : CUDA/ONNX-GPU reserve huge VA ranges that collide with ASan's gap
# detect_leaks=0           : Qt/Ice/ONNX leak harmlessly at exit; we only want the corruption
# halt_on_error=1          : stop at the first real heap error
# fast_unwind_on_malloc=0  : full, accurate allocation backtraces
# log_path                 : ASan writes its report straight to <path>.<pid>, bypassing the stdout
#                            pipe — so a SHUTDOWN crash on Ctrl+C is captured even though `tee` dies
#                            with the same SIGINT (the reason a prior stop produced no trace).
export ASAN_OPTIONS="verify_asan_link_order=0:detect_leaks=0:protect_shadow_gap=0:halt_on_error=1:abort_on_error=1:fast_unwind_on_malloc=0:log_path=/tmp/bottle_asan_report"
export LD_PRELOAD="$LIBASAN"

# If a locally-instrumented cortex (build-asan) exists, load ITS libs instead of /usr/local/lib
# so ASan's redzones catch the bad WRITE inside the graph viewer (not just the bad free). Build with:
#   cd ~/robocomp/components/cortex && cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# (do NOT install build-asan — keeping it local is what makes ASan affect ONLY this process.)
ASAN_BUILD="$HOME/robocomp/components/cortex/build-asan"
ASAN_GUI_LIB="$(find "$ASAN_BUILD" -name 'libdsr_gui.so*' 2>/dev/null | head -1)"
if [ -n "$ASAN_GUI_LIB" ]; then
    # Prepend every dir under build-asan that holds a cortex .so, so the instrumented
    # libdsr_gui (and its instrumented dsr_api/dsr_core siblings) win over /usr/local/lib.
    ASAN_LIB_DIRS="$(find "$ASAN_BUILD" -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd:)"
    echo "Using INSTRUMENTED cortex libs from: $ASAN_LIB_DIRS"
    export LD_LIBRARY_PATH="$ASAN_LIB_DIRS:${LD_LIBRARY_PATH:-}"
else
    echo "(no build-asan libdsr_gui found — running against the system libdsr_gui;"
    echo " ASan will catch the bad free but not the write. Build cortex build-asan to get the write.)"
fi

echo "Running ./bin/bottle_concept (ASan-instrumented) under LD_PRELOAD ASan; report -> /tmp/bottle_asan.txt"
exec ./bin/bottle_concept etc/config.toml 2>&1 | tee /tmp/bottle_asan.txt
