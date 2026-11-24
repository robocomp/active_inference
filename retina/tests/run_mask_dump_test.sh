#!/bin/bash
# Standalone test of the [ZedDump] mask dump (CSV writer/reader + gate). OpenCV + Eigen, no DSR runtime
# (only the header-only dsr_eigen_defs.h). Run from anywhere. -j1, tiny TUs: safe next to a live fleet.
set -e
D=$(cd "$(dirname "$0")" && pwd)
O=${TMPDIR:-/tmp}/retina_tests; mkdir -p "$O"
g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I/usr/local/include $(pkg-config --cflags opencv4) \
    "$D/test_mask_dump.cpp" "$D/../src/zed_frame_dump.cpp" "$D/../src/mask_dump_csv.cpp" \
    -o "$O/test_mask_dump" $(pkg-config --libs opencv4)
"$O/test_mask_dump"
