#!/bin/bash
# Standalone test of retina's per-mask colour summary (OpenCV + Eigen, no DSR). Run from anywhere.
set -e
D=$(cd "$(dirname "$0")" && pwd)
O=${TMPDIR:-/tmp}/retina_tests; mkdir -p "$O"
g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 $(pkg-config --cflags opencv4) "$D/test_mask_color_summary.cpp" \
    -o "$O/test_mask_color_summary" $(pkg-config --libs opencv4)
"$O/test_mask_color_summary"
