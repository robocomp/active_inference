#!/usr/bin/env bash
# Offline checks for residual_concept's pure pieces (no DSR, no Qt): every in-source self_test() and the
# try96 curtain replay. Build + run from anywhere:   residual_concept/tests/run_tests.sh
# A/B the absence channel on the replay:              <out>/test_curtain --control
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(mktemp -d)
CXX="g++ -O2 -std=c++23 -Wall -Wextra -Wno-unused-parameter -isystem /usr/include/eigen3"
$CXX -o "$out/test_self_tests" tests/test_self_tests.cpp src/residual_occupancy_grid.cpp \
    src/residual_floor_plane.cpp src/residual_clusterer.cpp src/residual_belief.cpp src/residual_zed_boost.cpp \
    src/residual_semantic.cpp
"$out/test_self_tests"
$CXX -o "$out/test_curtain" tests/test_curtain.cpp src/residual_occupancy_grid.cpp
"$out/test_curtain"
echo "residual_concept tests: ALL PASS"
