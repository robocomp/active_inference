#!/usr/bin/env bash
# Offline check of the ricoh azimuth-window rotation (Eigen only, no DSR).
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d); trap 'rm -rf "$out"' EXIT
g++ -std=c++23 -O0 -g -Wall -Wextra -I/usr/include/eigen3 -o "$out/t" mask_azimuth_windows_test.cpp
"$out/t"
