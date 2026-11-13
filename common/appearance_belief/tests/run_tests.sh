#!/bin/bash
# Header-only tests of common/appearance_belief (no DSR). Run from anywhere.
set -e
D=$(cd "$(dirname "$0")" && pwd)
O=${TMPDIR:-/tmp}/appearance_belief_tests; mkdir -p "$O"
g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 "$D/test_colour_name.cpp" -o "$O/test_colour_name"
"$O/test_colour_name"
g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 "$D/test_colour_trace.cpp" -o "$O/test_colour_trace"
"$O/test_colour_trace"
g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 "$D/test_foreign_mask.cpp" -o "$O/test_foreign_mask"
"$O/test_foreign_mask"
