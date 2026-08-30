#!/usr/bin/env bash
# Build and run every standalone test under common/. No CMake, no deps beyond Eigen — each test is a single
# .cpp that mirrors the header's geometry/algebra so it stays runnable in seconds while editing.
#
# ★They are STANDALONE ON PURPOSE, and that carries one trap worth knowing: a harness has no Qt, so it never
# calls setlocale(LC_ALL, "") and stays in the "C" locale. Any test that PARSES a data file must add that call
# itself or it answers a different question than the agent does (see CLAUDE.md, locale section).
# place_memory/place_map_test.cpp is the one test that PARSES A FILE, and it sets the locale itself
# (es_ES.UTF-8, falling back to ""). Any new file-reading test must do the same.
set -u
cd "$(dirname "$0")" || exit 1
EIGEN=$(pkg-config --cflags eigen3 2>/dev/null || echo -I/usr/include/eigen3)
BIN=$(mktemp -d)
trap 'rm -rf "$BIN"' EXIT

fails=0
for src in */*_test.cpp; do
    name=$(basename "$src" .cpp)
    if ! g++ -std=c++23 -O1 $EIGEN "$src" -o "$BIN/$name" 2>"$BIN/$name.log"; then
        echo "BUILD FAILED: $src"; sed -n '1,20p' "$BIN/$name.log"; fails=$((fails + 1)); continue
    fi
    if ! "$BIN/$name"; then fails=$((fails + 1)); fi
done

[ "$fails" -eq 0 ] && echo "common: all test binaries passed" || echo "common: $fails test binary(ies) FAILED"
exit $((fails > 0))
