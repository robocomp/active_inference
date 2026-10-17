#!/usr/bin/env bash
# Offline checks for the episodic memory's pure pieces (Qt Core only, no DSR, no DDS).
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
for t in em_schema_test em_archive_test em_consolidate_test; do
    g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/$t" "$t.cpp" \
        $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
    "$out/$t"
done
