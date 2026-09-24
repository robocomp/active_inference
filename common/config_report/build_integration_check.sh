#!/usr/bin/env bash
# Build+run integration_check.cpp. Separate from common/run_tests.sh because this one needs toml++
# and $ROBOCOMP/classes, which that runner deliberately does not provide.
set -eu
cd "$(dirname "$0")"
out=$(mktemp -d); trap 'rm -rf "$out"' EXIT
g++ -std=c++23 -O1 -I"${ROBOCOMP:-/home/pbustos/robocomp}/classes" \
    integration_check.cpp "${ROBOCOMP:-/home/pbustos/robocomp}/classes/ConfigLoader/ConfigLoader.cpp" \
    -o "$out/integration_check"
"$out/integration_check"
