#!/usr/bin/env bash
# Offline checks for chat_agent's pure pieces (no DSR, no ollama): NDJSON splitter, CommandGate, object facts,
# and look()'s frame preparation + request JSON (OpenCV + Qt Core).
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
for t in test_ndjson test_gate test_objects; do
    g++ -std=c++23 -O0 -g -Wall -Wextra -o "$out/$t" "$t.cpp"
    "$out/$t"
done
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-enum-enum-conversion -Wno-deprecated-literal-operator -fPIC \
    -o "$out/test_look" test_look.cpp \
    $(pkg-config --cflags opencv4 Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs opencv4 Qt6Core)
"$out/test_look"
rm -rf "$out"
