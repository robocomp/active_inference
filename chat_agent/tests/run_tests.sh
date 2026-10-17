#!/usr/bin/env bash
# Offline checks for chat_agent's pure pieces (no DSR, no ollama): NDJSON splitter, CommandGate, object facts,
# look()'s frame preparation + request JSON (OpenCV + Qt Core), and the mission IR + its JSON and text forms (Qt Core).
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
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/mission_ir_test" \
    ../../common/mission/mission_ir_test.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/mission_ir_test"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/mission_text_test" \
    ../../common/mission/mission_text_test.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/mission_text_test"
rm -rf "$out"
