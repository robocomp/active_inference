#!/usr/bin/env bash
# Live check of the look() path (needs ollama at localhost:11434 with a vision model pulled): a saved ZED frame through
# image_prep.h + look_messages.h + OllamaClient, as ChatSession does after a look() call.
#   tests/run_look_test.sh [model] [frame.png]
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
g++ -std=c++23 -O1 -fPIC -Wall -Wno-deprecated-enum-enum-conversion -Wno-deprecated-literal-operator \
    -o "$out/test_look_live" test_look_live.cpp ../src/ollama_client.cpp \
    $(pkg-config --cflags opencv4 Qt6Network Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs opencv4 Qt6Network Qt6Core)
"$out/test_look_live" "$@"
rm -rf "$out"
