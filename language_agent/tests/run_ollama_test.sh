#!/usr/bin/env bash
# Live check of the Qt ollama client (needs ollama at localhost:11434 with the model pulled).
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
g++ -std=c++23 -O1 -fPIC -Wall -Wno-deprecated-literal-operator -o "$out/test_ollama_client" test_ollama_client.cpp ../src/ollama_client.cpp \
    $(pkg-config --cflags --libs Qt6Network Qt6Core)
"$out/test_ollama_client" "${1:-qwen3:8b}"
rm -rf "$out"
