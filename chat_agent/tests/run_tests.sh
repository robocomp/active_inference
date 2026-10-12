#!/usr/bin/env bash
# Offline checks for chat_agent's pure pieces (no Qt, no DSR): NDJSON splitter, CommandGate, object facts.
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
for t in test_ndjson test_gate test_objects; do
    g++ -std=c++23 -O0 -g -Wall -Wextra -o "$out/$t" "$t.cpp"
    "$out/$t"
done
rm -rf "$out"
