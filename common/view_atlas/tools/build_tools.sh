#!/usr/bin/env bash
# Build the ORT-linked view_atlas tools (not part of run_tests.sh: they need ONNX Runtime and data).
set -euo pipefail
cd "$(dirname "$0")"
ORT=${ONNXRUNTIME_ROOT:-/usr/local/onnxruntime}
CFLAGS="-std=c++23 -O2 $(pkg-config --cflags eigen3 opencv4) -I$ORT/include"
LIBS="$(pkg-config --libs opencv4) -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib"
mkdir -p bin
g++ $CFLAGS encoder_parity.cpp ../view_encoder.cpp -o bin/encoder_parity $LIBS
g++ $CFLAGS crop_golden.cpp ../view_encoder.cpp -o bin/crop_golden $LIBS
if [ -f atlas_replay.cpp ]; then g++ $CFLAGS atlas_replay.cpp -o bin/atlas_replay $LIBS; fi
echo "built: $(ls bin)"
