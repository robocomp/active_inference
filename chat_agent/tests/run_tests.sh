#!/usr/bin/env bash
# Offline checks for chat_agent's pure pieces (no DSR, no ollama): NDJSON splitter, CommandGate, object facts,
# look()'s frame preparation + request JSON (OpenCV + Qt Core), the mission IR + its JSON and text forms (Qt Core), the mission interpreter's step machine, its progress/history JSON, the plan view's model, the system tools, the episodic-memory tools' answers, and the side canvas (fast-path phrases, the script runner + panel, the stock scripts).
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
for t in test_ndjson test_gate test_objects test_canvas; do
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
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -o "$out/mission_interpreter_test" \
    ../../common/mission/mission_interpreter_test.cpp
"$out/mission_interpreter_test"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/mission_progress_test" \
    ../../common/mission/mission_progress_test.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/mission_progress_test"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/test_plan_model" \
    test_plan_model.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_plan_model"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -o "$out/test_system" test_system.cpp
"$out/test_system"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/test_em" test_em.cpp \
    $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_em"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-literal-operator -fPIC \
    -o "$out/test_approach" test_approach.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_approach"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-literal-operator -fPIC \
    -o "$out/test_controller_facts" test_controller_facts.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_controller_facts"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-literal-operator -fPIC \
    -DCANVAS_DIR="\"$(realpath ../canvas)\"" -o "$out/test_canvas_runner" test_canvas_runner.cpp \
    ../src/canvas_runner.cpp ../src/canvas_panel.cpp ../src/plan_view.cpp \
    $(pkg-config --cflags Qt6Widgets Qt6OpenGLWidgets | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Widgets Qt6OpenGLWidgets)
"$out/test_canvas_runner" 2>&1 | grep -v propagateSizeHints
python3 test_canvas_stock.py
rm -rf "$out"
