#!/usr/bin/env bash
# Offline checks for language_agent's pure pieces (no DSR, no ollama): NDJSON splitter, CommandGate, object facts,
# look()'s frame preparation + request JSON (OpenCV + Qt Core), the mission IR + its JSON and text forms (Qt Core), the mission interpreter's step machine, its progress/history JSON, the plan view's model, the system tools, the episodic-memory tools' answers, and the side canvas (fast-path phrases, the script runner + panel, the stock scripts), the status bar's GPU memory reading, go_around's loop and sim_door's frame change.
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
for t in test_ndjson test_gate test_objects test_canvas test_gpu test_fleet; do
    g++ -std=c++23 -O0 -g -Wall -Wextra -o "$out/$t" "$t.cpp"
    "$out/$t"
done
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/test_history_trim" test_history_trim.cpp \
    $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_history_trim"
g++ -std=c++23 -O0 -g -Wall -Wextra -isystem /usr/include/eigen3 -o "$out/test_sim_door" test_sim_door.cpp
"$out/test_sim_door"
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
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/test_vlm_status" test_vlm_status.cpp \
    $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_vlm_status"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-literal-operator -fPIC \
    -o "$out/test_mission_plan" test_mission_plan.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_mission_plan"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-literal-operator -fPIC \
    -DSKILLS_DIR="\"$(realpath ../etc/skills)\"" -o "$out/test_skills" test_skills.cpp \
    $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_skills"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-literal-operator -fPIC \
    -o "$out/test_controller_facts" test_controller_facts.cpp $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_controller_facts"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-literal-operator -fPIC \
    -DCANVAS_DIR="\"$(realpath ../canvas)\"" -o "$out/test_canvas_runner" test_canvas_runner.cpp \
    ../src/canvas_runner.cpp ../src/canvas_panel.cpp ../src/plan_view.cpp \
    $(pkg-config --cflags Qt6Widgets Qt6OpenGLWidgets | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Widgets Qt6OpenGLWidgets)
"$out/test_canvas_runner" 2>&1 | grep -v propagateSizeHints
# the window's saved size: a real X display, never mapped (WA_DontShowOnScreen); skipped without one
if [[ -n "${DISPLAY:-}" ]]; then
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-missing-field-initializers -fPIC -o "$out/test_chat_window" test_chat_window.cpp \
    ../src/chat_window.cpp ../src/canvas_panel.cpp ../src/plan_view.cpp \
    $(pkg-config --cflags Qt6Widgets Qt6OpenGLWidgets | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Widgets Qt6OpenGLWidgets)
"$out/test_chat_window" 2>&1 | grep -v -E "propagateSizeHints|QOpenGLWidget"
else echo "test_chat_window: skipped (no DISPLAY)"; fi
python3 test_canvas_stock.py
MPLBACKEND=Agg python3 test_imagined_canvas.py
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -isystem /usr/include/eigen3 -I../../common \
    -I../../common/media_transport -o "$out/test_imagined_io" test_imagined_io.cpp \
    $(pkg-config --cflags Qt6Gui | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Gui)
"$out/test_imagined_io"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -isystem /usr/include/eigen3 -I../../common \
    -I../../common/media_transport -o "$out/test_imagine_view" test_imagine_view.cpp \
    $(pkg-config --cflags Qt6Gui | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Gui)
"$out/test_imagine_view"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -o "$out/test_topology" test_topology.cpp \
    $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_topology"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -isystem /usr/include/eigen3 -I../../common \
    -I../../common/media_transport -o "$out/test_can_see" test_can_see.cpp
"$out/test_can_see"
g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator -fPIC -isystem /usr/include/eigen3 -I../../common \
    -I../../common/media_transport -o "$out/test_imagine" test_imagine.cpp \
    $(pkg-config --cflags Qt6Core | sed "s/-I/-isystem /g") $(pkg-config --libs Qt6Core)
"$out/test_imagine"
rm -rf "$out"
