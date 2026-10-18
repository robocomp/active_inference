// Side canvas offline checks: the fast-path phrases (canvas_commands.h). Build+run: tests/run_tests.sh
#include "../src/canvas_commands.h"

#include <cassert>
#include <clocale>
#include <print>

int main()
{
    using namespace rc::chat;
    using K = ShowRequest::Kind;
    std::setlocale(LC_ALL, "");
    int checks = 0;
    const auto is = [](const char* t, K k, const char* arg = "")
    {
        const auto r = parse_show(t);
        return r.has_value() and r->kind == k and r->arg == arg;
    };

    assert(is("show me the zed camera", K::camera, "zed")); ++checks;
    assert(is("show the front camera", K::camera, "zed")); ++checks;
    assert(is("show me the camera", K::camera, "zed")); ++checks;
    assert(is("show me the 360 camera", K::camera, "ricoh")); ++checks;
    assert(is("show me the ricoh", K::camera, "ricoh")); ++checks;
    assert(is("muéstrame la cámara frontal", K::camera, "zed")); ++checks;
    assert(is("show me the layout of the room", K::layout)); ++checks;
    assert(is("show me the room", K::layout)); ++checks;
    assert(is("show me the map.", K::layout)); ++checks;
    assert(is("muestrame el mapa", K::layout)); ++checks;
    assert(is("show me the plan", K::plan)); ++checks;
    assert(is("close the canvas", K::hide)); ++checks;
    assert(is("show me the graph", K::graph)); ++checks;
    assert(is("show me the dsr graph", K::graph)); ++checks;
    assert(is("show me the working memory", K::graph)); ++checks;
    assert(is("show me the graph of room_1", K::graph, "room_1")); ++checks;
    assert(is("show me the graph around the door 1", K::graph, "door_1")); ++checks;
    assert(is("muéstrame el grafo", K::graph)); ++checks;
    assert(is("hide canvas", K::hide)); ++checks;
    assert(is("show me the robot", K::model, "robot")); ++checks;
    assert(is("draw object chair_1", K::model, "chair_1")); ++checks;
    assert(is("draw object chair 1", K::model, "chair_1")); ++checks;
    assert(is("show me the 3d model of the fridge", K::model, "fridge")); ++checks;
    assert(is("show me the table model", K::model, "table")); ++checks;
    assert(is("muéstrame el robot", K::model, "robot")); ++checks;
    // looser sentences go to the model
    assert(not parse_show("show me the doors in red").has_value()); ++checks;
    assert(not parse_show("show me where the chairs are").has_value()); ++checks;
    assert(is("draw the room", K::layout)); ++checks;   // a ready-made drawing, whatever the verb
    assert(is("draw graph", K::graph)); ++checks;
    assert(is("draw your working memory graph", K::graph)); ++checks;
    assert(not parse_show("draw the doors in red").has_value()); ++checks;
    assert(not parse_show("draw a bar chart of the distances").has_value()); ++checks;
    assert(not parse_show("cross the door").has_value()); ++checks;

    assert(canvas_script_name("Distance bars!") == "distance_bars"); ++checks;
    assert(canvas_script_name("  ../../etc/passwd") == "etcpasswd"); ++checks;
    assert(canvas_script_name("---").empty()); ++checks;

    std::println("test_canvas: {} checks passed", checks);
}
