// ── WHY COULD THE PLANNER NOT GET THERE? — replay a captured world offline ───────────────────────
// Loads controller/unreachable_world.txt (room polygon, obstacles, residual cells, start, goal) into
// the REAL GridPlanner and asks the questions the live logs cannot: is the goal footprint-feasible at
// any heading, can the body turn there, how far is the nearest clear spot, does a route exist — and
// how much of the free (cell,heading) space the yaw convention moves.
//
// build: g++ -std=c++23 -O2 -I/usr/include/eigen3 -Isrc -I../common -o /tmp/replay_world \
//          tools/replay_world.cpp src/grid_planner.cpp ../common/robot_footprint/robot_footprint.cpp
#include "grid_planner.h"
#include <charconv>          // locale-independent parsing (es_ES writes commas; CLAUDE.md)
#include <cstdio>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

namespace
{
bool num(std::string_view s, float &out)
{
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc{};
}
std::vector<std::string_view> split(std::string_view l)
{
    std::vector<std::string_view> v;
    std::size_t i = 0;
    while (i < l.size())
    {
        while (i < l.size() and l[i] == ' ') ++i;
        const std::size_t b = i;
        while (i < l.size() and l[i] != ' ') ++i;
        if (i > b) v.push_back(l.substr(b, i - b));
    }
    return v;
}
}   // namespace

int main(int argc, char **argv)
{
    const std::string path = argc > 1 ? argv[1] : "unreachable_world.txt";
    std::ifstream f(path);
    if (not f) { std::printf("cannot open %s\n", path.c_str()); return 1; }

    std::vector<Eigen::Vector2f> room;
    std::vector<std::vector<Eigen::Vector2f>> obstacles;
    std::vector<Eigen::Vector2f> cells;
    float cell_size = 0.f;
    Eigen::Vector2f start{0, 0}, goal{0, 0};
    int last_obs = -1;
    std::string line;
    while (std::getline(f, line))
    {
        const auto t = split(line);
        if (t.empty() or t[0].front() == '#') continue;
        float a = 0, b = 0;
        if (t[0] == "start" and t.size() > 2 and num(t[1], a) and num(t[2], b)) start = {a, b};
        else if (t[0] == "goal" and t.size() > 2 and num(t[1], a) and num(t[2], b)) goal = {a, b};
        else if (t[0] == "room" and t.size() > 2 and num(t[1], a) and num(t[2], b)) room.emplace_back(a, b);
        else if (t[0] == "cell" and t.size() > 2 and num(t[1], a) and num(t[2], b)) cells.emplace_back(a, b);
        else if (t[0] == "cell_size" and t.size() > 1) num(t[1], cell_size);
        else if (t[0] == "obs" and t.size() > 3)
        {
            float idx = 0;
            if (not num(t[1], idx) or not num(t[2], a) or not num(t[3], b)) continue;
            if (static_cast<int>(idx) != last_obs) { obstacles.emplace_back(); last_obs = static_cast<int>(idx); }
            obstacles.back().emplace_back(a, b);
        }
    }
    std::printf("loaded: room %zu pts | %zu obstacle polygons | %zu residual cells @ %.3f m\n",
                room.size(), obstacles.size(), cells.size(), cell_size);
    std::printf("start (%.2f,%.2f)  goal (%.2f,%.2f)  d=%.2f m\n\n",
                start.x(), start.y(), goal.x(), goal.y(), (goal - start).norm());

    rc::GridPlanner p;
    p.params.cell_size_m = 0.06f;                 // the controller's configured resolution
    p.set_world(room, obstacles, {.centres = cells, .cell_size_m = cell_size});
    std::printf("grid %d x %d cells\n", p.width(), p.height());

    // ── 1. IS THE GOAL EVEN STANDABLE, AND AT WHICH HEADINGS? ───────────────────────────────────
    std::printf("\ngoal footprint-feasible by heading:\n  ");
    int free_headings = 0;
    for (int k = 0; k < 8; ++k)
    {
        const float yaw = static_cast<float>(k) * std::numbers::pi_v<float> / 4.f;
        const bool ok = p.pose_free(goal, yaw);
        free_headings += ok ? 1 : 0;
        std::printf("%3d°:%s  ", k * 45, ok ? "yes" : "NO ");
    }
    std::printf("\n  clearance at the goal: %.3f m | can turn there: %s\n",
                p.pose_clearance(goal, 0.f), p.can_turn_here(goal) ? "yes" : "NO");
    if (const auto n = p.nearest_free(goal, 0.f); n.has_value())
        std::printf("  nearest footprint-free pose: (%.2f,%.2f), %.2f m away\n",
                    n->x(), n->y(), (*n - goal).norm());
    else std::printf("  nearest footprint-free pose: NONE within the search radius\n");

    // ── 2. AND THE START? ───────────────────────────────────────────────────────────────────────
    std::printf("\nstart: clearance %.3f m | free at travel heading 0°: %s | can turn: %s\n",
                p.pose_clearance(start, 0.f), p.pose_free(start, 0.f) ? "yes" : "NO",
                p.can_turn_here(start) ? "yes" : "NO");

    // ── 3. THE ROUTE ────────────────────────────────────────────────────────────────────────────
    const auto route = p.plan(start, goal);
    if (route.has_value())
    {
        float len = 0;
        for (std::size_t i = 1; i < route->size(); ++i) len += ((*route)[i] - (*route)[i - 1]).norm();
        std::printf("\nplan: ROUTE FOUND, %zu turning points, %.2f m\n", route->size(), len);
    }
    else std::printf("\nplan: NO ROUTE — %s\n", p.last_failure().c_str());

    // ── 4. WHAT THE YAW CONVENTION IS WORTH ─────────────────────────────────────────────────────
    // The corrected rasterisation orients the body along its direction of travel; the legacy one was
    // a quarter turn off. The totals are equal by construction — read `lost`, which is how much of
    // the free (cell,heading) space changes hands.
    const auto c = p.orientation_census();
    std::printf("\nyaw convention: %ld states | free now %ld, legacy %ld | changed hands %ld (%.1f%%)\n",
                c.states, c.free_now, c.free_legacy, c.lost,
                c.free_now ? 100.0 * static_cast<double>(c.lost) / static_cast<double>(c.free_now) : 0.0);
    return 0;
}
