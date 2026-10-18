// The system tools' pure side (src/system_facts.h): agents, sensor rates, graph summary, node inspection.
#include "../src/system_facts.h"

#include <clocale>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)
bool has(const std::string& s, std::string_view part) { return s.find(part) != std::string::npos; }

using namespace rc::chat;
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // es_ES here: numbers must still print with a point

    // ── agents: running = a heartbeat within kAgentStaleMs; a node with no recent beat is a leftover ────────────
    const std::vector<AgentFact> agents{   // a silent one first: answer_agents() must sort the running ones ahead
        {.name = "bottle_concept", .id = 23, .presence = "Operating", .fsm = "Compute", .heartbeat_age_ms = 95'000},
        {.name = "controller", .id = 7, .presence = "Operating", .fsm = "Compute", .heartbeat_age_ms = 400,
         .uptime_s = 3725, .cpu = 41.5f, .memory_kb = 812'000},
        {.name = "door_concept", .id = 12, .presence = "Degraded", .fsm = "Compute", .heartbeat_age_ms = 900},
        {.name = "retina", .id = 3, .presence = "Operating", .fsm = "Compute", .heartbeat_age_ms = 1200, .cpu = 180.f},
        {.name = "chat_agent", .id = 25, .presence = "", .fsm = "", .heartbeat_age_ms = std::nullopt},
    };
    CHECK(agent_running(agents[1]) and agent_running(agents[3]));
    CHECK(not agent_running(agents[0]) and not agent_running(agents[4]));
    {
        const std::string all = answer_agents(agents);
        CHECK(has(all, "3 agents running"));
        CHECK(has(all, "controller (id 7): Operating, Compute, up 1:02:05, CPU 41.5 %, memory 793 MB"));
        CHECK(has(all, "door_concept (id 12): Degraded"));
        CHECK(has(all, "retina (id 3): Operating, Compute, CPU 180.0 %"));   // >100 %: more than one core
        CHECK(has(all, "bottle_concept (id 23): not heard from for 95 s (its node is left over)"));
        CHECK(has(all, "chat_agent (id 25): no heartbeat yet"));
        CHECK(all.find("controller") < all.find("bottle_concept"));   // running first
    }

    // ── the STATE's agents line: the names alive, from the same heartbeat rule (live 2026-10-18: with no list in the
    // STATE the model answered "what agents are alive?" with the four EXAMPLE names of agents()'s description) ──
    // …and an alive agent not Operating is marked, so "is any agent in trouble?" is not answered "all fine"
    CHECK(alive_line(agents) == "agents alive now: controller, door_concept (running, but Degraded), retina (3; agents() for "
                                "health, CPU, memory); not running: bottle_concept, chat_agent");
    CHECK(alive_line({}) == "agents alive now: none known");

    // ── rate meter: frames counted by frame_id between samples, over the last kRateWindowNs ─────────────────────
    {
        RateMeter m;
        CHECK(not m.fps().has_value());
        m.push(100, 1'000'000'000);
        CHECK(not m.fps().has_value());   // one sample is no rate
        m.push(115, 1'500'000'000);       // 15 frames in 0.5 s
        m.push(130, 2'000'000'000);
        CHECK(m.fps().has_value() and *m.fps() == 30.0);
        m.push(130, 2'500'000'000);       // the same frame again: nothing new arrived, not a new sample
        CHECK(*m.fps() == 30.0 and m.last_ns() == 2'000'000'000);
        for (std::uint64_t k = 1; k <= 20; ++k) m.push(130 + 5 * k, 2'000'000'000 + k * 500'000'000);   // slows to 10 fps
        CHECK(*m.fps() == 10.0);   // the old 30 fps samples have left the window
        m.push(10, 20'000'000'000);       // the producer restarted (frame_id went back): start over
        CHECK(not m.fps().has_value());
    }

    // ── sensors ──────────────────────────────────────────────────────────────────────────────────────────────────
    {
        const std::vector<SensorFact> sensors{
            {.node = "ricoh", .label = "360-degree camera", .streams = {"rgb360"}, .fps = 14.96, .frame_age_ms = 70,
             .width = 1920, .height = 960, .media_bps = 41.2e6f, .nominal_hz = 15.f},
            {.node = "zed", .label = "front camera", .streams = {"rgb", "depth"}, .fps = std::nullopt,
             .frame_age_ms = 12'000, .media_bps = std::nullopt},
            {.node = "lidar3D", .label = "3D LiDAR", .streams = {"lidar"}, .media_bps = 3.5e6f, .nominal_hz = 10.f},
        };
        const std::string all = answer_sensors(sensors, "");
        CHECK(has(all, "ricoh (360-degree camera): streams rgb360; 15.0 frames/s measured (nominal 15 Hz), "
                       "1920x960, last frame 70 ms ago, 41.2 MB/s"));
        CHECK(has(all, "zed (front camera): streams rgb, depth; no frames for 12.0 s"));
        CHECK(has(all, "lidar3D (3D LiDAR): streams lidar; rate not measured (nominal 10 Hz), 3.5 MB/s"));
        const std::string one = answer_sensors(sensors, "360");
        CHECK(has(one, "ricoh") and not has(one, "zed"));
        CHECK(has(answer_sensors(sensors, "Ricoh"), "ricoh"));        // node name, any case
        CHECK(has(answer_sensors(sensors, "front camera"), "zed"));   // label
        CHECK(has(answer_sensors(sensors, "kinect"), "no sensor matches 'kinect'"));
        CHECK(has(answer_sensors(sensors, "kinect"), "ricoh, zed, lidar3D"));   // and says what there is
    }

    // ── graph summary ───────────────────────────────────────────────────────────────────────────────────────────
    {
        GraphSummary g{.nodes_by_type = {{"object", 14}, {"agent", 18}, {"room", 1}, {"affordance", 6}},
                       .edges_by_type = {{"RT", 40}, {"has", 19}},
                       .nodes_by_agent = {{"chair_concept", 4}, {"door_concept", 6}, {"robot_concept", 12}},
                       .room = "room_2"};
        const std::string s = answer_graph(g);
        CHECK(has(s, "39 nodes, 59 edges; current room room_2"));
        CHECK(has(s, "nodes by type: agent 18, object 14, affordance 6, room 1"));   // most first
        CHECK(has(s, "edges by type: RT 40, has 19"));
        CHECK(has(s, "nodes created by: robot_concept 12, door_concept 6, chair_concept 4"));
        // "how many agents are alive in the WM?" picks graph_summary as often as agents (the WM IS the graph), so the
        // summary carries the agents census too: whichever tool the model calls, the answer is in the result.
        g.agents = agents;
        const std::string w = answer_graph(g);
        CHECK(has(w, "agents: 3 running (controller, door_concept, retina); 2 not running (bottle_concept, chat_agent)"));
        CHECK(not has(answer_graph(GraphSummary{}), "agents:"));   // no agent nodes, no line
    }

    // ── inspect: every attribute, long values cut, edges both ways ─────────────────────────────────────────────
    {
        NodeView v{.name = "door_1", .type = "object", .id = 42, .created_by = "door_concept",
                   .attrs = {{"door_open_prob", "0.83"}, {"agent_config", std::string(500, 'x')}, {"width_m", "0.9"}},
                   .edges_out = {{"RT", "aff_door_1_approach"}}, .edges_in = {{"RT", "room_2"}}};
        const std::string s = answer_inspect(v);
        CHECK(has(s, "door_1 (type object, id 42, created by door_concept)"));
        CHECK(has(s, "door_open_prob = 0.83"));
        CHECK(has(s, "agent_config = xxxx") and has(s, "… (500 chars)"));
        CHECK(s.size() < 1200);
        CHECK(has(s, "edges out: RT → aff_door_1_approach"));
        CHECK(has(s, "edges in: RT ← room_2"));
        CHECK(s.find("agent_config") < s.find("door_open_prob"));   // alphabetical
    }

    std::println("test_system: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
