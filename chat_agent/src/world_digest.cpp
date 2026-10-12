#include "world_digest.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <format>
#include <numbers>
#include <ranges>

#include "../../common/room_resolve/room_resolve.h"
#include "../../common/world_query/door_query.h"

namespace rc::chat
{

namespace
{
constexpr double kRad2Deg = 180.0 / std::numbers::pi;

// door_concept's affordance protocol (affordance_manager.h decode_protocol_state): pending and not active = offered,
// both = executing, neither = completed. Mirrors the controller's DoorPassage::state_of.
std::string aff_state(DSR::DSRGraph& G, const std::string& name)
{
    const auto n = G.get_node(name);
    if (not n.has_value()) return "absent";
    const bool a = G.get_attrib_by_name<active_att>(n.value()).value_or(false);
    const bool p = G.get_attrib_by_name<epistemic_pending_att>(n.value()).value_or(false);
    if (p and not a) return "offered";
    if (p and a) return "executing";
    if (not p and not a) return "completed";
    return "active";
}

// Body frame: x right, y FORWARD (the robot drives along +y). Bearing 0 = straight ahead, + = to the left.
std::string bearing_words(double deg)
{
    const double a = std::abs(deg);
    if (a <= 20.0) return "ahead";
    if (a >= 150.0) return "behind";
    const char* side = deg > 0 ? "left" : "right";
    if (a <= 60.0) return std::format("ahead-{}", side);
    if (a <= 120.0) return std::format("to the {}", side);
    return std::format("behind-{}", side);
}

std::string clock_now()
{
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tmv{};
    localtime_r(&t, &tmv);
    return std::format("{:02}:{:02}:{:02}", tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
}
}   // namespace

WorldDigest::WorldDigest(std::shared_ptr<DSR::DSRGraph> G, std::string robot_frame)
    : G_(std::move(G)), robot_frame_(std::move(robot_frame)) {}

Digest WorldDigest::build(DSR::InnerEigenAPI& inner, bool controller_present, const IntentionStatus& intention) const
{
    Digest d;
    auto& G = *G_;
    d.facts.controller_present = controller_present;
    std::string s;
    s += std::format("time: {}\n", clock_now());
    s += std::format("controller (the agent that drives the robot): {}\n", controller_present ? "running" : "NOT running");

    // ── rooms ──
    std::uint64_t room_id = 0;
    if (const auto r = rc::room::current_room(G); r.has_value()) room_id = *r;
    std::string room_name;
    if (room_id != 0)
        if (const auto rn = G.get_node(room_id); rn.has_value()) room_name = rn->name();
    d.facts.room = room_name;
    std::string others;
    for (const auto& r : G.get_nodes_by_type("room"))
    {
        if (r.id() == room_id) continue;
        others += (others.empty() ? "" : ", ") + r.name() + (rc::room::is_proto(G, r.id()) ? " (provisional)" : "");
    }
    s += std::format("current room: {}\n", room_name.empty() ? std::string("unknown") : room_name);
    s += std::format("other rooms in the live map: {}\n", others.empty() ? std::string("none") : others);

    // ── robot pose in the room frame ──
    if (not room_name.empty())
    {
        if (const auto T = inner.get_transformation_matrix(room_name, robot_frame_); T.has_value())
        {
            const auto& M = T->matrix();
            const double yaw = std::atan2(M(1, 0), M(0, 0)) * kRad2Deg;
            s += std::format("robot pose in {}: x={:.2f} m, y={:.2f} m, yaw={:.0f} deg\n", room_name,
                             T->translation().x(), T->translation().y(), yaw);
        }
        else s += "robot pose: unknown (no transform room -> robot)\n";
    }
    else s += "robot pose: unknown (no current room)\n";

    // ── doors of the current room ──
    if (room_id != 0)
    {
        d.facts.doors = rc::world::doors_of_room(G, room_id);
        d.facts.nearest_door = rc::world::nearest_door(G, inner, room_name, robot_frame_);
    }
    // recall_<room> metaconcepts: parent = the door behind which that room is remembered.
    std::vector<std::pair<std::string, std::string>> recalls;   // (door, room)
    for (const auto& n : G.get_nodes_by_type("metaconcept"))
    {
        if (not n.name().starts_with("recall_")) continue;
        const auto parent = G.get_attrib_by_name<parent_att>(n);
        if (not parent.has_value()) continue;
        if (const auto door = G.get_node(parent.value()); door.has_value())
            recalls.emplace_back(door->name(), n.name().substr(7));
    }

    s += std::format("doors of the current room ({}):\n", d.facts.doors.size());
    if (d.facts.doors.empty()) s += "  none known\n";
    for (const auto& door : d.facts.doors)
    {
        std::string line = "  - " + door + ":";
        if (const auto T = inner.get_transformation_matrix(robot_frame_, door); T.has_value())
        {
            const double x = T->translation().x(), y = T->translation().y();
            const double dist = std::hypot(x, y);
            const double bearing = std::atan2(-x, y) * kRad2Deg;   // 0 = ahead (+y), + = left
            line += std::format(" {:.1f} m away, bearing {:+.0f} deg ({})", dist, bearing, bearing_words(bearing));
        }
        else line += " position unknown";
        if (const auto dn = G.get_node(door); dn.has_value())
        {
            if (const auto p = G.get_attrib_by_name<door_open_prob_att>(dn.value()); p.has_value())
                line += std::format(", P(open)={:.2f}", p.value());
            else line += ", P(open)=unknown";
        }
        line += std::format(", affordances: approach={} open={} cross={}", aff_state(G, "aff_" + door + "_approach"),
                            aff_state(G, "aff_" + door + "_open"), aff_state(G, "aff_" + door + "_cross"));
        for (const auto& [rd, rr] : recalls)
            if (rd == door) line += ", behind it: remembered " + rr;
        if (d.facts.nearest_door.has_value() and *d.facts.nearest_door == door) line += " [nearest]";
        s += line + "\n";
    }
    std::string rem;
    for (const auto& [rd, rr] : recalls) rem += (rem.empty() ? "" : "; ") + rr + " (behind " + rd + ")";
    s += std::format("remembered rooms (not in the live map, known from memory): {}\n", rem.empty() ? std::string("none") : rem);

    // ── the chat's own command ──
    if (intention.command.empty()) s += "last command sent from this chat: none\n";
    else
    {
        // bt_state = "<step>|<last_event>|driving=on|off"
        std::vector<std::string> parts;
        for (auto&& p : intention.bt_state | std::views::split('|')) parts.emplace_back(p.begin(), p.end());
        s += std::format("last command sent from this chat: \"{}\", status: {}", intention.command,
                         intention.state.empty() ? std::string("not yet acknowledged") : intention.state);
        if (parts.size() >= 1 and not parts[0].empty() and parts[0] != "-") s += ", step: " + parts[0];
        if (parts.size() >= 2 and not parts[1].empty()) s += ", last event: " + parts[1];
        if (parts.size() >= 3 and parts[2] == "driving=off") s += ", NOTE: the base is not armed (Run not pressed)";
        s += "\n";
    }
    d.text = std::move(s);
    return d;
}

}   // namespace rc::chat
