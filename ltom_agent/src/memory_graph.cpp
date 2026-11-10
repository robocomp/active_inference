#include "memory_graph.h"
#include "recall_publisher.h"
#include "../../common/rt_covariance/rt_covariance.h"

#include <array>
#include <ctime>
#include <format>
#include <print>

namespace ltom
{
namespace
{
// Canvas layout of the viewer only: rooms in a row below the root, each object below its room, laid out as it stood.
constexpr float kRoomSpacing = 300.f, kRoomRow = 120.f, kObjectRow = 300.f, kCanvasPerMetre = 25.f;

std::string room_name(std::int64_t key) { return key < 0 ? "room_unknown" : "room_" + std::to_string(key); }
float room_canvas_x(std::int64_t key) { return kRoomSpacing * static_cast<float>(key - 1); }   // room_1 under the root

std::string clock_of(std::int64_t ms)
{
    const std::time_t t = ms / 1000;
    std::tm local{};
    localtime_r(&t, &local);
    std::array<char, 16> text{};
    const auto n = std::strftime(text.data(), text.size(), "%H:%M:%S", &local);
    return {text.data(), n};
}
}   // namespace

std::string history_text(const Entry& e)
{
    std::string out = std::format("sightings: {}\nkeyframes: {}", e.n_sightings, e.keyframes.size());
    int i = 0;
    for (const auto& l : e.lives)
        out += std::format("\nlife {}: {} -> {} {}", ++i, clock_of(l.first_seen_ms), l.end == End::alive ? "now" : clock_of(l.end_ms), end_name(l.end));
    return out;
}

MemoryGraph::MemoryGraph(std::shared_ptr<DSR::DSRGraph> G) : G_(std::move(G)), rt_api_(G_->get_rt_api()) {}

void MemoryGraph::warn_once(const std::string& name, const char* what)
{
    if (warned_.insert(name + "|" + what).second)
        std::println(stderr, "[ltom] memory graph {}: {}", name, what);
}

std::uint64_t MemoryGraph::ensure_room(std::int64_t key)
{
    const std::string name = room_name(key);
    if (const auto room = G_->get_node(name)) return room->id();
    auto root = G_->get_node("root");
    if (not root) { warn_once(name, "no root node"); return 0; }
    auto n = DSR::Node::create<room_node_type>(name);
    G_->add_or_modify_attrib_local<parent_att>(n, root->id());
    G_->add_or_modify_attrib_local<level_att>(n, 1);
    G_->add_or_modify_attrib_local<pos_x_att>(n, room_canvas_x(key));
    G_->add_or_modify_attrib_local<pos_y_att>(n, kRoomRow);
    if (key >= 0) G_->add_or_modify_attrib_local<room_id_att>(n, static_cast<std::uint64_t>(key));
    const auto id = G_->insert_node(n);
    if (not id) { warn_once(name, "insert_node failed"); return 0; }
    rt_api_->insert_or_assign_edge_RT(*root, *id, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
    return *id;
}

void MemoryGraph::write_pose(std::uint64_t room_id, std::uint64_t id, const std::string& name, const Keyframe& kf)
{
    auto room = G_->get_node(room_id);
    if (not room) { warn_once(name, "room node vanished"); return; }
    rt_api_->insert_or_assign_edge_RT(*room, id, {static_cast<float>(kf.pose.x), static_cast<float>(kf.pose.y), 0.f},
                                      {0.f, 0.f, static_cast<float>(kf.pose.yaw)});
    float last_trace = 0.f;
    if (not rc::rtcov::publish(*G_, room_id, id, keyframe_variances(kf), last_trace, true, name))
        warn_once(name, "rt_covariance write failed");
}

void MemoryGraph::show(const Entry& e, std::uint64_t room_id, const std::string& room, const std::string& state)
{
    const auto history = history_text(e);
    const bool has_kf = not e.keyframes.empty();
    const Keyframe kf = has_kf ? e.keyframes.back() : Keyframe{};
    auto node = G_->get_node(e.name);
    const auto seen = written_.find(e.uid);
    const bool moved_room = node.has_value() and seen != written_.end() and seen->second.room != room;
    const bool attrs_stale = not node or seen == written_.end() or moved_room or seen->second.state != state or seen->second.history != history;

    if (moved_room)
        if (const auto old_parent = G_->get_attrib_by_name<parent_att>(*node))
            G_->delete_edge(*old_parent, node->id(), "RT");

    if (attrs_stale)
    {
        DSR::Node n = node ? *node : DSR::Node::create<object_node_type>(e.name);
        G_->add_or_modify_attrib_local<object_uid_att>(n, e.uid);
        G_->add_or_modify_attrib_local<object_subtype_att>(n, e.cls);
        G_->add_or_modify_attrib_local<width_m_att>(n, e.w);
        G_->add_or_modify_attrib_local<depth_m_att>(n, e.d);
        G_->add_or_modify_attrib_local<height_m_att>(n, e.h);
        G_->add_or_modify_attrib_local<parent_att>(n, room_id);
        G_->add_or_modify_attrib_local<level_att>(n, 2);
        G_->add_or_modify_attrib_local<state_att>(n, state);
        if (not e.lives.empty())
            G_->add_or_modify_attrib_local<timestamp_alivetime_att>(n, static_cast<std::uint64_t>(e.lives.back().last_seen_ms));
        G_->add_or_modify_attrib_local<em_record_att>(n, history);
        if (not node or moved_room)
        {
            const auto rn = G_->get_node(room_id);
            const float rx = rn ? G_->get_attrib_by_name<pos_x_att>(*rn).value_or(0.f) : 0.f;
            G_->add_or_modify_attrib_local<pos_x_att>(n, rx + kCanvasPerMetre * static_cast<float>(kf.pose.x));
            G_->add_or_modify_attrib_local<pos_y_att>(n, kObjectRow + kCanvasPerMetre * static_cast<float>(kf.pose.y));
        }
        if (node)
        {
            if (not G_->update_node(n)) { warn_once(e.name, "update_node failed"); return; }
        }
        else
        {
            const auto id = G_->insert_node(n);
            if (not id) { warn_once(e.name, "insert_node failed"); return; }
            node = G_->get_node(*id);
            if (not node) return;
        }
    }

    const auto same_pose = [&](const Pose2& a, const Pose2& b)
    { return a.x == b.x and a.y == b.y and a.yaw == b.yaw and a.vx == b.vx and a.vy == b.vy and a.vyaw == b.vyaw; };
    const bool pose_stale = moved_room or seen == written_.end() or seen->second.has_kf != has_kf or (has_kf and not same_pose(seen->second.kf.pose, kf.pose));
    if (pose_stale)
    {
        if (has_kf) write_pose(room_id, node->id(), e.name, kf);
        else if (auto rn = G_->get_node(room_id)) rt_api_->insert_or_assign_edge_RT(*rn, node->id(), {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
    }
    written_[e.uid] = {.room = room, .state = state, .history = history, .kf = kf, .has_kf = has_kf};
}

void MemoryGraph::sync(const std::map<std::uint64_t, Entry>& entries, const std::set<std::uint64_t>& present)
{
    for (const auto& [uid, e] : entries)
    {
        const std::int64_t key = e.keyframes.empty() ? -1 : e.keyframes.back().room_key;
        const auto room_id = ensure_room(key);
        if (room_id == 0) continue;
        show(e, room_id, room_name(key), present.contains(uid) ? "live" : "remembered");
    }
}
}   // namespace ltom
