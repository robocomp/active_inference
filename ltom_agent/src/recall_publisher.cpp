#include "recall_publisher.h"

#include <map>
#include <print>
#include <string>
#include <vector>

namespace ltom
{
namespace
{
constexpr std::string_view kPrefix = "ltom_recall_";
// Not read by any consumer of a recall node (they use x, y, yaw): z is the floor, roll/pitch the shared floor prior.
constexpr float kVarZ = 1e-4f;

std::string name_of(std::uint64_t uid) { return std::string(kPrefix) + std::to_string(uid); }
}   // namespace

rc::rtcov::Se3Var keyframe_variances(const Keyframe& k)
{
    return {static_cast<float>(k.pose.vx), static_cast<float>(k.pose.vy), kVarZ,
            rc::rtcov::kFlatRollPitchVar, rc::rtcov::kFlatRollPitchVar, static_cast<float>(k.pose.vyaw)};
}

RecallPublisher::RecallPublisher(std::shared_ptr<DSR::DSRGraph> G) : G_(std::move(G)), rt_api_(G_->get_rt_api()) {}

void RecallPublisher::warn_once(const std::string& name, const char* what)
{
    if (warned_.insert(name + "|" + what).second)
        std::println(stderr, "[ltom] recall {}: {}", name, what);
}

// What this class last wrote for the node. The RT edge holds a history ring, so reading the pose back means picking the
// head block; comparing against our own record is exact and simpler (this class is the only writer of these edges).
bool RecallPublisher::pose_current(const Recall& r) const
{
    const auto it = written_.find(r.uid);
    if (it == written_.end()) return false;
    const auto& a = it->second.pose;
    const auto& b = r.kf.pose;
    return a.x == b.x and a.y == b.y and a.yaw == b.yaw and a.vx == b.vx and a.vy == b.vy and a.vyaw == b.vyaw;
}

void RecallPublisher::write_pose(std::uint64_t room_id, std::uint64_t node_id, const std::string& name, const Recall& r)
{
    const auto& p = r.kf.pose;
    auto room = G_->get_node(room_id);
    if (not room) { warn_once(name, "room node vanished"); return; }
    rt_api_->insert_or_assign_edge_RT(*room, node_id,
                                      {static_cast<float>(p.x), static_cast<float>(p.y), 0.0f},
                                      {0.0f, 0.0f, static_cast<float>(p.yaw)});
    float last_trace = 0.0f;
    // Record the pose as written only when the edge exists AND its covariance landed: otherwise the next sync retries.
    if (not G_->get_edge(room_id, node_id, "RT")) { warn_once(name, "RT edge write failed"); return; }
    if (not rc::rtcov::publish(*G_, room_id, node_id, keyframe_variances(r.kf), last_trace, true, name))
    {
        warn_once(name, "rt_covariance write failed");
        return;
    }
    written_[r.uid] = r.kf;
}

void RecallPublisher::insert(std::uint64_t room_id, const Recall& r)
{
    const std::string name = name_of(r.uid);
    const auto room = G_->get_node(room_id);
    if (not room) { warn_once(name, "room node vanished"); return; }
    auto n = DSR::Node::create<metaconcept_node_type>(name);
    G_->add_or_modify_attrib_local<object_uid_att>(n, r.uid);
    G_->add_or_modify_attrib_local<object_subtype_att>(n, r.cls);
    G_->add_or_modify_attrib_local<parent_att>(n, room_id);
    G_->add_or_modify_attrib_local<level_att>(n, G_->get_attrib_by_name<level_att>(*room).value_or(1) + 1);
    const auto id = G_->insert_node(n);
    if (not id) { warn_once(name, "insert_node failed"); return; }
    write_pose(room_id, *id, name, r);
    std::println("[ltom] recall offered {} ({}) in room {}", name, r.cls, room->name());
}

void RecallPublisher::sync(std::optional<std::uint64_t> room_id, std::span<const Recall> want)
{
    std::map<std::string, const Recall*> wanted;
    if (room_id)
        for (const auto& r : want) wanted.emplace(name_of(r.uid), &r);

    std::set<std::string> kept;
    std::vector<std::pair<std::uint64_t, std::string>> to_delete;
    for (const auto& n : G_->get_nodes_by_type("metaconcept"))
    {
        if (not n.name().starts_with(kPrefix)) continue;
        const auto it = wanted.find(n.name());
        // Not wanted, or hanging under another room (we re-create it under this one): delete.
        if (it == wanted.end() or not G_->get_edge(*room_id, n.id(), "RT")) { to_delete.emplace_back(n.id(), n.name()); continue; }
        kept.insert(n.name());
        if (not pose_current(*it->second))
            write_pose(*room_id, n.id(), n.name(), *it->second);   // move: same node, new RT pose
    }
    for (const auto& [id, name] : to_delete)
        if (G_->delete_node(id)) { std::println("[ltom] recall withdrawn {}", name); }
        else warn_once(name, "delete_node failed");
    for (auto it = written_.begin(); it != written_.end();)   // forget what is no longer on the graph
        it = wanted.contains(name_of(it->first)) and kept.contains(name_of(it->first)) ? std::next(it) : written_.erase(it);

    for (const auto& [name, r] : wanted)
        if (not kept.contains(name)) insert(*room_id, *r);
}
}   // namespace ltom
