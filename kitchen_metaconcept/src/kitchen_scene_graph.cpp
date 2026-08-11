/*
 * kitchen_scene_graph.cpp — see the header. Every DSR write this agent makes is here.
 */

#include "kitchen_scene_graph.h"

#include <chrono>
#include <cmath>
#include <print>

namespace rc {

namespace {

std::uint64_t now_ms()
{
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

// Refresh the stamp at least this often even when the payload has not changed, so a consumer's
// staleness check sees a live frame. Without it the payload self-gate would starve the very check
// that protects a member from a frame that has died.
constexpr std::uint64_t kHeartbeatMs = 1000;

bool same(const KitchenMemberPrior& a, const KitchenMemberPrior& b)
{
    const auto eq = [](float x, float y) { return std::abs(x - y) < 1e-4f; };
    return eq(a.yaw, b.yaw) and eq(a.yaw_kappa, b.yaw_kappa)
       and eq(a.end_lo_info, b.end_lo_info) and eq(a.end_hi_info, b.end_hi_info)
       and (a.end_lo - b.end_lo).norm() < 1e-3f and (a.end_hi - b.end_hi).norm() < 1e-3f;
}

}  // namespace

KitchenSceneGraph::KitchenSceneGraph(std::shared_ptr<DSR::DSRGraph> G, DSR::RT_API* rt_api,
                                     const KitchenConfig& cfg)
    : G_(std::move(G)), rt_api_(rt_api), cfg_(cfg) {}

std::uint64_t KitchenSceneGraph::ensure_node(const KitchenBelief& belief, std::uint64_t room_id)
{
    if (not G_ or room_id == 0)
        return 0;

    if (node_id_ == 0)
    {
        const std::string name = cfg_.node_prefix + "0";
        // DSR type `metaconcept`, not `object` — see the header. A relation is not a solid.
        DSR::Node node = DSR::Node::create<metaconcept_node_type>(name);
        G_->add_or_modify_attrib_local<object_subtype_att>(node, cfg_.node_subtype);
        G_->add_or_modify_attrib_local<level_att> (node, 3);
        G_->add_or_modify_attrib_local<parent_att>(node, room_id);
        if (const auto room = G_->get_node(room_id); room.has_value())
        {
            const float rpx = G_->get_attrib_by_name<pos_x_att>(room.value()).value_or(200.f);
            const float rpy = G_->get_attrib_by_name<pos_y_att>(room.value()).value_or(200.f);
            G_->add_or_modify_attrib_local<pos_x_att>(node, rpx - 150.f);
            G_->add_or_modify_attrib_local<pos_y_att>(node, rpy - 120.f);
        }
        const auto id = G_->insert_node(node);
        if (not id.has_value())
            return 0;
        node_id_ = id.value();
        std::print("kitchen_metaconcept: created '{}' id={}\n", name, node_id_);
    }

    auto node = G_->get_node(node_id_);
    if (not node.has_value()) { node_id_ = 0; return 0; }
    const auto& s = belief.state();
    G_->add_or_modify_attrib_local<rig_schema_att> (node.value(), std::string("rectilinear"));
    G_->add_or_modify_attrib_local<rig_logodds_att>(node.value(), belief.log_odds());
    G_->add_or_modify_attrib_local<rig_yaw_prior_att>(node.value(), s.axis);
    G_->update_node(node.value());
    return node_id_;
}

void KitchenSceneGraph::remove_node()
{
    if (not G_ or node_id_ == 0)
        return;
    G_->delete_node(node_id_);       // its outgoing group_member edges go with it
    std::print("kitchen_metaconcept: removed node id={}\n", node_id_);
    node_id_ = 0;
    members_.clear();
    last_published_.clear();
    last_publish_ms_.clear();
}

void KitchenSceneGraph::publish_member_prior(const KitchenMemberPrior& p)
{
    if (not G_ or node_id_ == 0 or p.member_id == 0)
        return;

    // Self-gate on the payload, with a liveness heartbeat so the consumer's staleness check still ticks.
    const std::uint64_t t = now_ms();
    if (const auto prev = last_published_.find(p.member_id);
        prev != last_published_.end() and same(prev->second, p))
    {
        const auto lt = last_publish_ms_.find(p.member_id);
        if (lt != last_publish_ms_.end() and t - lt->second < kHeartbeatMs)
            return;
    }

    auto edge_opt = G_->get_edge(node_id_, p.member_id, "group_member");
    DSR::Edge edge = edge_opt.has_value()
                         ? edge_opt.value()
                         : DSR::Edge::create<group_member_edge_type>(node_id_, p.member_id);
    G_->add_or_modify_attrib_local<rig_id_att>       (edge, node_id_);
    G_->add_or_modify_attrib_local<rig_stamp_ms_att> (edge, t);
    G_->add_or_modify_attrib_local<rig_yaw_prior_att>(edge, p.yaw);
    G_->add_or_modify_attrib_local<rig_yaw_kappa_att>(edge, p.yaw_kappa);
    G_->add_or_modify_attrib_local<rig_end_lo_x_att>   (edge, p.end_lo.x());
    G_->add_or_modify_attrib_local<rig_end_lo_y_att>   (edge, p.end_lo.y());
    G_->add_or_modify_attrib_local<rig_end_lo_info_att>(edge, p.end_lo_info);
    G_->add_or_modify_attrib_local<rig_end_hi_x_att>   (edge, p.end_hi.x());
    G_->add_or_modify_attrib_local<rig_end_hi_y_att>   (edge, p.end_hi.y());
    G_->add_or_modify_attrib_local<rig_end_hi_info_att>(edge, p.end_hi_info);
    G_->insert_or_assign_edge(edge);

    members_.insert(p.member_id);
    last_published_[p.member_id] = p;
    last_publish_ms_[p.member_id] = t;
}

void KitchenSceneGraph::drop_member(std::uint64_t member_id)
{
    if (not G_ or node_id_ == 0)
        return;
    // Zero the precisions BEFORE removing the edge. A consumer that reads between the two sees an
    // inert message rather than a stale one it would keep obeying.
    if (auto e = G_->get_edge(node_id_, member_id, "group_member"); e.has_value())
    {
        G_->add_or_modify_attrib_local<rig_yaw_kappa_att>  (e.value(), 0.0f);
        G_->add_or_modify_attrib_local<rig_end_lo_info_att>(e.value(), 0.0f);
        G_->add_or_modify_attrib_local<rig_end_hi_info_att>(e.value(), 0.0f);
        G_->insert_or_assign_edge(e.value());
    }
    G_->delete_edge(node_id_, member_id, "group_member");
    members_.erase(member_id);
    last_published_.erase(member_id);
    last_publish_ms_.erase(member_id);
}

void KitchenSceneGraph::retain_members(const std::unordered_set<std::uint64_t>& keep)
{
    std::vector<std::uint64_t> gone;
    for (const auto id : members_)
        if (not keep.contains(id)) gone.push_back(id);
    for (const auto id : gone) drop_member(id);
}

}  // namespace rc
