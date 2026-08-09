/*
 * ring_scene_graph.cpp — DSR writes for the ring_metaconcept agent.
 */

#include "ring_scene_graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::stamp_creation

namespace rc
{

namespace
{
std::uint64_t now_ms()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

// Republish thresholds. These are CHURN controls on the wire, not model quantities: below them the
// message is unchanged for any consumer, so rewriting it would only spam the CRDT.
constexpr float kYawEpsRad   = 0.009f;   // ~0.5°
constexpr float kKappaRelEps = 0.05f;    // 5 %

// ★HEARTBEAT. The payload self-gate above and the consumer's STALENESS check are in direct conflict:
// on a settled arrangement the message never changes, so rig_stamp_ms freezes, and the consumer
// correctly concludes that nobody is maintaining the prior and ignores it. Measured 2026-07-26: the
// chair read the edge once, then rejected it for the rest of the run (rig_found=0 with a plausible
// rig_prior_deg still cached). So the message IS the liveness signal — republish it periodically even
// when unchanged. Must stay well below the consumer's RigPrior.StaleMs (5000 ms).
constexpr std::uint64_t kHeartbeatMs = 1000;
}   // namespace

RingSceneGraph::RingSceneGraph(std::shared_ptr<DSR::DSRGraph> G, DSR::RT_API* rt_api, const RingConfig& cfg)
    : G_(std::move(G)), rt_api_(rt_api), cfg_(cfg)
{
}

std::uint64_t RingSceneGraph::ensure_rig_node(const RingBelief& belief, std::uint64_t room_id)
{
    if (not G_ or room_id == 0)
        return 0;

    auto room_opt = G_->get_node(room_id);
    if (not room_opt.has_value())
        return 0;

    const auto& s = belief.state();

    // ── Birth ────────────────────────────────────────────────────────────────
    if (rig_node_id_ == 0)
    {
        // Auto-name one past the highest existing dining_set_<N>, same NAME convention as the
        // level-1 agents (the class itself stays in object_subtype).
        int max_n = 0;
        const auto plen = cfg_.node_prefix.size();
        for (const auto& n : G_->get_nodes_by_type("metaconcept"))
            if (n.name().rfind(cfg_.node_prefix, 0) == 0)
                try { max_n = std::max(max_n, std::stoi(n.name().substr(plen))); } catch (...) {}
        const std::string name = cfg_.node_prefix + std::to_string(max_n + 1);

        // ★Type `metaconcept`, NOT `object`. A rig is a belief about a RELATION among nodes, not a
        // body: it has a footprint (the ring's extent) but nothing occupies it. Every furniture
        // consumer in the fleet gathers with get_nodes_by_type("object") — the voxelizer's three
        // draw passes, residual_concept's carve, the level-1 association scans — so publishing it
        // as an `object` made a 2r×2r phantom box that got drawn, carved and associated against.
        // The dedicated type is what excludes it everywhere at once; graph3d's SceneBuilder keys
        // Kind::Meta off it and is the one view that DOES draw it.
        DSR::Node node = DSR::Node::create<metaconcept_node_type>(name);
        G_->add_or_modify_attrib_local<object_subtype_att>(node, cfg_.node_subtype);
        G_->add_or_modify_attrib_local<level_att> (node, 3);
        G_->add_or_modify_attrib_local<parent_att>(node, room_id);
        {
            const float rpx = G_->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
            const float rpy = G_->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
            G_->add_or_modify_attrib_local<pos_x_att>(node, rpx - 150.f);
            G_->add_or_modify_attrib_local<pos_y_att>(node, rpy + 120.f);
        }
        // No mesh_path: a rig is an ABSTRACT grouping, not a solid. The viewer falls back to the
        // fitted box, which here is the ring's footprint — a flat disc-ish extent at floor level.
        rc::provenance::stamp_creation(*G_, node);   // birth stamp: epoch ms + local ISO-8601
        const auto id_opt = G_->insert_node(node);
        if (not id_opt.has_value())
            return 0;
        rig_node_id_ = id_opt.value();
        std::print("ring_metaconcept: BIRTH '{}' id={} at ({:.2f},{:.2f}) r={:.2f} slots={} logodds={:.2f}\n",
                   name, rig_node_id_, s.cx, s.cy, s.radius, belief.slot_count(), belief.log_odds());
    }

    auto node_opt = G_->get_node(rig_node_id_);
    if (not node_opt.has_value())   // someone deleted it under us — forget and re-birth next cycle
    {
        rig_node_id_ = 0;
        members_.clear();
        last_published_.clear();
        last_publish_ms_.clear();
        return 0;
    }
    auto node = node_opt.value();

    // ── Latent ───────────────────────────────────────────────────────────────
    G_->add_or_modify_attrib_local<rig_schema_att> (node, std::string("ring"));
    G_->add_or_modify_attrib_local<rig_radius_att> (node, s.radius);
    G_->add_or_modify_attrib_local<rig_n_slots_att>(node, belief.slot_count());
    G_->add_or_modify_attrib_local<rig_logodds_att>(node, belief.log_odds());

    // Footprint for the viewer: the ring's extent, flat on the floor.
    const bool geometry_changed = std::abs(s.radius - last_pub_radius_) > 0.02f
                               or belief.slot_count() != last_pub_slots_;
    if (geometry_changed)
    {
        G_->add_or_modify_attrib_local<width_m_att> (node, 2.0f * s.radius);
        G_->add_or_modify_attrib_local<depth_m_att> (node, 2.0f * s.radius);
        G_->add_or_modify_attrib_local<height_m_att>(node, 0.02f);
        last_pub_radius_ = s.radius;
        last_pub_slots_  = belief.slot_count();
    }
    G_->update_node(node);

    // Centre + phase as the room→rig RT edge (the rig IS placed in the world, it just has no body).
    if (rt_api_)
        rt_api_->insert_or_assign_edge_RT(room_opt.value(), rig_node_id_,
                                          {s.cx, s.cy, 0.0f}, {0.0f, 0.0f, s.yaw});
    return rig_node_id_;
}

void RingSceneGraph::remove_rig_node()
{
    if (not G_ or rig_node_id_ == 0)
        return;
    const auto id = rig_node_id_;
    rig_node_id_ = 0;
    members_.clear();
    last_published_.clear();
    last_publish_ms_.clear();
    last_pub_radius_ = -1.0f;
    last_pub_slots_  = -1;
    if (G_->delete_node(id))   // outgoing group_member edges go with the node
        std::print("ring_metaconcept: removed rig node id={}\n", id);
}

void RingSceneGraph::publish_member_prior(const MemberPrior& prior)
{
    if (not G_ or rig_node_id_ == 0 or prior.member_id == 0)
        return;

    // Self-gate: an unchanged message is not rewritten. A settled dining set would otherwise
    // republish four edges every cycle for no consumer-visible change. ★But an unchanged message
    // still has to prove it is LIVE — see kHeartbeatMs — so the gate lapses periodically.
    const std::uint64_t t = now_ms();
    if (const auto it = last_published_.find(prior.member_id); it != last_published_.end())
    {
        const auto& p = it->second;
        const float dyaw   = std::abs(std::remainder(prior.yaw_prior - p.yaw_prior, 2.0f * static_cast<float>(M_PI)));
        const float dkappa = std::abs(prior.yaw_kappa - p.yaw_kappa);
        const bool  same   = dyaw < kYawEpsRad
                         and dkappa < kKappaRelEps * std::max(1e-6f, p.yaw_kappa)
                         and prior.slot_index == p.slot_index;
        const auto  ht = last_publish_ms_.find(prior.member_id);
        const bool  fresh_enough = ht != last_publish_ms_.end() and (t - ht->second) < kHeartbeatMs;
        if (same and fresh_enough)
            return;
    }

    // Reuse the existing edge when present so we modify rather than churn it.
    auto edge_opt = G_->get_edge(rig_node_id_, prior.member_id, "group_member");
    DSR::Edge edge = edge_opt.has_value()
                         ? edge_opt.value()
                         : DSR::Edge::create<group_member_edge_type>(rig_node_id_, prior.member_id);

    G_->add_or_modify_attrib_local<rig_id_att>        (edge, rig_node_id_);
    G_->add_or_modify_attrib_local<rig_stamp_ms_att>  (edge, t);
    G_->add_or_modify_attrib_local<rig_slot_index_att>(edge, prior.slot_index);
    G_->add_or_modify_attrib_local<rig_yaw_prior_att> (edge, prior.yaw_prior);
    G_->add_or_modify_attrib_local<rig_yaw_kappa_att> (edge, prior.yaw_kappa);
    G_->add_or_modify_attrib_local<rig_slot_x_att>    (edge, prior.slot_xy.x());
    G_->add_or_modify_attrib_local<rig_slot_y_att>    (edge, prior.slot_xy.y());
    G_->add_or_modify_attrib_local<rig_slot_info_xx_att>(edge, prior.info_xx);
    G_->add_or_modify_attrib_local<rig_slot_info_yy_att>(edge, prior.info_yy);

    G_->insert_or_assign_edge(edge);
    members_.insert(prior.member_id);
    last_published_[prior.member_id]  = prior;
    last_publish_ms_[prior.member_id] = t;
}

// Zero the precision BEFORE deleting: a consumer that reads between our two writes must see an
// INERT prior, never a stale confident one. κ=0 means "ignore me" in the contract, so this is safe
// even if the delete is lost.
void RingSceneGraph::drop_member(std::uint64_t member_id)
{
    if (not G_ or rig_node_id_ == 0)
        return;
    if (auto edge_opt = G_->get_edge(rig_node_id_, member_id, "group_member"); edge_opt.has_value())
    {
        auto edge = edge_opt.value();
        G_->add_or_modify_attrib_local<rig_yaw_kappa_att>(edge, 0.0f);
        G_->insert_or_assign_edge(edge);
        G_->delete_edge(rig_node_id_, member_id, "group_member");
    }
    members_.erase(member_id);
    last_published_.erase(member_id);
    last_publish_ms_.erase(member_id);
}

void RingSceneGraph::retain_members(const std::unordered_set<std::uint64_t>& keep)
{
    std::vector<std::uint64_t> departed;
    for (const auto id : members_)
        if (not keep.contains(id))
            departed.push_back(id);
    for (const auto id : departed)
        drop_member(id);
}

}  // namespace rc
