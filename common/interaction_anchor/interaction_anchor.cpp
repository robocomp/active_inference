/*
 * interaction_anchor.cpp — see interaction_anchor.h for the contract (who writes what, and why the
 * link to the original is interaction_node_id rather than `parent`).
 */

#include "interaction_anchor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <print>
#include <string>
#include <vector>

#include "../graph_provenance/creation_stamp.h"

namespace rc::interaction
{

namespace
{
// (x, y, yaw) inside the row-major 6x6 SE(3) block [x,y,z,rx,ry,rz] RT_API stores on an edge.
constexpr std::array<int, 3> kSe3OfSe2{0, 1, 5};

std::optional<DSR::Node> robot_node(DSR::DSRGraph& G)
{
    // By TYPE, as rc::rt and mask_ingestor do: the name ("Shadow") belongs to one robot model.
    const auto robots = G.get_nodes_by_type("robot");
    if (robots.empty()) return std::nullopt;
    return robots.front();
}

// Drop affordance.interaction_node_id if (and only if) it still names `anchor_id`. Fresh read, then a
// whole-node update: update_node is a replace, so a cached copy would revert the producer's other
// attributes.
void unlink_affordance(DSR::DSRGraph& G, std::uint64_t affordance_id, std::uint64_t anchor_id)
{
    auto aff = G.get_node(affordance_id);
    if (not aff.has_value()) return;
    const auto link = G.get_attrib_by_name<interaction_node_id_att>(aff.value());
    if (not link.has_value() or link.value() != anchor_id) return;      // relinked to another, or never set
    G.remove_attrib_local<interaction_node_id_att>(aff.value());
    G.update_node(aff.value());
}
}  // namespace

std::optional<Anchor> Anchor::create(DSR::DSRGraph& G, std::uint64_t object_id, std::uint64_t affordance_id,
                                     int agent_id, const RelPose& init)
{
    const auto obj   = G.get_node(object_id);
    const auto aff   = G.get_node(affordance_id);
    const auto robot = robot_node(G);
    if (not obj.has_value() or not aff.has_value() or not robot.has_value())
        return std::nullopt;

    // One anchor per object. A same-named node of OURS is a copy a previous execution left behind (or a
    // crashed run's, if the sweep has not reached it yet — possibly with no RT edge): reclaim it.
    // Anything else, including another agent's copy, is not ours to delete.
    const std::string name = obj->name() + std::string(kAnchorSuffix);
    if (const auto old = G.get_node(name); old.has_value())
    {
        if (not is_own_anchor_node(G, old.value(), agent_id)) return std::nullopt;
        if (not G.delete_node(old->id())) return std::nullopt;
    }

    // Shallow copy: identity and size, nothing the owner's belief keeps (the copy is a pose handle).
    DSR::Node n = DSR::Node::create<object_node_type>(name);
    if (const auto v = G.get_attrib_by_name<object_subtype_att>(obj.value()); v.has_value())
        G.add_or_modify_attrib_local<object_subtype_att>(n, v.value());
    if (const auto v = G.get_attrib_by_name<width_m_att>(obj.value()); v.has_value())
        G.add_or_modify_attrib_local<width_m_att>(n, v.value());
    if (const auto v = G.get_attrib_by_name<height_m_att>(obj.value()); v.has_value())
        G.add_or_modify_attrib_local<height_m_att>(n, v.value());
    if (const auto v = G.get_attrib_by_name<depth_m_att>(obj.value()); v.has_value())
        G.add_or_modify_attrib_local<depth_m_att>(n, v.value());
    G.add_or_modify_attrib_local<interaction_node_id_att>(n, object_id);
    G.add_or_modify_attrib_local<agent_id_att>(n, static_cast<std::uint32_t>(agent_id));
    G.add_or_modify_attrib_local<valid_att>(n, false);   // I-2: born NOT drivable; the owner marks it once it has a sighting
    rc::provenance::stamp_creation(G, n);
    const auto id = G.insert_node(n);
    if (not id.has_value())
        return std::nullopt;

    Anchor a;
    a.id_ = id.value();
    a.object_id_ = object_id;
    a.affordance_id_ = affordance_id;
    a.robot_id_ = robot->id();
    a.rt_ = G.get_rt_api();
    if (not a.write(G, init))
    {
        G.delete_node(a.id_);
        return std::nullopt;
    }

    // Name the copy on the affordance only once the edge exists: a reader that follows the link must
    // never find a node with no pose.
    auto aff_now = G.get_node(affordance_id);
    if (not aff_now.has_value())
    {
        G.delete_node(a.id_);
        return std::nullopt;
    }
    G.add_or_modify_attrib_local<interaction_node_id_att>(aff_now.value(), a.id_);
    if (not G.update_node(aff_now.value()))
    {
        G.delete_node(a.id_);
        return std::nullopt;
    }
    std::print("[interaction_anchor] '{}' id={} under '{}' for affordance {}\n", name, a.id_, robot->name(),
               affordance_id);
    return a;
}

bool Anchor::write(DSR::DSRGraph& G, const RelPose& r)
{
    if (id_ == 0 or not rt_) return false;
    auto robot = G.get_node(robot_id_);
    if (not robot.has_value()) return false;

    std::vector<float> cov(36, 0.f);                  // z, roll, pitch: not estimated, left at 0 (room_concept does the same)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cov[kSe3OfSe2[i] * 6 + kSe3OfSe2[j]] = r.P(i, j);
    const std::uint64_t ts = r.stamp_ms > 0 ? static_cast<std::uint64_t>(r.stamp_ms) : rc::provenance::now_ms();
    try
    {
        rt_->insert_or_assign_edge_RT(robot.value(), id_,
                                      DSR::RT_API::RTBlock{
                                          .translation    = {r.x.x(), r.x.y(), 0.f},
                                          .rotation_euler = {0.f, 0.f, r.x.z()},
                                          .covariance     = std::move(cov)},
                                      ts);
    }
    catch (const std::exception& e)
    {
        std::print(stderr, "[interaction_anchor] RT write to {} failed: {}\n", id_, e.what());
        return false;
    }
    return true;
}

bool Anchor::relink(DSR::DSRGraph& G, std::uint64_t new_object_id)
{
    if (id_ == 0) return false;
    auto n = G.get_node(id_);
    if (not n.has_value()) return false;
    G.add_or_modify_attrib_local<interaction_node_id_att>(n.value(), new_object_id);
    if (not G.update_node(n.value())) return false;
    object_id_ = new_object_id;                       // the NAME keeps the old original's: DSR names are fixed
    return true;
}

bool Anchor::link_affordance(DSR::DSRGraph& G, std::uint64_t affordance_id)
{
    if (id_ == 0 or affordance_id == 0) return false;
    if (affordance_id != affordance_id_)
    {
        unlink_affordance(G, affordance_id_, id_);   // the old one must not keep pointing at a copy it no longer uses
        affordance_id_ = affordance_id;
    }
    auto aff = G.get_node(affordance_id);
    if (not aff.has_value()) return false;
    if (const auto link = G.get_attrib_by_name<interaction_node_id_att>(aff.value());
        link.has_value() and link.value() == id_)
        return true;                                 // already there: no write, no churn
    G.add_or_modify_attrib_local<interaction_node_id_att>(aff.value(), id_);
    return G.update_node(aff.value());
}

void Anchor::remove(DSR::DSRGraph& G)
{
    if (id_ == 0) return;
    unlink_affordance(G, affordance_id_, id_);        // the link first: nobody may follow it to a dead node
    G.delete_node(id_);                               // drops the robot→anchor RT edge with it
    id_ = 0;
}

void sweep_stale(DSR::DSRGraph& G, int agent_id)
{
    std::vector<std::uint64_t> gone;
    for (const auto& n : G.get_nodes_by_type("object"))
    {
        if (not is_own_anchor_node(G, n, agent_id)) continue;   // edge-independent: parentless leaks too
        if (G.delete_node(n.id())) gone.push_back(n.id());
    }
    if (gone.empty()) return;
    // Only links pointing at anchors THIS agent just deleted: those affordances are ours to write.
    for (const auto& a : G.get_nodes_by_type("affordance"))
        if (const auto link = G.get_attrib_by_name<interaction_node_id_att>(a);
            link.has_value() and std::ranges::contains(gone, link.value()))
            unlink_affordance(G, a.id(), link.value());
    std::print("[interaction_anchor] swept {} stale anchor(s) of agent {}\n", gone.size(), agent_id);
}

bool Anchor::set_width(DSR::DSRGraph& G, float width_m)
{
    auto n = G.get_node(id_);
    if (not n.has_value()) return false;
    if (const auto v = G.get_attrib_by_name<width_m_att>(n.value()); v.has_value() and v.value() == width_m) return true;
    G.add_or_modify_attrib_local<width_m_att>(n.value(), width_m);
    return G.update_node(n.value());
}

bool Anchor::set_unseen_ms(DSR::DSRGraph& G, int unseen_ms)
{
    auto n = G.get_node(id_);
    if (not n.has_value()) return false;
    if (const auto v = G.get_attrib_by_name<unseen_time_att>(n.value()); v.has_value() and v.value() == unseen_ms) return true;
    G.add_or_modify_attrib_local<unseen_time_att>(n.value(), unseen_ms);
    return G.update_node(n.value());
}

bool Anchor::set_lost_prob(DSR::DSRGraph& G, float p_lost)
{
    if (not std::isfinite(p_lost)) return false;
    auto n = G.get_node(id_);
    if (not n.has_value()) return false;
    const float exist = 1.f - std::clamp(p_lost, 0.f, 1.f);
    if (const auto v = G.get_attrib_by_name<obj_exist_prob_att>(n.value()); v.has_value() and v.value() == exist) return true;
    G.add_or_modify_attrib_local<obj_exist_prob_att>(n.value(), exist);
    return G.update_node(n.value());
}

bool Anchor::clear_lost_prob(DSR::DSRGraph& G)
{
    auto n = G.get_node(id_);
    if (not n.has_value()) return false;
    if (not G.get_attrib_by_name<obj_exist_prob_att>(n.value()).has_value()) return true;
    G.remove_attrib_local<obj_exist_prob_att>(n.value());
    return G.update_node(n.value());
}

std::optional<float> read_anchor_lost_prob(DSR::DSRGraph& G, std::uint64_t anchor_id)
{
    const auto n = G.get_node(anchor_id);
    if (not n.has_value()) return std::nullopt;
    const auto e = G.get_attrib_by_name<obj_exist_prob_att>(n.value());
    if (not e.has_value() or not std::isfinite(e.value())) return std::nullopt;
    return 1.f - std::clamp(e.value(), 0.f, 1.f);
}

bool Anchor::set_drivable(DSR::DSRGraph& G, bool drivable)
{
    auto n = G.get_node(id_);
    if (not n.has_value()) return false;
    if (const auto v = G.get_attrib_by_name<valid_att>(n.value()); v.has_value() and v.value() == drivable) return true;
    G.add_or_modify_attrib_local<valid_att>(n.value(), drivable);
    return G.update_node(n.value());
}

bool Anchor::set_leaves(DSR::DSRGraph& G, const std::vector<AnchorLeaf>& leaves)
{
    auto n = G.get_node(id_);
    if (not n.has_value()) return false;
    const auto px = G.get_attrib_by_name<delimiting_polygon_x_att>(n.value());
    const auto py = G.get_attrib_by_name<delimiting_polygon_y_att>(n.value());
    const auto ph = G.get_attrib_by_name<door_phi_rad_att>(n.value());
    if (leaves.empty())
    {
        if (not px.has_value() and not py.has_value() and not ph.has_value()) return true;
        if (px.has_value()) G.remove_attrib_local<delimiting_polygon_x_att>(n.value());
        if (py.has_value()) G.remove_attrib_local<delimiting_polygon_y_att>(n.value());
        if (ph.has_value()) G.remove_attrib_local<door_phi_rad_att>(n.value());
        return G.update_node(n.value());
    }
    auto [xs, ys] = encode_anchor_leaves(leaves);
    const float phi = leaves.front().phi;
    if (px.has_value() and py.has_value() and ph.has_value() and px->get() == xs and py->get() == ys and ph.value() == phi)
        return true;
    G.add_or_modify_attrib_local<delimiting_polygon_x_att>(n.value(), std::move(xs));
    G.add_or_modify_attrib_local<delimiting_polygon_y_att>(n.value(), std::move(ys));
    G.add_or_modify_attrib_local<door_phi_rad_att>(n.value(), phi);
    return G.update_node(n.value());
}

std::vector<AnchorLeaf> read_anchor_leaves(DSR::DSRGraph& G, std::uint64_t anchor_id)
{
    const auto n = G.get_node(anchor_id);
    if (not n.has_value()) return {};
    const auto px = G.get_attrib_by_name<delimiting_polygon_x_att>(n.value());
    const auto py = G.get_attrib_by_name<delimiting_polygon_y_att>(n.value());
    if (not px.has_value() or not py.has_value()) return {};
    const auto ph = G.get_attrib_by_name<door_phi_rad_att>(n.value());
    return decode_anchor_leaves(px->get(), py->get(), ph.value_or(0.f));
}

std::optional<std::pair<RelPose, float>> read_affordance_anchor(DSR::DSRGraph& G, const DSR::Node& affordance,
                                                                std::int64_t now_ms)
{
    const auto link = G.get_attrib_by_name<interaction_node_id_att>(affordance);
    if (not link.has_value() or link.value() == 0) return std::nullopt;
    return read_anchor(G, link.value(), now_ms);
}

std::optional<std::pair<RelPose, float>> read_anchor(DSR::DSRGraph& G, std::uint64_t anchor_id, std::int64_t now_ms)
{
    auto r = read_anchor_latest(G, anchor_id);
    if (not r.has_value()) return std::nullopt;
    const auto anchor = G.get_node(anchor_id);
    if (not anchor.has_value() or not anchor_fresh(G, anchor.value(), r->first.stamp_ms, now_ms)) return std::nullopt;
    return r;
}

std::optional<std::pair<RelPose, float>> read_anchor_latest(DSR::DSRGraph& G, std::uint64_t anchor_id)
{
    const auto anchor = G.get_node(anchor_id);
    if (not anchor.has_value() or not is_anchor(G, anchor.value())) return std::nullopt;
    const auto parent = G.get_attrib_by_name<parent_att>(anchor.value());
    if (not parent.has_value()) return std::nullopt;
    const auto edge = G.get_edge(parent.value(), anchor->id(), "RT");
    if (not edge.has_value()) return std::nullopt;

    // The newest block, by scanning the stamps: the ring is overwritten in place (see rc::rt::newest_twist). The owner stamps
    // monotonically (I-2: "not drivable" is the node's `valid`, never an older stamp), so the newest stamp is the last write.
    const auto stamps = G.get_attrib_by_name<rt_timestamps_att>(edge.value());
    if (not stamps.has_value()) return std::nullopt;
    std::uint64_t newest = 0;
    for (const auto s : stamps->get()) newest = std::max(newest, s);
    if (newest == 0) return std::nullopt;

    const auto rt = G.get_rt_api();
    const auto T = rt->get_edge_RT_as_rtmat(edge.value(), newest);
    const auto C = rt->get_edge_RT_covariance(edge.value(), newest);
    if (not T.has_value() or not C.has_value()) return std::nullopt;   // no covariance = no claim to execute on

    RelPose r;
    r.x = {static_cast<float>(T->translation().x()), static_cast<float>(T->translation().y()),
           wrap_pi(static_cast<float>(std::atan2(T->linear()(1, 0), T->linear()(0, 0))))};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.P(i, j) = static_cast<float>((*C)(kSe3OfSe2[i], kSe3OfSe2[j]));
    r.stamp_ms = static_cast<std::int64_t>(newest);
    return std::pair{r, G.get_attrib_by_name<width_m_att>(anchor.value()).value_or(0.f)};
}

}  // namespace rc::interaction
