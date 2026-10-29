/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
#include "proposal_adoption.h"

#include <dsr/api/dsr_rt_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <ranges>

namespace rc::proposal
{
namespace
{
bool is_proposal(DSR::DSRGraph& G, const DSR::Node& n)
{
    return n.type() == "metaconcept" and G.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == kSubtype;
}

// The planar part of an RT edge's latest block: (x, y, yaw) and z (kept as it is).
std::optional<std::pair<Eigen::Vector3f, float>> planar_rt(DSR::DSRGraph& G, DSR::RT_API& rt, std::uint64_t from, std::uint64_t to)
{
    const auto e = G.get_edge(from, to, "RT");
    if (not e.has_value()) return std::nullopt;
    const auto m = rt.get_edge_RT_as_rtmat(e.value());
    if (not m.has_value()) return std::nullopt;
    const auto& t = m->translation();
    const auto& R = m->rotation();
    return std::pair{Eigen::Vector3f(static_cast<float>(t.x()), static_cast<float>(t.y()),
                                     static_cast<float>(std::atan2(R(1, 0), R(0, 0)))),
                     static_cast<float>(t.z())};
}
}   // namespace

float angle_of(const Eigen::Affine2f& T) { return std::atan2(T.linear()(1, 0), T.linear()(0, 0)); }

Eigen::Vector3f transform_pose(const Eigen::Affine2f& T, const Eigen::Vector3f& pose)
{
    const Eigen::Vector2f p = T * pose.head<2>();
    return {p.x(), p.y(), std::remainder(pose.z() + angle_of(T), 2.f * static_cast<float>(M_PI))};
}

std::vector<rc::reloc::PriorMode> modes_in(const std::vector<float>& v, const Eigen::Affine2f& T_dst_K)
{
    std::vector<rc::reloc::PriorMode> out;
    if (v.size() % 10 != 0) return out;
    Eigen::Matrix3f J = Eigen::Matrix3f::Identity();
    J.topLeftCorner<2, 2>() = T_dst_K.linear();
    for (std::size_t c = 0; c < v.size(); c += 10)   // [w,x,y,th,Sxx,Sxy,Sxth,Syy,Syth,Sthth]
    {
        rc::reloc::PriorMode pm;
        pm.w = v[c];
        pm.mu = transform_pose(T_dst_K, Eigen::Vector3f(v[c + 1], v[c + 2], v[c + 3]));
        Eigen::Matrix3f S;
        S << v[c + 4], v[c + 5], v[c + 6],
             v[c + 5], v[c + 7], v[c + 8],
             v[c + 6], v[c + 8], v[c + 9];
        pm.S = J * S * J.transpose();
        if (pm.mu.allFinite() and pm.S.allFinite() and pm.w > 0.f) out.push_back(pm);
    }
    return out;
}

std::vector<Proposal> pending(DSR::DSRGraph& G)
{
    std::vector<Proposal> out;
    const auto rt = G.get_rt_api();
    for (const auto& n : G.get_nodes_by_type("metaconcept"))
    {
        if (not is_proposal(G, n)) continue;
        if (G.get_attrib_by_name<state_att>(n).value_or(std::string(kPending)) != kPending) continue;
        Proposal p;
        p.node_id    = n.id();
        p.parent     = G.get_attrib_by_name<parent_att>(n).value_or(0);
        p.room_index = G.get_attrib_by_name<room_id_att>(n).value_or(0);
        p.live_name  = n.name().starts_with(kPrefix) ? n.name().substr(kPrefix.size()) : n.name();
        const auto px = G.get_attrib_by_name<delimiting_polygon_x_att>(n);
        const auto py = G.get_attrib_by_name<delimiting_polygon_y_att>(n);
        const auto pp = G.get_attrib_by_name<recall_pose_prior_att>(n);
        if (px.has_value() and py.has_value() and px->get().size() == py->get().size())
            for (std::size_t i = 0; i < px->get().size(); ++i) p.outline_K.emplace_back(px->get()[i], py->get()[i]);
        if (pp.has_value()) p.pose_prior = pp->get();
        const auto rtp = p.parent != 0 ? planar_rt(G, *rt, p.parent, n.id()) : std::nullopt;
        if (rtp.has_value())
            p.T_P_K = Eigen::Translation2f(rtp->first.head<2>()) * Eigen::Rotation2Df(rtp->first.z());
        p.well_formed = rtp.has_value() and p.outline_K.size() >= 3 and not p.pose_prior.empty()
                        and p.pose_prior.size() % 10 == 0 and p.room_index != 0 and not p.live_name.empty();
        out.push_back(std::move(p));
    }
    return out;
}

const char* to_string(Admission a)
{
    switch (a)
    {
    case Admission::Judge:          return "judge";
    case Admission::NotProvisional: return "P is not this agent's provisional room";
    case Admission::NameTaken:      return "K's name is held by another node";
    case Admission::Malformed:      return "malformed proposal";
    }
    return "?";
}

Admission admit(DSR::DSRGraph& G, const Proposal& p, std::uint64_t provisional_room)
{
    if (not p.well_formed) return Admission::Malformed;
    if (provisional_room == 0 or p.parent != provisional_room or not G.get_node(p.parent).has_value()
        or not G.get_edge(p.parent, p.parent, "proto").has_value())
        return Admission::NotProvisional;
    if (const auto other = G.get_node(p.live_name); other.has_value() and other->id() != p.parent)
        return Admission::NameTaken;
    return Admission::Judge;
}

bool answer(DSR::DSRGraph& G, std::uint64_t proposal_id, std::string_view state)
{
    auto n = G.get_node(proposal_id);
    if (not n.has_value()) return false;
    G.add_or_modify_attrib_local<state_att>(n.value(), std::string(state));
    return G.update_node(n.value());
}

Adoption adopt_in_place(DSR::DSRGraph& G, const Proposal& p, std::uint64_t robot_id, std::optional<std::uint64_t> robot_ts)
{
    Adoption a;
    auto P = G.get_node(p.parent);
    if (not P.has_value()) { a.why = "P is gone"; return a; }
    if (not G.get_edge(p.parent, p.parent, "proto").has_value()) { a.why = "P is not provisional (no proto edge)"; return a; }
    if (const auto other = G.get_node(p.live_name); other.has_value() and other->id() != p.parent)
    { a.why = "the name " + p.live_name + " is held by id " + std::to_string(other->id()); return a; }

    // P's own RT children and their poses, read BEFORE anything is written.
    const auto rt = G.get_rt_api();
    struct Move { std::uint64_t from, to; Eigen::Vector3f pose; float z; bool robot; };
    std::vector<Move> moves;
    const auto collect = [&](std::uint64_t from, bool robot_allowed)
    {
        const auto fn = G.get_node(from);
        if (not fn.has_value()) return;
        for (const auto& e : G.get_node_edges_by_type(fn.value(), "RT"))
        {
            if (e.from() != from) continue;
            const auto child = G.get_node(e.to());
            if (not child.has_value()) continue;
            const bool robot = robot_allowed and e.to() == robot_id;
            if (not robot and child->type() != "wall") continue;
            if (const auto q = planar_rt(G, *rt, from, e.to()); q.has_value())
                moves.push_back({from, e.to(), q->first, q->second, robot});
        }
    };
    collect(p.parent, true);
    for (const auto& e : G.get_node_edges_by_type(P.value(), "RT"))
        if (e.from() == p.parent)
            if (const auto c = G.get_node(e.to()); c.has_value() and c->type() == "floor") collect(c->id(), false);

    // 1. The identity change, in ONE update_node: name, room_id, outline. Refused ⇒ nothing else is written.
    P->name(p.live_name);
    G.add_or_modify_attrib_local<room_id_att>(P.value(), p.room_index);
    std::vector<float> xs, ys;
    for (const auto& v : p.outline_K) { xs.push_back(v.x()); ys.push_back(v.y()); }
    G.add_or_modify_attrib_local<delimiting_polygon_x_att>(P.value(), std::move(xs));
    G.add_or_modify_attrib_local<delimiting_polygon_y_att>(P.value(), std::move(ys));
    try
    {
        if (not G.update_node(P.value())) { a.why = "cortex refused the rename to " + p.live_name; return a; }
    }
    catch (const std::exception& e)
    {
        a.why = std::string("cortex threw on the rename: ") + e.what();
        return a;
    }

    // 2. P's own geometry follows the frame: T(K←P) = T(P←K)⁻¹.
    const Eigen::Affine2f T_K_P = p.T_P_K.inverse();
    for (const auto& m : moves)
    {
        auto from = G.get_node(m.from);
        if (not from.has_value()) continue;
        const Eigen::Vector3f q = transform_pose(T_K_P, m.pose);
        rt->insert_or_assign_edge_RT(from.value(), m.to, std::vector<float>{q.x(), q.y(), m.z},
                                     std::vector<float>{0.f, 0.f, q.z()}, m.robot ? robot_ts : std::nullopt);
        ++a.reexpressed;
    }

    // 3. Not provisional any more.
    G.delete_edge(p.parent, p.parent, "proto");
    a.ok = true;
    return a;
}
}   // namespace rc::proposal
