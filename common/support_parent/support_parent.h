/*
 * common/support_parent/support_parent.h — "what am I standing on?", as a model comparison. SHARED, header-only.
 *
 * An object that RESTS on something has to answer two coupled questions every cycle: which node is its
 * support, and therefore what its base height is. Extracted from bottle_concept 2026-08-17, which was the only
 * agent that had ever answered them — in bottle_scene_graph.cpp, bespoke — and the next object that rests on
 * something is a MICROWAVE on a worktop. Cloning bottle's copy is exactly the inheritance the shared core
 * exists to stop, so the rule moves here first.
 *
 * ★IT IS A LIKELIHOOD COMPARISON, NOT A GATE, and that is why it was worth keeping intact rather than
 * simplifying. Each candidate support and the FLOOR are competing hypotheses:
 *
 *     ll(support) = −½·r_z²/σ_z²  −  λ_xy·d_xy²          r_z = base_z − support_top (in the support's frame)
 *     ll(floor)   = −½·(base_z/σ_z)²                     d_xy = how far outside the footprint the centre is
 *
 * and the best support wins only if it beats the floor by `decision_margin` nats. So an object hovering
 * ambiguously between a worktop edge and the floor is not forced onto either by a distance cutoff — it stays
 * with the floor until the evidence says otherwise, which is the safe direction (the room is always a valid
 * parent; a wrong support parent moves the object with a surface it is not on).
 *
 * ★σ_z IS INFLATED BY THE SUPPORT'S OWN PUBLISHED z-VARIANCE (the room→support RT covariance). A poorly-fitted
 * table is a WEAK anchor and says so, instead of pulling the object onto a height nobody knows. This is the
 * one place the two agents' beliefs meet, and it meets them as precisions rather than as a hand-off.
 *
 * ★★AND IT ANSWERS THE CROSS-AGENT SPAN QUESTION BY ITSELF. A counter_top object's base is not a class
 * constant — it is the worktop's FITTED top, i.e. another agent's belief, which a static manifest z_base_m
 * cannot express. `Decision::top_z` IS that number, live, per cycle. Declare the nominal in the manifest for
 * the cold start and let this override it; do not try to make the manifest carry a fitted value.
 *
 * THE SEAM: the caller says which node NAMES may support it (`support_prefixes`) and supplies the params. The
 * fleet convention makes that enough — every fitted object is node type "object" with its class in the name
 * ("table_1", "cabinet_w13_base", "cabinet_peninsula"), so a bottle passes {"table"} and a microwave passes
 * {"cabinet", "table"}. Everything after that is object-independent.
 *
 * Main-thread only: ts==0 transforms (the InnerEigenAPI ts==0 cache is unlocked — see CLAUDE.md).
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc::support
{

struct Params
{
    float sigma_z_m           = 0.04f;   // vertical residual σ for "resting on" (m)
    float footprint_margin_m  = 0.05f;   // slack on the support's footprint (m) — overhang is normal
    float lambda_xy           = 50.0f;   // penalty weight per m² outside the footprint
    float decision_margin     = 2.0f;    // nats the best support must beat the FLOOR by
};

struct Decision
{
    std::uint64_t parent_id = 0;
    // Defaults to "room" because the room IS this fleet's universal fallback parent — every object hangs off
    // it when nothing else wins. A default of "" would read as "no parent decided", which never happens.
    std::string   parent_name = "room";
    // The chosen support's TOP in room z. NaN ⇒ the room/floor won, so there is NO z anchor and the caller
    // must not invent one. Distinguishing "no support" from "support at z=0" matters: the floor hypothesis is
    // about the object's own base being near zero, not about a surface at zero.
    float top_z  = std::numeric_limits<float>::quiet_NaN();
    float margin = 0.0f;                 // how much the winner beat the runner-up by (nats)

    bool on_support() const { return std::isfinite(top_z); }
};

// ─── The decision arithmetic, separable from the graph so it can be TESTED ───────────────────────
//
// Kept as free functions on purpose: everything else in this file needs a live DSR graph and an InnerEigenAPI,
// so it can only be exercised by running an agent. These two carry the actual judgement, and the sweep that
// motivated this module found its lesson exactly here — a path with no test is one the next agent pays for.

// Log-evidence that the object rests on a candidate: vertical residual against the support's top, weighted by
// σ_z² (already inflated by the support's own published z-variance), minus a penalty for the centre lying
// outside the footprint.
inline float log_evidence(float r_z, float sigma_z2, float d_xy2, float lambda_xy)
{
    return -0.5f * (r_z * r_z) / std::max(1e-9f, sigma_z2) - lambda_xy * d_xy2;
}

// Log-evidence for the FLOOR: the object's own base is near zero. Note this is a statement about the OBJECT,
// not about a surface at z = 0 — which is why a floor decision carries NO z anchor.
inline float floor_log_evidence(float base_z, float sigma_z)
{
    const float sz = std::max(0.005f, sigma_z);
    return -0.5f * (base_z / sz) * (base_z / sz);
}

// A candidate's top in room z = its RT origin z + height_m. Relies on the fleet-wide convention that a node's
// origin is its BASE (see the identical comment in every <concept>_scene_graph.cpp). NaN when the node is
// gone, unsized, or unreachable in the RT chain — never a guess.
inline float support_top_z(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner_eigen, std::uint64_t id,
                           std::span<const std::string_view> support_prefixes)
{
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    if (inner_eigen == nullptr or id == 0)
        return nan;
    const auto n = G.get_node(id);
    if (not n.has_value())
        return nan;
    bool named = support_prefixes.empty();
    for (const auto& p : support_prefixes)
        if (std::string_view(n->name()).starts_with(p)) { named = true; break; }
    if (not named)
        return nan;
    const auto h = G.get_attrib_by_name<height_m_att>(n.value());
    if (not h.has_value() or h.value() <= 0.0f)
        return nan;
    const auto org = inner_eigen->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), n->name(), 0);
    if (not org.has_value())
        return nan;
    return static_cast<float>(org->z()) + h.value();
}

// Which node is this object resting on, given its footprint centre and the height of its BASE (room frame)?
// Returns the room/floor decision when nothing wins by the margin — the room is always a valid parent.
inline Decision decide(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner_eigen, std::uint64_t room_node_id,
                       float cx, float cy, float base_z,
                       std::span<const std::string_view> support_prefixes,
                       const Params& p = {})
{
    Decision room_dec;                            // default: parented to the room, NaN top ⇒ no z anchor
    room_dec.parent_id = room_node_id;
    if (const auto r = G.get_node(room_node_id); r.has_value())
        room_dec.parent_name = r->name();
    if (inner_eigen == nullptr)
        return room_dec;

    const float sz0 = std::max(0.005f, p.sigma_z_m);
    const float ll_room = floor_log_evidence(base_z, p.sigma_z_m);

    float best_ll = -std::numeric_limits<float>::infinity();
    Decision best;
    for (const auto& t : G.get_nodes_by_type("object"))
    {
        bool named = false;
        for (const auto& pre : support_prefixes)
            if (std::string_view(t.name()).starts_with(pre)) { named = true; break; }
        if (not named)
            continue;
        const float w = G.get_attrib_by_name<width_m_att> (t).value_or(0.0f);
        const float d = G.get_attrib_by_name<depth_m_att> (t).value_or(0.0f);
        const float h = G.get_attrib_by_name<height_m_att>(t).value_or(0.0f);
        if (w <= 0.0f or d <= 0.0f or h <= 0.0f)
            continue;                             // unsized ⇒ cannot reason about it geometrically

        // The base point in the SUPPORT'S LOCAL frame — oriented footprint test and vertical residual in one
        // step, so a rotated worktop needs no separate yaw handling.
        const auto loc = inner_eigen->transform(t.name(), Mat::Vector3d(cx, cy, base_z), "room", 0);
        if (not loc.has_value())
            continue;
        const float lx = std::abs(static_cast<float>(loc->x()));
        const float ly = std::abs(static_cast<float>(loc->y()));
        const float hw = 0.5f * w, hd = 0.5f * d;
        if (lx > hw + p.footprint_margin_m or ly > hd + p.footprint_margin_m)
            continue;                             // centre not over this support at all

        const float dxo = std::max(0.0f, lx - hw);
        const float dyo = std::max(0.0f, ly - hd);
        const float d_xy2 = dxo * dxo + dyo * dyo;
        const float r_z = static_cast<float>(loc->z()) - h;    // base vs the support's top (local top = z = h)

        // ★A POORLY-KNOWN SUPPORT IS A WEAK ANCHOR. Inflate σ_z² by the support's published z-variance from
        // the room→support RT covariance, so the two agents' beliefs meet as precisions.
        float sz2 = sz0 * sz0;
        if (const auto e = G.get_edge(room_node_id, t.id(), "RT"); e.has_value())
            if (const auto cov = G.get_attrib_by_name<rt_covariance_att>(e.value()); cov.has_value())
            {
                const auto& v = cov->get();
                if (v.size() >= 36) sz2 += std::max(0.0f, v[2 * 6 + 2]);   // the z-variance block
            }

        const float ll = log_evidence(r_z, sz2, d_xy2, p.lambda_xy);
        if (ll > best_ll)
        {
            best_ll          = ll;
            best.parent_id   = t.id();
            best.parent_name = t.name();
            best.top_z       = support_top_z(G, inner_eigen, t.id(), support_prefixes);
        }
    }

    if (std::isfinite(best_ll) and best_ll > ll_room + p.decision_margin)
    {
        best.margin = best_ll - ll_room;
        return best;
    }
    room_dec.margin = ll_room - best_ll;
    return room_dec;
}

}  // namespace rc::support
