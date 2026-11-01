// recall_candidates.h — what chair_concept may resurrect: the chairs ltom offers back in this room.
// Reads `metaconcept` nodes named `ltom_recall_<uid>` that hang under the room (RT child), class "chair", whose uid
// belongs to THIS agent (uid >> 40 == agent id; inline, chair_concept does not include ltom headers) and whose
// chair_<N> is not live and N <= max_n (the persisted identity counter's last()). The decision itself (MAP over {candidates, new}) is common/recall_match; this file only
// gathers candidates and removes a taken one. Main thread only.
#pragma once

#include <dsr/api/dsr_api.h>

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <print>
#include <set>
#include <string>
#include <vector>

#include "../../common/recall_match/recall_match.h"

namespace rc
{
struct RecallCand
{
    recall::Candidate cand;   // uid, room-frame mu, diagonal cov from the recall edge's x/y variances
    int n = 0;                // the chair_<N> number carried by the uid (low 40 bits)
};

inline constexpr std::uint64_t kUidNumberMask = (std::uint64_t{1} << 40) - 1;

// The room to read recall candidates under, or none. A birth is parented under chair's LATCHED room
// (room_node_id_, which every existing chair's RT edge is written from), while ltom publishes recalls under the
// CURRENT room. They differ only for the few cycles of a room change (the latch re-resolves when the old room is
// evicted). Resurrect only when they agree, so candidates, birth parent and ltom's room are one frame; otherwise
// the newborn simply takes a fresh number.
inline std::optional<std::uint64_t> recall_room(std::uint64_t latched, std::optional<std::uint64_t> current)
{
    if (latched == 0 or not current.has_value() or *current != latched) return std::nullopt;
    return latched;
}

inline std::vector<RecallCand> recall_candidates(DSR::DSRGraph& G, std::uint64_t room_id, std::uint32_t my_agent_id, int max_n,
                                                 const std::set<std::string>& live_names)
{
    static std::set<std::string> warned;   // log a malformed candidate once, not every cycle
    std::vector<RecallCand> out;
    for (const auto& node : G.get_nodes_by_type("metaconcept"))
    {
        if (not node.name().starts_with("ltom_recall_")) continue;
        if (G.get_attrib_by_name<object_subtype_att>(node).value_or("") != "chair") continue;
        const auto uid = G.get_attrib_by_name<object_uid_att>(node);
        if (not uid or (*uid >> 40) != my_agent_id) continue;
        const int n = static_cast<int>(*uid & kUidNumberMask);
        if (n > max_n)   // above the persisted counter: it could be re-issued to another birth, so never offer it
        {
            if (warned.insert(node.name()).second)
                std::println(stderr, "chair_concept: [recall] {} is above the identity counter ({}) — not offered", node.name(), max_n);
            continue;
        }
        if (live_names.contains("chair_" + std::to_string(n))) continue;
        const auto edge = G.get_edge(room_id, node.id(), "RT");
        if (not edge) continue;                                    // hangs under another room
        const auto T = G.get_rt_api()->get_edge_RT_as_rtmat(*edge);
        if (not T) continue;
        const auto cov = G.get_attrib_by_name<rt_covariance_att>(*edge);
        if (not cov or cov.value().get().size() < 8)
        {
            if (warned.insert(node.name()).second)
                std::println(stderr, "chair_concept: [recall] {} has no rt_covariance — skipped", node.name());
            continue;
        }
        const auto& c = cov.value().get();
        RecallCand rc;
        rc.n = n;
        rc.cand.uid = *uid;
        rc.cand.mu = {T->translation().x(), T->translation().y()};
        rc.cand.cov = Eigen::Matrix2f::Zero();
        rc.cand.cov(0, 0) = c[0];
        rc.cand.cov(1, 1) = c[7];
        out.push_back(rc);
    }
    return out;
}

/// Take-once: remove candidate `index` from the cycle's list and return it.
inline RecallCand take_candidate(std::vector<RecallCand>& cands, std::size_t index)
{
    RecallCand taken = cands[index];
    cands.erase(cands.begin() + static_cast<std::ptrdiff_t>(index));
    return taken;
}

/// The bare candidates, in list order (what rc::recall::decide takes; its index refers to this order).
inline std::vector<recall::Candidate> as_candidates(const std::vector<RecallCand>& v)
{
    std::vector<recall::Candidate> out;
    out.reserve(v.size());
    for (const auto& r : v) out.push_back(r.cand);
    return out;
}

/// Shoelace area of a room polygon (m²); 0 for fewer than 3 vertices.
inline float polygon_area(const std::vector<Eigen::Vector2f>& poly)
{
    if (poly.size() < 3) return 0.f;
    double s = 0.0;
    for (std::size_t i = 0, n = poly.size(); i < n; ++i)
    {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % n];
        s += static_cast<double>(a.x()) * b.y() - static_cast<double>(b.x()) * a.y();
    }
    return static_cast<float>(std::abs(s) * 0.5);
}
}   // namespace rc
