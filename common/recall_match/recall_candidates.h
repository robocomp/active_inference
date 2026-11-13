// recall_candidates.h — what a concept agent may resurrect: the objects of its class ltom offers back in this room.
// Reads `metaconcept` nodes named `ltom_recall_<uid>` that hang under the room (RT child), whose object_subtype
// is `cls`, whose uid belongs to THIS agent (uid >> 40 == agent id; inline, agents do not include ltom headers)
// and whose <cls>_<N> is not live and N <= max_n (the persisted identity counter's last()). The decision itself
// (MAP over {candidates, new}) is recall_match.h; this file only gathers candidates. Main thread only.
#pragma once

#include <dsr/api/dsr_api.h>

#include <Eigen/Dense>

#include <cstdint>
#include <print>
#include <set>
#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "recall_match.h"

namespace rc
{
inline std::vector<RecallCand> recall_candidates(DSR::DSRGraph& G, std::uint64_t room_id, std::string_view cls,
                                                 std::uint32_t my_agent_id, int max_n,
                                                 const std::set<std::string>& live_names)
{
    static std::set<std::string> warned;   // log a malformed candidate once, not every cycle
    std::vector<RecallCand> out;
    for (const auto& node : G.get_nodes_by_type("metaconcept"))
    {
        if (not node.name().starts_with("ltom_recall_")) continue;
        if (G.get_attrib_by_name<object_subtype_att>(node).value_or("") != cls) continue;
        const auto uid = G.get_attrib_by_name<object_uid_att>(node);
        if (not uid or (*uid >> 40) != my_agent_id) continue;
        const int n = static_cast<int>(*uid & kUidNumberMask);
        if (n > max_n)   // above the persisted counter: it could be re-issued to another birth, so never offer it
        {
            if (warned.insert(node.name()).second)
                std::println(stderr, "[recall] {}_{} is above the identity counter ({}) — not offered", cls, n, max_n);
            continue;
        }
        if (live_names.contains(std::string(cls) + "_" + std::to_string(n))) continue;
        const auto edge = G.get_edge(room_id, node.id(), "RT");
        if (not edge) continue;                                    // hangs under another room
        const auto T = G.get_rt_api()->get_edge_RT_as_rtmat(*edge);
        if (not T) continue;
        const auto cov = G.get_attrib_by_name<rt_covariance_att>(*edge);
        if (not cov or cov.value().get().size() < 8)
        {
            if (warned.insert(node.name()).second)
                std::println(stderr, "[recall] {} has no rt_covariance — skipped", node.name());
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
        // The owner's last existence belief, as ltom carried it (2026-11-11): the candidate's prior. Absent = not
        // stated, which stays log 1 (position alone), never read as "believed absent".
        if (const auto pe = G.get_attrib_by_name<obj_exist_prob_att>(node); pe.has_value() and std::isfinite(pe.value()))
            rc.cand.log_prior = std::log(std::clamp(pe.value(), 0.f, 1.f));
        out.push_back(rc);
    }
    return out;
}
}   // namespace rc
