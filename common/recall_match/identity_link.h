// identity_link.h — after birth: is this live object a remembered one that MOVED, or a new arrival?
// Pure maths, no graph, no Qt (spec docs/superpowers/specs/2026-11-15-changed-object-design.md, owner's choice B:
// the live object keeps its number; this posterior is the link the memory records).
//
// At birth (recall_match.h) the newborn was compared with the room's recalls on POSITION alone and lost: a chair
// moved across the room is far from its remembered spot. Position is uniform over the room under both stories
// below, so it cancels here; what decides is the rest of the owner's common sense:
//
//   same_k : recall k MOVED (π_move). Its old spot is then empty by construction (evidence likelihood 1).
//   new    : a NEW object arrived (λ_new), and recall k either STAYED where it was (π_stay = 1 − π_move − π_gone,
//            weighed by the absence evidence its old spot has gathered, exp(still_there_llr)) or is GONE (π_gone).
//   colour : the likelihood ratio of the live colour posterior under "same as the remembered colour" against
//            "any colour" (uniform over the palette) = 12·Σ_i q_now[i]·q_then[i]. q_then is prob_then on the
//            remembered term and (1 − prob_then)/11 elsewhere. Unmeasured on either side = no evidence (0).
//
//   logodds_k = log π_move + colour_lr_k + log_prior_k − log λ_new − log(π_stay·exp(still_there_llr_k) + π_gone)
//
// The posterior is a softmax over {logodds_k..., 0 (new)}. The room's COUNT enters through the normalisation: with
// one remembered chair the "which" is sure; with several alike the mass is shared. A stale memory (low existence
// prior, no colour) adds little. The decision is the MAP, as in recall::decide: a candidate only when its posterior
// is strictly larger than "new"'s.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "../appearance_belief/colour_terms.h"

namespace rc::recall
{
static_assert(rc::appearance::kPalette.size() == 12, "the colour ratio below is written for the 12-term palette");

struct LinkCand
{
    std::uint64_t uid = 0;
    float log_prior = 0.f;          // log obj_exist_prob, as in Candidate (0 = not stated)
    std::string colour_name;        // "" = never measured
    float colour_prob = 0.f;
    float still_there_llr = 0.f;    // the ghost at its old spot: L_now − L0 (log P(evidence|still there) − log P(evidence|not))
};
// Per-visit rates; guesses until the learned "gets moved" belief replaces π_move (config Tracker.Link*).
struct LinkParams { float pi_move = 0.10f; float pi_gone = 0.10f; float lambda_new = 0.10f; };
struct Link { std::optional<std::size_t> index; std::vector<float> posterior; };   // candidates..., new

[[nodiscard]] inline float colour_log_ratio(const std::optional<std::array<float, 12>>& colour_now,
                                            std::string_view name_then, float prob_then)
{
    if (not colour_now.has_value() or name_then.empty()) return 0.f;
    const auto* then = rc::appearance::find_term(name_then);
    if (then == nullptr) return 0.f;                              // a word outside the palette says nothing
    const float p = std::clamp(prob_then, 0.f, 1.f);
    const float rest = (1.f - p) / 11.f;
    double s = 0.0;
    for (std::size_t i = 0; i < rc::appearance::kPalette.size(); ++i)
        s += static_cast<double>((*colour_now)[i]) * (rc::appearance::kPalette[i].name == then->name ? p : rest);
    if (s <= 0.0) return -std::numeric_limits<float>::infinity();
    return static_cast<float>(std::log(12.0 * s));
}

[[nodiscard]] inline Link link_posterior(std::span<const LinkCand> cands,
                                         const std::optional<std::array<float, 12>>& colour_now, const LinkParams& p)
{
    const double pi_stay = std::max(0.0, 1.0 - p.pi_move - p.pi_gone);
    std::vector<double> lo;
    lo.reserve(cands.size() + 1);
    for (const auto& c : cands)
        lo.push_back(std::log(p.pi_move) + colour_log_ratio(colour_now, c.colour_name, c.colour_prob) + c.log_prior
                     - std::log(p.lambda_new) - std::log(pi_stay * std::exp(c.still_there_llr) + p.pi_gone));
    lo.push_back(0.0);                                             // new
    const double mx = std::ranges::max(lo);
    double z = 0.0;
    for (const double l : lo) z += std::exp(l - mx);
    Link out;
    out.posterior.reserve(lo.size());
    for (const double l : lo) out.posterior.push_back(static_cast<float>(std::exp(l - mx) / z));
    if (cands.empty()) return out;
    const auto best = std::max_element(out.posterior.begin(), out.posterior.end() - 1);
    if (*best > out.posterior.back()) out.index = static_cast<std::size_t>(best - out.posterior.begin());
    return out;
}
}   // namespace rc::recall
