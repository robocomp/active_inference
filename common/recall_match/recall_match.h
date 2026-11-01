// MAP re-identification of a newborn object against recalled (dead, remembered) candidates.
// Pure maths on top of view_atlas' location mixture and re-id posterior; no graph, no Qt.
// Birth-time evidence is position only, so the appearance ratio is 0 (no evidence). The decision is the MAP
// hypothesis of {candidates..., new}: a candidate is taken only if its posterior is STRICTLY larger than
// "new"'s and it is the argmax; ties go to "new".
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "../view_atlas/view_atlas.h"

namespace rc::recall
{

struct Candidate { std::uint64_t uid; Eigen::Vector2f mu; Eigen::Matrix2f cov; };
struct Params    { float pi_move = 0.5f; float log_prior_new = 0.f; float room_area_m2 = 20.f; };
struct Decision  { std::optional<std::size_t> index; std::vector<float> posterior; };   // posterior: candidates..., new

/// x: newborn's room-frame centroid; meas_cov: its measurement covariance (added to each candidate's).
inline Decision decide(const Eigen::Vector2f& x, const Eigen::Matrix2f& meas_cov,
                       std::span<const Candidate> cands, const Params& p)
{
    std::vector<rc::atlas::ReidCandidate> atlas_cands;
    atlas_cands.reserve(cands.size());
    for (const auto& c : cands)
        atlas_cands.push_back({0.f, rc::atlas::location_log_lik(x, c.mu, c.cov + meas_cov, p.pi_move, p.room_area_m2), 0.f});

    // "new" is uniform over the room. Written as log(1/A), the very expression the candidate mixture reduces to
    // when pi_move = 1, so the two terms are bit-identical there (no float-rounding lead on uninformative evidence).
    const float log_loc_new = std::log(1.f / p.room_area_m2);

    Decision d;
    // reid_posterior returns candidates..., new: the last element is "new".
    d.posterior = rc::atlas::reid_posterior(atlas_cands, p.log_prior_new, log_loc_new);
    if (cands.empty()) return d;
    const auto best = std::max_element(d.posterior.begin(), d.posterior.end() - 1);
    if (*best > d.posterior.back()) d.index = static_cast<std::size_t>(best - d.posterior.begin());
    return d;
}

} // namespace rc::recall
