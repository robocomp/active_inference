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

namespace rc
{
/// A resurrection (ltom recall): the newborn takes the remembered object's number and uid instead of issuing new ones.
struct Resurrect { int n; std::uint64_t uid; };

struct RecallCand
{
    recall::Candidate cand;   // uid, room-frame mu, diagonal cov from the recall edge's x/y variances
    int n = 0;                // the <class>_<N> number carried by the uid (low 40 bits)
};

inline constexpr std::uint64_t kUidNumberMask = (std::uint64_t{1} << 40) - 1;

// The room to read recall candidates under, or none. A birth is parented under the agent's LATCHED room
// (every existing instance's RT edge is written from it), while ltom publishes recalls under the CURRENT room.
// They differ only for the few cycles of a room change (the latch re-resolves when the old room is evicted).
// Resurrect only when they agree, so candidates, birth parent and ltom's room are one frame; otherwise the
// newborn simply takes a fresh number.
inline std::optional<std::uint64_t> recall_room(std::uint64_t latched, std::optional<std::uint64_t> current)
{
    if (latched == 0 or not current.has_value() or *current != latched) return std::nullopt;
    return latched;
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

struct RecallChoice { Resurrect who; float p; };

/// MAP over {cands, new} for one newborn at room-frame xy, measurement cov = detection_noise_m² I. On a win the
/// candidate leaves `cands` (take once) and its posterior is returned; otherwise nullopt (the newborn is new).
inline std::optional<RecallChoice> choose_recall(const Eigen::Vector2f& xy, float detection_noise_m,
                                                 std::vector<RecallCand>& cands, const recall::Params& p)
{
    if (cands.empty()) return std::nullopt;
    const Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * (detection_noise_m * detection_noise_m);
    const auto dec = recall::decide(xy, R, as_candidates(cands), p);
    if (not dec.index) return std::nullopt;
    const float post = dec.posterior[*dec.index];
    const auto taken = take_candidate(cands, *dec.index);
    return RecallChoice{Resurrect{taken.n, taken.cand.uid}, post};
}
}   // namespace rc
