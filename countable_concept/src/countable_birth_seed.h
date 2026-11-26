/*
 * countable_birth_seed.h — the birth's seed placement (F1 round 9 a): the cold-start footprint, or the back on the nearest wall
 * for a class that declares a back-face attachment (rc::birth_on_wall). Pure (no DSR): the fitter calls rc::seed_birth with its
 * nearest-wall query, and birth_seed_rule_test calls it on synthetic clouds.
 */
#pragma once

#include <functional>
#include <optional>
#include <print>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "countable_belief.h"
#include "countable_birth_footprint.h"
#include "countable_config.h"

namespace rc {

using NearestWall = std::function<WallRef(const Eigen::Vector2f&)>;

// BIRTH ON THE PRIORS. A wall-standing object's first view is its FRONT; the principal-axis yaw of that view is
// noise on a near-square footprint and the depth is unseen. So the birth takes what the priors already know
// (spec 7c.8 rev 3.3.14, rc::birth_footprint):
//  · the wall: the one nearest the cloud (the class prior says it stands against one — WallFlushPrior);
//  · yaw: back face onto that wall, front into the room (local +Y = back = −n);
//  · width/depth: the CLASS MEANS. The cloud's extent is not the object's: a mask that took in a neighbour
//    (live14fixA: 2.46 m for a 0.51 m nightstand) seeded a 2.13 m-deep box under the old depth-band rule;
//  · centre: the cloud's middle along the wall, half the class depth in from it;
//  · the cloud's extent widens the first update's POSITION covariance instead (F1.3): along the wall (w̄ − E)²/12 for
//    either sign (a cloud wider than the class, or one clipped/partial and shorter); across it max(0, E − h̄)²/12 (the
//    centre across is set by the wall, so only a cloud reaching deeper than the class says the object may stand off);
//  · H: NOT here — re-derived after the belief is built (CountableBelief::seed_height, F1 round 6.2: the class-prior MAP
//    with the top of the points the box explains, box-vs-headboard shares at this born pose).
// Not a gate on the fit: the belief starts here and the data then moves and sizes it.
inline std::optional<Eigen::Matrix2f> birth_on_priors(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo,
                                                      float z_hi, const CountableConfig& cfg, const NearestWall& nearest_wall,
                                                      const std::string& name)
{
    std::vector<Eigen::Vector2f> xy;
    for (const auto& p : pts) if (p.z() >= z_lo and p.z() <= z_hi) xy.emplace_back(p.x(), p.y());
    if (xy.size() < 10) return std::nullopt;
    Eigen::Vector2f c = Eigen::Vector2f::Zero();
    for (const auto& q : xy) c += q;
    c /= static_cast<float>(xy.size());
    const WallRef w = nearest_wall(c);
    if (not w.ok) return std::nullopt;
    const float prior_d = cfg.ai2_prior_depth_m > 0.0f ? cfg.ai2_prior_depth_m : cfg.ai2_prior_footprint_m;
    const rc::BirthFootprint bf = rc::birth_footprint(xy, w.p, w.n, cfg.ai2_prior_footprint_m, prior_d, 0.0f);   // the EXCESS; the belief adds its σ_pos²
    if (not bf.ok) return std::nullopt;
    s0.w = bf.w; s0.h = bf.h; s0.yaw = bf.yaw;
    s0.cx = bf.centre.x(); s0.cy = bf.centre.y();
    std::print("[{}] birth on priors: wall n=({:.2f},{:.2f}) → yaw={:.1f}° w={:.2f} h={:.2f} (class means) centre ({:.2f},{:.2f}) "
               "[cloud centre ({:.2f},{:.2f}), extent along {:.2f} across {:.2f} → extra pos std along {:.3f} across {:.3f}]\n",
               name, w.n.x(), w.n.y(), s0.yaw * 57.29578f, s0.w, s0.h, s0.cx, s0.cy, c.x(), c.y(), bf.extent_t, bf.extent_n,
               std::sqrt(bf.var_t), std::sqrt(bf.var_n));
    return bf.pos_cov;
}

// The birth's seed: s0 arrives as the cold-start footprint (the cloud's moment centre and extents, the fitter's); for a class
// that declares a back-face attachment, and when room geometry is known, the back goes on the nearest wall (birth_on_priors).
// Returns the cloud's EXCESS position covariance when the wall placed the seed, nullopt when s0 is left as it came.
inline std::optional<Eigen::Matrix2f> seed_birth(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo,
                                                 float z_hi, const CountableConfig& cfg, bool have_room_geometry,
                                                 const NearestWall& nearest_wall, const std::string& name)
{
    if (not (have_room_geometry and birth_on_wall(cfg))) return std::nullopt;
    return birth_on_priors(s0, pts, z_lo, z_hi, cfg, nearest_wall, name);
}

}  // namespace rc
