/*
 * countable_birth_seed.h — the birth's seed placement (F1 round 9 a): the cold-start footprint, or the back on the nearest wall
 * for a class that declares a back-face attachment (rc::birth_on_wall). Pure (no DSR): the fitter calls rc::seed_birth with its
 * nearest-wall query, and birth_seed_rule_test calls it on synthetic clouds.
 */
#pragma once

#include <cmath>
#include <functional>
#include <limits>
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
// The wall seed's ingredients: the banded cloud's moment centre, the wall nearest it, and the footprint on that wall.
struct WallSeed { rc::BirthFootprint bf; WallRef w; Eigen::Vector2f c = Eigen::Vector2f::Zero(); };
inline std::optional<WallSeed> wall_seed(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi, const CountableConfig& cfg,
                                         const NearestWall& nearest_wall)
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
    return WallSeed{bf, w, c};
}

inline void apply_wall_seed(CountableBeliefState& s0, const WallSeed& ws, const std::string& name)
{
    const auto& bf = ws.bf;
    s0.w = bf.w; s0.h = bf.h; s0.yaw = bf.yaw;
    s0.cx = bf.centre.x(); s0.cy = bf.centre.y();
    std::print("[{}] birth on priors: wall n=({:.2f},{:.2f}) → yaw={:.1f}° w={:.2f} h={:.2f} (class means) centre ({:.2f},{:.2f}) "
               "[cloud centre ({:.2f},{:.2f}), extent along {:.2f} across {:.2f} → extra pos std along {:.3f} across {:.3f}]\n",
               name, ws.w.n.x(), ws.w.n.y(), s0.yaw * 57.29578f, s0.w, s0.h, s0.cx, s0.cy, ws.c.x(), ws.c.y(), bf.extent_t, bf.extent_n,
               std::sqrt(bf.var_t), std::sqrt(bf.var_n));
}

inline std::optional<Eigen::Matrix2f> birth_on_priors(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo,
                                                      float z_hi, const CountableConfig& cfg, const NearestWall& nearest_wall,
                                                      const std::string& name)
{
    const auto ws = wall_seed(pts, z_lo, z_hi, cfg, nearest_wall);
    if (not ws) return std::nullopt;
    apply_wall_seed(s0, *ws, name);
    return ws->bf.pos_cov;
}

// log N(r; 0, v). v = 0 is a point mass (+inf at r = 0, −inf elsewhere): the arithmetic of a degenerate variance, not a cutoff.
inline double log_normal_1d(double r, double v)
{
    if (not (v > 0.0)) return r == 0.0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    return -0.5 * r * r / v - 0.5 * std::log(2.0 * M_PI * v);
}

// The placement mixture's MAP seed (final fix wave F2). A class whose manifest declares [prior.placement] (the chair: p_free 0.5,
// p_against_wall 0.5) is EITHER against its nearest wall OR free in the room. The seed is the hypothesis with the larger
//     p · L(cloud | wall seed)   vs   (1 − p) · L(cloud | free seed at the cloud's moment centre),
// both in the ONE coordinate the placement speaks about: across the wall (along it both seeds take the cloud's middle).
//  · The cloud's across-wall evidence is its middle m = (p5 + p95)/2 from the wall — the same statistic the wall seed uses along
//    the wall. A cloud of extent E inside an object of class depth h̄ puts the object's centre anywhere over |h̄ − E| around m:
//    uniform, variance V_obs = (h̄ − E)²/12 (rc::birth_footprint's along-wall argument, applied across).
//  · Against the wall the centre is the stand-off from it plus h̄/2: x_w = s + h̄/2, spread σ_s² ([prior.attachment]).
//  · Free, the centre is the cold-start seed x_f (the cloud's moment centre) with the born belief's position std σ_pos
//    (CountableBeliefParams::prior_pos_std — the position covariance the free seed is born with).
//  So L_wall = N(m − x_w; V_obs + σ_s²), L_free = N(m − x_f; V_obs + σ_pos²). No new constant and no reach cutoff: a chair whose
//  cloud sits well off every wall has L_wall ≈ 0; a cloud at the wall is explained σ_pos/σ_s times more sharply by the wall.
inline bool placement_map_on_wall(const WallSeed& ws, const CountableBeliefState& free_seed, const CountableConfig& cfg, float p,
                                  const std::string& name)
{
    const float h_bar = ws.bf.h;
    const double m = ws.bf.mid_n;
    const double e = static_cast<double>(h_bar) - ws.bf.extent_n;
    const double v_obs = e * e / 12.0;
    const double x_w = cfg.wall_standoff_m + 0.5 * h_bar;
    const double x_f = (Eigen::Vector2f(free_seed.cx, free_seed.cy) - ws.w.p).dot(ws.w.n);
    const double sp = CountableBeliefParams{}.prior_pos_std;
    const double ss = cfg.wall_standoff_std_m;
    const double lw = std::log(static_cast<double>(p))        + log_normal_1d(m - x_w, v_obs + ss * ss);
    const double lf = std::log(1.0 - static_cast<double>(p))  + log_normal_1d(m - x_f, v_obs + sp * sp);
    const bool wall = lw > lf;
    std::print("[{}] birth placement MAP: p(wall) {:.2f}; cloud middle {:.3f} m off the wall (extent {:.3f}, class depth {:.3f}); "
               "wall seed {:.3f} ± {:.3f} → log {:.2f}; free seed {:.3f} ± {:.3f} → log {:.2f} → {}\n",
               name, p, m, ws.bf.extent_n, h_bar, x_w, std::sqrt(v_obs + ss * ss), lw, x_f, std::sqrt(v_obs + sp * sp), lf,
               wall ? "AGAINST THE WALL" : "FREE (cloud seed)");
    return wall;
}

// The birth's seed: s0 arrives as the cold-start footprint (the cloud's moment centre and extents, the fitter's); for a class
// that declares a back-face attachment, and when room geometry is known, the back goes on the nearest wall (birth_on_priors) —
// with p = 1 implied unless the manifest declares a placement mixture ([prior.placement], the chair), in which case the seed is
// the mixture's MAP (placement_map_on_wall). p = 1 classes: nightstand, bed, toilet, hood (flush prior + stand-off) and the
// refrigerator (stand-off, flush 0): their path is today's, bit for bit. Never a c2v class (rc::birth_on_wall).
// Returns the cloud's EXCESS position covariance when the wall placed the seed, nullopt when s0 is left as it came.
inline std::optional<Eigen::Matrix2f> seed_birth(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo,
                                                 float z_hi, const CountableConfig& cfg, bool have_room_geometry,
                                                 const NearestWall& nearest_wall, const std::string& name)
{
    if (not (have_room_geometry and birth_on_wall(cfg))) return std::nullopt;
    if (not cfg.placement_p_against_wall) return birth_on_priors(s0, pts, z_lo, z_hi, cfg, nearest_wall, name);
    const auto ws = wall_seed(pts, z_lo, z_hi, cfg, nearest_wall);
    if (not ws) return std::nullopt;
    if (not placement_map_on_wall(*ws, s0, cfg, *cfg.placement_p_against_wall, name)) return std::nullopt;
    apply_wall_seed(s0, *ws, name);
    return ws->bf.pos_cov;
}

}  // namespace rc
