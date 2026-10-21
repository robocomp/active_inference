// near_field_clearance.h — the clearance field the spline route optimiser climbs, with a STRUCTURED near field.
//
// ★ WHY (user, 2026-10-21, Task 6 of the doorway-release plan). The smooth spline joining the robot to its target is the
//   right local<->global bridge, but in a NARROW passage its clearance term gets no usable traction from the raw LiDAR
//   ESDF: in a 0.90 m doorway the margin is 1-3 cm a side, the size of the return noise, and the ESDF's min-distance
//   gradient flips between the jambs, is flat across the opening and steep at its edges. The optimiser
//   (route_optimizer.h) is UNCHANGED — it is handed a better field through RouteOptimizerConfig::distance /
//   distance_gradient.
//
// THE MODEL. Near the robot the clearance at p is the EXPECTED distance to the nearest of the FITTED surfaces the local
// policy already models (local_policy/contact.h denoise_returns: surface samples — segments on a fitted line —, the
// surfaces' ends and isolated hits, each with its sigma, anisotropic for samples and ends), plus, when a measured gap is
// held for the goal, its two jambs (local_policy/doorway_corridor.h DoorwayJambs: the edges at the measured half-width,
// sigma_lat / sigma_n / sigma_s / sigma_hw). ONE obstacle model shared with the policy — no second fit.
//   · each element j is a segment a_j b_j whose position is Gaussian with covariance S_j; its distance from p is
//     D_j ~ N(mu_j, s_j^2), mu_j = |p - q_j| (q_j the closest point of the segment), s_j^2 = n' S_j n along the clearance
//     direction n = (p - q_j)/mu_j — the projection contact.cpp uses for its per-cell sigma;
//   · the clearance is min_j D_j, and its expectation is folded element by element with CLARK's moment-matched minimum of
//     two Gaussians (Clark 1961, "The greatest of a finite set of random variables"): exact for a pair, a Gaussian
//     approximation of the running minimum thereafter. It is smooth everywhere: where two surfaces are equally near (the
//     doorway's centre line) the min's kink is rounded over the width the surfaces' own sigmas give it — the ridge the
//     optimiser climbs is where the MODEL puts it, not where the noise of one scan does. No temperature, no floor.
//   · an element that may not exist (p_exist < 1) enters as the mixture p E[min(M, D)] + (1 - p) E[M], folded after the
//     certain ones;
//   · the two doorway hypotheses (anchor / measured gap) are mixed with the gap's weight p_gap, exactly as F4's risk is
//     (doorway_corridor.h doorway_risk_density): V = p_gap V_gap + (1 - p_gap) V_anchor.
//   Far away the field IS the existing ESDF (the callables the caller already had).
//
// THE JOIN IS A WEIGHT FROM THE MODEL, NOT A SWITCH RADIUS. The near model knows every surface within `near_radius_m` of
// `centre` (the robot) — that disc is its COVERAGE, R = the body's circumscribed radius + the route look-ahead the band
// deforms (stated by the caller). Its clearance at p is complete iff the clearance disc around p lies inside the covered
// disc: |p - centre| + M(p) <= R. The near weight is the probability of that under the model,
//     w(p) = Phi( (R - |p - centre| - E[M](p)) / sqrt(Var[M](p) + sigma_far^2) ),
//   sigma_far = the far field's own resolution (the ESDF cell). Then V = w V_near + (1 - w) V_far, with the exact gradient
//   w grad V_near + (1 - w) grad V_far + (V_near - V_far) grad w. Value and gradient are continuous across the join.
//   Where the near model holds nothing (no surface within R), E[M] = +inf and w = 0: the ESDF, exactly.
//
// The only cuts are NUMERICAL (a fold whose effect is below double precision is skipped; a weight below 1e-17 is zero) —
// documented at the site, not model thresholds.
//
// Pure Eigen/STL (plus the policy's value types) — no grid, no Qt, no DSR.
#pragma once

#include <Eigen/Dense>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "local_policy/doorway_corridor.h"
#include "local_policy/policy_types.h"

namespace rc
{
// One surface element of the near model: the segment a b (a == b: a point) whose position is N(nominal, S).
struct NearFieldElement
{
    Eigen::Vector2d a = Eigen::Vector2d::Zero(), b = Eigen::Vector2d::Zero();
    Eigen::Matrix2d S = Eigen::Matrix2d::Zero();   // m^2
    double p_exist = 1.0;
};

struct NearFieldEval
{
    double value = 0.0;                                  // the blended clearance (m)
    Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
    double near_weight = 0.0;                            // w(p) (mixed over the doorway hypotheses)
    double near_mean = 0.0, near_sigma = 0.0;            // E[M], sd[M] of the near model (no doorway: the surfaces')
};

class NearFieldClearance
{
public:
    using DistFn = std::function<float(const Eigen::Vector2f &)>;
    using GradFn = std::function<Eigen::Vector2f(const Eigen::Vector2f &)>;

    // centre: the robot (room frame); near_radius_m: the near model's coverage R; far_sigma_m: the far field's resolution.
    NearFieldClearance(const Eigen::Vector2f &centre, float near_radius_m, float far_sigma_m, DistFn far_distance,
                       GradFn far_gradient);

    // The policy's surface model (denoise_returns OUTPUT). Elements whose nearest point is outside the coverage disc are
    // not part of the near model (the coverage is what defines it). aggregate == false returns (unseen-contact hypotheses)
    // are kept with their p_exist.
    void set_surfaces(std::span<const policy::Return> denoised);
    // The room model the ESDF injects (TrajectoryController::build_esdf, inject_model_: the room outline minus the passage
    // being crossed, and the furniture polygons, each deferring to a LiDAR return near it) as near-model elements — so the
    // near model sees the SAME obstacle set as the field it replaces. Elements outside the coverage disc are dropped.
    void set_model(std::span<const NearFieldElement> model);
    // The edges of a closed polygon as near-model segments, sampled EXACTLY as TrajectoryController samples them for the
    // ESDF (num = max(1, int(len / spacing)), t = k / num), a sample kept iff keep(q); each maximal run of kept consecutive
    // samples on one edge becomes ONE segment (a lone sample, a point). sigma: isotropic position sigma.
    static std::vector<NearFieldElement> polygon_elements(std::span<const Eigen::Vector2f> poly, double sigma, float spacing,
                                                          const std::function<bool(const Eigen::Vector2f &)> &keep = {});
    // The measured doorway for the goal (F4's two hypotheses, weighted by p_gap). Absent = surfaces only.
    // wall_half_depth_m: half the wall's thickness — each jamb is then its FACE along the normal through the wall's depth
    // plus the wall's two faces (0 = a thin wall's line).
    void set_doorway(const policy::DoorwayRisk &risk, double wall_half_depth_m = 0.0);

    NearFieldEval evaluate(const Eigen::Vector2f &p) const;
    // The optimiser asks distance(p) and gradient(p) at the same sample: one evaluation serves both (last-point memo).
    // ★ Not thread-safe — an instance belongs to the one solve that built it (the band builds one per cycle).
    float distance(const Eigen::Vector2f &p) const { return static_cast<float>(memo(p).value); }
    Eigen::Vector2f gradient(const Eigen::Vector2f &p) const { return memo(p).gradient.cast<float>(); }

    std::size_t surface_count() const { return surfaces_.size(); }
    std::size_t model_count() const { return model_.size(); }
    float near_radius() const { return static_cast<float>(R_); }

    // A measured opening's two jambs as near-model elements: each the wall continued outward from its edge at the
    // measured half-width (with wall_half_depth_m > 0: the jamb's face along the normal and the wall's two faces), over one
    // body placement (j.placement_m — the circumscribed diameter, the span a body beside
    // the opening can reach), with lateral sigma^2 = sigma_lat^2 + sigma_s^2 + sigma_hw^2 and normal sigma^2 =
    // sigma_n^2 + sigma_s^2.
    static std::vector<NearFieldElement> jamb_elements(const policy::DoorwayJambs &j, double wall_half_depth_m = 0.0);
    // One surface-model return as an element (p -+ seg, S = sigma^2 I + scatter).
    static NearFieldElement element_of(const policy::Return &r);

private:
    struct Moments;   // (mean, variance) with their gradients — defined in the .cpp
    bool covered(const NearFieldElement &e) const;
    NearFieldEval evaluate_hypothesis(const Eigen::Vector2d &p, std::span<const NearFieldElement> extra,
                                      double far_v, const Eigen::Vector2d &far_g) const;

    const NearFieldEval &memo(const Eigen::Vector2f &p) const
    {
        if (not memo_valid_ or p != memo_p_) { memo_ = evaluate(p); memo_p_ = p; memo_valid_ = true; }
        return memo_;
    }
    mutable bool memo_valid_ = false;
    mutable Eigen::Vector2f memo_p_ = Eigen::Vector2f::Zero();
    mutable NearFieldEval memo_;

    Eigen::Vector2d centre_;
    double R_ = 0.0, sigma_far_ = 0.0;
    DistFn far_d_;
    GradFn far_g_;
    std::vector<NearFieldElement> surfaces_, model_;
    std::vector<NearFieldElement> anchor_jambs_, gap_jambs_;
    double p_gap_ = 0.0;
};
}  // namespace rc
