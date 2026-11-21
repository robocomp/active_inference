/*
 * lidar_select.h — which LiDAR returns belong to THIS instance, and what one look was worth. SHARED.
 *
 * Extracted 2026-08-14 from four copies (cabinet · hood · refrigerator · table) measured at 85.6% pairwise
 * identical lines with the object's name normalised away — the highest family in the fleet after support_bank
 * was taken. refrigerator and table were BYTE-IDENTICAL: the normalised diff between them is empty.
 *
 * ★THIS FILE HAS ALREADY PRODUCED TWO LIVE DEFECTS, both in the part that is NOT shared:
 *
 *   1. THE BAND. hood inherited refrigerator's floor-referenced `[−margin, BirthHeightM + margin]`. A hood
 *      hangs, so that band was [−0.10, 0.85] m against a body at [1.79, 2.29] — DISJOINT. 109 returns per
 *      cycle were selected off the floor and the worktop at a mean 1.33 m from the model, every real hood
 *      return was excluded before the factor saw one, and `coverage` still reported 1.0 because 109 rays is
 *      more than LidarCoverageN0 = 60. The fit was silently mask-only.
 *   2. THE COVERAGE WEIGHT reads the RAY COUNT, and a count says nothing about whether those rays could
 *      have resolved the object. Same file, still open — the LiDAR has no counterpart of the camera's
 *      p_detect. Noted here because this is where it would go.
 *
 * So the seam is drawn where the damage was: the agent answers WHERE ITS BODY IS (a radius and a z-band)
 * and WHAT ITS SURFACE IS (an SDF for the residual probe). Everything after those two — the selection loop,
 * the ray-count and angular-coverage down-weights, the "near?" raw count, the z-calibration probe — is
 * identical across the fleet and lives here.
 *
 * Pure: Eigen + the standard library + rc::ai::LidarRays. No DSR, no config type, no instance type.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../ai_belief/beam_ray.h"          // rc::ai::beam::BeamRays
#include "../ai_belief/lidar_ray_factor.h"   // rc::ai::LidarRays

namespace rc::lidar_select
{

// WHERE THE BODY IS — the agent's answer, and the only part of the selection that is per-object.
// ★The z-band must be the BODY's, never the floor's. See the header note: a floor-referenced band on a
// hanging object is disjoint from it, and nothing downstream can tell.
struct Box
{
    float radius_xy = 1.0f;   // rotation-agnostic footprint radius + margin
    float z_lo = 0.0f;        // band low  (already includes the agent's margin)
    float z_hi = 0.0f;        // band high
};

// The two continuous informativeness down-weights on the staged precision. Identical in all four agents.
struct Weights
{
    float coverage_n0        = 60.0f;   // ray count that constitutes a resolving look
    float coverage_ang_power = 1.0f;    // exponent on the angular-spread weight
    float robust_c_m         = 0.10f;   // Huber scale handed to the ray factor
};

// What the probe measured, for the agent's own diagnostics/CSV.
struct Diag
{
    int   rays = 0, raw = 0;
    float cov_ang = 1.0f;
    float resid_m = -1.0f;              // mean |SDF| of the selected returns — the fit's own error
    float meanz = -1.0f, topz = -1.0f, floorz = -1.0f;   // z-calibration probe
};

// Stage the returns inside `box` (centred on `centroid_xy`) into `out`, and apply the two down-weights to
// `precision_base`. Returns the staged ray count.
//
// The angular weight is 1 − R where R is the resultant length of the returns' bearings: rays all arriving
// from one direction see one face and constrain far less than the same number spread around the object.
inline int select_into(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                       const Eigen::Vector2f& centroid_xy, const Box& box,
                       float precision_base, const Weights& w,
                       rc::ai::LidarRays& out, float* cov_ang = nullptr)
{
    const float r2 = box.radius_xy * box.radius_xy;
    out.endpoints.clear();
    out.weights.clear();                      // unweighted selection ⇒ every ray weighs 1
    out.endpoints.reserve(256);
    double bear_c = 0.0, bear_s = 0.0;
    for (const auto& p : sweep)
    {
        const float dx = p.x() - centroid_xy.x(), dy = p.y() - centroid_xy.y();
        if (dx * dx + dy * dy > r2 or p.z() < box.z_lo or p.z() > box.z_hi) continue;
        out.endpoints.push_back(p);
        const float phi = std::atan2(dy, dx);
        bear_c += std::cos(phi); bear_s += std::sin(phi);
    }
    const int n = static_cast<int>(out.endpoints.size());
    float ca = 1.0f;
    if (n > 0)
    {
        const float coverage = std::min(1.0f, static_cast<float>(n) / std::max(1.0f, w.coverage_n0));
        const float R        = static_cast<float>(std::hypot(bear_c, bear_s) / n);
        ca = std::pow(std::max(0.0f, 1.0f - R), w.coverage_ang_power);
        out.origin     = origin;
        out.precision  = precision_base * coverage * ca;
        out.robust_c_m = w.robust_c_m;
    }
    if (cov_ang) *cov_ang = ca;
    return n;
}

// select_into with a per-return MEMBERSHIP m(p) in [0, 1] (the probability the return belongs to THIS object, e.g. from
// the camera masks). Same loop and box as select_into; a return with m <= 0 is not staged (it would carry no term), the
// others are staged with weight m. The ray-count coverage then counts Σm instead of n (a return that is 30% this object is
// 0.3 of a ray), and the bearing resultant is m-weighted: R = |Σ m·e^{iφ}| / Σm. precision = base · coverage · ca when
// Σm > 0, else 0 (the factor is then skipped). *m_sum receives Σm. With m ≡ 1 this is select_into bit for bit.
inline int select_into_weighted(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                                const Eigen::Vector2f& centroid_xy, const Box& box,
                                float precision_base, const Weights& w,
                                const std::function<float(const Eigen::Vector3f&)>& membership,
                                rc::ai::LidarRays& out, float* cov_ang = nullptr, float* m_sum = nullptr)
{
    const float r2 = box.radius_xy * box.radius_xy;
    out.endpoints.clear();
    out.weights.clear();
    out.endpoints.reserve(256);
    out.weights.reserve(256);
    double bear_c = 0.0, bear_s = 0.0, msum = 0.0;
    for (const auto& p : sweep)
    {
        const float dx = p.x() - centroid_xy.x(), dy = p.y() - centroid_xy.y();
        if (dx * dx + dy * dy > r2 or p.z() < box.z_lo or p.z() > box.z_hi) continue;
        const float m = membership(p);
        if (not (m > 0.0f)) continue;
        out.endpoints.push_back(p);
        out.weights.push_back(m);
        const float phi = std::atan2(dy, dx);
        bear_c += m * std::cos(phi); bear_s += m * std::sin(phi);
        msum += m;
    }
    const int n = static_cast<int>(out.endpoints.size());
    float ca = 1.0f;
    if (msum > 0.0)
    {
        const float coverage = std::min(1.0f, static_cast<float>(msum) / std::max(1.0f, w.coverage_n0));
        const float R        = static_cast<float>(std::hypot(bear_c, bear_s) / msum);
        ca = std::pow(std::max(0.0f, 1.0f - R), w.coverage_ang_power);
        out.origin     = origin;
        out.precision  = precision_base * coverage * ca;
        out.robust_c_m = w.robust_c_m;
    }
    else
        out.precision = 0.0f;
    if (cov_ang) *cov_ang = ca;
    if (m_sum)   *m_sum   = static_cast<float>(msum);
    return n;
}

// What one beam-staging pass counted (spec 7c.8). staged = rays handed to the beam model; no_label = returns whose ray
// crosses the select box but whose camera label was unavailable (nullopt: behind / outside the image) so they are not
// staged; in_box = returns whose ENDPOINT lies in the select box (the set kappa is computed over); q_sum = their sum of q.
struct BeamStageStats
{
    int   staged = 0, no_label = 0, in_box = 0;
    float kappa = 0.0f, q_sum = 0.0f;
};

// Ray staging for the smooth beam model (spec 2026-11-15 lidar-measures 7c.8, ray staging). Stages EVERY return whose
// ray (origin -> return, continued to rho_max) crosses the select cylinder (radius_xy about centroid_xy) within the box's
// z-band, whether the return stops short of the box, lands in it or goes through (a miss is information). The label gives
// per return {m, q}: m = eps_px + (1 - 2 eps_px) q goes to out.m (the persistent label), q is the mask membership proper;
// nullopt => the return is not staged (it has no camera evidence) and is counted in stats->no_label. A return farther than
// rho_max is staged AT rho_max (a beam that saw nothing inside the sensor's useful range).
//
// kappa is select_into_weighted's `precision / base` (coverage * ca) computed by that very function, on the SAME sweep and
// box, over the endpoint-in-box returns weighted by q: reused, not re-derived, so the two cannot drift. No floor: q = 0
// everywhere gives kappa = 0.
inline int select_rays_beam(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                            const Eigen::Vector2f& centroid_xy, const Box& box, const Weights& w, double rho_max,
                            const std::function<std::optional<std::pair<double, float>>(const Eigen::Vector3f&)>& label,
                            rc::ai::beam::BeamRays& out, BeamStageStats* stats = nullptr)
{
    out.origin = origin.cast<double>();
    out.endpoints.clear();
    out.m.clear();
    out.rho_max = rho_max;
    out.kappa   = 0.0;
    BeamStageStats st;

    const double r2 = static_cast<double>(box.radius_xy) * box.radius_xy;
    const double cx = centroid_xy.x(), cy = centroid_xy.y();
    const Eigen::Vector3d o = origin.cast<double>();
    std::vector<std::optional<std::pair<double, float>>> lab(sweep.size());
    std::vector<char> have(sweep.size(), 0);
    const auto label_of = [&](std::size_t i) -> const std::optional<std::pair<double, float>>&
    {
        if (not have[i]) { lab[i] = label(sweep[i]); have[i] = 1; }
        return lab[i];
    };
    // Does the segment o + t d, t in [0, rho_max], d a unit vector, meet the cylinder ∩ z-band?
    const auto crosses = [&](const Eigen::Vector3d& d) -> bool
    {
        double t0 = 0.0, t1 = rho_max;
        const double ox = o.x() - cx, oy = o.y() - cy;
        const double a = d.x() * d.x() + d.y() * d.y();
        const double c = ox * ox + oy * oy - r2;
        if (a > 0.0)
        {
            const double b = ox * d.x() + oy * d.y();
            const double disc = b * b - a * c;
            if (disc < 0.0) return false;
            const double sq = std::sqrt(disc);
            t0 = std::max(t0, (-b - sq) / a);
            t1 = std::min(t1, (-b + sq) / a);
        }
        else if (c > 0.0) return false;                   // vertical ray outside the cylinder
        if (d.z() != 0.0)
        {
            double za = (box.z_lo - o.z()) / d.z(), zb = (box.z_hi - o.z()) / d.z();
            if (za > zb) std::swap(za, zb);
            t0 = std::max(t0, za);
            t1 = std::min(t1, zb);
        }
        else if (o.z() < box.z_lo or o.z() > box.z_hi) return false;
        return t0 <= t1;
    };

    const float r2f = box.radius_xy * box.radius_xy;
    for (std::size_t i = 0; i < sweep.size(); ++i)
    {
        const Eigen::Vector3f& p = sweep[i];
        const float dx = p.x() - centroid_xy.x(), dy = p.y() - centroid_xy.y();
        const bool endpoint_in = dx * dx + dy * dy <= r2f and p.z() >= box.z_lo and p.z() <= box.z_hi;
        const Eigen::Vector3d v = p.cast<double>() - o;
        const double rho = v.norm();
        const bool hit = rho > 0.0 and crosses(v / rho);
        if (not (hit or endpoint_in)) continue;
        const auto& lb = label_of(i);
        if (endpoint_in) ++st.in_box;
        if (not hit) continue;                            // an endpoint in the box on a degenerate ray: kappa only
        if (not lb.has_value()) { ++st.no_label; continue; }
        const double r = std::min(rho, rho_max);
        out.endpoints.push_back(o + (v / rho) * r);
        out.m.push_back(lb->first);
    }
    st.staged = static_cast<int>(out.endpoints.size());

    // kappa: select_into_weighted over the endpoint-in-box returns, weight q, base 1 => out.precision IS coverage * ca.
    rc::ai::LidarRays kr;
    float msum = 0.0f;
    const Eigen::Vector3f* base = sweep.data();
    select_into_weighted(sweep, origin, centroid_xy, box, 1.0f, w,
                         [&](const Eigen::Vector3f& p) -> float
                         {
                             const auto& lb = label_of(static_cast<std::size_t>(&p - base));
                             return lb.has_value() ? lb->second : 0.0f;
                         }, kr, nullptr, &msum);
    st.kappa = kr.precision;
    st.q_sum = msum;
    out.kappa = static_cast<double>(st.kappa);
    if (stats) *stats = st;
    return st.staged;
}

// A generous "is anything near this object at all?" count — a diagnostic, deliberately wider than the
// selection so an empty selection can be told apart from an empty neighbourhood.
//
// `scale` widens the XY radius only; the caller widens the z band by passing a taller Box, because how far
// to look vertically is an object question (bottle looks 2x its own half-height, furniture reuses its band).
// It must stay under the birth min-separation or the count grabs a neighbour's returns.
inline int raw_count(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector2f& c, const Box& box,
                     float scale = 1.5f)
{
    const float r2 = (scale * box.radius_xy) * (scale * box.radius_xy);
    int raw = 0;
    for (const auto& p : sweep)
    {
        const float dx = p.x() - c.x(), dy = p.y() - c.y();
        if (dx * dx + dy * dy <= r2 and p.z() >= box.z_lo and p.z() <= box.z_hi) ++raw;
    }
    return raw;
}

// The fit-quality residual + the z-calibration probe, over the returns already staged in `rays`.
//
// ★resid_m IS THE INSTRUMENT THAT NAMED THE BAND DEFECT. It read 1.33 m on hood while the selection was
// picking up the floor, and 0.07 m once the band was corrected. `sdf` is the agent's own surface — a
// compound SDF for a box with legs, a plain box for a run — which is why it is a callback rather than
// something this file could compute.
//
// The z probe reports the mean of ALL selected returns and of the highest / lowest 20%. A persistent gap
// between topz and the fitted top is a lidar3D→room z-offset, not a fit error — the two are indistinguishable
// from the residual alone, which is why both are reported.
inline void probe(const rc::ai::LidarRays& rays,
                  const std::function<float(const Eigen::Vector3f&)>& sdf, Diag& d)
{
    const std::size_t n = rays.endpoints.size();
    if (n == 0) return;
    double resid_sum = 0.0, zsum = 0.0;
    std::vector<float> zs;
    zs.reserve(n);
    for (const auto& p : rays.endpoints)
    {
        resid_sum += std::abs(sdf(p));
        zs.push_back(p.z());
        zsum += p.z();
    }
    d.resid_m = static_cast<float>(resid_sum / static_cast<double>(n));
    std::sort(zs.begin(), zs.end());
    const std::size_t k = std::max<std::size_t>(1, zs.size() / 5);
    double topsum = 0.0; for (std::size_t i = zs.size() - k; i < zs.size(); ++i) topsum += zs[i];
    double botsum = 0.0; for (std::size_t i = 0; i < k; ++i) botsum += zs[i];
    d.meanz  = static_cast<float>(zsum   / static_cast<double>(n));
    d.topz   = static_cast<float>(topsum / static_cast<double>(k));
    d.floorz = static_cast<float>(botsum / static_cast<double>(k));
}

}  // namespace rc::lidar_select
