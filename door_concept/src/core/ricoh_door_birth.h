// ricoh_door_birth.h — how much is ONE 360-camera door slice worth toward BIRTHING a door? (door_concept only; pure, Eigen.)
//
// The ricoh slice's range comes from LiDAR reprojected onto the mask, which can land on whatever is behind the door, so a
// ricoh-only birth at that range is the churn risk: born at a wrong range, killed by absence, reborn. Three continuous
// terms, none a cutoff, all multiplied into the shared birth evidence (common/instance_tracker/birth_evidence.h):
//   1. WALL: a door lies in a wall, so the bearing ray meets the room polygon at r_w. The slice's LiDAR range r_s is
//      scored against it (Gaussian agreement over their summed variances); no wall along the bearing ⇒ weight 0.
//      The position is the precision-weighted fusion of r_s and r_w, so a LiDAR range that agrees but is noisy is
//      pulled onto the wall. The wall-range variance is the bearing error lever arm, (σ_b r_w)² tan²φ (φ = incidence
//      from the wall normal): grazing rays are ill-conditioned and say little.
//   2. VARIANCE: R0 / (R0 + depth_var), R0 = the fit's own R for a ZED slice at that range (door_fit.cpp).
//   3. VIEWPOINT: frames from one viewpoint are one observation (the same view model as door_view_pool.h: bearing from
//      the door in units of its angular half-size, range in units of its half-size). A frame is credited 1 − s, s its
//      similarity to the candidate's pooled viewpoint, so a parked robot's 40 frames are worth one.
#pragma once
#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <vector>
#include <Eigen/Dense>

namespace rc::door::core
{

struct WallHit { float r = 0.0f; float cos_inc = 1.0f; };

/// Nearest crossing of the ray O + r·dir (r > 0) with the polygon's edges; cos_inc = |dir · wall normal|.
[[nodiscard]] inline std::optional<WallHit> wall_hit(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& O,
                                                     const Eigen::Vector2f& dir)
{
    std::optional<WallHit> best;
    const std::size_t n = poly.size();
    if (n < 3) return best;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f a = poly[i], e = poly[(i + 1) % n] - a;
        const float den = dir.x() * e.y() - dir.y() * e.x();
        if (std::abs(den) < 1e-9f) continue;                      // parallel
        const Eigen::Vector2f ao = a - O;
        const float r = (ao.x() * e.y() - ao.y() * e.x()) / den;
        const float t = (ao.x() * dir.y() - ao.y() * dir.x()) / den;
        if (r <= 1e-4f or t < 0.0f or t > 1.0f) continue;
        if (not best or r < best->r)
            best = WallHit{r, std::abs(dir.x() * e.y() - dir.y() * e.x()) / std::max(e.norm(), 1e-9f)};
    }
    return best;
}

struct RicohRange { float agree = 0.0f; float weight = 0.0f; Eigen::Vector2f xy = Eigen::Vector2f::Zero(); float var = -1.0f; };

/// Terms 1 and 2 for one slice seen from O: its birth weight and the wall-fused position.
[[nodiscard]] inline RicohRange ricoh_range(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& O,
                                            const Eigen::Vector2f& centroid, float depth_var, float sigma_base_m,
                                            float lat_per_m, float base_var = 0.0f)
{
    RicohRange o; o.xy = centroid;
    const Eigen::Vector2f d = centroid - O;
    const float rs = d.norm();
    if (rs < 1e-3f) return o;
    const Eigen::Vector2f dir = d / rs;
    const auto h = wall_hit(poly, O, dir);
    if (not h) return o;                                           // no wall along the bearing ⇒ nothing to birth on
    const float R0     = sigma_base_m * sigma_base_m + (lat_per_m * rs) * (lat_per_m * rs);
    const float var_s  = R0 + std::max(0.0f, depth_var);
    const float tanphi = std::sqrt(std::max(0.0f, 1.0f - h->cos_inc * h->cos_inc)) / std::max(h->cos_inc, 1e-3f);
    const float lever  = lat_per_m * h->r;
    const float var_w  = std::max(lever * lever * tanphi * tanphi, 1e-6f);
    const float dr     = rs - h->r;
    const float agree  = std::exp(-0.5f * dr * dr / (var_s + var_w));
    o.agree  = agree;                                              // is it the wall's door at all (no variance factor)
    o.weight = agree * (R0 / var_s);                                // ...and how much position information it carries
    const float r = (rs / var_s + h->r / var_w) / (1.0f / var_s + 1.0f / var_w);
    o.xy = O + dir * r;
    // position variance (isotropic): the base measurement noise + the bearing's lateral lever (sigma_b r)^2 + the fused along-ray variance
    o.var = base_var + (lat_per_m * r) * (lat_per_m * r) + 1.0f / (1.0f / var_s + 1.0f / var_w);
    return o;
}

/// LINE OF SIGHT from the 360 camera to a door's aperture centre, WALLS included: a door lies IN a wall, so the ray meets
/// the polygon at ~the door's distance; a wall met clearly BEFORE it blocks the view. Continuous: the shortfall in units of
/// the same wall-range and measurement variances ricoh_range uses. 1 = clear. No polygon / no hit ⇒ 1 (nothing blocks).
[[nodiscard]] inline float los_weight(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& O,
                                      const Eigen::Vector2f& target, float sigma_base_m, float lat_per_m)
{
    const Eigen::Vector2f d = target - O;
    const float dist = d.norm();
    if (dist < 1e-3f) return 1.0f;
    const auto h = wall_hit(poly, O, d / dist);
    if (not h) return 1.0f;
    const float R0     = sigma_base_m * sigma_base_m + (lat_per_m * dist) * (lat_per_m * dist);
    const float tanphi = std::sqrt(std::max(0.0f, 1.0f - h->cos_inc * h->cos_inc)) / std::max(h->cos_inc, 1e-3f);
    const float lever  = lat_per_m * h->r;
    const float var    = R0 + std::max(lever * lever * tanphi * tanphi, 1e-6f);
    const float short_by = std::max(0.0f, dist - h->r);
    return std::exp(-0.5f * short_by * short_by / var);
}

/// Term 3: the pooled viewpoints of the ricoh birth candidates (one per door position).
class RicohViewPool
{
public:
    /// Credit in [0,1] for a frame seen from `cam` of a door at `tgt` (half-size R); folds the view in.
    /// The credit the frame WOULD get, folding nothing in (the slice may turn out to be associated to a live door).
    [[nodiscard]] float peek(const Eigen::Vector2f& cam, const Eigen::Vector2f& tgt, float R, float match_m) const
    {
        RicohViewPool copy = *this;
        return copy.credit(cam, tgt, R, match_m);
    }
    float credit(const Eigen::Vector2f& cam, const Eigen::Vector2f& tgt, float R, float match_m)
    {
        const Eigen::Vector2f dv = cam - tgt;
        const float d = dv.norm();
        const float b = std::atan2(dv.y(), dv.x());
        const float Rr = std::max(R, 1e-3f);
        View* v = nullptr;
        for (auto& c : views_) if ((c.tgt - tgt).norm() < match_m) { v = &c; break; }
        float s = 0.0f;
        if (v == nullptr)
        {
            views_.push_back({tgt, b, d, 1.0f, true});
            return 1.0f;
        }
        const float half = std::max(std::atan2(Rr, std::max(d, 1e-3f)), 1e-3f);
        const float db = std::remainder(b - v->bearing, 2.0f * std::numbers::pi_v<float>);
        const float dr = d - v->range;
        s = std::exp(-std::min((db / half) * (db / half) + (dr / Rr) * (dr / Rr), 60.0f));
        const float n_old = s * v->n;
        v->n = n_old + 1.0f;
        v->bearing += std::remainder(b - v->bearing, 2.0f * std::numbers::pi_v<float>) / v->n;
        v->range   += (d - v->range) / v->n;
        v->tgt = tgt;
        v->seen = true;
        return 1.0f - s;
    }
    /// End of an observation: candidates not seen in it are gone (mirrors the tracker's candidate expiry).
    void end_frame()
    {
        std::erase_if(views_, [](const View& v){ return not v.seen; });
        for (auto& v : views_) v.seen = false;
    }
private:
    struct View { Eigen::Vector2f tgt; float bearing, range, n; bool seen; };
    std::vector<View> views_;
};

}  // namespace rc::door::core
