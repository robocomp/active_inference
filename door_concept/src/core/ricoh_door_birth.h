// ricoh_door_birth.h — what is ONE 360-camera door slice worth, and where is it? (door_concept only; pure, Eigen.)
//
// The ricoh slice's range comes from LiDAR reprojected onto the mask, which can land on whatever is behind the door, so its
// position is not taken at face value. Two continuous terms, none a cutoff, multiplied into the detection's reliability:
//   1. WALL: a door lies in a wall, so the bearing ray meets the room polygon at r_w. The slice's LiDAR range r_s is
//      scored against it (Gaussian agreement over their summed variances); no wall along the bearing => weight 0.
//      The position is the precision-weighted fusion of r_s and r_w, so a LiDAR range that agrees but is noisy is
//      pulled onto the wall. The wall-range variance is the bearing error lever arm, (sigma_b r_w)^2 tan^2(phi) (phi = incidence
//      from the wall normal): grazing rays are ill-conditioned and say little.
//   2. VARIANCE: R0 / (R0 + depth_var), R0 = the fit's own R for a ZED slice at that range (door_fit.cpp).
// Also here: which wall hits a bearing sees (wall_hit, wall_hypotheses), which strips the 360 detector processed
// (ricoh_looked: EMPTY windows = unknown = not looked at). The ricoh birth BANK (ripening bar) and
// the separate ricoh view pool are GONE (docs/superpowers/plans/2026-11-07-door-existence-model.md): a ricoh frame is a Bernoulli
// trial in the same view ledger as the ZED's, and a place is born by comparison (L > 0), not by ripening.
#pragma once
#include <algorithm>
#include <ranges>
#include <cmath>
#include <numbers>
#include <optional>
#include <utility>
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

/// The wall planes a door seen along O + r·dir may be IN, as (r, cos_inc) hypotheses. The nearest crossing is the door's wall
/// only on a plain wall. A DOORWAY NICHE (a recess the outline makes at the door: side faces + a back face) interrupts the
/// wall's line, and the ray into it meets the BACK face, not the wall the door belongs to. So a crossing edge whose two
/// neighbours are perpendicular to it (a niche back) also offers the LINE of the wall it interrupts, taken from the edges
/// beyond those neighbours when they are collinear. The caller scores each hypothesis and keeps the best explanation.
[[nodiscard]] inline std::vector<WallHit> wall_hypotheses(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& O,
                                                          const Eigen::Vector2f& dir)
{
    std::vector<WallHit> hs;
    const std::size_t n = poly.size();
    if (n < 3) return hs;
    const auto edge = [&](std::size_t i) { return poly[(i + 1) % n] - poly[i % n]; };
    const auto cross_at = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& e) -> std::optional<float>
    {
        const float den = dir.x() * e.y() - dir.y() * e.x();
        if (std::abs(den) < 1e-9f) return std::nullopt;
        const Eigen::Vector2f ao = a - O;
        return (ao.x() * e.y() - ao.y() * e.x()) / den;
    };
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f a = poly[i], e = edge(i);
        const auto r = cross_at(a, e);
        if (not r or *r <= 1e-4f) continue;
        const float den = dir.x() * e.y() - dir.y() * e.x();
        const Eigen::Vector2f ao = a - O;
        const float t = (ao.x() * dir.y() - ao.y() * dir.x()) / den;
        if (t < 0.0f or t > 1.0f) continue;
        hs.push_back({*r, std::abs(den) / std::max(e.norm(), 1e-9f)});
        // niche back? neighbours perpendicular to it, the edges beyond them collinear with each other
        const Eigen::Vector2f eu = e.normalized();
        const Eigen::Vector2f pe = edge(i + n - 1), ne = edge(i + 1), pp = edge(i + n - 2), nn = edge(i + 2);
        if (n < 5 or pe.norm() < 1e-6f or ne.norm() < 1e-6f or pp.norm() < 1e-6f or nn.norm() < 1e-6f) continue;
        const bool perp = std::abs(pe.normalized().dot(eu)) < 0.2f and std::abs(ne.normalized().dot(eu)) < 0.2f;
        const Eigen::Vector2f lu = pp.normalized();
        const bool beyond = std::abs(lu.dot(eu)) > 0.9f and std::abs(nn.normalized().dot(lu)) > 0.9f
                            and std::abs((poly[(i + 2) % n] - poly[(i + n - 1) % n]).dot(Eigen::Vector2f(-lu.y(), lu.x()))) < 0.05f;
        if (not (perp and beyond)) continue;
        const Eigen::Vector2f La = poly[(i + n - 1) % n];               // a point of the interrupted wall line
        const auto rl = cross_at(La, lu);
        if (rl and *rl > 1e-4f)
            hs.push_back({*rl, std::abs(dir.x() * lu.y() - dir.y() * lu.x())});
    }
    return hs;
}

/// Was the bearing from the 360 camera at O to `xy` inside the strips the detector ran on this frame? `windows` are the ROOM-frame
/// azimuth intervals (from, to), counter-clockwise (MaskPacket::ricoh_windows); EMPTY = unknown ⇒ looked at (nothing is withheld).
/// 'Not looked at' is NOT 'not seen': a door outside the processed strip gets no evidence and no absence from that frame.
[[nodiscard]] inline bool ricoh_looked(const std::vector<Eigen::Matrix<float, 2, 1>>& windows, const Eigen::Vector2f& O,
                                       const Eigen::Vector2f& xy)
{
    if (windows.empty()) return true;
    const Eigen::Vector2f d = xy - O;
    if (d.squaredNorm() < 1e-6f) return true;
    const float a = std::atan2(d.y(), d.x());
    for (const auto& w : windows)
    {
        const float two_pi = 2.0f * std::numbers::pi_v<float>;
        float width = w.y() - w.x();                                    // the interval is counter-clockwise from w.x to w.y
        if (width >= two_pi - 1e-4f) return true;                       // the whole panorama
        width = std::fmod(width, two_pi);
        if (width < 0.0f) width += two_pi;
        float off = std::fmod(a - w.x(), two_pi);
        if (off < 0.0f) off += two_pi;
        if (off <= width + 1e-6f) return true;                          // closed: a bearing on the seam of two adjacent strips is in both
    }
    return false;
}

/// ★THREE STATES of the 360 camera on a cycle's mask packet (the merged packet carries ricoh windows ONLY when the ricoh node was fresh):
///   (a) a fresh ricoh frame WITH windows           => a bearing is looked at iff inside them;
///   (b) a fresh ricoh frame WITHOUT the attribute   => unknown (an old producer) => every bearing looked at (legacy);
///   (c) NO fresh ricoh frame in this packet         => NOTHING looked: no evidence and no absence for the ricoh at all.
/// A fresh frame is recognised by its windows or by any ricoh-source slice (an old producer's empty frame is then (c): it said nothing).
template <class Packet>
[[nodiscard]] inline bool ricoh_fresh(const Packet& pkt)
{
    return not pkt.ricoh_windows.empty()
        or std::ranges::any_of(pkt.slices, [](const auto& q) { return q.source != 0; });
}
template <class Packet>
[[nodiscard]] inline bool ricoh_looked_in(const Packet& pkt, const Eigen::Vector2f& O, const Eigen::Vector2f& xy)
{
    return ricoh_fresh(pkt) and ricoh_looked(pkt.ricoh_windows, O, xy);
}

struct RicohRange
{
    float agree = 0.0f; float weight = 0.0f; Eigen::Vector2f xy = Eigen::Vector2f::Zero(); float var = -1.0f;
    float agree_shut = 0.0f, agree_open = 0.0f;   // the two terms `agree` mixes (agree_open = 0 unless p_open > 0); for the logs
    float rho_open = 0.0f;                         // P(the slice is an open leaf | it was seen), the mixture's posterior
};

/// Gaussian CDF.
[[nodiscard]] inline float normal_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }

/// Terms 1 and 2 for one slice seen from O: its birth weight and the wall-fused position.
/// ★OPEN-AWARE (Task D2, gold12): `p_open` is the prior probability that the slice is a leaf standing OPEN, `leaf_w` the leaf's
/// width (m). A shut leaf lies in the wall: the slice's range agrees with the wall's (the original term). An open leaf stands off
/// the wall along the ray by anything in [-leaf_w, +leaf_w] (it swings through that arc): its range likelihood is the Gaussian
/// range error integrated over that uniform offset, Phi((dr+w)/s) - Phi((dr-w)/s) (peak ~1, flat over the arc, no wall-hit gate).
/// `agree` is the mixture (1-p_open)·shut + p_open·open: a slice ON the wall keeps weight 1 whatever p_open is, a slice a leaf-width
/// off it gets ~p_open, one farther than the arc ~0. The door is IN the wall, so under "open" its position is the wall point (the
/// slice says only that a leaf is near it); the fused position is the posterior mixture of the two, moment-matched. p_open = 0
/// reproduces the original function exactly (every other caller).
[[nodiscard]] inline RicohRange ricoh_range(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& O,
                                            const Eigen::Vector2f& centroid, float depth_var, float sigma_base_m,
                                            float lat_per_m, float base_var = 0.0f, float plane_var = 0.0f, bool niche_aware = false,
                                            float p_open = 0.0f, float leaf_w = 0.0f)
{
    RicohRange o; o.xy = centroid;
    const Eigen::Vector2f d = centroid - O;
    const float rs = d.norm();
    if (rs < 1e-3f) return o;
    const Eigen::Vector2f dir = d / rs;
    std::vector<WallHit> hyps;
    if (niche_aware) hyps = wall_hypotheses(poly, O, dir);
    else if (const auto h0 = wall_hit(poly, O, dir)) hyps.push_back(*h0);
    if (hyps.empty()) return o;                                    // no wall along the bearing ⇒ nothing to birth on
    const float R0     = sigma_base_m * sigma_base_m + (lat_per_m * rs) * (lat_per_m * rs);
    const float var_s  = R0 + std::max(0.0f, depth_var);
    // each wall hypothesis is scored; the door is IN one wall, so the best explanation stands (niche back face vs the wall line it interrupts)
    float best_agree = -1.0f; WallHit hb{}; float var_wb = 0.0f;
    for (const auto& h : hyps)
    {
        const float tanphi = std::sqrt(std::max(0.0f, 1.0f - h.cos_inc * h.cos_inc)) / std::max(h.cos_inc, 1e-3f);
        const float lever  = lat_per_m * h.r;
        const float var_w  = std::max(lever * lever * tanphi * tanphi, 1e-6f) + std::max(0.0f, plane_var);
        const float dr     = rs - h.r;
        const float ag     = std::exp(-0.5f * dr * dr / (var_s + var_w));
        if (ag > best_agree) { best_agree = ag; hb = h; var_wb = var_w; }
    }
    const float var_w = var_wb;
    const WallHit* h = &hb;
    const float agree = best_agree;
    const float r_shut = (rs / var_s + h->r / var_w) / (1.0f / var_s + 1.0f / var_w);
    const float var_shut = 1.0f / (1.0f / var_s + 1.0f / var_w);
    o.agree_shut = agree;
    if (p_open <= 0.0f or leaf_w <= 0.0f)
    {
        o.agree  = agree;                                          // is it the wall's door at all (no variance factor)
        o.weight = agree * (R0 / var_s);                           // ...and how much position information it carries
        o.xy = O + dir * r_shut;
        // position variance (isotropic): the base measurement noise + the bearing's lateral lever (sigma_b r)^2 + the fused along-ray variance
        o.var = base_var + (lat_per_m * r_shut) * (lat_per_m * r_shut) + var_shut;
        return o;
    }
    const float po = std::clamp(p_open, 0.0f, 1.0f);
    const float s  = std::sqrt(var_s + var_w);                     // range error of (slice - wall) when nothing else is known
    const float dr = rs - h->r;
    const float ag_open = std::max(0.0f, normal_cdf((dr + leaf_w) / s) - normal_cdf((dr - leaf_w) / s));
    const float mix = (1.0f - po) * agree + po * ag_open;
    const float rho = mix > 1e-12f ? po * ag_open / mix : 0.0f;
    // open: the door is in the wall (the slice sees only the leaf near it); its position is the wall point, with the wall's variance
    const float r_open = h->r, var_open = var_w;
    const float r = (1.0f - rho) * r_shut + rho * r_open;
    const float var_r = (1.0f - rho) * (var_shut + (r_shut - r) * (r_shut - r)) + rho * (var_open + (r_open - r) * (r_open - r));
    o.agree = mix; o.agree_open = ag_open; o.rho_open = rho;
    o.weight = mix * (R0 / var_s);
    o.xy = O + dir * r;
    o.var = base_var + (lat_per_m * r) * (lat_per_m * r) + var_r;
    return o;
}

}  // namespace rc::door::core
