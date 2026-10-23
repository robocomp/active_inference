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
#include <ranges>
#include <cmath>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "../door_view_pool.h"

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
        const float width = std::remainder(w.y() - w.x(), 2.0f * std::numbers::pi_v<float>);
        const float off   = std::remainder(a - w.x(), 2.0f * std::numbers::pi_v<float>);
        const float offp  = off < 0.0f ? off + 2.0f * std::numbers::pi_v<float> : off;
        const float wp    = width < 0.0f ? width + 2.0f * std::numbers::pi_v<float> : width;
        if (offp <= wp) return true;
    }
    return false;
}

struct RicohRange { float agree = 0.0f; float weight = 0.0f; Eigen::Vector2f xy = Eigen::Vector2f::Zero(); float var = -1.0f; };

/// Terms 1 and 2 for one slice seen from O: its birth weight and the wall-fused position.
[[nodiscard]] inline RicohRange ricoh_range(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& O,
                                            const Eigen::Vector2f& centroid, float depth_var, float sigma_base_m,
                                            float lat_per_m, float base_var = 0.0f, float plane_var = 0.0f, bool niche_aware = false)
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
    o.agree  = agree;                                              // is it the wall's door at all (no variance factor)
    o.weight = agree * (R0 / var_s);                               // ...and how much position information it carries
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
    /// `looked(tgt)` false for a candidate whose strip was not processed this frame: it is HELD (its view survives), not expired.
    template <class Looked>
    void end_frame(Looked&& looked)
    {
        std::erase_if(views_, [&](const View& v){ return not v.seen and looked(v.tgt); });
        for (auto& v : views_) v.seen = false;
    }
    void end_frame() { end_frame([](const Eigen::Vector2f&) { return true; }); }
private:
    struct View { Eigen::Vector2f tgt; float bearing, range, n; bool seen; };
    std::vector<View> views_;
};

/// One door's calibration of the 360 detector's hit rate, POOLED PER VIEWPOINT: the frames of one view are one Bernoulli trial
/// (door_view_pool.h's view model), whose outcome is the view's detected fraction; a new view is a new trial. add() returns the
/// (hits, trials) increments to fold into the shared Beta posterior - 100 parked frames add 1 trial in total, K views add K.
struct RicohHitPool
{
    rc::door::ViewPool hit{}, trial{};
    std::pair<float, float> add(const Eigen::Vector2f& cam, const Eigen::Vector2f& door, float R, bool seen)
    {
        const float t = trial.add_logodds(cam, door, R, 1.0f);
        const float h = hit.add_logodds(cam, door, R, seen ? 1.0f : 0.0f);
        return {h, t};
    }
};

/// The 360 camera's BIRTH EVIDENCE BANK. Why it exists (gold4, 2026-10-23, replay): DOOR_0 was seen by the ricoh 1962 times
/// from a parked pose and never born, because the tracker's candidate EXPIRES the cycle it gets no detection and the 360 detector
/// flickers (~1 look in 3): the longest streak was 4 hits, each worth ~0.15 ideal observations against a bar of 6.6.
/// The bank keeps a ricoh-seen place alive across the gaps and settles it by the same Bernoulli LLR the existence channel uses:
///   hit  : + agree · ln(h / pc)               (agree = ricoh_range's wall agreement; h = the 360's hit rate, pc = clutter)
///   miss : + ln((1 - q h) / (1 - q pc))        (q = P(detectable from here) x line of sight, so an unresolvable look says nothing)
/// L < 0 means no evidence for the place: the entry is dropped (that is the prior, not a cut). A phantom that is detected at a
/// rate below ~pc/h ... drifts down by its misses and never ripens; a door detected at rate >= the break-even ripens.
/// It only RIPENS an entry (L >= birth_frames ideal ricoh hits); the birth itself is still the tracker's (a ripe place's next
/// detection is credited a full birth_frames), so min-separation, association and exclusion claims still apply.
class RicohBirthBank
{
public:
    struct Entry { Eigen::Vector2f xy; float var; float L; };
    struct Look { Eigen::Vector2f xy; float var; float agree; };
    /// One new observation. `hits` are this frame's UNASSOCIATED ricoh door detections; `q_of(xy)` the detectability x LOS
    /// of a place from the camera; entries near a live track are dropped.
    template <class Q>
    void update(const std::vector<Look>& hits, float hit_rate, float clutter, float gate_mahalanobis, Q&& q_of,
                const std::vector<Eigen::Vector2f>& track_xy, float min_sep)
    {
        const float h  = std::clamp(hit_rate, 1e-3f, 1.0f - 1e-3f), pc = std::clamp(clutter, 1e-3f, 1.0f - 1e-3f);
        const float llr_hit = std::log(h / pc);
        std::vector<char> matched(entries_.size(), 0);
        for (const auto& k : hits)
        {
            int best = -1; float best_m = gate_mahalanobis;
            for (std::size_t i = 0; i < entries_.size(); ++i)
            {
                const float m = (entries_[i].xy - k.xy).squaredNorm() / std::max(entries_[i].var + k.var, 1e-9f);
                if (m <= best_m) { best_m = m; best = static_cast<int>(i); }
            }
            if (best < 0)
            {
                entries_.push_back({k.xy, k.var, k.agree * llr_hit});
                matched.push_back(1);
                continue;
            }
            auto& e = entries_[static_cast<std::size_t>(best)];
            const float wc = 1.0f / std::max(e.var, 1e-9f), wd = 1.0f / std::max(k.var, 1e-9f);
            e.xy = (e.xy * wc + k.xy * wd) / (wc + wd);
            e.var = 1.0f / (wc + wd);
            e.L += k.agree * llr_hit;
            matched[static_cast<std::size_t>(best)] = 1;
        }
        for (std::size_t i = 0; i < entries_.size(); ++i)
        {
            if (matched[i]) continue;
            const float q = std::clamp(q_of(entries_[i].xy), 0.0f, 1.0f);
            entries_[i].L += std::log((1.0f - q * h) / (1.0f - q * pc));
        }
        std::erase_if(entries_, [&](const Entry& e)
        {
            if (e.L <= 0.0f) return true;
            return std::ranges::any_of(track_xy, [&](const Eigen::Vector2f& t) { return (t - e.xy).norm() < min_sep; });
        });
    }
    /// The entry a detection at `xy` (variance `var`) belongs to when that place is ripe (L >= bar), else nullptr.
    [[nodiscard]] const Entry* ripe_at(const Eigen::Vector2f& xy, float var, float gate_mahalanobis, float bar) const
    {
        for (const auto& e : entries_)
            if (e.L >= bar and (e.xy - xy).squaredNorm() / std::max(e.var + var, 1e-9f) <= gate_mahalanobis) return &e;
        return nullptr;
    }
    void drop_near(const Eigen::Vector2f& xy, float r)
    { std::erase_if(entries_, [&](const Entry& e) { return (e.xy - xy).norm() < r; }); }
    [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }
private:
    std::vector<Entry> entries_;
};

}  // namespace rc::door::core
