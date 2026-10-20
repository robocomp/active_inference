/*
 * door_look.h — the EPISTEMIC "look here" affordance for a REMEMBERED door that is not instantiated.
 *
 * WHY (2026-10-21, tourP5 leg 1): room_1's identity table (etc/door_identities.csv) still held door_2's name and
 * aperture, but door_2 was not in the graph — the 60 s exploration went to the far side of room_1 and to door_1 and
 * never looked at door_2, so `cross door_2` had nothing to cross. A remembered door is not nothing: it is a PRIOR on
 * a door at a known place, and the one action that resolves it is to look at that place. So door_concept offers that
 * look, valued like every other epistemic affordance in this fleet: by the information it is expected to buy — here
 * about the door's EXISTENCE, a binary variable with prior P(exists) — and it withdraws it as soon as the question is
 * answered: the door is re-born (identity match by name or by hole), or absence evidence from the view rules it out
 * (then the row is not offered again this visit).
 *
 * The geometry is the shared NBV model (common/nbv/viewpoint_score.h): the remembered aperture as a Target (a leaf of
 * the remembered — or prior — width/height, flush in its wall), the walls as occluders with the doorways punched out,
 * the stand-off band from the detector envelope, P(detect | pose) from fill + visible fraction. Nothing here is a new
 * threshold: candidates are a quadrature over the band and a fan about the wall normal, and selection is argmax EIG.
 *
 * Pure (Eigen + std + common headers): tests/test_door_look.cpp.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "door_identity_table.h"
#include "door_standpoint.h"
#include "door_leaf_channel.h"   // ViewBearings / view_bearings / ViewpointLedger — the viewpoint, shared
#include "../../common/nbv/viewpoint_score.h"
#include "../../common/existence_belief/existence_belief.h"

namespace rc::door::look
{

struct Params
{
    float robot_radius_m = 0.30f;   // feasibility: the standpoint must hold the robot's body
    float clutter_prob   = 0.05f;   // P(detect | ¬exists)  (Existence.ClutterProb)
    float detection_prob = 0.85f;   // P(detect | exists, observable)  (Existence.DetectionProb)
    float prior_w_m      = 0.90f;   // geometry of a row that remembers only its place (DoorPriorW/H, thickness)
    float prior_h_m      = 2.00f;
    float thickness_m    = 0.05f;
};

struct Plan
{
    bool            valid = false;
    Eigen::Vector2f pos{0.f, 0.f};
    float           yaw = 0.f;
    float           p_detect = 0.f;       // P(detector fires | door there, this pose) — envelope × visibility
    float           visible_frac = 0.f;
    float           eig = 0.f;            // nats: expected information gain about existence from one look
    float           standoff_m = 0.f, band_near_m = 0.f, band_far_m = 0.f;
};

inline float binary_entropy(float p)
{
    const auto h = [](float q) { return q > 0.f ? -q * std::log(q) : 0.f; };
    p = std::clamp(p, 0.f, 1.f);
    return h(p) + h(1.f - p);
}

/// Mutual information (nats) between existence E ~ Bernoulli(p_exists) and one detector outcome Y with
/// P(Y=1|E=1) = p_hit, P(Y=1|E=0) = p_false. Zero when the prior is certain or the look cannot tell the two apart.
inline float existence_eig(float p_exists, float p_hit, float p_false)
{
    const float p = std::clamp(p_exists, 0.f, 1.f);
    const float py = p * std::clamp(p_hit, 0.f, 1.f) + (1.f - p) * std::clamp(p_false, 0.f, 1.f);
    return std::max(0.f, binary_entropy(py) - p * binary_entropy(p_hit) - (1.f - p) * binary_entropy(p_false));
}

/// The remembered aperture's along-wall direction: its remembered yaw, or (identity-only row) the wall of the room
/// polygon it lies on. Unit, room frame; nullopt when neither is available.
inline std::optional<Eigen::Vector2f> along_wall(const IdentityRow& r, std::span<const Eigen::Vector2f> poly)
{
    if (r.pose.has_value() and std::isfinite(r.pose->yaw))
        return Eigen::Vector2f(std::cos(r.pose->yaw), std::sin(r.pose->yaw));
    const std::vector<Eigen::Vector2f> pts{r.xy};
    if (const WallRun w = associate_wall(poly, pts, std::nullopt); w.ok)
        return w.u;
    return std::nullopt;
}

/// The look plan for one remembered door. `other_gaps` = the other doorways (their walls are open too);
/// `extra` = furniture obstacles. Refuses (invalid) on an incomplete camera model or no feasible standpoint.
inline Plan plan(const IdentityRow& r, float p_exists, std::span<const Eigen::Vector2f> poly, const rc::nbv::Sensor& sensor,
                 std::span<const rc::nbv::Obstacle> extra, std::span<const rc::nbv::WallGap> other_gaps, const Params& prm)
{
    Plan best;
    if (not sensor.complete() or poly.size() < 3 or not r.xy.allFinite())
        return best;
    const auto u_opt = along_wall(r, poly);
    if (not u_opt.has_value())
        return best;
    const Eigen::Vector2f u = *u_opt;
    const float w = r.pose.has_value() ? r.pose->width  : prm.prior_w_m;
    const float h = r.pose.has_value() ? r.pose->height : prm.prior_h_m;
    const float t = r.pose.has_value() ? r.pose->depth  : prm.thickness_m;

    // The room side of the wall: the normal whose stand-off lies inside the polygon (a region question, as in
    // interior_side; a door in a partition with the same room on both sides keeps +n).
    Eigen::Vector2f n(-u.y(), u.x());
    const float probe = prm.robot_radius_m + w;
    if (not polygon_contains(r.xy + probe * n, poly) and polygon_contains(r.xy - probe * n, poly))
        n = -n;

    rc::nbv::Target tg;
    tg.cx = r.xy.x(); tg.cy = r.xy.y(); tg.yaw = std::atan2(u.y(), u.x());
    tg.w = w; tg.h = std::max(1e-3f, t); tg.z0 = 0.f; tg.z1 = h;

    std::vector<rc::nbv::WallGap> gaps(other_gaps.begin(), other_gaps.end());
    gaps.push_back({r.xy, 0.5f * w});
    auto obstacles = rc::nbv::wall_obstacles(poly, std::max(1e-3f, t), gaps);
    obstacles.insert(obstacles.end(), extra.begin(), extra.end());

    const rc::nbv::StandoffBand band =
        rc::nbv::standoff_band(tg, sensor, r.xy, n, prm.robot_radius_m + 0.5f * t);
    // Quadrature over the band (near / best / mid-far) and a fan about the normal: the detector envelope and the
    // walls decide among them through p_detect, not a rule.
    const std::array<float, 3> dists{band.near_m, band.best, 0.5f * (band.best + band.far_m)};
    constexpr std::array<float, 7> kFanDeg{0.f, -15.f, 15.f, -30.f, 30.f, -45.f, 45.f};
    for (const float deg : kFanDeg)
    {
        const float a = deg * std::numbers::pi_v<float> / 180.f;
        const Eigen::Vector2f dir = Eigen::Rotation2Df(a) * n;
        for (const float d : dists)
        {
            const Eigen::Vector2f pos = r.xy + d * dir;
            if (not polygon_contains(pos, poly) or rc::nbv::detail::distance_to_boundary(pos, poly) < prm.robot_radius_m)
                continue;
            const Eigen::Vector2f to = r.xy - pos;
            const float yaw = std::atan2(to.y(), to.x());
            const auto s = rc::nbv::score({pos, yaw, 1.f, -1}, tg, sensor, obstacles, prm.robot_radius_m);
            if (s.stands_inside)
                continue;
            // A look resolves the question only as far as it could see the place: weighted by p_detect exactly as the
            // existence channel weights the evidence it then gets (rc::exist integrate: delta = p_vis · llr).
            const float eig = s.p_detect * existence_eig(p_exists, prm.detection_prob, prm.clutter_prob);
            if (not best.valid or eig > best.eig)
            {
                best.valid = true;
                best.pos = pos; best.yaw = yaw;
                best.p_detect = s.p_detect; best.visible_frac = s.visible_frac;
                best.eig = eig; best.standoff_m = d;
            }
        }
    }
    best.band_near_m = band.near_m;
    best.band_far_m  = band.far_m;
    if (best.valid and not (best.eig > 0.f))
        best.valid = false;   // nothing any reachable pose could learn: no look to offer
    return best;
}

/// P(detect | door there) from an ACTUAL camera pose (the agent's current zed pose), for the absence channel.
inline float p_detect_from(const IdentityRow& r, const Eigen::Vector2f& cam, float cam_yaw,
                           std::span<const Eigen::Vector2f> poly, const rc::nbv::Sensor& sensor,
                           std::span<const rc::nbv::Obstacle> extra, std::span<const rc::nbv::WallGap> other_gaps,
                           const Params& prm)
{
    if (not sensor.complete() or poly.size() < 3)
        return 0.f;
    const auto u_opt = along_wall(r, poly);
    if (not u_opt.has_value())
        return 0.f;
    const float w = r.pose.has_value() ? r.pose->width  : prm.prior_w_m;
    const float h = r.pose.has_value() ? r.pose->height : prm.prior_h_m;
    const float t = r.pose.has_value() ? r.pose->depth  : prm.thickness_m;
    rc::nbv::Target tg;
    tg.cx = r.xy.x(); tg.cy = r.xy.y(); tg.yaw = std::atan2(u_opt->y(), u_opt->x());
    tg.w = w; tg.h = std::max(1e-3f, t); tg.z0 = 0.f; tg.z1 = h;
    std::vector<rc::nbv::WallGap> gaps(other_gaps.begin(), other_gaps.end());
    gaps.push_back({r.xy, 0.5f * w});
    auto obstacles = rc::nbv::wall_obstacles(poly, std::max(1e-3f, t), gaps);
    obstacles.insert(obstacles.end(), extra.begin(), extra.end());
    const auto s = rc::nbv::score({cam, cam_yaw, 1.f, -1}, tg, sensor, obstacles, 0.f);
    return std::isfinite(s.p_detect) ? s.p_detect : 0.f;
}

/// Which of `room`'s remembered doors to offer a look at: not live (by name, or a live door on the same hole —
/// reacquire radius + remembered width), and not ruled out this visit.
inline std::vector<IdentityRow> offer_rows(std::span<const IdentityRow> rows, std::string_view room,
                                           std::span<const LiveDoor> live, std::span<const std::string> ruled_out,
                                           float reacquire_radius_m)
{
    std::vector<IdentityRow> out;
    for (const auto& r : rows)
    {
        if (r.room != room) continue;
        if (std::ranges::find(ruled_out, r.name) != ruled_out.end()) continue;
        const float same_hole = std::max(0.f, reacquire_radius_m) + (r.pose.has_value() ? std::max(0.f, r.pose->width) : 0.f);
        const bool is_live = std::ranges::any_of(live, [&](const LiveDoor& l)
        {
            return l.name == r.name or (l.xy_in_room.has_value() and l.xy_in_room->allFinite()
                                        and (*l.xy_in_room - r.xy).norm() < same_hole);
        });
        if (not is_live and std::ranges::none_of(out, [&](const IdentityRow& k) { return k.name == r.name; }))
            out.push_back(r);
    }
    return out;
}

/// The viewpoint a camera at `cam` has on the remembered aperture: its bearing from EACH jamb, in the door's own frame —
/// the leaf channel's viewpoint (door_leaf_channel.h, try102), so "the same viewpoint" means the same thing in both.
inline ViewBearings view_of(const IdentityRow& r, std::span<const Eigen::Vector2f> poly, const Eigen::Vector2f& cam,
                            float prior_w_m = 0.90f)
{
    Aperture ap;
    const Eigen::Vector2f u = along_wall(r, poly).value_or(Eigen::Vector2f(1.f, 0.f));
    ap.wall_u = u;
    ap.w = r.pose.has_value() ? r.pose->width : prior_w_m;
    ap.wall_O = r.xy - 0.5f * ap.w * u;
    ap.s = 0.f;
    return view_bearings(ap, cam);
}

/// A viewpoint is the jamb bearings within the leaf channel's viewpoint radius: the φ grid's step (5°), the model's
/// own resolution (door_leaf_channel.h, kViewpointRad).
inline constexpr double kLookViewpointRad = 5.0 * std::numbers::pi / 180.0;

/// The existence belief of ONE remembered-but-absent door, fed from the views the robot actually has of its place.
/// ★POOLED PER VIEWPOINT (tourP9, 2026-10-21). door_2 was ruled out (P 0.018) from ONE standpoint 2.8 m away at
///  p_detect 0.41 by many frames of non-detection treated as independent — a real, closed door. Frames from the same
///  viewpoint share everything that decides whether the detector fires there (the framing, the lighting, the occluder,
///  the detector's own blind spot for that aspect): they are copies of ONE look. So a viewpoint contributes the MEAN of
///  its frames' evidence (each frame p_detect · log-ratio, the rc::exist weighting), i.e. one look's worth, bounded however
///  many frames it sends; a revisit re-uses its record; only NEW viewpoints add evidence — the leaf channel's rule (term 2,
///  commit 6f267858) in its simplest form. The removal decision is the shared rc::exist::decide_removal.
///  A look that could not have seen the door (p_detect 0: behind, occluded, out of range) HOLDS.
class Evidence
{
public:
    Evidence(float p_prior, const rc::exist::RemovalPolicy& pol)
    {
        rc::exist::arm(b_, pol);
        const float p = std::clamp(p_prior, 1e-4f, 1.f - 1e-4f);
        L0_ = std::log(p / (1.f - p));
        b_.set(L0_);
    }
    /// One mask frame seen from viewpoint `at`. `detected_here` = a door detection landed on this hole. True ⇒ ruled out.
    bool observe(float p_detect, bool detected_here, float p_hit, float p_false, const ViewBearings& at,
                 const rc::exist::RemovalPolicy& pol)
    {
        rc::exist::arm(b_, pol);
        const float pd = std::isfinite(p_detect) ? std::clamp(p_detect, 0.f, 1.f) : 0.f;
        if (pd > 0.f)
        {
            const float ph = std::clamp(p_hit, 1e-3f, 1.f - 1e-3f), pf = std::clamp(p_false, 1e-3f, 1.f - 1e-3f);
            const double llr = detected_here ? std::log(ph / pf) : std::log((1.f - ph) / (1.f - pf));
            ledger_.advance(at, kLookViewpointRad);
            if (pool_.size() < ledger_.views.size()) pool_.resize(ledger_.views.size());
            auto& v = pool_[std::size_t(ledger_.current)];
            v.sum += double(pd) * llr;
            v.n   += 1;
            double L = L0_;
            for (const auto& q : pool_) if (q.n > 0) L += q.sum / q.n;
            b_.set(static_cast<float>(L));
        }
        return rc::exist::decide_removal(b_, deb_, pol, pd).remove;
    }
    float p_exists()   const { return b_.p_exists(); }
    float logodds()    const { return b_.logodds(); }
    int   viewpoints() const { return static_cast<int>(ledger_.views.size()); }

private:
    struct Pool { double sum = 0.0; int n = 0; };
    float L0_ = 0.f;
    rc::exist::ExistenceBelief b_{};
    rc::exist::RemovalDebounce deb_{};
    ViewpointLedger ledger_;
    std::vector<Pool> pool_;
};

}   // namespace rc::door::look
