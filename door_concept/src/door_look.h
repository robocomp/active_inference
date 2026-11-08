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

#include "door_existence_info.h"
#include "door_identity_table.h"
#include "door_standpoint.h"
#include "../../common/nbv/viewpoint_score.h"
#include "core/door_decision.h"   // decide, Costs, Role
#include "core/detector_llr.h"    // DetectorModel
#include "core/place_belief.h"    // PlaceBelief (the one belief of a door place)

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

inline float binary_entropy(float p) { return rc::door::binary_entropy_nats(p); }

/// Mutual information (nats) between existence E ~ Bernoulli(p_exists) and one detector outcome (door_existence_info.h).
inline float existence_eig(float p_exists, float p_hit, float p_false) { return rc::door::existence_mutual_info(p_exists, p_hit, p_false); }

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
/// identity radius + remembered width), not removed, and not ruled out this visit. Only in a room that IS the remembered one
/// (`room_recalled`: entered by an ltsm recall or an accepted proposal): a room entered fresh may share the name, but the
/// rows' places are in another frame (recog_proto10), so nothing is offered there.
inline std::vector<IdentityRow> offer_rows(std::span<const IdentityRow> rows, std::string_view room,
                                           std::span<const LiveDoor> live, std::span<const std::string> ruled_out,
                                           float reacquire_radius_m, bool room_recalled)
{
    std::vector<IdentityRow> out;
    if (not room_recalled) return out;
    for (const auto& r : rows)
    {
        if (r.room != room) continue;
        if (r.removed_ms.has_value()) continue;   // a place already decided "no door here": nothing to look for
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

/// The existence belief of ONE remembered-but-absent door, fed from the views the robot actually has of its place.
/// It is the SAME model as a live door's (spec §3: one belief, one ledger, one decision): L = the remembered evidence
/// (the place row's aged L, no other prior) + the view ledger; each frame is a Bernoulli trial of the ZED detector with
/// q = hit_rate · p_detect (p_detect: envelope × visibility of THIS standpoint, nbv::score) and the false-alarm rate that
/// passes through the same front end (detector_llr.h); a view that could not see the place (q = 0) holds. The ledger pools
/// the frames of one viewpoint into ONE trial (view_ledger.h), so a parked stare is one look. Ruled out = the costed
/// deletion line of door_decision.h on L (no debounce, no streak): the look has no role, it is an Ordinary place.
class Evidence
{
public:
    /// remembered_L = the row's aged L_evidence, remembered_physical its physical part (PlaceBelief::remember): vision saturates across the memory.
    explicit Evidence(float remembered_L, float remembered_physical = 0.f, float lookalike = rc::door::core::kLookalikeProbDefault, bool crossed = false)
    {
        b_.lookalike = lookalike;
        b_.crossed = crossed;
        b_.remember(remembered_L, remembered_physical);
    }
    /// One mask frame of the ZED seen from `cam`. `detected_here` = a door detection landed on this hole. True ⇒ ruled out.
    bool observe(const IdentityRow& r, std::span<const Eigen::Vector2f> poly, const Eigen::Vector2f& cam, float p_detect,
                 bool detected_here, const rc::door::core::DetectorModel& zed, const rc::door::core::Costs& costs,
                 float prior_w_m, float prior_h_m)
    {
        // p_detect is a probability by construction; a non-finite or out-of-range value is "could not look", not a datum.
        const float pd = std::isfinite(p_detect) ? p_detect : 0.f;   // (nbv::score already returns it in [0, 1])
        const float q = zed.hit_rate * pd;
        if (q > 0.f)
        {
            const Eigen::Vector2f u = along_wall(r, poly).value_or(Eigen::Vector2f(1.f, 0.f));
            const float w = r.pose.has_value() ? r.pose->width : prior_w_m, h = r.pose.has_value() ? r.pose->height : prior_h_m;
            const float R = 0.5f * std::hypot(w, h);   // the door's own scale: the view kernel's width (view_ledger.h)
            const auto v = b_.ledger.observe(cam, r.xy, Eigen::Vector2f(-u.y(), u.x()), R, 0.f, 0);
            b_.ledger.add_bernoulli(v, rc::door::core::Channel::Zed, q, zed, detected_here ? 1.f : 0.f);
            b_.ledger.recompute(v, false);
        }
        return rc::door::core::decide(b_.L(), b_.crossed ? rc::door::core::Role::BeingCrossed : rc::door::core::Role::Ordinary, costs).remove;   // a crossed place is never ruled out
    }
    float p_exists()   const { return b_.p(); }
    float logodds()    const { return b_.L(); }
    int   viewpoints() const { return static_cast<int>(b_.ledger.views().size()); }

private:
    rc::door::core::PlaceBelief b_{};
};

}   // namespace rc::door::look
