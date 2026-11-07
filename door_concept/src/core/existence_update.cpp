// existence_update.cpp — see existence_update.h.
#include "existence_update.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

#include "../door_geometry.h"
#include "../door_model.h"
#include "../../../common/contour_edge/contour_edge_project.h"

namespace rc::door::core
{
namespace
{
// Is a detection of this camera from an inadmissible frame inside the place's association gate?
bool inadmissible_near(std::span<const FrameDetection> dets, int sensor, const Vec2f& xy, float var, const rc::DoorConfig& cfg)
{
    const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    for (const auto& d : dets)
        if ((d.sensor != 0) == (sensor != 0) and not d.admissible
            and (d.xy - xy).squaredNorm() / std::max(1e-9f, var + (d.xy_var >= 0.0f ? d.xy_var : rn2)) <= cfg.tracker_gate_mahalanobis)
            return true;
    return false;
}

float nan() { return std::numeric_limits<float>::quiet_NaN(); }

Vec2f xy_of(const Mat4d& T) { return {static_cast<float>(T(0, 3)), static_cast<float>(T(1, 3))}; }

// The door's own scale (the view kernel's width): its half-diagonal (spec §3.5, a flagged kernel width).
float half_diag(const rc::DoorFitState& t) { return 0.5f * std::hypot(t.aperture.w, t.aperture.h); }

// One camera's Bernoulli channel for one frame. Returns the view index (npos when the camera has no pixel of θ).
struct CamResult { std::size_t v = ViewLedger::npos; float q = 0.f, hit = 0.f, kappa = 0.f; ViewGeometry g; };

CamResult observe_camera(PlaceBelief& b, const rc::DoorFitState& th, const ExistenceFrame& fr, const CameraSpec& spec,
                         const Mat4d& room_T_cam, std::span<const Vec2f> windows, bool windows_known, Channel ch,
                         const DetectorModel& m, int sensor, int visit, int self_sensor, float self_w)
{
    CamResult r;
    const Vec2f door_xy = th.aperture.centre_xy();
    const float R = half_diag(th);
    const PoseOverlap po = pose_overlap_factor(fr.rt_cov, room_T_cam, door_xy, R);
    r.kappa = po.kappa;
    const float pos_var = th.ai2_initialized ? std::max(0.0f, th.ai2_belief.covariance()(0, 0)) : 0.0f;
    // ★A frame the fit may not use may not judge the place either (fix round 1, review I-1): a detection of this camera that lands on
    //  the place from a truncated / motion-smeared frame makes this camera's frame NO TRIAL — neither a hit nor a miss.
    if (inadmissible_near(fr.dets, sensor, door_xy, pos_var + th.chain_cov_xx, fr.cfg))
        return r;
    r.hit = sensor == self_sensor ? po.kappa * self_w
                                  : association_weight(fr.dets, sensor, door_xy, pos_var + th.chain_cov_xx, po.kappa, fr.cfg);
    // ★Ruling R4: a 360 frame whose processed strips are unknown looked at NOTHING (no miss), but a detection of THIS door
    //  proves the look at its bearing: a window spanning the door's own angular extent from the camera.
    std::array<Vec2f, 1> hit_window{};
    if (not windows_known and r.hit > 0.0f and spec.model != Projection::Pinhole)
    {
        const Vec2f d = door_xy - xy_of(room_T_cam);
        const float az = std::atan2(d.y(), d.x()), hw = std::atan2(R, std::max(1e-3f, d.norm()));
        hit_window[0] = {az - hw, az + hw};
        windows = hit_window;
        windows_known = true;
    }
    const CameraView view{spec, room_T_cam, windows, windows_known};
    r.g = door_view(view, fr.scene, th);
    r.q = detect_prob(m, r.g, po.kappa);
    if (not (r.q > 0.0f) and not (r.hit > 0.0f))
        return r;   // nothing reportable and nothing reported: no trial, no view (unseen is not evidence)
    r.v = b.ledger.observe(xy_of(room_T_cam), door_xy, th.aperture.across_u(), R, po.spread_m, visit);
    b.ledger.add_bernoulli(r.v, ch, r.q, m, r.hit);
    return r;
}
}  // namespace

rc::edges::ContourSet leaf_contour(const ExistenceScene& sc, const rc::DoorFitState& inst)
{
    if (not inst.ai2_initialized or not camera_usable(sc.cam) or not sc.room_T_zed.has_value())
        return {};
    const Eigen::Matrix4d zed_T_room = sc.room_T_zed->inverse();
    // (compute_silhouette_existence's contour block, unchanged: the four leaf-face corners projected with the cycle's pose,
    //  the controls slid ±1/±1.6 widths along the leaf's OWN plane in 3-D, all-or-nothing when a corner is behind the camera.)
    const auto project = [&](const Eigen::Vector3d& Pr) -> std::optional<rc::edges::ProjectedVertex>
    {
        const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
        if (Pc.y() <= 0.20) return std::nullopt;
        const Eigen::Vector2d uv = camera_project(sc.cam, Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
        if (not std::isfinite(uv.x()) or not std::isfinite(uv.y())) return std::nullopt;
        rc::edges::ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
        v.depth_m = static_cast<float>(Pc.y());
        return v;
    };
    const float hw = inst.leaf_pose.half_w, half_h = inst.leaf_pose.half_h, cz = inst.leaf_pose.centre_z;
    const auto corner = [&](float lx, float lz) -> Eigen::Vector3d
    { return door::leaf_point(inst.leaf_pose, lx, 0.0f, lz).cast<double>(); };
    const std::array<Eigen::Vector3d, 4> quad{corner(-hw, cz - half_h), corner(hw, cz - half_h),
                                              corner(hw, cz + half_h), corner(-hw, cz + half_h)};
    const Eigen::Vector3d along = corner(1.0f, cz) - corner(0.0f, cz);
    return rc::edges::project_quad(quad, along, 2.0f * hw, project);
}

float association_weight(std::span<const FrameDetection> dets, int sensor, const Vec2f& xy, float var, float kappa,
                         const rc::DoorConfig& cfg)
{
    const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    float best = 0.0f;
    for (const auto& d : dets)
    {
        if ((d.sensor != 0) != (sensor != 0) or not d.admissible) continue;
        const float S = std::max(1e-9f, var + (d.xy_var >= 0.0f ? d.xy_var : rn2));
        const float m2 = (d.xy - xy).squaredNorm() / S;
        if (m2 <= cfg.tracker_gate_mahalanobis)   // the tracker's own association gate (squared, 3σ)
            best = std::max(best, d.w * std::exp(-0.5f * m2));
    }
    return kappa * best;
}

ExistenceTrace observe_place(PlaceBelief& b, const rc::DoorFitState& th, const ExistenceFrame& fr, int visit,
                             float actuation_llr, std::vector<ViewRow>* rows, int self_sensor, float self_w)
{
    ExistenceTrace t;
    std::vector<std::size_t> touched;
    const float sum0 = b.ledger.sum();
    float zed_view_kappa = 0.0f, zed_vis = 0.0f;
    std::size_t zed_v = ViewLedger::npos;
    const auto row = [&](int cam, const CamResult& r)
    {
        if (rows == nullptr or r.v == ViewLedger::npos) return;
        rows->push_back({cam, static_cast<int>(r.v), r.q, r.g.fill_h, r.g.fill_v, r.g.vis, r.kappa, r.g.aspect, r.hit, true});
    };
    // ── the ZED ──
    if (fr.zed_ran and fr.zed_spec.has_value() and fr.scene.room_T_zed.has_value())
    {
        const float before = b.ledger.sum();
        const CamResult r = observe_camera(b, th, fr, *fr.zed_spec, *fr.scene.room_T_zed, {}, false, Channel::Zed, fr.det.zed,
                                           0, visit, self_sensor, self_w);
        t.zed_q = r.q; t.zed_hit = r.hit; t.kappa = r.kappa;
        zed_view_kappa = r.kappa; zed_vis = r.g.vis; zed_v = r.v;
        if (r.v != ViewLedger::npos)
        {
            b.ledger.recompute(r.v, fr.geometry_may_refute);
            touched.push_back(r.v);
            t.zed_view = static_cast<int>(r.v);
            row(0, r);
        }
        t.zed_dl = b.ledger.sum() - before;
    }
    // ── the ricoh ──
    if (fr.ricoh_ran and fr.ricoh_spec.has_value() and fr.room_T_ricoh.has_value())
    {
        const float before = b.ledger.sum();
        const auto& w = fr.scene.pkt.ricoh_windows;
        const CamResult r = observe_camera(b, th, fr, *fr.ricoh_spec, *fr.room_T_ricoh, w, not w.empty(), Channel::Ricoh,
                                           fr.det.ricoh, 1, visit, self_sensor, self_w);
        t.ric_q = r.q; t.ric_hit = r.hit;
        if (r.v != ViewLedger::npos)
        {
            b.ledger.recompute(r.v, fr.geometry_may_refute);
            touched.push_back(r.v);
            t.ric_view = static_cast<int>(r.v);
            row(1, r);
        }
        t.ric_dl = b.ledger.sum() - before;
    }
    else if (fr.ricoh_hits_only and fr.room_T_ricoh.has_value())
    {
        const float before = b.ledger.sum();
        const Vec2f door_xy = th.aperture.centre_xy();
        const PoseOverlap po = pose_overlap_factor(fr.rt_cov, fr.room_T_ricoh, door_xy, half_diag(th));
        const float pos_var = th.ai2_initialized ? std::max(0.0f, th.ai2_belief.covariance()(0, 0)) : 0.0f;
        const float hit = inadmissible_near(fr.dets, 1, door_xy, pos_var + th.chain_cov_xx, fr.cfg) ? 0.0f
                        : self_sensor == 1 ? po.kappa * self_w
                                           : association_weight(fr.dets, 1, door_xy, pos_var + th.chain_cov_xx, po.kappa, fr.cfg);
        t.ric_hit = hit;
        if (hit > 0.0f)
        {
            // q unknown but positive: its limit q → 0⁺ (the smallest positive float) keeps the hit's ratio and makes the miss part 0.
            const std::size_t v = b.ledger.observe(xy_of(*fr.room_T_ricoh), door_xy, th.aperture.across_u(), half_diag(th), po.spread_m, visit);
            b.ledger.add_bernoulli(v, Channel::Ricoh, std::numeric_limits<float>::min(), fr.det.ricoh, hit);
            b.ledger.recompute(v, fr.geometry_may_refute);
            touched.push_back(v);
            t.ric_view = static_cast<int>(v);
            if (rows) rows->push_back({1, static_cast<int>(v), 0.f, 0.f, 0.f, 0.f, po.kappa, 0.f, hit, false});
        }
        t.ric_dl = b.ledger.sum() - before;
    }
    // ── geometry: the leaf contour against the same frame's controls (an RGB frame, mask or not) ──
    if (fr.rgb != nullptr and fr.rgb->valid() and fr.scene.room_T_zed.has_value())
    {
        const auto contour = leaf_contour(fr.scene, th);
        if (contour.face.valid() and contour.controls.size() >= 2)
        {
            if (fr.note_poly)
            {
                fr.note_poly(contour.face.px);
                for (const auto& c : contour.controls) fr.note_poly(c.px);
            }
            const auto s_true = rc::edges::contour_edge_support(*fr.rgb, contour.face.px, {});
            std::vector<double> sk;
            for (const auto& c : contour.controls)
                if (const auto sc = rc::edges::contour_edge_support(*fr.rgb, c.px, {}); sc.n_samples > 0) sk.push_back(sc.s_true);
            const float ref = fr.rgb->frame_ref;
            if (s_true.n_samples > 0 and sk.size() >= 2 and ref > 0.0f)
            {
                double mean = 0.0, var = 0.0;
                for (const double v : sk) mean += v;
                mean /= static_cast<double>(sk.size());
                for (const double v : sk) var += (v - mean) * (v - mean);
                var /= static_cast<double>(sk.size() - 1);
                const float x = s_true.s_true / ref, mu0 = static_cast<float>(mean) / ref, s2 = static_cast<float>(var) / (ref * ref);
                t.geo_x = x; t.geo_mu0 = mu0; t.geo_s = std::sqrt(s2); t.geo_mu1 = mu0 + fr.geo_mu1_sd * t.geo_s;
                // The ZED's view of θ (the camera the contour is drawn in); a geometry-only cycle still opens/joins it.
                std::size_t v = zed_v;
                float kappa = zed_view_kappa, vis = zed_vis;
                if (v == ViewLedger::npos)
                {
                    const PoseOverlap po = pose_overlap_factor(fr.rt_cov, fr.scene.room_T_zed, th.aperture.centre_xy(), half_diag(th));
                    kappa = po.kappa;
                    if (fr.zed_spec.has_value())
                        vis = door_view({*fr.zed_spec, *fr.scene.room_T_zed, {}, false}, fr.scene, th).vis;
                    if (kappa * vis > 0.0f)
                        v = b.ledger.observe(xy_of(*fr.scene.room_T_zed), th.aperture.centre_xy(), th.aperture.across_u(),
                                             half_diag(th), po.spread_m, visit);
                }
                const float before = b.ledger.sum();
                if (v != ViewLedger::npos and s2 > 0.0f)
                {
                    b.ledger.add_geometry(v, x, vis * kappa, mu0, t.geo_mu1, s2);
                    b.ledger.recompute(v, fr.geometry_may_refute);
                    touched.push_back(v);
                }
                t.geo_dl = b.ledger.sum() - before;
            }
        }
    }
    // ── actuation: once per satisfied command, into the view current now ──
    if (actuation_llr != 0.0f)
    {
        std::size_t v = touched.empty() ? ViewLedger::npos : touched.front();
        if (v == ViewLedger::npos and fr.scene.room_T_zed.has_value())
            v = b.ledger.observe(xy_of(*fr.scene.room_T_zed), th.aperture.centre_xy(), th.aperture.across_u(), half_diag(th),
                                 0.0f, visit);
        if (v != ViewLedger::npos)   // no view and no ZED pose: the evidence stays pending (act_dl NaN tells the caller)
        {
            const float before = b.ledger.sum();
            b.ledger.add_actuation(v, actuation_llr);
            b.ledger.recompute(v, fr.geometry_may_refute);
            t.act_dl = b.ledger.sum() - before;
        }
    }
    (void) sum0;
    return t;
}

PlacePrior prior_of(const rc::DoorFitState& th, const std::vector<Vec2f>& polygon, bool proto, const rc::door::RoomBand& band,
                    int doors_on_wall_before, float width_sd, float height_sd, bool size_resolved)
{
    const PlacePriorInput in{polygon, proto, band, th.aperture.centre_xy(), th.aperture.wall_u, th.aperture.w, th.aperture.h,
                             width_sd, height_sd, size_resolved, doors_on_wall_before};
    return place_prior(in);
}

int wall_of(const Vec2f& xy, const std::vector<Vec2f>& polygon, const std::vector<int>& groups)
{
    const std::size_t n = polygon.size();
    if (n < 3 or groups.size() != n)
        return -1;
    float best = std::numeric_limits<float>::max();
    int g = -1;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Vec2f a = polygon[i], ab = polygon[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        const float u = len2 > 1e-12f ? std::clamp((xy - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        if (const float d = (xy - (a + u * ab)).squaredNorm(); d < best) { best = d; g = groups[i]; }
    }
    return g;
}

rc::DoorFitState typical_door_at(const Vec2f& xy, float width, const std::vector<Vec2f>& polygon)
{
    // The wall the blob is on: the nearest polygon edge's direction (no polygon: +x, an unknown wall).
    Vec2f u(1.0f, 0.0f);
    const std::size_t n = polygon.size();
    float best = std::numeric_limits<float>::max();
    for (std::size_t i = 0; n >= 3 and i < n; ++i)
    {
        const Vec2f a = polygon[i], ab = polygon[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-12f) continue;
        const float t = std::clamp((xy - a).dot(ab) / len2, 0.0f, 1.0f);
        if (const float d = (xy - (a + t * ab)).squaredNorm(); d < best) { best = d; u = ab / std::sqrt(len2); }
    }
    rc::DoorFitState th;
    th.node_id = 0;
    th.node_name = "candidate";
    rc::DoorState ms;
    ms.cx = xy.x(); ms.cy = xy.y(); ms.cz = 0.0f;
    ms.yaw = std::atan2(u.y(), u.x());
    ms.w = width; ms.h = 2.05f; ms.thickness = 0.05f;   // 2.05 m: the size prior's typical height (§6, flagged there)
    ms.phi = 0.0f;
    const rc::PreBeliefGeometry g = rc::pre_belief_geometry(ms);
    th.leaf_pose = g.leaf_pose; th.leaf = g.leaf; th.aperture = g.aperture;
    rc::read_back_leaf(ms, th.leaf_pose, th.leaf);
    th.model.set_state(ms);
    th.ai2_initialized = true;   // a resolved hypothesis: θ is fully specified, so door_view may render it
    return th;
}
}  // namespace rc::door::core
