/*
 * door_fitter.cpp — door_concept's DSR-bound side of the fit: the instance map, ensure_instance's graph read, and the
 * existence probes that still query cortex (silhouette, aperture frame, obliquity). The FIT itself — observe /
 * run_inference / estimate_phi / the pre-belief geometry and read-back / the ROI / the chain covariance — moved into the
 * DSR-free DoorCore (src/core/door_fit.{h,cpp}, Task 4b of the door-core replay plan).
 */

#include "door_fitter.h"
#include "core/door_core.h"   // point_in_polygon — the containment rule, shared with the birth test
#include "door_standpoint.h"
#include "door_phi_posterior.h"   // the phi channel as pure functions (tested offline)
#include "door_opening_edges.h"   // try96: wall registration, the opening's through-span, the registration offset

#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits>
#include <cstdint>
#include <print>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <array>
#include <opencv2/imgproc.hpp>   // fillConvexPoly / bitwise_and — the rasterised phi score
#include "../../common/rt_query_probe/rt_query_probe.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room_frame — the room is room_1/room_2/…, never the literal "room"

namespace
{
// Hashed pixel-cell coverage of the current YOLO foreground, split door (occupancy) vs every other label
// (occluder). A CELL-px cell absorbs mask-boundary jitter and makes membership O(1); the key packs the two
// cell indices. CELL is a discretization constant (quadrature), not a belief gate — no config key.
// Used by BOTH the leaf silhouette and the aperture-frame probe, so "what hides a door" is one definition.
struct MaskCells
{
    static constexpr float CELL = 6.0f;
    std::unordered_set<std::int64_t> door, occluder;
    [[nodiscard]] static std::int64_t key(float col, float row)
    {
        const std::int64_t kx = static_cast<std::int64_t>(std::floor(col / CELL));
        const std::int64_t ky = static_cast<std::int64_t>(std::floor(row / CELL));
        return (kx << 32) ^ (ky & 0xffffffffLL);
    }
    // A pixel covered by some NON-door mask and by no door mask: a nearer masked object hides it.
    [[nodiscard]] bool hides(float col, float row) const
    {
        const std::int64_t k = key(col, row);
        return occluder.contains(k) and not door.contains(k);
    }
};

template <class Packet>
MaskCells build_mask_cells(const Packet& pkt)
{
    MaskCells c;
    if (not pkt.valid)
        return c;
    for (const auto& sl : pkt.slices)
    {
        const std::size_t b = std::min(sl.pixel_begin, pkt.mask_pixels.size());
        const std::size_t e = std::min(sl.pixel_end,   pkt.mask_pixels.size());
        auto& dst = (sl.label == "door") ? c.door : c.occluder;
        for (std::size_t i = b; i < e; ++i)
            dst.insert(MaskCells::key(pkt.mask_pixels[i].x(), pkt.mask_pixels[i].y()));
    }
    return c;
}
}   // namespace

namespace rc {

DoorFitter::DoorFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         DoorConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         DoorSceneGraph* scene_graph,
                         rc::door::core::DoorFit& fit)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph), fit_(fit)
{}

// ─── Instance lifecycle ──────────────────────────────────────────────────────

bool DoorFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    DoorState init_state;
    init_state.cx        = 0.0f;
    init_state.cy        = 0.0f;
    init_state.yaw       = 0.0f;
    init_state.w         = cfg_.door_prior_w_m;
    init_state.h         = cfg_.door_prior_h_m;
    init_state.thickness = cfg_.door_thickness_m;

    // Standard DSR geometry attrs → panel dims: width_m→w, depth_m→thickness, height_m→h.
    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.w         = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.thickness = v.value();
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.h         = v.value();

    // Read RT pose from room→door edge
    if (room_node_id_ != 0)
    {
        if (const auto edge = G_->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G_->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 2) { init_state.cx = tvec[0]; init_state.cy = tvec[1]; }
            }
            if (const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value()); rot.has_value())
            {
                const auto& rvec = rot.value().get();
                if (rvec.size() >= 3) init_state.yaw = rvec[2];
            }
        }
    }

    // The DSR-free rest — the birth seed (noted by the DoorCore at birth) overrides the centre, non-finite fields are
    // reset, the model is built and the cached geometry seeded from the box — is the core's (DoorFit::init_instance).
    DoorInstance inst;
    fit_.init_instance(inst, node.id(), node.name(), init_state);
    inst.affordance.init(G_, node.id(), node.name(), "door");

    instances_.emplace(node.id(), std::move(inst));
    std::print("door_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}


// ─── RGB-mask ROI projection (active-perception aid) ──────────────────────────

std::optional<Eigen::Matrix4d> DoorFitter::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin the moving room→body hop to the frame's capture stamp (Nearest); keep the rigid body→zed mount
    // at latest (it carries only a bootstrap stamp — a pinned query would fail). ts=0 → current pose.
    // ★LISTENING ONLY — no behaviour change. Asks cortex what this query DID (see
    // common/rt_query_probe/rt_query_probe.h). A timestamped query that falls outside the ring
    // returns the end block and is indistinguishable from a success, so a fitter cannot currently
    // tell whether it is placing detections at the pose the camera actually had. One throttled line
    // per 15 s appends a row to etc/rt_query_probe.csv saying whether this call site ever clamps,
    // by how much, and — the column that makes it readable — how fast the robot was moving in the
    // same window. A clamp only costs geometry while the robot moves; the controller's rate goes
    // 1-3% parked to 35% moving, so a clamp share without a motion column cannot be interpreted.
    static rc::rtprobe::Probe rt_probe{"door_fitter room<-body"};
    DSR::RT_API::TimeQueryInfo rt_info;
    const auto rtb = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G_), "body", pose_ts_ms, "RT",
                                                             DSR::RT_API::TimeQuery::Interpolated,
                                                             &rt_info);
    rt_probe.note(rt_info, G_);
    const auto btz = inner_eigen_->get_transformation_matrix("body", "zed", 0);
    if (not (rtb.has_value() and btz.has_value()))
        return std::nullopt;
    const auto to_mat4 = [](const Mat::RTMat& T)
    {
        Eigen::Matrix4d m;
        const auto& s = T.matrix();
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = s(i, j);   // no aligned load
        return m;
    };
    return to_mat4(rtb.value()) * to_mat4(btz.value());
}

// ─── Silhouette existence (pixel-level) ─────────────────────────────────────────────────────────

// PIXEL-LEVEL silhouette existence evidence — see DoorSilhouette / DoorInstance::existence.
//
// Projects the door PANEL FACE (a w×h rectangle in the wall plane) into the ZED and, over the predicted
// samples, splits into: lit by a "door" mask (occupancy), lit by a non-door mask (occlusion ⇒ excluded from
// the detectable footprint), or lit by nothing (ABSENCE — the "gone" signal that fires even on a frame where
// YOLO produced no door mask at all). A sample that does not land inside the real image is NOT detectable, so
// n_detectable==0 means "the door was never looked at this frame" and rc::exist HOLDs.
//
// This is what makes turning the robot around harmless: the old scheme asked a bearing-free range term how
// detectable the door was and got ≈0.4 for a door squarely BEHIND the camera, then charged absence evidence
// against it every frame until it died. Here the frustum test is the detectability.
DoorSilhouette DoorFitter::compute_silhouette_existence(const DoorInstance& inst,
                                                        const rc::SemanticProbField* field)
{
    DoorSilhouette out;
    if (not inner_eigen_ or not inst.ai2_initialized)
        return out;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return out;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return out;
    }
    const auto Mopt = room_T_zed_matrix();   // camera→room, current pose
    if (not Mopt.has_value())
        return out;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();

    const float W    = static_cast<float>(camera_api_->get_width());
    const float Himg = static_cast<float>(camera_api_->get_height());
    if (W <= 0.f or Himg <= 0.f)
        return out;

    if (not mask_ingestor_)
        return out;
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid or pkt.mask_pixels.empty())
        return out;

    // Hashed pixel-cell coverage of the current YOLO foreground, split door (occupancy) vs other (occluder).
    // Shared with compute_frame_probe — see MaskCells.
    const MaskCells cells = build_mask_cells(pkt);
    const auto key = [&cells](float col, float row) { return cells.key(col, row); };
    const auto& door_cells     = cells.door;
    const auto& occluder_cells = cells.occluder;

    const auto& s = inst.model.state();
    const float hw = inst.leaf_pose.half_w;

    // Camera position (room frame) for the GEOMETRIC occlusion tests below.
    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)),
                            static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const Eigen::Vector2f Oxy(O.x(), O.y());
    const float occ_margin = std::max(0.0f, cfg_.exist_occlusion_margin_m);

    // ── THE LEAF BEYOND THE APERTURE (door_room_band.h) ─────────────────────────────────────────────────
    // A leaf swung OUT of the room the camera is in lies in the NEXT room. What reaches the camera of it is
    // a sliver seen edge-on THROUGH the hole, and the P(detect) below (envelope × obliquity × resolvability
    // × in-view × central) is a model of a leaf in THIS room — it does not represent that view. Live try34:
    // the crossed door, carried into room_2 with its leaf at ~95-110° out of room_2, scored free=112-122 of
    // 120 detectable samples (the part within walls_block's 0.30 m own-wall skip; the rest was already
    // wall-occluded) on a door that plainly exists. The model has no term for the opening's jambs/casing as
    // occluders of what lies past the room-side face, so each sample carries weight 1 − P(it lies past that
    // face), measured from the APERTURE LINE: a sample on it (a shut leaf, the hinge edge) weighs exactly 1,
    // one a few σ out weighs ~0. A weight, not a skip: occupancy and absence both scale by it, and
    // so does the visible fraction (det_w), so a door whose whole leaf is beyond is simply not probed by
    // this channel — the frame channel (compute_frame_probe) is what speaks for it then.
    const auto band = room_band();
    const float delta_ap = has_room_polygon()
                         ? rc::door::signed_boundary_distance(room_polygon(), inst.aperture.centre_xy())
                         : std::numeric_limits<float>::quiet_NaN();
    const bool  band_known = std::isfinite(delta_ap);
    const Eigen::Vector2f n_out = band_known
        ? rc::door::outward_normal(room_polygon(), inst.aperture.centre_xy(), inst.aperture.across_u(), band)
        : Eigen::Vector2f::Zero();
    const float r0 = rc::door::face_offset_from_aperture(delta_ap);
    const auto room_weight = [&](const Eigen::Vector3f& P) -> float
    {
        if (not band_known) return 1.0f;
        const float a = (Eigen::Vector2f(P.x(), P.y()) - inst.aperture.wall_O).dot(n_out);
        return 1.0f - rc::door::p_beyond_far_face(a, r0, band.sigma_m);
    };

    // Classify ONE LEAF-face sample (local x hinge→free edge, z absolute): project it, then vote.
    //
    // ★ The sample comes from door::leaf_point at the instance's CURRENT leaf pose. It used to be spelled
    // out here as a rectangle in the WALL PLANE, which meant that the moment a door actually opened, its
    // predicted silhouette landed where the leaf no longer was: every sample came back unlit, e_free
    // spiked, and the existence channel deleted the door BECAUSE it opened. With phi pinned at 0 (M0) this
    // yields the identical samples; when phi becomes a fitted DOF in M1 it follows the leaf for free.
    std::unordered_set<std::int64_t> covered_cells;   // distinct cells the DETECTABLE silhouette occupies
    double range_sum = 0.0;
    // ★PARAMETERISED BY THE LEAF POSE, AND BY WHERE IT ACCUMULATES. One pass renders the panel at ONE
    // opening angle; the caller runs it once per candidate phi and combines. `extras` marks the pass
    // whose by-products describe the scene for everybody else (centroid, semantic field, covered cells,
    // range) — those are reported at the modal pose, because they answer "where is it on screen", which
    // is a statement about one drawing and not something to average over hypotheses.
    const auto classify = [&](const door::LeafPose& L, DoorSilhouette& out, bool extras,
                              float lx, float lz)
    {
        ++out.n_total;                                                  // one sample of the WHOLE panel
        const Eigen::Vector3f Ps = door::leaf_point(L, lx, 0.0f, lz);   // face at mid-thickness
        const Eigen::Vector4d Pr(Ps.x(), Ps.y(), Ps.z(), 1.0);
        const Eigen::Vector4d Pc = zed_T_room * Pr;
        const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
        if (Y <= 0.20) return;                                          // behind / at the near clip
        const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(X, Y, Z));
        const float col = static_cast<float>(uv.x()), row = static_cast<float>(uv.y());
        if (col < 0.f or col >= W or row < 0.f or row >= Himg) return;  // out of the REAL frustum ⇒ not detectable
        const std::int64_t k = key(col, row);
        if (occluder_cells.contains(k) and not door_cells.contains(k))
        { ++out.n_occluded; return; }                                   // a nearer MASKED object hides it ⇒ no vote
        // GEOMETRIC occluders, which carry no YOLO mask and so are invisible to the cell test above:
        // (a) a room WALL crossing this sightline (the robot is around a corner / the door is in another
        //     room's wall) — the structural case, and the one that would otherwise read as clean absence;
        // (b) another door instance's panel standing in front of this sample.
        // Per SAMPLE, not per instance: a partially-hidden door keeps the visible part of its footprint and
        // votes with it, instead of the old all-or-nothing `continue` that froze a phantom indefinitely.
        const Eigen::Vector3f Pw(static_cast<float>(Pr.x()), static_cast<float>(Pr.y()), static_cast<float>(Pr.z()));
        if (rc::occlusion::walls_block(Oxy, {Pw.x(), Pw.y()}, room_polygon(), /*own_wall_skip_m=*/0.30f))
        { ++out.n_occluded; return; }
        {
            Eigen::Vector3f dc = Pw - O;
            const float rc_len = dc.norm();
            if (rc_len > 1e-3f)
            {
                dc /= rc_len;
                for (const auto& [jid, jinst] : instances_)
                {
                    if (jid == inst.node_id or not jinst.ai2_initialized) continue;
                    const auto& js = jinst.model.state();
                    const Eigen::Vector3f Cj(js.cx, js.cy, js.cz + 0.5f * js.h);
                    if (rc::occlusion::cone_blocks(O, dc, rc_len, Cj, 0.5f * std::max(js.w, js.thickness),
                                                   (Cj - O).norm(), occ_margin))
                    { ++out.n_occluded; return; }
                }
            }
        }
        const float w_room = room_weight(Pw);                           // 1 − P(seen through the opening)
        if (not (w_room > 0.0f))
        { ++out.n_occluded; return; }                                   // wholly through the opening: not this view
        ++out.n_detectable;
        out.det_w += w_room;
        if (extras) covered_cells.insert(k);
        const float f = central_region_frac_, g = 1.0f - central_region_frac_;
        // Silhouette centroid over ALL detectable samples — the size-invariant input to
        // central_frac(). Deliberately OUTSIDE the central-box test below: it must describe
        // where the whole visible object sits, not only the part already inside the box.
        out.sum_col += col;
        out.sum_row += row;

        // Sample the graded posterior right here, at a pixel we have just established the camera could
        // actually see. Doing it in this loop rather than from a bounding box is the point: the contour
        // is the door's own projected shape, so the statistic is about the door and not about the
        // rectangle around it.
        if (extras and field != nullptr and field->valid())
            if (const float p = field->at(static_cast<int>(col), static_cast<int>(row)); p >= 0.0f)
            {
                out.field_sum += p;
                out.field_max = std::max(out.field_max, p);
                ++out.field_n;
            }
        out.img_w = static_cast<int>(W);
        out.img_h = static_cast<int>(Himg);
        if (col > f * W and col < g * W and row > f * Himg and row < g * Himg)
            ++out.n_central;                                            // the robot is looking AT it
        if (extras) range_sum += std::sqrt(X * X + Y * Y + Z * Z);
        if (door_cells.contains(k)) out.e_occ  += w_room;               // still there
        else                        out.e_free += w_room;               // predicted-but-absent
    };

    // Regular grid over the panel face. NX/NZ are numeric SAMPLING RESOLUTION (a quadrature density for the
    // occupancy/detectability counts), not decision thresholds — denser is smoother at linear cost. A door is
    // much taller than wide, so the vertical grid is finer.
    constexpr int NX = 14, NZ = 30;
    const auto sweep = [&](const door::LeafPose& L, DoorSilhouette& acc, bool extras)
    {
        for (int ix = 0; ix < NX; ++ix)
            for (int iz = 0; iz < NZ; ++iz)
                classify(L, acc, extras,
                         (-1.0f + 2.0f * (ix + 0.5f) / NX) * hw, s.cz + s.h * (iz + 0.5f) / NZ);
    };

    // The modal pass: the leaf where phi_est puts it. This is the drawing — face_px, the centroid, the
    // semantic-field statistics and the covered cells all come from here, and so does everything the UI
    // and the logs show.
    sweep(inst.leaf_pose, out, /*extras=*/true);
    out.n_cells = static_cast<int>(covered_cells.size());

    // ── ABSENCE IS MARGINALISED OVER phi, NOT CONDITIONED ON ITS ARGMAX ──────────────────────────────
    // ★WHY. The silhouette is the only channel permitted to remove a door, and until now it rendered the
    // leaf at a single fitted angle. Measured 2026-09-10 over one approach, the phi likelihood peaks at
    // 0.007-0.245 — flat — so that angle is chosen from noise; twice on that run the leaf was projected
    // where the panel was not, occ fell 207 -> 0, free_eff rose 0 -> 33 and L ran +4.00 -> -4.00 in 13
    // cycles. The door was not gone; the model had merely guessed its hinge angle wrong. Charging a
    // removal to an unidentified nuisance parameter is the defect, and picking the peak better, or
    // smoothing it, would not fix it — the information is not there to be extracted.
    // ★THE FIX IS THE STANDARD ONE: integrate the nuisance out. e_free under P(phi | mask) instead of
    // e_free at phi-hat. A sample counts as absent to the extent that it is absent at EVERY angle the
    // data still permits, so a flat curve nearly cancels the absence (the panel is SOMEWHERE, we just
    // cannot say where) while a sharply-peaked one charges it in full. There is no flatness test and no
    // angle gate: the sharpness of the curve does the weighting by itself, which is what makes this a
    // model term and not another threshold.
    // ★THE WEIGHTS ARE THE POSTERIOR the angle estimate itself is drawn from — image support times the
    // persistence/command prior — so the absence is marginalised over exactly what this agent believes
    // about the leaf, and the two consumers of phi cannot disagree. An empty curve means the frame
    // carried no door mask to score phi against, and then there is nothing to marginalise: the modal
    // pass stands, which is exactly the old behaviour on exactly the frames where it was never at fault
    // (with no mask anywhere, every angle is equally unlit and the marginal equals the point estimate).
    double w_sum = 0.0;
    for (const auto& [phi, sup] : inst.phi_curve)
        w_sum += std::max(0.0f, sup);
    if (inst.phi_curve.size() > 1 and w_sum > 0.0)
    {
        double m_occ = 0.0, m_free = 0.0, m_total = 0.0, m_det = 0.0;
        for (const auto& [phi, sup] : inst.phi_curve)
        {
            const double w = std::max(0.0f, sup) / w_sum;
            if (w <= 0.0) continue;
            door::LeafState ls = inst.leaf;
            ls.phi = phi;
            DoorSilhouette acc;
            sweep(door::leaf_pose(inst.aperture, ls), acc, /*extras=*/false);
            m_occ     += w * acc.e_occ;
            m_free    += w * acc.e_free;
            m_total   += w * acc.n_total;
            m_det     += w * acc.det_w;
        }
        // Only the ABSENCE/OCCUPANCY evidence is replaced. The modal counts stay as the modal pass left
        // them, because resolvability() and central_frac() are ratios built from that one rendering.
        out.e_occ  = static_cast<float>(m_occ);
        out.e_free = static_cast<float>(m_free);
        out.n_det_marg   = static_cast<float>(m_det);
        out.n_total_marg = static_cast<float>(m_total);
        out.phi_marginalised = true;
        out.phi_n_hyp = static_cast<int>(inst.phi_curve.size());
        out.phi_w_max = 0.0f;
        for (const auto& [phi, sup] : inst.phi_curve)
            out.phi_w_max = std::max(out.phi_w_max, static_cast<float>(std::max(0.0f, sup) / w_sum));
    }

    if (out.n_detectable > 0)
        out.mean_range_m = static_cast<float>(range_sum / out.n_detectable);

    // ★THE LEAF-FACE QUAD, projected with the SAME camera and the SAME transform as every sample above.
    // This is what the RGB and depth contour checks are scored on, so the thing being defended and the
    // thing being measured are one contour. Order is a closed loop (bottom-hinge, bottom-free, top-free,
    // top-hinge) — a polygon whose points are not in loop order produces a bow-tie whose "edges" cross
    // the object and would score whatever happens to lie under the diagonals.
    //
    // ★THE CONSTRUCTION NOW LIVES IN common/contour_edge/contour_edge_project.h, unchanged in behaviour:
    // the same four corners, the same ±1/±1.6 slide along the leaf's OWN plane in 3-D, the same
    // all-or-nothing rejection when a corner falls behind the camera. It moved because the trap it
    // encodes is not door-specific — a control displaced in IMAGE pixels stops being a like-for-like
    // comparison the moment the object leaves the surface its neighbours are made of, and every concept
    // agent adopting this channel would otherwise rediscover that the expensive way (we did: the channel
    // voted to DELETE the door, hardest when it was open). It also returns the per-vertex DEPTH the
    // projection already computed, which is what the depth channel tests the belief against.
    {
        const auto project = [&](const Eigen::Vector3d& Pr) -> std::optional<rc::edges::ProjectedVertex>
        {
            const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
            if (Pc.y() <= 0.20) return std::nullopt;                       // a corner behind the camera
            const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
            if (not std::isfinite(uv.x()) or not std::isfinite(uv.y())) return std::nullopt;
            rc::edges::ProjectedVertex v;
            v.px = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
            // The ZED depth plane stores the camera-frame FORWARD coordinate, not the Euclidean norm
            // (retina deprojects with `py = depth`). Handing back a norm here would read high off-axis
            // and look exactly like a door believed slightly too far away.
            v.depth_m = static_cast<float>(Pc.y());
            return v;
        };
        const float half_h = inst.leaf_pose.half_h;
        const float cz     = inst.leaf_pose.centre_z;
        const auto corner = [&](float lx, float lz) -> Eigen::Vector3d
        { return door::leaf_point(inst.leaf_pose, lx, 0.0f, lz).cast<double>(); };
        const std::array<Eigen::Vector3d, 4> quad{
            corner(-hw, cz - half_h), corner(hw, cz - half_h),
            corner( hw, cz + half_h), corner(-hw, cz + half_h)};
        // Slide direction: the leaf's own +x (hinge → free edge), so a displaced copy stays in the leaf's
        // plane at any phi. Span is the leaf's full width.
        const Eigen::Vector3d along = corner(1.0f, cz) - corner(0.0f, cz);
        out.contour = rc::edges::project_quad(quad, along, 2.0f * hw, project);
    }
    if (field != nullptr and field->valid())
        out.field_bg = field->background_mean();
    return out;
}

// ── THE DOORWAY FRAME: the part of a door that does not move ────────────────────────────────────────────
// Projects the aperture's jambs + lintel and weights each sample by whether the camera could see it. The
// scoring against the image and the reasoning behind the statistic live in door_frame_evidence.h; this
// method owns only what needs the agent: the camera, the transform, the masks, the walls, the other doors
// and the leaf's opening-angle posterior.
//
// Visibility per frame point P (a weight, never a skip-the-cycle switch):
//   · outside the real frustum / behind the near clip                 → 0
//   · a NON-door mask covers its pixel (a person, a chair in front)   → 0   (same cells as the silhouette)
//   · a room WALL crosses the sightline (rc::occlusion::walls_block)  → 0   (own wall skipped, as everywhere)
//   · another door instance stands in front (cone_blocks)             → 0
//   · OUR OWN LEAF crosses the sightline                              → 1 − P_φ(blocked), marginalised over
//     the same phi posterior the silhouette's absence is marginalised over, so the two cannot disagree
//     about where the leaf is. A leaf swung toward the camera hides the jamb behind it; one swung away
//     (out of this room — the try32 case) hides nothing, and the frame speaks in full.
rc::door::FrameProbe DoorFitter::compute_frame_probe(const DoorInstance& inst)
{
    if (not inner_eigen_ or not inst.ai2_initialized)
        return {};
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return {};
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return {};
    }
    const auto Mopt = room_T_zed_matrix();   // camera→room, current pose — the SAME transform the silhouette uses
    if (not Mopt.has_value())
        return {};
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();
    const int W = camera_api_->get_width(), H = camera_api_->get_height();
    if (W <= 0 or H <= 0)
        return {};

    // No mask packet ⇒ no KNOWN masked occluder: the cells are empty and that factor is 1. The frame is a
    // geometric probe of the RGB image, so it does not need masks to exist — only to know what hides it.
    const MaskCells cells = mask_ingestor_ ? build_mask_cells(mask_ingestor_->packet()) : MaskCells{};

    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)), static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const Eigen::Vector2f Oxy(O.x(), O.y());
    const float occ_margin = std::max(0.0f, cfg_.exist_occlusion_margin_m);

    // The leaf hypotheses and their weights: the phi posterior when there is one, else the modal pose.
    std::vector<std::pair<door::LeafPose, float>> leaves;
    {
        double w_sum = 0.0;
        for (const auto& [phi, sup] : inst.phi_curve)
            w_sum += std::max(0.0f, sup);
        if (inst.phi_curve.size() > 1 and w_sum > 0.0)
            for (const auto& [phi, sup] : inst.phi_curve)
            {
                const float w = static_cast<float>(std::max(0.0f, sup) / w_sum);
                if (w <= 0.0f) continue;
                door::LeafState ls = inst.leaf;
                ls.phi = phi;
                leaves.emplace_back(door::leaf_pose(inst.aperture, ls), w);
            }
        else
            leaves.emplace_back(inst.leaf_pose, 1.0f);
    }
    const auto own_leaf_clear = [&](const Eigen::Vector3f& P) -> float
    {
        float blocked = 0.0f;
        for (const auto& [L, w] : leaves)
            if (door::leaf_blocks_sightline(L, O, P))
                blocked += w;
        return std::clamp(1.0f - blocked, 0.0f, 1.0f);
    };

    const auto project = [&](const Eigen::Vector3d& Pr) -> std::optional<rc::edges::ProjectedVertex>
    {
        const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
        if (Pc.y() <= 0.20) return std::nullopt;                       // behind / at the near clip
        const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
        if (not std::isfinite(uv.x()) or not std::isfinite(uv.y())) return std::nullopt;
        rc::edges::ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
        v.depth_m = static_cast<float>(Pc.y());                        // ZED depth = forward coordinate
        return v;
    };

    // The frame is scored where the belief puts the aperture. (Tried 2026-10-13 and NOT adopted: seating it on
    // the room polygon's face. With a real casing standing proud of the wall the visible occluding edge is the
    // casing's front, which is where the fit already puts the aperture — try34's carried door sat 6 cm in front
    // of room_2's face, a 5 cm proud Webots casing — and the synthetic scene in tests/test_door_frame_evidence.cpp
    // scored the seated probe WORSE in 3 of 5 views, down to +0.16 from +28 on an oblique view.)
    const door::Aperture& ap_seen = inst.aperture;

    const auto visible = [&](const Eigen::Vector3f& P) -> float
    {
        const auto v = project(P.cast<double>());
        if (not v.has_value()) return 0.0f;
        if (v->px.x < 0.f or v->px.y < 0.f or v->px.x >= static_cast<float>(W) or v->px.y >= static_cast<float>(H))
            return 0.0f;
        if (cells.hides(v->px.x, v->px.y))
            return 0.0f;
        if (rc::occlusion::walls_block(Oxy, {P.x(), P.y()}, room_polygon(), /*own_wall_skip_m=*/0.30f))
            return 0.0f;
        Eigen::Vector3f dc = P - O;
        const float rc_len = dc.norm();
        if (rc_len > 1e-3f)
        {
            dc /= rc_len;
            for (const auto& [jid, jinst] : instances_)
            {
                if (jid == inst.node_id or not jinst.ai2_initialized) continue;
                const auto& js = jinst.model.state();
                const Eigen::Vector3f Cj(js.cx, js.cy, js.cz + 0.5f * js.h);
                if (rc::occlusion::cone_blocks(O, dc, rc_len, Cj, 0.5f * std::max(js.w, js.thickness),
                                               (Cj - O).norm(), occ_margin))
                    return 0.0f;
            }
        }
        return own_leaf_clear(P);
    };
    return rc::door::build_frame_probe(ap_seen, project, visible, own_leaf_clear, W, H, cfg_.frame_casing_m,
                                       cfg_.exist_boundary_sigma_m);
}

// |cos| of the camera→leaf ray against the LEAF's face normal. 1 = square-on, 0 = grazing. Diagnostic only
// (a log column) — see the note in door_fitter.h. Keyed on the leaf, not the wall, so it stays meaningful
// once phi is fitted: an open door presents its face at a different angle than its aperture does.
float DoorFitter::door_view_obliquity(const DoorInstance& inst) const
{
    if (not inst.ai2_initialized)
        return 1.0f;
    const auto M = room_T_zed_matrix();   // camera→room (current pose)
    if (not M.has_value())
        return 1.0f;                      // no extrinsic → can't judge → don't suppress
    const Eigen::Vector2f cam(static_cast<float>(M->coeff(0, 3)), static_cast<float>(M->coeff(1, 3)));
    Eigen::Vector2f r = inst.leaf_pose.centre_xy - cam;
    if (r.norm() < 1e-6f)
        return 1.0f;
    r.normalize();
    return std::clamp(std::abs(r.dot(inst.leaf_pose.ey)), 0.0f, 1.0f);   // ey = the leaf's face normal
}

float DoorFitter::aperture_band_offset(const DoorInstance& inst) const
{
    if (not has_room_polygon())
        return std::numeric_limits<float>::quiet_NaN();
    const auto& ms = inst.model.state();
    return rc::door::signed_boundary_distance(room_polygon(), Eigen::Vector2f(ms.ap_cx, ms.ap_cy));
}

bool DoorFitter::point_in_room(const Eigen::Vector2f& q, float margin_m) const
{
    // ONE rule for the containment prior: the DoorCore's birth test and this one (core/door_core.cpp).
    return rc::door::core::point_in_polygon(room_polygon(), q, margin_m);
}

}  // namespace rc
