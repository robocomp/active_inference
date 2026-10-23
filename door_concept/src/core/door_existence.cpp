// door_existence.cpp — see door_existence.h. The probes are DoorFitter's (door_fitter.cpp as of 8a1abeac) and
// DoorCore::update_existence is SpecificWorker::update_existence_beliefs (specificworker.cpp as of 8a1abeac), moved
// with their comments; only the sources of the values and the destinations of the outputs changed (door_existence.h).
#include "door_existence.h"
#include "door_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <locale>
#include <numbers>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgproc.hpp>
#include <Eigen/Eigenvalues>

#include "../../../common/instance_tracker/birth_evidence.h"   // rc::birth::observation_weight (SHARED): the ricoh channel's P(detect)
#include "../door_geometry.h"
#include "../door_phi_posterior.h"                                // room_walls_beside, phi_field_nuisance (one own-wall definition)
#include "../../../common/occlusion/occlusion.h"                 // rc::occlusion::{cone_blocks, walls_block} (SHARED)
#include "../../../common/contour_edge/contour_edge_check.h"     // rc::edges::PreparedFrame / contour_edge_support
#include "../../../common/contour_edge/contour_depth_check.h"    // rc::edges::contour_depth_support
#include "../../../common/detectability/detectability.h"         // rc::detect — the YOLO inverse model (SHARED)
#include "../../../common/existence_belief/existence_belief.h"   // rc::exist (SHARED)

namespace rc::door::core
{
namespace
{
using K = rc::door::DoorDecisionKind;

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

// ★Task D-C (rtD1i door_5): LINE OF SIGHT AGAINST EVERY WALL BUT THE DOOR'S OWN. rc::occlusion::walls_block's radius form
// skips any polygon edge closer than 0.30 m to the target as "the target's own wall"; DOOR_0's hinge jamb stands 7 cm from
// the corner where a PARTITION meets its wall, so both partition faces were skipped and from behind it the door's frame
// and leaf read VISIBLE and empty — the image channels walked a shut door from +4 to −4 while it was hidden. The own wall
// is the run of polygon edges the doorway is cut in (door_wall_edges: one definition with the leaf field's background
// walls); the shared walls_block overload takes them explicitly (review M-4).
std::vector<std::size_t> own_wall_edges(const rc::DoorFitState& inst, const std::vector<Eigen::Vector2f>& polygon)
{
    const auto nz = rc::door::phi_field_nuisance(inst.aperture);
    return rc::door::door_wall_edges(inst.aperture, polygon, nz.depth_lo, nz.depth_hi);
}

MaskCells build_mask_cells(const MaskPacket& pkt)
{
    MaskCells c;
    if (not pkt.valid)
        return c;
    for (const auto& sl : pkt.slices)
    {
        const std::size_t b = std::min<std::size_t>(sl.pixel_begin, pkt.mask_pixels.size());
        const std::size_t e = std::min<std::size_t>(sl.pixel_end,   pkt.mask_pixels.size());
        auto& dst = (sl.label == "door") ? c.door : c.occluder;
        for (std::size_t i = b; i < e; ++i)
            dst.insert(MaskCells::key(pkt.mask_pixels[i].x(), pkt.mask_pixels[i].y()));
    }
    return c;
}
}  // namespace

const char* const kExistenceCsvHeader =
    "cycle,node,L,p_exists,cx,cy,inroom,roomprior_loaded,won,since_det,"
    "sil_occ,sil_free,sil_free_eff,n_detectable,n_total,n_occluded,"
    "resolvability,central_frac,p_detect,oblq,remove_streak,"
    "obs_top_z,obs_top_conf,trunc,gated,gate_fresh,"
    // Contour check against retina's graded posterior. ★fld_n==0 means the field
    // was UNAVAILABLE and this cycle behaved exactly as it did before the channel
    // existed — a different fact from "no support found", which is fld_sup<=0.5.
    "fld_sup,fld_in,fld_bg,fld_n,"
    // RGB contour check (classifier-free). edge_n==0 ⇒ NOT MEASURED.
    "edge_sup,edge_true,edge_ctrl,edge_n,edge_dL,edge_exc,edge_nctl,"
    // DEPTH half of the same channel — the one a poster of a door cannot pass.
    // dep_n==0 ⇒ NOT MEASURED (no depth plane, or no valid reads along the
    // contour), which is NOT the same as dep_v==0 (measured, and it says nothing:
    // a surface at the predicted distance with nothing behind it, i.e. flat wall).
    // dep_bias is SIGNED (observed − predicted, m): a channel scoring low with a
    // large bias is a FIT error, not an absence, and only this column separates them.
    "dep_v,dep_bias,dep_n,"
    // M1: the leaf angle is estimated now, not pinned. phi_support is how much of
    // the leaf face the image actually backs at that angle; phi_cmd=1 means a swing
    // WE commanded is being anticipated, which is legitimately evidence-free.
    // phi_marg=1 ⇒ the occ/free split above is the MARGINAL over the opening
    // angle, not a single-angle rendering. phi_wmax is the largest normalised
    // weight: ~1/phi_nhyp means the frame could not identify the angle at all, so
    // the absence was averaged across hypotheses and is correctly near zero.
    "phi,phi_support,phi_cmd,phi_marg,phi_nhyp,phi_wmax,"
    // The doorway-FRAME channel. frame_n==0 ⇒ NOT MEASURED; frame_vis is the
    // fraction of the jambs+lintel the camera could resolve (its P(detect)).
    "frame_n,frame_vis,frame_exc,frame_true,frame_ctrl,frame_dep,frame_ndep,"
    // The WALL-BAND prior (door_room_band.h): signed aperture offset from the polygon
    // (> 0 outside; [0, WallThicknessM] is the normal place for a door) and π = e^term.
    "band_delta,band_p,"
    // The detector envelope's input (door_fit.cpp compute_projected_roi): the leaf box's
    // extent INSIDE the image per axis (≤ 1), and p_env, the envelope factor of p_detect.
    "roi_fill,roi_fill_h,roi_fill_v,roi_valid,p_env\n";

// ─── SemanticField: rc::SemanticProbField over the recorded plane ───────────────────────────────────────────────────

void SemanticField::apply(SemanticState state, const std::optional<SemanticPlane>& plane)
{
    if (state == SemanticState::Unchanged)
        return;                                   // the field the worker holds is the last refresh's
    valid_ = false;
    if (state != SemanticState::Fresh or not plane.has_value())
        return;
    const auto& p = plane.value();
    if (p.pw <= 0 or p.ph <= 0 or p.frame_w <= 0 or p.frame_h <= 0
        or p.plane.size() < static_cast<std::size_t>(p.pw) * static_cast<std::size_t>(p.ph))
        return;
    plane_ = p.plane;
    // Background = the whole frame's mean door-ness. NaN cells are "not looked at" (a 360 strip the
    // scheduler skipped) and are excluded rather than counted as zero (SemanticProbField::refresh, unchanged).
    double sum = 0.0;
    int    n   = 0;
    for (const float v : plane_)
        if (std::isfinite(v)) { sum += v; ++n; }
    if (n == 0)
        return;
    bg_mean_ = static_cast<float>(sum / n);
    pw_ = p.pw; ph_ = p.ph; frame_w_ = p.frame_w; frame_h_ = p.frame_h;
    valid_ = true;
}

float SemanticField::at(int u, int v) const noexcept
{
    if (not valid_ or u < 0 or v < 0 or u >= frame_w_ or v >= frame_h_)
        return -1.0f;
    // Frame pixel → native cell, cell centres, nearest (SemanticProbField::at, unchanged).
    const int cx = std::clamp(static_cast<int>((static_cast<float>(u) + 0.5f) * pw_
                                               / static_cast<float>(frame_w_)), 0, pw_ - 1);
    const int cy = std::clamp(static_cast<int>((static_cast<float>(v) + 0.5f) * ph_
                                               / static_cast<float>(frame_h_)), 0, ph_ - 1);
    return plane_[static_cast<std::size_t>(cy) * static_cast<std::size_t>(pw_) + static_cast<std::size_t>(cx)];
}

// ─── ImageSource ─────────────────────────────────────────────────────────────────────────────────────────────────────

void ImageSource::apply(const std::optional<ImageFrame>& rgb_gray, const std::optional<ImageFrame>& depth)
{
    if (rgb_gray.has_value() and rgb_gray->format == PixelFormat::Gray8 and rgb_gray->width > 0 and rgb_gray->height > 0)
    {
        cv::Mat canvas(rgb_gray->height, rgb_gray->width, CV_8UC1, cv::Scalar(0));
        for (const auto& p : rgb_gray->patches)
        {
            if (p.w <= 0 or p.h <= 0 or p.bytes.size() < static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h)) continue;
            const cv::Rect r = cv::Rect(p.x0, p.y0, p.w, p.h) & cv::Rect(0, 0, canvas.cols, canvas.rows);
            if (r != cv::Rect(p.x0, p.y0, p.w, p.h)) continue;
            cv::Mat src(p.h, p.w, CV_8UC1, const_cast<std::uint8_t*>(p.bytes.data()));
            src.copyTo(canvas(r));
        }
        rgb_ = canvas;
        frame_ref_ = rgb_gray->frame_ref;
    }
    if (depth.has_value() and depth->format == PixelFormat::DepthF32 and depth->width > 0 and depth->height > 0)
    {
        cv::Mat canvas(depth->height, depth->width, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        for (const auto& p : depth->patches)
        {
            if (p.w <= 0 or p.h <= 0
                or p.bytes.size() < static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h) * sizeof(float)) continue;
            const cv::Rect r = cv::Rect(p.x0, p.y0, p.w, p.h) & cv::Rect(0, 0, canvas.cols, canvas.rows);
            if (r != cv::Rect(p.x0, p.y0, p.w, p.h)) continue;
            cv::Mat src(p.h, p.w, CV_32FC1);
            std::memcpy(src.data, p.bytes.data(), static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h) * sizeof(float));
            src.copyTo(canvas(r));
        }
        depth_ = canvas;
    }
    // (DepthZ16 is never written: the recorder stores the ingestor's DECODED frame, DepthF32.)
}

void ImageSource::stage_live(const cv::Mat& bgr, const cv::Mat& depth_m)
{
    rgb_ = bgr;
    depth_ = depth_m;
    frame_ref_.reset();
}

// ─── the pose-uncertainty overlap (Task D-C, door_existence.h) ─────────────────────────────────────────────────────

PoseOverlap pose_overlap_factor(const std::optional<Mat6d>& rt_cov, const std::optional<Mat4d>& room_T_zed,
                                const Eigen::Vector2f& door_xy, float half_diag_m)
{
    if (not rt_cov.has_value() or not room_T_zed.has_value() or not rt_cov->allFinite())
        return {};
    const Eigen::Vector2d cam(room_T_zed->coeff(0, 3), room_T_zed->coeff(1, 3));
    const double range = std::max(1e-2, (door_xy.cast<double>() - cam).norm());
    const Eigen::Matrix2d Pxy = rt_cov->topLeftCorner<2, 2>();
    const double var_xy = std::max(0.0, Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(Pxy).eigenvalues().maxCoeff());
    const double var_b = std::max(0.0, rt_cov->coeff(5, 5)) + var_xy / (range * range);
    if (not (var_b > 0.0))
        return {};
    const double sb = std::sqrt(var_b);
    const double a = 2.0 * std::atan(std::max(1e-3, static_cast<double>(half_diag_m)) / range);
    const double z = a / sb;
    const double k = std::erf(z / std::sqrt(2.0))
                   - (2.0 / (z * std::sqrt(2.0 * std::numbers::pi))) * (1.0 - std::exp(-0.5 * z * z));
    return {static_cast<float>(std::clamp(k, 0.0, 1.0)), static_cast<float>(sb * range)};
}

// ─── the probes (DoorFitter's, moved) ────────────────────────────────────────────────────────────────────────────────

bool camera_usable(const CameraIntrinsics& c)
{
    return c.present and c.projection_model == 0 /*CameraAPI::ProjectionModel::Pinhole*/ and c.width > 0 and c.height > 0;
}

// PIXEL-LEVEL silhouette existence evidence — see DoorSilhouette / DoorFitState::existence.
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
DoorSilhouette compute_silhouette_existence(const ExistenceScene& sc, const rc::DoorFitState& inst,
                                            const SemanticField* field)
{
    DoorSilhouette out;
    if (not inst.ai2_initialized)
        return out;
    if (not camera_usable(sc.cam))
        return out;
    const auto& Mopt = sc.room_T_zed;   // camera→room, the cycle's pose
    if (not Mopt.has_value())
        return out;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();

    const float W    = static_cast<float>(sc.cam.width);
    const float Himg = static_cast<float>(sc.cam.height);
    if (W <= 0.f or Himg <= 0.f)
        return out;

    const auto& pkt = sc.pkt;
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
    const bool has_poly = sc.polygon.size() >= 3;

    // Camera position (room frame) for the GEOMETRIC occlusion tests below.
    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)),
                            static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const Eigen::Vector2f Oxy(O.x(), O.y());
    const auto own_edges = own_wall_edges(inst, sc.polygon);   // Task D-C: every wall but the door's own occludes
    const float occ_margin = std::max(0.0f, sc.cfg.exist_occlusion_margin_m);

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
    const auto band = sc.band;
    const float delta_ap = has_poly
                         ? rc::door::signed_boundary_distance(sc.polygon, inst.aperture.centre_xy())
                         : std::numeric_limits<float>::quiet_NaN();
    const bool  band_known = std::isfinite(delta_ap);
    const Eigen::Vector2f n_out = band_known
        ? rc::door::outward_normal(sc.polygon, inst.aperture.centre_xy(), inst.aperture.across_u(), band)
        : Eigen::Vector2f::Zero();
    const float r0 = rc::door::face_offset_from_aperture(delta_ap);
    const auto room_weight = [&](const Eigen::Vector3f& P) -> float
    {
        if (not band_known) return 1.0f;
        const float a = (Eigen::Vector2f(P.x(), P.y()) - inst.aperture.plane_O()).dot(n_out);   // from the LEAF's line (5b)
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
    const float central_region_frac = sc.cfg.exist_central_region_frac;
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
        const Eigen::Vector2d uv = camera_project(sc.cam, Eigen::Vector3d(X, Y, Z));
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
        if (rc::occlusion::walls_block(Oxy, {Pw.x(), Pw.y()}, sc.polygon, own_edges))
        { ++out.n_occluded; return; }
        {
            Eigen::Vector3f dc = Pw - O;
            const float rc_len = dc.norm();
            if (rc_len > 1e-3f)
            {
                dc /= rc_len;
                for (const auto& [jid, jp] : sc.doors)
                {
                    const rc::DoorFitState& jinst = *jp;
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
        const float f = central_region_frac, g = 1.0f - central_region_frac;
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
    // all-or-nothing rejection when a corner falls behind the camera. It also returns the per-vertex DEPTH
    // the projection already computed, which is what the depth channel tests the belief against.
    {
        const auto project = [&](const Eigen::Vector3d& Pr) -> std::optional<rc::edges::ProjectedVertex>
        {
            const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
            if (Pc.y() <= 0.20) return std::nullopt;                       // a corner behind the camera
            const Eigen::Vector2d uv = camera_project(sc.cam, Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
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
// function owns only what needs the scene: the camera, the transform, the masks, the walls, the other doors
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
rc::door::FrameProbe compute_frame_probe(const ExistenceScene& sc, const rc::DoorFitState& inst, float s_shift)
{
    // Task D-C review C1: the probe at the aperture slid by s_shift along its wall (a placement hypothesis); 0 = the belief.
    door::Aperture ap_shift = inst.aperture;
    ap_shift.s += s_shift;
    if (not inst.ai2_initialized)
        return {};
    if (not camera_usable(sc.cam))
        return {};
    const auto& Mopt = sc.room_T_zed;   // camera→room — the SAME transform the silhouette uses
    if (not Mopt.has_value())
        return {};
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();
    const int W = static_cast<int>(sc.cam.width), H = static_cast<int>(sc.cam.height);
    if (W <= 0 or H <= 0)
        return {};

    // No mask packet ⇒ no KNOWN masked occluder: the cells are empty and that factor is 1. The frame is a
    // geometric probe of the RGB image, so it does not need masks to exist — only to know what hides it.
    const MaskCells cells = build_mask_cells(sc.pkt);

    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)), static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const Eigen::Vector2f Oxy(O.x(), O.y());
    const auto own_edges = own_wall_edges(inst, sc.polygon);   // Task D-C: every wall but the door's own occludes
    const float occ_margin = std::max(0.0f, sc.cfg.exist_occlusion_margin_m);

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
                leaves.emplace_back(door::leaf_pose(ap_shift, ls), w);
            }
        else
            leaves.emplace_back(s_shift == 0.0f ? inst.leaf_pose : door::leaf_pose(ap_shift, inst.leaf), 1.0f);
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
        const Eigen::Vector2d uv = camera_project(sc.cam, Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
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
    const door::Aperture& ap_seen = ap_shift;

    const auto visible = [&](const Eigen::Vector3f& P) -> float
    {
        const auto v = project(P.cast<double>());
        if (not v.has_value()) return 0.0f;
        if (v->px.x < 0.f or v->px.y < 0.f or v->px.x >= static_cast<float>(W) or v->px.y >= static_cast<float>(H))
            return 0.0f;
        if (cells.hides(v->px.x, v->px.y))
            return 0.0f;
        if (rc::occlusion::walls_block(Oxy, {P.x(), P.y()}, sc.polygon, own_edges))
            return 0.0f;
        Eigen::Vector3f dc = P - O;
        const float rc_len = dc.norm();
        if (rc_len > 1e-3f)
        {
            dc /= rc_len;
            for (const auto& [jid, jp] : sc.doors)
            {
                const rc::DoorFitState& jinst = *jp;
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
    const float sd_s = inst.ai2_initialized ? std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(0, 0))) : 0.0f;
    return rc::door::build_frame_probe(ap_seen, project, visible, own_leaf_clear, W, H, sc.cfg.frame_casing_m,
                                       sc.cfg.exist_boundary_sigma_m, sd_s);
}

// |cos| of the camera→leaf ray against the LEAF's face normal. 1 = square-on, 0 = grazing. Keyed on the leaf, not
// the wall, so it stays meaningful once phi is fitted: an open door presents its face at a different angle than its
// aperture does.
float door_view_obliquity(const std::optional<Mat4d>& room_T_zed, const rc::DoorFitState& inst)
{
    if (not inst.ai2_initialized)
        return 1.0f;
    const auto& M = room_T_zed;           // camera→room (the cycle's pose)
    if (not M.has_value())
        return 1.0f;                      // no extrinsic → can't judge → don't suppress
    const Eigen::Vector2f cam(static_cast<float>(M->coeff(0, 3)), static_cast<float>(M->coeff(1, 3)));
    Eigen::Vector2f r = inst.leaf_pose.centre_xy - cam;
    if (r.norm() < 1e-6f)
        return 1.0f;
    r.normalize();
    return std::clamp(std::abs(r.dot(inst.leaf_pose.ey)), 0.0f, 1.0f);   // ey = the leaf's face normal
}

float aperture_band_offset(const std::vector<Eigen::Vector2f>& polygon, const rc::DoorFitState& inst)
{
    if (polygon.size() < 3)
        return std::numeric_limits<float>::quiet_NaN();
    const auto& ms = inst.model.state();
    return rc::door::signed_boundary_distance(polygon, Eigen::Vector2f(ms.ap_cx, ms.ap_cy));
}

// ═══ THE UPDATE (SpecificWorker::update_existence_beliefs as of 8a1abeac, moved with its comments) ═════════════════
// Integrate at the SENSOR rate; the priors, the silhouette, the image geometry, P(detect); the shared removal policy;
// the decision rows, the stdout readout and the CSV row; the removals through the world. See door_existence.h for what
// changed (the sources of the values, the destinations of the outputs) and why.

void DoorCore::update_existence(const DoorCycleInput& in, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    // Integrate at the SENSOR rate, not the compute rate: only when a new mask frame arrived. Otherwise a fast
    // compute loop would decay a briefly-occluded real door away between two sensor frames.
    const auto& pkt = masks_.packet();
    const bool sensor_fresh = pkt.valid and static_cast<int>(pkt.frame_id) != exist_last_mask_frame_;
    if (not sensor_fresh)
        return;
    exist_last_mask_frame_ = static_cast<int>(pkt.frame_id);

    // The silhouette channel needs the producer's raw 2D mask pixels (mask_pixels_xy — an OPTIONAL, newer
    // producer channel). Without them EVERY door reports n_detectable==0 and therefore HOLDs forever, which
    // looks exactly like "phantoms are never removed". Say so once rather than failing silently.
    // The 360 camera's channel needs no ZED pixels: with none, only a packet that carries no ricoh slice either has nothing to say.
    if (pkt.mask_pixels.empty()
        and std::none_of(pkt.slices.begin(), pkt.slices.end(), [](const MaskSlice& q) { return q.source != 0; }))
    {
        if (not exist_warned_)
        {
            exist_warned_ = true;
            diag("stdout", "door_concept: [existence] WARNING — the masks node carries no mask_pixels_xy; the "
                           "silhouette channel has no evidence, so NO door can be confirmed or removed. Check the "
                           "retina's mask-pixel publish.\n", out);
        }
        return;
    }

    // Physical sensor rates (interpretable detection/clutter probabilities), shared with table/chair.
    rc::exist::SensorModel sm;
    sm.sensor_sigma_m = cfg.exist_sensor_sigma_m;
    sm.detection_prob = cfg.exist_detection_prob;
    sm.clutter_prob   = cfg.exist_clutter_prob;

    // ★ONE removal policy, the SHARED one (rc::exist::RemovalPolicy / decide_removal). door was the last agent
    // still hand-writing this decision, in THREE places — and two of them counted CYCLES (`+= 1.0f`), the exact
    // form the shared module was extracted to eliminate. Hand-written copies is how the debounce drifted three
    // ways before, and it is why the confidence-scaled `required` fix would otherwise have reached five agents
    // out of six. The two PRIOR channels below keep their full-weight advance — that is a real, argued
    // difference (a prior is visibility-independent by construction) and it is expressed by passing a
    // resolvability of 1.0, not by rewriting the policy.
    rc::exist::RemovalPolicy policy;
    policy.logodds_max   = cfg.exist_max_logodds;
    policy.removal_prob  = cfg.exist_removal_prob;
    policy.remove_frames = static_cast<float>(cfg.exist_remove_frames);
    // Task 5b item 4 (fix round I-2): both channels are pooled by view (door_view_pool.h); the shared belief is armed with
    // ρ 0 so no second correlation discount is applied on top (the review's double counting).
    policy.frame_correlation = 0.0f;

    // ★Task D-C (rtD1i door_5): REMOVAL IS DECIDED ON THE EVIDENCE GATHERED SINCE ENTRY, NOT ON THE PRIOR THE DOOR ENTERED
    // WITH. A door restored from memory came back at its remembered L = −4, already below the removal bound (−1.99), so the
    // shared debounce counted P(detect)-weighted looks as "sustained condemnation" although the data since the restore were
    // net POSITIVE (L −4.00 → −3.04, +0.96 nats) — the prior alone completed the remove streak.
    // (Review I-1) The decision reads L_eff = L_bound + (L − L_entry) for a door that ENTERED below the bound: the test starts
    // AT the bound and moves only by what happened since entry — the Wald/SPRT reading, the prior entering once (in the
    // posterior L the agent publishes), never as a free condemnation. Continuous, not a sign switch: the streak and the
    // confidence-scaled `required` see how far below the bound the data since entry have carried it. A door that entered at
    // or above the bound (every birth, +1) is decided on L itself, exactly as before.
    const float L_bound = [&] { const float p = std::clamp(policy.removal_prob, 1e-3f, 0.5f); return std::log(p / (1.0f - p)); }();
    // ★Review I1: the record is UNCLAMPED BELOW (the −4 entry must stay condemnable) and CAPPED ABOVE at L_max − L_bound — the
    //  most the published belief can say in favour. Uncapped, a restored door confirmed for many views banked tens of nats;
    //  absent later, eff sat pinned at +L_max until absence repaid them all — immortal, the mirror of the clamp defect.
    const float since_entry_cap = policy.logodds_max - L_bound;
    const auto bank = [&](rc::DoorFitState& inst, float dL)
    { inst.existence_since_entry = std::min(inst.existence_since_entry + dL, since_entry_cap); };
    const auto decide = [&](rc::DoorFitState& inst, float look)
    {
        if (not (inst.existence_entry_L < L_bound))
            return rc::exist::decide_removal(inst.existence, inst.existence_debounce, policy, look);
        rc::exist::ExistenceBelief eff = inst.existence;
        eff.set(L_bound + inst.existence_since_entry);   // round 2: the unclamped record (DoorFitState::existence_since_entry)
        return rc::exist::decide_removal(eff, inst.existence_debounce, policy, look);
    };

    // ★ONE PreparedFrame (blur + Sobel + whole-frame gradient reference) per cycle, shared by every door's
    // leaf contour AND its aperture frame. The one-shot contour_edge_support(img, ...) recomputed all of it
    // per door for a result identical every time; the arithmetic of the score is unchanged.
    // ★The frames are the core's ImageSource (door_existence.h): live, the ingestors' frames the world staged after
    //  their pump; on a tape, the recorded ROI patches — with the RECORDED whole-frame gradient reference, which the
    //  patches alone cannot give.
    rc::edges::PreparedFrame rgb_prep;
    if (cfg.rgb_contour_check and not images_.rgb().empty())
    {
        rgb_prep = rc::edges::prepare_frame(images_.rgb());
        if (images_.frame_ref().has_value()) rgb_prep.frame_ref = images_.frame_ref().value();
    }
    // Input recording: the frames the checks below may score this cycle (cropped at the end of the cycle to the ROIs
    // they actually sample — the ROI notes below). Same conditions as the reads. (Live: InputSnapshot::note_images /
    // note_roi through the sinks; the adapter adds the frames and their stamps.)
    {
        const bool depth_used = not images_.depth().empty()
            and ((cfg.rgb_contour_check and cfg.contour_depth_check) or (cfg.frame_contour_check and cfg.frame_depth_check));
        if (sinks_.image_use) sinks_.image_use({not rgb_prep.gx.empty(), rgb_prep.frame_ref, depth_used});
    }
    const auto note_poly = [&](const std::vector<cv::Point>& poly, int extra, bool rgb, bool depth)
    { if (sinks_.image_roi) sinks_.image_roi({poly, {}, extra, rgb, depth}); };
    const auto note_point = [&](const cv::Point2f& p, int extra, bool rgb, bool depth)
    { if (sinks_.image_roi) sinks_.image_roi({{}, p, extra, rgb, depth}); };
    // The RGB frame's size: the depth checks scale their reads by depth / RGB (1 when there is no RGB frame).
    const int rgb_cols = images_.rgb().cols, rgb_rows = images_.rgb().rows;

    // The scene every probe reads: the cycle's camera (the snapshot's room ← zed — was a ts=0 cortex query inside each
    // probe), the packet, the room, and every live door (the occluders), in the world's order.
    const auto states = world_->fit_states();
    const std::optional<Mat4d> room_T_zed = CameraModel::room_T_zed(in.robot);
    const ExistenceScene scene{cfg, room_T_zed, in.camera, pkt, fit_.room_polygon(), fit_.room_band(), states};

    std::vector<std::uint64_t> to_remove;
    for (const auto& [id, instp] : states)
    {
        rc::DoorFitState& inst = *instp;
        rc::exist::arm(inst.existence, policy);      // the clamp and the frame correlation (shared policy)
        if (not inst.existence_seeded)               // seed on first visit (fresh birth OR adopted graph node)
        {
            inst.existence.set(cfg.exist_birth_logodds);
            inst.existence_seeded = true;
            inst.existence_entry_L = inst.existence.logodds();
            inst.existence_since_entry = 0.0f;
        }

        // ── ROOM-CONTAINMENT PRIOR: THE WALL BAND (door_room_band.h) ────────────────────────────────────
        // A door's aperture lies IN a wall: between the room polygon (drawn at the room-side face) and the far
        // face a wall thickness T beyond it. The old test — aperture centre inside the polygon ± 0.40 m, else
        // −1.5 nats PER FRAME and skip every sensor channel — had both halves wrong: the normal place for a real
        // door is OUTSIDE the polygon (it was only ever saved by the margin), and a fact about where a door is
        // does not become new evidence ten times a second. Integrated per frame, ANY penalty short of exactly
        // zero walks a still, unobserved door to the floor.
        // Now: a static log-prior log π(δ) = −½(dist(δ,[0,T])/σ)², exactly 0 anywhere on the band, falling off
        // in the door-registration ⊕ polygon-face σ on BOTH sides — a door well inside the room is as wrong as
        // one far outside. It REPLACES its previous value in L each cycle (take the old term out, put the new
        // one in), so a door on the band pays nothing ever, and one far off it is pushed to the floor once and
        // held there. The removal debounce is advanced by 1 − π (how sure the prior is that this is not a door
        // of this wall), on every cycle, probed or not — so an off-band door behind a wall, which no sensor can
        // reach, is still removed after RemoveFrames ideal looks' worth, as before.
        // ★ONLY ON A RESOLVED APERTURE (ai2_initialized). Before the first fit the "aperture" is the published
        // LEAF box (door_fitter.cpp refresh_geometry), whose centre sits up to half a width INSIDE the room
        // for a leaf open into it — a two-sided band would floor a healthy newborn. Births outside the room are
        // suppressed at birth (RoomMarginM); a newborn holds here as it does on every sensor channel.
        float p_band = 1.0f;
        if (cfg.exist_room_prior and not inst.is_bearing_hypothesis and inst.ai2_initialized)
        {
            const float delta = aperture_band_offset(fit_.room_polygon(), inst);   // NaN with no polygon ⇒ term 0
            const float term  = rc::door::band_log_prior(delta, fit_.room_band());
            inst.existence.set(inst.existence.logodds() - inst.room_prior_term + term);
            bank(inst, term - inst.room_prior_term);
            inst.room_prior_term = term;
            p_band = std::exp(term);
            inst.dbg_band_delta = delta;
            inst.dbg_band_p     = p_band;
        }
        else if (inst.room_prior_term != 0.0f)
        {
            inst.existence.set(inst.existence.logodds() - inst.room_prior_term);   // prior switched off: take it out
            bank(inst, -inst.room_prior_term);
            inst.room_prior_term = 0.0f;
        }
        const float prior_look = 1.0f - p_band;   // the prior's own resolvability toward removal

        // MINIMUM-HEIGHT prior: same categorical prior as at birth, for an instance that is ALREADY alive —
        // either it predates the birth check, or it drifted onto a low blob. A door is an aperture a person
        // walks through, so a support that CONFIDENTLY tops out below MinHeightM says "this is not a door"
        // no matter how well its silhouette is explained.
        //
        // obs_top_z accumulates untruncated views only, and obs_top_conf is the weight behind it, so a door
        // whose top is always clipped by the image border never reaches MinHeightConf and is never judged —
        // the prior stays silent rather than guessing. Not gated on visibility: like the room prior, this is
        // a fact about what a door IS, so it applies whether or not the door is in view right now.
        if (cfg.exist_min_height_prior and not inst.is_bearing_hypothesis
            and std::isfinite(inst.obs_top_z) and inst.obs_top_conf >= cfg.exist_min_height_conf
            and inst.obs_top_z < cfg.exist_min_height_m)
        {
            inst.existence.set(inst.existence.logodds() - cfg.exist_short_gain);
            bank(inst, -cfg.exist_short_gain);
            // Same as the room prior: a categorical fact about what a door IS, so it needs no visibility.
            if (decide(inst, 1.0f).remove)
                to_remove.push_back(id);
            continue;   // too short to be a door → no amount of silhouette agreement makes it one
        }

        // A bearing-only hypothesis has no depth (existence unjudgeable) and an un-initialised newborn has no
        // silhouette to project — both HOLD.
        if (inst.is_bearing_hypothesis or not inst.ai2_initialized)
            continue;

        // ── SILHOUETTE CHANNEL (the only one that may remove a door) ──────────────────────────────────
        const auto sil = compute_silhouette_existence(scene, inst, &semantic_);
        // Task D-C: how much of the door's predicted image lands on the door under the pose's uncertainty (κ, see
        // pose_overlap_factor) — every image channel below is weighted by it.
        const float view_R = 0.5f * std::hypot(inst.aperture.w, inst.aperture.h);   // the door's half-diagonal
        const PoseOverlap po = pose_overlap_factor(in.robot.rt_cov, room_T_zed, inst.aperture.centre_xy(), view_R);
        const float kappa = po.kappa;
        const float pool_R = std::hypot(view_R, po.spread_m);   // views closer than the pose spread are one look
        inst.dbg_pose_kappa = kappa;

        // ── THE 360 CAMERA'S OWN EXISTENCE CHANNEL (task Q) ───────────────────────────────────────────────────────
        // Symmetric to its BIRTH evidence: the same detection model P(detect | ricoh, viewpoint) = the birth's range term
        // (rc::birth, the SAME Detectability) x line of sight with WALLS (ricoh_door_birth.h), pooled per viewpoint
        // (ViewPool, the model every other channel uses). A ricoh detection of this door confirms it, worth its
        // variance/wall weight; a ricoh frame (any ricoh slice in the packet = the camera ran) that did NOT detect it,
        // while the ZED did not either, is absence worth P(detect). Where the ricoh rarely resolves a door, not
        // seeing it is weak evidence — and a ricoh-born phantom BEHIND the robot, which the ZED can never probe, is
        // removable by it. `ricoh_look` is the resolvability of that look, handed to the removal debounce.
        float ricoh_look = 0.0f;
        if (in.robot.room_T_body.has_value()
            and ricoh_fresh(pkt))
        {
            const Eigen::Vector2f cam_b(static_cast<float>(in.robot.room_T_body->coeff(0, 3)),
                                        static_cast<float>(in.robot.room_T_body->coeff(1, 3)));
            const Eigen::Vector2f door_c = inst.aperture.centre_xy();
            const bool det_now = inst.frames_since_detection == 0;
            // ★NOT LOOKED AT IS NOT NOT-SEEN. The 360 detector runs on one of three panorama strips per frame; a door outside
            // the processed strip is neither confirmed nor missed by this frame - no trial, no absence (MaskPacket::ricoh_windows).
            const bool looked = ricoh_looked_in(pkt, cam_b, door_c);
            // P(ricoh detects a door that is there), LEARNED on the doors the ZED confirms this frame: a ZED-observed door is
            // there for certain, so whether the ricoh also reports it is a clean Bernoulli trial of the 360 detector.
            if (det_now and inst.last_source == 0 and looked)
            {
                bool seen = false;
                for (const auto& q : pkt.slices)
                {
                    if (q.source == 0 or q.label != "door" or not q.has_depth) continue;
                    const auto rr = ricoh_range(fit_.room_polygon(), cam_b, Vec2f(q.centroid.x(), q.centroid.y()), q.depth_var,
                                                cfg.ai2_sigma_base_m, cfg.ai2_range_noise_lat_per_m,
                                                cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m,
                                                cfg.ricoh_plane_std_m * cfg.ricoh_plane_std_m, true);   // as the birth path (niche-aware)
                    const float v = std::max(0.0f, inst.ai2_belief.covariance()(0, 0)) + std::max(0.0f, rr.var);
                    if (rr.var >= 0.0f and (Vec2f(q.centroid.x(), q.centroid.y()) - door_c).squaredNorm() <= cfg.tracker_gate_mahalanobis * v)
                    { seen = true; break; }
                }
                // ONE trial per distinct (viewpoint, door): the frames of a view share its pooled detected fraction
                const auto [dh, dt] = inst.ricoh_hit_pool.add(cam_b, door_c, pool_R, seen);
                ricoh_trials_ += dt; ricoh_hits_ += dh;
            }
            const float hit = (ricoh_hits_ + 1.0f) / (ricoh_trials_ + 2.0f);
            const float pc  = std::clamp(sm.clutter_prob, 1e-3f, 1.0f - 1e-3f);
            float dL_look = 0.0f;
            if (det_now and inst.last_source != 0)
                // presence: the Bernoulli LLR of a detection, ln(hit / pc), worth its association likelihood
                dL_look = inst.last_detect_w * kappa * std::log(std::clamp(hit, pc + 1e-3f, 1.0f - 1e-3f) / pc);
            else if (not det_now and looked)
            {
                const float range = (door_c - cam_b).norm();
                const rc::birth::Detectability bd{cfg.ai2_periph_ref, 2.5f, 2.0f};
                const float p_ricoh = rc::birth::observation_weight({1.0f, range}, bd)
                                    * los_weight(fit_.room_polygon(), cam_b, door_c, cfg.ai2_sigma_base_m,
                                                 cfg.ai2_range_noise_lat_per_m);
                ricoh_look = p_ricoh;
                // absence: a Bernoulli look. The door yields a detection with prob q = P(detect | ricoh, viewpoint) x hit (clutter
                // scales with q too, so q = 0 says NOTHING, never a small ratchet): LLR = ln((1 - q hit) / (1 - q pc)).
                dL_look = kappa * std::log((1.0f - p_ricoh * hit) / (1.0f - p_ricoh * pc));
            }
            if (dL_look != 0.0f)
            {
                const float dL = inst.ricoh_view_pool.add_logodds(cam_b, door_c, pool_R, dL_look);
                const float L0 = inst.existence.logodds();
                inst.existence.integrate(1.0f, dL);
                bank(inst, inst.existence.last_increment());
                inst.ricoh_view_pool.commit(inst.existence.logodds() - L0);
            }
        }
        if (pkt.mask_pixels.empty())   // no ZED silhouette this frame: the ricoh channel is all there is
        {
            if (decide(inst, 1.0f - (1.0f - prior_look) * (1.0f - ricoh_look)).remove)
                to_remove.push_back(id);
            continue;
        }

        // ── RGB CONTOUR CHECK ────────────────────────────────────────────────────────────────────
        // Does the IMAGE have a boundary where this door's own projected leaf face predicts one? No
        // classifier is asked, so this does not go blind when ADE20K does — which is the entire reason
        // it exists. Scored against the same quad displaced along the wall, so the number is "better
        // than its own neighbours", not an absolute that would need a per-scene constant.
        // ★LIVE, not diagnostic: the delta computed below is integrated into inst.existence. (This line
        // said "DIAGNOSTIC ONLY, wired into no belief" long after that stopped being true — it was
        // written when the channel was first measured and never updated when it was armed.)
        // ★THE CONTOUR CHECK ABSTAINS ONCE THE LEAF LEAVES THE WALL PLANE.
        // Its null is "the same quad displaced sideways ALONG THE WALL", which is only a comparison
        // between like and like while the contour lies IN that wall. A swung leaf is out of the wall,
        // so the displaced copies stop sampling wall and start sampling the open doorway and the room
        // beyond — both edge-rich. Measured 2026-09-09 with phi at 35-45 deg: s_true fell below
        // s_control, excess went negative, and the channel contributed dL -1.4 to -2.6 EVERY cycle
        // while the silhouette was reporting occ 255 against free 15. It deleted the door it exists to
        // defend, and it did so most confidently exactly when the door was open.
        //
        // The test is geometric, not a tuned angle: the leaf is "in the wall" while its free edge has
        // displaced across the wall by less than the wall's own thickness. Beyond that the control
        // construction is invalid and the honest contribution is NONE — abstaining is a statement about
        // where this measurement is defined, not a way of ignoring inconvenient evidence.
        // ⚠The proper fix is a null built from the leaf's OWN kinematics — the same leaf at a DIFFERENT
        // phi — which is valid at any angle and is the same comparison estimate_phi already makes.
        // ★NO ABSTENTION ANY MORE. It was introduced because the controls were displaced along the wall
        // IN IMAGE SPACE, a null valid only for a flush leaf — but it was gated on phi, and phi wanders
        // to 20-31 deg on a door that is shut, so the channel stood down almost always and stopped
        // defending the door it was built for. (It DID defend it before phi became a DOF: dL +2.83
        // through 180 fully-blind cycles, which is the behaviour this restores.)
        // The controls now come from door_fitter, built by sliding the quad along the LEAF'S OWN plane
        // in 3-D and reprojecting — identical at phi = 0, still valid at any angle. So the null holds
        // open or shut, and there is nothing to gate.
        rc::edges::ContourEdgeScore edge{};
        if (cfg.rgb_contour_check and not images_.rgb().empty()
            and sil.contour.face.valid() and not sil.contour.controls.empty())
        {
            std::vector<std::vector<cv::Point>> ctl;
            ctl.reserve(sil.contour.controls.size());
            for (const auto& c : sil.contour.controls)
                ctl.push_back(c.px);
            edge = rc::edges::contour_edge_support(rgb_prep, sil.contour.face.px, ctl);
        }
        if (sil.contour.face.valid())   // the leaf quad + its controls: RGB and depth reads (input recording)
        {
            const bool rgb_read   = cfg.rgb_contour_check and not rgb_prep.gx.empty();
            const bool depth_read = cfg.rgb_contour_check and cfg.contour_depth_check;
            note_poly(sil.contour.face.px, 0, rgb_read, depth_read);
            for (const auto& c : sil.contour.controls) note_poly(c.px, 0, rgb_read, depth_read);
        }
        // ── THE METRIC HALF ─────────────────────────────────────────────────────────────────────────
        // The RGB check confirms "a rectangular thing with these borders is here" — its own header says
        // so, and a poster, a door-shaped panel or a cupboard front all pass it. Depth is the observable
        // that separates them: a photograph of a door has the borders and none of the step. It asks the
        // model's own question rather than an edge detector's — is the surface at the DISTANCE the belief
        // predicts, and does the world recede behind its boundary — so it is absolute where the RGB
        // statistic is relative, and it needs no control to refute.
        rc::edges::ContourDepthScore depth{};
        // ★contour_depth_check gates ONLY this half. Left off, `depth` stays default-constructed with
        // n_samples == 0, so every consumer below reduces to RGB-only by its own existing logic: the
        // summed verdict becomes edge.excess alone, and the depth-stands-alone branch cannot fire. No
        // separate "disabled" path to keep in step — which is the point of gating the measurement
        // rather than the integration.
        if (cfg.rgb_contour_check and cfg.contour_depth_check
            and not images_.depth().empty()
            and sil.contour.face.valid())
        {
            rc::edges::ContourDepthParams dp;
            const cv::Mat& dimg = images_.depth();
            const int fc = rgb_cols;   // (live, rgb_ingestor_ always exists: its frame's size, 0 when empty)
            const int fr = rgb_rows;
            dp.depth_scale_x = (fc > 0) ? static_cast<float>(dimg.cols) / fc : 1.0f;
            dp.depth_scale_y = (fr > 0) ? static_cast<float>(dimg.rows) / fr : 1.0f;
            depth = rc::edges::contour_depth_support(dimg, sil.contour.face, sil.contour.controls, dp);
        }
        inst.dbg_depth_verdict = depth.verdict;
        inst.dbg_depth_bias_m  = depth.mean_bias_m;
        inst.dbg_depth_n       = depth.n_samples;
        inst.dbg_edge_support = edge.support;
        inst.dbg_edge_excess  = edge.excess;
        inst.dbg_edge_nctl    = edge.n_controls;
        inst.dbg_edge_true    = edge.s_true;
        inst.dbg_edge_ctrl    = edge.s_control;
        inst.dbg_edge_n       = edge.n_samples;
        inst.dbg_sil_occ = sil.e_occ;     inst.dbg_sil_free  = sil.e_free;
        inst.dbg_phi_marg = sil.phi_marginalised; inst.dbg_phi_nhyp = sil.phi_n_hyp;
        inst.dbg_phi_wmax = sil.phi_w_max;
        inst.dbg_sil_ndet = sil.n_detectable; inst.dbg_sil_ntotal = sil.n_total;
        inst.dbg_sil_noccl = sil.n_occluded;  inst.dbg_sil_ncells = sil.n_cells;
        inst.dbg_sil_central = sil.central_frac(); inst.dbg_sil_resolv = sil.resolvability();
        // ── THE DOORWAY FRAME (door_frame_evidence.h) ──────────────────────────────────────────────────
        // The jambs + lintel of the APERTURE, which do not move when the leaf swings. Every channel above
        // looks at the leaf; this is the one that still sees the door when the leaf is open, and above all
        // when it has swung OUT of the room the robot is in (live try32: L +3.6 -> -4.0, room_2's only door
        // deleted while the camera looked straight at the doorway). Computed BEFORE the probed() test on
        // purpose: a leaf the camera cannot see does not make the frame around it unobservable.
        rc::door::FrameScore frame{};
        float frame_verdict = 0.0f;   // the placement-marginal verdict (below); frame.* stays the believed placement's
        if (cfg.frame_contour_check)
        {
            const auto probe = compute_frame_probe(scene, inst);
            rc::edges::ContourDepthParams fdp;
            const cv::Mat* fdimg = nullptr;
            if (cfg.frame_depth_check and not images_.depth().empty())
            {
                fdimg = &images_.depth();
                const int fc = rgb_cols;
                const int fr = rgb_rows;
                fdp.depth_scale_x = (fc > 0) ? static_cast<float>(fdimg->cols) / fc : 1.0f;
                fdp.depth_scale_y = (fr > 0) ? static_cast<float>(fdimg->rows) / fr : 1.0f;
            }
            frame = rc::door::score_frame(probe, rgb_prep.valid() ? &rgb_prep : nullptr, fdimg, fdp);
            frame_verdict = frame.verdict();
            // ★Task D-C review C1: THE FRAME IS SCORED OVER THE DOOR'S OWN PLACEMENT UNCERTAINTY. The frame channel asks
            //  "is there a doorway frame where the belief puts it" — at one placement, that conflates WHERE with WHETHER:
            //  rtD1i's real DOOR_0, with its aperture 0.1-0.2 m off along the wall, scored −6…−21 a frame (true 10-25 vs
            //  controls 97-199 sliding onto the real jamb) and was removed while YOLO detected it. The verdict is now the
            //  mixture over s ~ N(ŝ, σ_s²) (σ_s the fit's own posterior along the wall), log Σ_k π_k e^{v_k}, on the
            //  Gauss-Hermite nodes {0, ±√3·σ_s} (weights 2/3, 1/6, 1/6 — exact for a Gaussian up to 5th moments). The
            //  jamb controls slide kFrameJambOutward·w + √3σ_s outward (round 2), beyond the placement a ±√3σ_s node reaches.
            const float sd_s = std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(0, 0)));
            if (sd_s > 0.0f and frame.n_samples() > 0)
            {
                constexpr std::array<std::pair<float, double>, 2> kNodes{{{1.7320508f, 1.0 / 6.0}, {-1.7320508f, 1.0 / 6.0}}};
                double acc = 0.0;
                double mx = frame_verdict;
                std::array<float, 2> v{};
                for (std::size_t k = 0; k < kNodes.size(); ++k)
                {
                    const auto pk = compute_frame_probe(scene, inst, kNodes[k].first * sd_s);
                    const auto fk = rc::door::score_frame(pk, rgb_prep.valid() ? &rgb_prep : nullptr, fdimg, fdp);
                    v[k] = fk.n_samples() > 0 ? fk.verdict() : frame_verdict;
                    mx = std::max<double>(mx, v[k]);
                    for (const auto& smp : pk.samples)
                    {
                        const int reach = static_cast<int>(std::ceil(std::max(0.0f, smp.side_window_px)));
                        note_point(smp.px, reach, rgb_prep.valid(), fdimg != nullptr);
                        for (const auto& c : smp.ctl) if (c.w > 0.0f) note_point(c.px, 0, rgb_prep.valid(), false);
                    }
                }
                acc = (2.0 / 3.0) * std::exp(frame_verdict - mx);
                for (std::size_t k = 0; k < kNodes.size(); ++k) acc += kNodes[k].second * std::exp(v[k] - mx);
                frame_verdict = static_cast<float>(mx + std::log(acc));
            }
            // the aperture frame's samples, their controls, and the depth read windows (input recording)
            for (const auto& smp : probe.samples)
            {
                const int reach = static_cast<int>(std::ceil(std::max(0.0f, smp.side_window_px)));
                note_point(smp.px, reach, rgb_prep.valid(), fdimg != nullptr);
                for (const auto& c : smp.ctl)   // the controls are the RGB half's null only (score_frame)
                    if (c.w > 0.0f) note_point(c.px, 0, rgb_prep.valid(), false);
            }
        }
        inst.dbg_frame_excess = frame.excess;  inst.dbg_frame_true = frame.s_true; inst.dbg_frame_ctrl = frame.s_control;
        inst.dbg_frame_depth  = frame.depth_verdict; inst.dbg_frame_vis = frame.vis_frac;
        inst.dbg_frame_n      = frame.n_edge;  inst.dbg_frame_ndepth = frame.n_depth;

        // ── IMAGE-GEOMETRY EVIDENCE: the leaf contour (where the leaf is visible) + the doorway frame ─────
        // ★ONE tanh, ONE observation's worth per cycle, whichever of them speak. On a SHUT door the leaf
        // quad and the aperture frame are nearly the same pixels, so summing their verdicts inside one
        // saturation is the common-mode treatment — two full observations would count one boundary twice.
        // The leaf halves keep exactly their old form (RGB confirm-only + depth), so with the frame silent
        // this reduces term for term to what ran before. The frame enters SYMMETRIC and scaled by vis_frac
        // (how much of it the camera could resolve): on a blank wall both its halves are zero-mean, so it
        // neither confirms nor refutes a phantom — and a confirm-only form fed that noise would be a ratchet.
        const auto integrate_image_geometry = [&](bool leaf_visible)
        {
            float verdict = 0.0f;
            int   n = 0;
            if (leaf_visible and inst.dbg_edge_n > 0 and edge.n_controls > 0)
            { verdict += std::max(0.0f, edge.excess); n = std::max(n, inst.dbg_edge_n); }
            if (leaf_visible and depth.n_samples > 0)
            { verdict += depth.verdict; n = std::max(n, depth.n_samples); }
            if (frame.n_samples() > 0)
            { verdict += frame_verdict; n = std::max(n, frame.n_samples()); }
            if (n == 0)
            { inst.dbg_edge_delta = 0.0f; return; }   // nothing measured — NOT a refutation
            // ★D4 (Task 5b): POOLED BY VIEW NOVELTY (door_view_pool.h). contour_evidence's llr·tanh(v) is the worth of ONE
            // observation; the frames of one view are that observation repeated, so they pool (mean verdict) instead of
            // adding a full llr each — which on the tapes walked correct doors from the +4 clamp to the −4 floor in a few
            // frames. A new view (the camera moved by several of the door's angular half-sizes) is a new observation.
            // No camera pose ⇒ no viewpoint to pool by ⇒ HOLD (nothing integrated), as for an unmeasured cycle.
            if (not room_T_zed.has_value())
            { inst.dbg_edge_delta = 0.0f; return; }
            // one confident observation, worth κ of one under the pose's uncertainty (Task D-C)
            const float llr = kappa * rc::exist::contour_evidence(1e9f, n, sm).log_odds_delta;
            const Eigen::Vector2f cam(static_cast<float>(room_T_zed->coeff(0, 3)), static_cast<float>(room_T_zed->coeff(1, 3)));
            // The view's scale is the door's half-DIAGONAL, the same as the silhouette's pool (add_logodds below): "a new
            // view" is one question about one object, and two channels judging it at different scales let the one that
            // sees novelty sooner (the narrower) out-run the other — measured, tourR1 DOOR_1 at RecessPriorStd 0.2:
            // the image channel at 0.5·w walked a door the silhouette read occupied (150/30) to the floor in 13 frames.
            const float dL = inst.image_view_pool.add(cam, inst.aperture.centre_xy(), pool_R, verdict, llr);
            const float L0 = inst.existence.logodds();
            inst.existence.integrate(1.0f, dL);
            bank(inst, inst.existence.last_increment());
            inst.image_view_pool.commit(inst.existence.logodds() - L0);   // what the view really moved L (the clamp)
            inst.dbg_edge_delta = dL;
        };

        // ★"WAS IT LOOKED AT" IS A QUESTION ABOUT THE DOOR, NOT ABOUT ONE GUESS AT ITS HINGE ANGLE.
        // probed() answers it under the phi posterior, so a leaf the fit happened to swing out of frame
        // is not mistaken for a door the camera never pointed at. See DoorSilhouette::probed().
        if (not sil.probed())
        {
            // NOT PROBED this frame — behind the robot, out of the frustum, or fully occluded. rc::exist HOLDs.
            // This branch is the whole fix for "the door disappears when the robot turns around": there is no
            // pd to fall back on, because a sensor that did not look produces no evidence either way.
            inst.dbg_sil_pdetect = 0.0f; inst.dbg_sil_free_eff = 0.0f;
            // The leaf was not looked at; the FRAME may have been. Its evidence is integrated, but no
            // removal decision is taken on a cycle the removal channel (the silhouette) did not probe.
            integrate_image_geometry(/*leaf_visible=*/false);
            // The wall-band prior needs no look: an off-band door advances its debounce here too (a door the
            // sensor can never reach — behind a wall — is exactly the case that prior exists for).
            if (decide(inst, 1.0f - (1.0f - prior_look) * (1.0f - ricoh_look)).remove)
                to_remove.push_back(id);
            continue;
        }

        // OCCUPANCY, weighted by how much of its mask the door MODEL actually explains. last_clutter_frac is the
        // mixture's clutter responsibility — the same per-point posterior the free energy is built from — so a
        // blob the door model fits none of confirms nothing. This is the missing term that let door_2 (FE 1.33
        // vs 0.17 for the real door, clutter 0.38 → 0.98) climb to L=2.29 and then sit there: the old channel
        // scored point MASS, never fit QUALITY. Stale between wins, but e_occ is only non-zero on a frame whose
        // door mask overlaps this silhouette, i.e. one where the instance is being detected.
        const float q_explain = std::clamp(1.0f - inst.last_clutter_frac, 0.0f, 1.0f);
        const float e_occ = sil.e_occ * q_explain;

        // ABSENCE. Suppressed entirely when the door was DETECTED this frame by any sensor (it is not gone —
        // a ZED false-negative, or a door only the ricoh resolved, must never vote itself away). This is the
        // honest form of "ricoh confirms, never removes": the old code enforced it on the positive side while
        // the negative side removed through range_detectability, which is precisely the ricoh's own reach.
        // ★"OBSERVED" IS A WEIGHT, NOT A FLAG. A ZED detection is a full observation; a ricoh one is worth its own
        // variance/wall-agreement weight (DoorFit: last_detect_w), so a noisy 360 detection cannot silence the ZED's
        // absence evidence at full strength.
        const float observed_w = inst.frames_since_detection == 0 ? (inst.last_source == 0 ? 1.0f : inst.last_detect_w) : 0.0f;
        // ★A frame that may not MOVE the geometry may not DESTROY the object either. `dbg_gated` is the fit's
        // own admissibility verdict (door_fitter.cpp: truncated mask, or the robot moving with the mask
        // off-centre ⇒ predict-only). When it is set the belief is FROZEN, so the projected silhouette is a
        // stale panel compared against a live image and any mismatch measures our registration, not the door's
        // existence. Ported from refrigerator_concept, where the absence of this rule deleted a healthy fridge
        // in 5 s: the robot drove 2.3 m → 1.15 m, the motion gate froze the belief, and the frozen box stopped
        // projecting onto a mask that was still arriving with ~55000 points. Every cycle that contributed
        // removal evidence had gated=1. Occupancy is untouched — a lit sample can only ever confirm.
        // ⚠A phantom is now only removable from an admissible viewpoint; removal waits for a good look.
        //
        // ★2026-08-07 — THE GUARD MUST READ THE GATE'S *FRESHNESS*, NOT THE BARE VERDICT (same defect table_concept
        // fixed on 08-06). `run_inference` returns early when no mask reaches the fit, so `dbg_gated` — and the
        // `last_trunc_frac` it is computed from — are STALE on exactly the cycles this guard fires: they are the
        // verdict of whenever the door was last SEEN. A phantom is by definition never seen again, so one
        // truncated frame at birth pinned dbg_gated=true for the rest of its life and its absence evidence was
        // zeroed for good — the channel demanded a good mask on the door in order to believe there is no mask on
        // the door. Live proof (etc/door_existence_log.csv, door_3 phantom at (1.82,−4.57), robot parked and
        // staring at it): n_detectable 409/420, e_free 409, occ 0, central 0.76, p_detect 0.76 — a textbook
        // resolving look — yet sil_free_eff ≡ 0.00, remove_streak ≡ 0 and L pinned at the +4 clamp for 1928
        // frames, because trunc=0.155 > TruncGateFrac 0.10 was carried over from its last detection.
        // Reading gate_fresh makes the guard mean what its comment always claimed: it suppresses absence only
        // when THIS frame's fit really was frozen while a mask was in fact arriving.
        const bool view_untrustworthy = inst.dbg_gate_fresh and (inst.dbg_trunc_gated or inst.dbg_motion_gated);
        const float raw_free = view_untrustworthy ? 0.0f : (1.0f - observed_w) * sil.e_free;
        // P(detect | present, geometry): could this view have resolved a door that IS there?
        //   resolvability — is the panel big enough in the image to segment (range × foreshortening),
        //   in_fov_frac  — how much of it the real frustum + occluders actually left visible,
        //   central_frac — whether the robot is LOOKING at it rather than clipping the frustum edge.
        // Predicted-but-absent counts toward REMOVAL only in proportion to p_detect; the remainder is epistemic
        // surprise ("I cannot resolve this from here"), which should send the robot to look, never delete.
        //
        // ★ p_detect scales the SATURATED delta, not the raw pixel COUNT. Scaling the count does not work:
        // mask_evidence tanh-saturates the summed log-odds, and with hundreds of samples the sum sits far
        // past the knee, so p_detect changes almost nothing until it hits exactly 0 — at which point the
        // absence term vanishes entirely and the channel becomes a monotonic CONFIRMER. That is precisely
        // what the live log showed: central_frac ≡ 0 ⇒ p_detect ≡ 0 ⇒ sil_free_eff ≡ 0, so the phantom at
        // (−4.22,−2.16) sat at L = 4.0 with 230 of its 420 silhouette samples unlit, forever. (Same defect
        // family as the table's "p_detect inert" — see the dining-set fix, which is where this form comes
        // from.) Scaling the SATURATED delta by p_detect makes it act linearly over its whole range, and keeps
        // confirmation (+) untouched. (It once interpolated from a confirm-only FLOOR; that floor was removed
        // — see the note at ev.log_odds_delta below — so the confirm-only delta is no longer computed.)
        //
        // ★THE DETECTOR ENVELOPE, which this channel was documented to use and did not. door_config.h's
        // own comment says "the removal channel weights absence by it, so a missing mask from a pose the
        // detector could never fire from reads as EXPECTED rather than as evidence the object is gone" —
        // but only the epistemic planner ever read it. The three geometric factors above all IMPROVE
        // monotonically as a face-on door gets closer, so p_detect climbed toward 1 exactly where the
        // detector was failing hardest.
        //
        // Measured 2026-09-08 over one approach, 198 distinct viewpoints, de-duplicated by pose:
        //     roi_fill 0.15-0.20 -> fired 31%   |  p_detect said 0.44
        //     roi_fill 0.20-0.25 -> fired 90%   |               0.67
        //     roi_fill 0.25-0.30 -> fired 55%   |               0.80
        //     roi_fill 0.30-0.35 -> fired 23%   |               0.97
        // Detectability is UNIMODAL in fill and the geometric product is monotone: the wrong SHAPE, not
        // wrong constants. Past ~4 m of range (fill ~0.31) absence was charged ~4x harder than the truth,
        // which is what drove L to its -4.00 clamp and deleted a door in plain view.
        //
        // visible_frac is passed as 1.0 on purpose: occlusion/frustum loss is already carried by
        // in_fov_frac() below, and passing it twice would square it.
        const rc::detect::DetectorEnvelope env{cfg.detect_min_fill, cfg.detect_max_fill, cfg.detect_soft};
        const float p_env = rc::detect::p_detect(inst.roi_fill, 1.0f, env);
        inst.dbg_p_env = p_env;
        // ★OBLIQUITY OF THE LEAF, which only became a live variable when phi did. A panel seen edge-on
        // presents almost no area to segment: |ray . leaf face normal| falls from 0.99 at phi=5 deg to
        // 0.43 at 60 deg and 0.21 at 85 deg. door_view_obliquity() has computed exactly this all along —
        // against inst.leaf_pose.ey, so it follows the leaf — and was labelled "diagnostic only" and
        // never used. With phi pinned at 0 that was harmless: a flush door is always face-on, the term
        // is ~1, and it changed nothing. The moment the leaf can swing it becomes the dominant
        // detectability factor, and leaving it out charges a full-strength absence against a look that
        // physically could not have resolved the door.
        //
        // Measured 2026-09-09 on the deaths: absence was charged at p_detect 0.55 with the leaf at 37
        // deg and 0.28 at 60 deg; with obliquity those become 0.39 and 0.12.
        //
        // ⚠It cuts BOTH ways, as every p_detect does: an edge-on leaf can no longer refute the door,
        // and also can no longer defend it. That is the honest reading of an unresolvable view, but it
        // is the same shape as the documented hood failure, so a door believed permanently edge-on
        // would be hard to remove. The aperture-side evidence is what has to keep that in check.
        const float leaf_oblq = door_view_obliquity(room_T_zed, inst);
        float p_detect = p_env * leaf_oblq * sil.resolvability() * sil.in_fov_frac() * sil.central_frac() * kappa;

        // ★THE CONTOUR CHECK. Everything above is geometry: it says how well the camera could have
        // resolved the door IN PRINCIPLE. This asks what the classifier actually computed AT the door's
        // own projected pixels, which is the only place the question is really settled.
        //
        // The statistic is a CONTRAST, not an absolute. At 4.2 m over a plainly visible door the field
        // reads P(door)=0.265 while calling it `wall` at 0.676 — so "P(door) is high" would fail on the
        // exact case this exists for. What still holds is WHERE the door-ness sits: if the contour is
        // markedly more door-like than the rest of the frame, the door is where the belief says it is,
        // even while the argmax disagrees.
        //
        //     support = mean P over contour / (mean P over contour + mean P over the whole field)
        //
        // 0.5 means the contour is no more door-like than anywhere else — the look said nothing, and
        // absence must not be charged for it. Above 0.5 the contour is concentrated door-ness and
        // absence is damped in proportion. BELOW 0.5 the contour is LESS door-like than background,
        // which is genuine evidence against, and absence is charged in full — that asymmetry is what
        // keeps this a test the belief can fail rather than a shield it wears.
        //
        // ★No threshold, and no constant: the 0.5 is the neutral point of a ratio, not a tuned cutoff.
        float field_support = 0.5f;
        if (sil.field_n > 0 and sil.field_bg > 0.0f)
        {
            const float fin = sil.field_mean();   // (`in` is the cycle's input here)
            field_support = fin / (fin + sil.field_bg);
            // ★NO LONGER ACTS. It damped p_detect until 2026-09-09, and that was wrong twice over.
            //  (1) It could only ever HOLD, never DEFEND: integrate() collapses to the confirm-only term
            //      as p_detect -> 0, and that term is zero on a frame with no detection. The belief sat
            //      at p_exists 0.03 while the node survived — a corpse that could not be buried.
            //  (2) The background is the whole frame, which is mostly floor and ceiling at P(door)~0.015,
            //      so `support` saturated at 0.98 for any contour touching anything door-ish. p_detect
            //      collapsed to ~0.017 and STARVED decide_removal's accumulator, which measures the
            //      debounce in ideal observations — so a genuine phantom became unremovable too.
            // It stays computed and logged because it is an excellent DIAGNOSTIC: watching fld_in fall
            // 0.935 -> 0.485 -> 0.058 is what showed the classifier going blind in real time. The
            // defending is now done by the RGB contour channel below, which is not derived from the
            // classifier and therefore does not go blind with it.
        }
        inst.dbg_field_support = field_support;
        inst.dbg_field_mean    = sil.field_mean();
        inst.dbg_field_bg      = sil.field_bg;
        inst.dbg_field_n       = sil.field_n;
        inst.dbg_sil_pdetect = p_detect; inst.dbg_sil_free_eff = raw_free * p_detect;

        const float d_full = rc::exist::mask_evidence(e_occ, raw_free, sil.n_detectable, sm).log_odds_delta;
        rc::exist::Evidence ev;
        ev.e_occ = e_occ; ev.e_free = raw_free; ev.n_reached = sil.n_detectable;
        // Interpolate from ZERO, not from the occupancy-only floor — see the note in
        // refrigerator_existence.cpp: the old form left a positive push at p_detect -> 0, which is the
        // ratchet that makes a phantom immortal. A probe that could not resolve the object is a HOLD.
        ev.log_odds_delta = p_detect * d_full;
        // ★Task 5b item 4: the looks of ONE view are one observation repeated; a NEW view is a new one. The silhouette is
        // pooled by the same view model and scale as the image channel (door_view_pool.h add_logodds: the view's
        // contribution is the mean increment of its looks, renewed in proportion to the novelty of each new look). Before
        // this the door counted a parked robot's every miss as a new one. Measured on tourR1+R2: parked look pairs
        // P(miss | miss) 0.784 vs P(miss) 0.024 — ρ 0.78, near the ρ = 1 this pooling assumes for one view (no config key:
        // the pooling has no ρ to set).
        bool pooled_sil = false;
        if (room_T_zed.has_value())
        {
            const Eigen::Vector2f cam(static_cast<float>(room_T_zed->coeff(0, 3)), static_cast<float>(room_T_zed->coeff(1, 3)));
            ev.log_odds_delta = inst.silhouette_view_pool.add_logodds(cam, inst.aperture.centre_xy(), pool_R,
                                                                      ev.log_odds_delta);
            pooled_sil = true;
        }
        const float L_before_sil = inst.existence.logodds();
        // p_detect is passed EXPLICITLY: log_odds_delta already carries it, but the second argument is also
        // what the frame-correlation weighting reads (rho_eff = rho*(1-p_detect)). Defaulting it to 1.0 told
        // the belief every miss was a maximally-unambiguous look, so consecutive correlated misses counted as
        // independent. (Task 5b item 4: the silhouette is pooled by view above; the belief itself is armed with ρ 0, so
        // this p_detect no longer feeds a second correlation discount.)
        inst.existence.integrate(ev, p_detect);
        bank(inst, inst.existence.last_increment());
        if (pooled_sil) inst.silhouette_view_pool.commit(inst.existence.logodds() - L_before_sil);   // what it REALLY moved L

        // ── RGB CONTOUR CHANNEL: an INDEPENDENT sensor, integrated separately ────────────────────────
        //
        // Measured 2026-09-09 across a death, on this door: as the robot closed, the semantic posterior
        // under the contour fell 0.935 -> 0.485 -> 0.058 while the image's across-boundary gradient on
        // that same contour ROSE, s_true 47 -> 58 against controls falling 16 -> 11 (support 0.75 ->
        // 0.84, on 150 -> 267 boundary samples). The door's jamb and lintel become MORE distinct as you
        // approach; only the network's label collapses. So this channel carries information exactly
        // where every classifier-derived one is empty.
        //
        // ★SYMMETRIC, which is what makes it a test and not a shield. Above 0.5 the contour is more
        // distinct than the same shape displaced along the wall -> occupancy. Below 0.5 it is LESS
        // distinct -> free, at the same weight. The magnitude is how decisive the image was, and 0.5 is
        // the neutral point of a ratio rather than a tuned cutoff.
        //
        // ★n_reached is the count of boundary samples actually scored, so a contour half out of frame
        // contributes proportionally less on its own, with no special case.
        //
        // ★p_detect = 1: for THIS channel, having scored the boundary IS having resolved the question.
        // Its detectability is already carried by n_reached and by edge_n == 0 meaning "not measured".
        //
        // ⚠CORRELATION, stated because the framework has no term for it: this and the silhouette channel
        // are different measurements (geometric boundary vs semantic label) but not independent events —
        // both depend on the door being there, so on a frame where a mask IS present both push positive.
        // The +/-4 log-odds clamp bounds the resulting overconfidence; a proper common-mode treatment
        // (as the mask-point Woodbury marginalisation does for sigma) is the right long-term fix.
        // (What follows is the history of the LEAF half of integrate_image_geometry above — its confirm-only
        // RGB term and its depth term. The doorway-FRAME half is symmetric; the reasons are at the lambda.)
        // ★DIFFERENCE, NOT RATIO, and NOT through mask_evidence. Both halves are scars:
        //  · The ratio form threw away magnitude, so a near-blank frame (s_true 5.2 against controls
        //    7.8) produced support 0.400 and a full-strength refutation that deleted a live door.
        //    The excess for that same frame is -0.26: weak, which is the truth.
        //  · Routing through mask_evidence() applied its tanh common-mode cap, and my sample counts
        //    (75-182) sat so far past its knee that EVERY cycle clipped to the identical +-2.83.
        //    The graded weight I thought I had was a binary vote on the sign. Measured: the log
        //    contained exactly two distinct dL values across the whole run.
        // The conversion is now rc::exist::contour_evidence (MODALITY 3), which is that reasoning
        // written once for the fleet: scaled by ONE confident observation's worth, log(pd/pc), the
        // same quantity every other channel is denominated in, with tanh bounding the cycle rather
        // than quantising it. The gate stays HERE because it is this channel's own precondition —
        // the RGB statistic is relative and means nothing without a surviving control.
        // ★THE TWO HALVES SUM INSIDE ONE tanh, and that IS the common-mode treatment. A boundary in
        // the image and a step in the depth at the same place are not independent events — both are
        // consequences of the door being there — so letting each contribute a full observation's
        // worth would count one fact twice. Both are dimensionless and bounded, so the sum is well
        // defined, and one confident observation still bounds the cycle.
        // ★CONFIRM-ONLY: this channel may RAISE the belief and never lower it, and that asymmetry is
        // stated, not tuned. A contour that MATCHES is evidence the door is there. A contour that does
        // NOT match is evidence our POSE is wrong — the leaf is not at the angle we drew it at — which
        // is a fit error and none of the existence channel's business. The channel was introduced for
        // exactly the first job: positive evidence that stops a blind detector from removing a door.
        // ★MEASURED, AND IT IS NOT A SMALL EFFECT. With the leaf tracked to its true ~97 deg the
        // contour scores a foreshortened quad out in the room against copies of itself displaced
        // along a wall that is no longer behind it, and the excess flips sign nearly every cycle:
        // -2.35 +0.85 -5.80 -5.68 +2.40 +1.11 -4.29 ... -7.48 over consecutive frames, up to 7.5 nats
        // from ONE frame against a band only 8 wide. The silhouette channel was healthy throughout
        // (free_eff 0, occ 92, detector firing) — the belief was being thrown across its whole range
        // by a measurement with no notion of its own reliability.
        // ⚠THE PROPER FIX IS PRECISION, NOT A CLAMP: the excess is a difference of means whose own
        // spread across the control placements is discarded, so a wild measurement and a confident
        // one currently weigh the same. Returning that spread and weighting by it would let a noisy
        // contour contribute little in BOTH directions, which is strictly better than refusing one
        // direction. This asymmetry is the honest interim: it cannot make the door immortal — the
        // silhouette channel still removes, and it is the one with a detectability model.
        // ★THE DEPTH HALF STANDS ALONE when the RGB half cannot speak — no surviving control, or no
        // RGB frame at all. Its statistic is absolute, so it does not need the null the gradient
        // check does, and making it wait for one would silence the only channel that can say "we are
        // looking straight through the place the door is supposed to be".
        integrate_image_geometry(/*leaf_visible=*/true);

        // Debounce on consecutive EVIDENCE cycles (not wall-clock), so a transient hiccup cannot delete a real
        // door, and removal always reflects sustained agreement across frames.
        // ★DEBOUNCE ON LOOKS, NOT CYCLES. `++` counted every cycle in which the decision said remove — including
        // the ones where p_detect was 0 and the channel had just, correctly, HELD. So a door that dipped below
        // the boundary once was deleted `RemoveFrames` cycles later no matter what happened in between, and the
        // live record shows exactly that: 12 of 12 door deaths had fixated=0, five at p_detect=0.000, at ranges
        // of 2.1-6.7 m. Accumulating p_detect makes the debounce measure how much genuine LOOKING has happened,
        // so RemoveFrames is a number of IDEAL observations - the same unit table and bottle use.
        // ★And RemoveFrames is what is demanded AT the boundary. Charging the full count of ideal observations
        // AFTER L has already crossed double-counts the same evidence and is what freezes a condemned door
        // forever once the robot looks away; the shared policy lets confidence discharge the repetition.
        // Either the look (p_detect) or the wall-band prior (1 − π) can resolve this cycle: probabilistic OR.
        // ★Task D-C review C1 (lifecycle §3: occupancy confirms; only PREDICTED-VISIBLE-BUT-ABSENT removes): the look a
        //  cycle contributes toward REMOVAL is the silhouette's, and only when the silhouette actually charged absence —
        //  not on a frame where the door was detected (observed) or the view was not admissible. rtD1i: the real DOOR_0
        //  was removed at cycle 614 while YOLO detected it (since_det 0-3, occ 330 / free 90) — the image-geometry channel
        //  had pulled L down and the debounce counted P(detect) on detected frames.
        // ★Task D-C round 2: CONTINUOUS, not a switch on raw_free > 0. The look this cycle contributes toward removal is
        //  P(detect) times the share of the silhouette's reached samples that charged ABSENCE (free / (occ + free)): a frame
        //  that saw the door mostly lit and a few samples free is mostly a confirmation, not a full look at an absence.
        const float absence_share = raw_free > 0.0f ? raw_free / std::max(1e-6f, raw_free + std::max(0.0f, e_occ)) : 0.0f;
        const float absence_look = p_detect * absence_share;
        const float cycle_look = 1.0f - (1.0f - absence_look) * (1.0f - prior_look) * (1.0f - ricoh_look);
        const auto verdict = decide(inst, cycle_look);
        if (verdict.remove)
            to_remove.push_back(id);
        if (verdict.stalled)
            diag("stdout", std::format("door_concept: {}\n", rc::exist::stall_note(inst.node_name, inst.existence,
                                                                                   inst.existence_debounce, verdict)), out);

        // ★AND THE MIRROR CASE, structurally unreportable until 2026-08-15: NOT condemned, and nothing
        // has resolved it for a long time. `stalled` is computed AFTER decide_removal's not-condemned
        // early return, so a belief pinned at the +4 clamp by evidence no look can contradict said
        // NOTHING — a phantom refrigerator held there by a DOOR's returns sat silent for an hour with
        // streak 0 on all 9714 rows. A REPORT only; nothing branches on it.
        if (verdict.unfalsifiable)
            diag("stdout", std::format("door_concept: {}\n",
                                       rc::exist::unfalsifiable_note(inst.node_name, inst.existence, verdict)), out);
    }

    // Existence readout. Throttled to 1-in-60 so a "why is this phantom still here?" case stays
    // diagnosable without flooding — but ★THE THROTTLE MUST NOT SWALLOW THE EVENT THE FILE EXISTS FOR.
    //
    // Three removals tonight were invisible: the last sampled row read L +4.00, occ 420, free 0,
    // free_eff 0.00, won 1, edge_dL +2.83 — every channel positive — and 60 cycles later the door was
    // gone at L -4.00. Eight nats crossed INSIDE one sampling interval, so the transition that actually
    // did it was never written down, and the min-height and room-containment priors could only be
    // eliminated by inference rather than read off. A log that samples uniformly is the wrong shape for
    // a question about a transient: it records the steady states either side of the thing you want.
    //
    // So a row is forced whenever the belief MOVES sharply (|dL| > 1 nat in one cycle) or is about to
    // be removed, regardless of the counter. Rare by construction — a healthy door sits at its clamp —
    // and it costs nothing on the cycles where nothing happens.
    // (the throttle counter and the last-L memory were function statics; they are the core's now)
    ++ex_dbg_;
    std::unordered_set<std::uint64_t> force_row;
    for (const auto& [id, instp] : states)
    {
        const auto& inst = *instp;
        if (inst.is_bearing_hypothesis) continue;
        const float L = inst.existence.logodds();
        if (const auto it = last_L_.find(id); it != last_L_.end() and std::abs(L - it->second) > 1.0f)
            force_row.insert(id);
        last_L_[id] = L;
    }
    // Decision log: the existence belief of every door after this fresh-mask update (the input of the REMOVE below).
    // (a = L, b = p_exists, c = x, d = y — the floats widened, as DoorDecisionLog::record widened them.)
    for (const auto& [id, instp] : states)
        if (const auto& inst = *instp; not inst.is_bearing_hypothesis)
        {
            rc::door::DoorDecision d{out.cycle, out.t_ms, static_cast<std::uint16_t>(K::Existence), id, inst.node_name,
                                     inst.existence.logodds(), inst.existence.p_exists(), inst.model.state().cx,
                                     inst.model.state().cy, {}};
            if (sinks_.decision) sinks_.decision(d);
            out.decisions.push_back(std::move(d));
        }
    for (const auto id : to_remove)     // the removal cycle itself, always
        force_row.insert(id);
    for (const auto& [id, instp] : states)
        if (const auto& inst = *instp; not inst.is_bearing_hypothesis and (ex_dbg_ % 60 == 0 or force_row.contains(id)))
            {
                const auto& ms = inst.model.state();
                const bool has_poly = fit_.has_room_polygon();
                const bool inroom = (not has_poly)
                                    or point_in_polygon(fit_.room_polygon(), Eigen::Vector2f(ms.cx, ms.cy), cfg.exist_room_margin_m);
                const float oblq = door_view_obliquity(room_T_zed, inst);   // diagnostic only
                diag("stdout", std::format("door_concept: [existence] {} L={:.2f} p={:.2f} pos=({:.2f},{:.2f}) inroom={} roomprior={} "
                           "band d={:+.2f} pi={:.2f} "
                           "won={} since_det={} | sil occ={:.0f} free={:.0f} free_eff={:.1f} ndet={}/{} occl={} "
                           "resolv={:.2f} central={:.2f} pdet={:.2f} oblq={:.2f} gate={}/{} strk={}"
                           // The RGB-reprojection contour channel, on the line where pdet is read: without
                           // these three, pdet=0.02 beside resolv=0.60 central=0.98 is unexplainable from
                           // the terminal (0.60*0.98 = 0.59, not 0.02) and looks like a bug in the geometry.
                           // fld_n=0 means the field was UNAVAILABLE, which is not the same as no support.
                           " | fld n={} in={:.3f} bg={:.3f} sup={:.3f}"
                           // The classifier-free channel, beside the classifier one. When these two
                           // disagree — edge high while P(door) collapses — the disagreement IS the
                           // finding, and it was previously only visible by joining two files afterwards.
                           " | edge n={}/{}ctl sup={:.2f} exc={:+.2f} (t={:.0f} c={:.0f}) dL={:+.2f}"
                           " | phi={:.1f}deg sup={:.2f} cmd={}"
                           // The doorway-FRAME channel (door_frame_evidence.h). n=0 ⇒ NOT MEASURED. Its
                           // contribution is inside the edge dL above (one tanh for all image geometry).
                           " | frame n={} vis={:.2f} exc={:+.2f} (t={:.0f} c={:.0f}) dep={:+.2f}/{}\n",
                           inst.node_name, inst.existence.logodds(), inst.existence.p_exists(), ms.cx, ms.cy,
                           inroom ? 1 : 0, has_poly ? 1 : 0, inst.dbg_band_delta, inst.dbg_band_p,
                           inst.assigned_mask_idx >= 0 ? 1 : 0, inst.frames_since_detection,
                           inst.dbg_sil_occ, inst.dbg_sil_free, inst.dbg_sil_free_eff,
                           inst.dbg_sil_ndet, inst.dbg_sil_ntotal, inst.dbg_sil_noccl,
                           inst.dbg_sil_resolv, inst.dbg_sil_central, inst.dbg_sil_pdetect, oblq,
                           inst.dbg_gated ? 1 : 0, inst.dbg_gate_fresh ? 1 : 0,
                           inst.existence_debounce.streak,
                           inst.dbg_field_n, inst.dbg_field_mean, inst.dbg_field_bg, inst.dbg_field_support,
                           inst.dbg_edge_n, inst.dbg_edge_nctl, inst.dbg_edge_support, inst.dbg_edge_excess,
                           inst.dbg_edge_true, inst.dbg_edge_ctrl, inst.dbg_edge_delta,
                           inst.phi_est * 180.0f / static_cast<float>(M_PI), inst.phi_support,
                           inst.phi_cmd_active ? 1 : 0,
                           inst.dbg_frame_n, inst.dbg_frame_vis, inst.dbg_frame_excess, inst.dbg_frame_true,
                           inst.dbg_frame_ctrl, inst.dbg_frame_depth, inst.dbg_frame_ndepth), out);
                // Minimum-height evidence: obs_top is the support top over UNTRUNCATED views only, so a door
                // whose top is always clipped by the image border shows conf→0 and is never judged short.
                diag("stdout", std::format("door_concept: [existence] {} obs_top={:.2f} m (conf {:.2f}, trunc {:.2f}) min={:.2f} m\n",
                           inst.node_name, inst.obs_top_z, inst.obs_top_conf, inst.last_trunc_frac,
                           cfg.exist_min_height_m), out);
                // Same diagnostic to a CSV. How to read it: ndet=0 ⇒ the door was NOT looked at (out of frustum
                // or fully occluded) ⇒ HOLD, L must not move — that is the "robot turned around" case. A real
                // door in view shows occ≫free. A phantom in a resolving view (pdet high) shows free≫occ and a
                // rising strk. free_eff≪free means "seen, but this view could not resolve it" ⇒ go verify, not delete.
                // (The header — kExistenceCsvHeader, with its column notes — is written by the adapter, which owns the
                //  file; the row is formatted HERE exactly as the old std::ofstream formatted it: operator<<, default
                //  precision, the classic locale — the global C++ locale the old stream used.)
                std::ostringstream ex_csv;
                ex_csv.imbue(std::locale::classic());
                {
                    ex_csv << ex_dbg_ << ',' << inst.node_name << ',' << inst.existence.logodds() << ','
                           << inst.existence.p_exists() << ',' << ms.cx << ',' << ms.cy
                           << ',' << (inroom ? 1 : 0) << ',' << (has_poly ? 1 : 0)
                           << ',' << (inst.assigned_mask_idx >= 0 ? 1 : 0) << ',' << inst.frames_since_detection
                           << ',' << inst.dbg_sil_occ << ',' << inst.dbg_sil_free << ',' << inst.dbg_sil_free_eff
                           << ',' << inst.dbg_sil_ndet << ',' << inst.dbg_sil_ntotal << ',' << inst.dbg_sil_noccl
                           << ',' << inst.dbg_sil_resolv << ',' << inst.dbg_sil_central << ',' << inst.dbg_sil_pdetect
                           << ',' << oblq << ',' << inst.existence_debounce.streak
                           << ',' << inst.obs_top_z << ',' << inst.obs_top_conf << ',' << inst.last_trunc_frac
                           // gated=1 with gate_fresh=0 is the stale-verdict trap: the flag is left over from the
                           // last frame that carried a mask, and must NOT suppress absence (see view_untrustworthy).
                           << ',' << (inst.dbg_gated ? 1 : 0) << ',' << (inst.dbg_gate_fresh ? 1 : 0)
                           << ',' << inst.dbg_field_support << ',' << inst.dbg_field_mean
                           << ',' << inst.dbg_field_bg << ',' << inst.dbg_field_n
                           << ',' << inst.dbg_edge_support << ',' << inst.dbg_edge_true
                           << ',' << inst.dbg_edge_ctrl << ',' << inst.dbg_edge_n
                           << ',' << inst.dbg_edge_delta << ',' << inst.dbg_edge_excess
                           << ',' << inst.dbg_edge_nctl
                           << ',' << inst.dbg_depth_verdict << ',' << inst.dbg_depth_bias_m
                           << ',' << inst.dbg_depth_n
                           << ',' << inst.phi_est << ',' << inst.phi_support
                           << ',' << (inst.phi_cmd_active ? 1 : 0)
                           << ',' << (inst.dbg_phi_marg ? 1 : 0) << ',' << inst.dbg_phi_nhyp
                           << ',' << inst.dbg_phi_wmax
                           << ',' << inst.dbg_frame_n << ',' << inst.dbg_frame_vis << ',' << inst.dbg_frame_excess
                           << ',' << inst.dbg_frame_true << ',' << inst.dbg_frame_ctrl
                           << ',' << inst.dbg_frame_depth << ',' << inst.dbg_frame_ndepth
                           << ',' << inst.dbg_band_delta << ',' << inst.dbg_band_p
                           << ',' << inst.roi_fill << ',' << inst.roi_fill_h << ',' << inst.roi_fill_v
                           << ',' << (inst.roi_valid ? 1 : 0) << ',' << inst.dbg_p_env << '\n';
                }
                diag("door_existence_log.csv", ex_csv.str(), out);
            }

    // The removals: the decision is the core's; the graph half is the world's (TrackerWorld::before_remove /
    // remove_door), called where the old loop made those calls, so the decision log keeps its order: remember_ghost
    // (and the stored identity it forgets) + the affordance-withdrawal rows BEFORE the REMOVE row; the phantom DEATH
    // record, the affordance node removals, forget_node and the node delete AFTER it.
    for (const std::uint64_t id : to_remove)
    {
        const auto it = std::ranges::find(states, id, &std::pair<std::uint64_t, rc::DoorFitState*>::first);
        const bool found = it != states.end();
        const float L = found ? it->second->existence.logodds() : 0.0f;
        diag("stdout", std::format("door_concept: [existence] REMOVE id={} (L={:.2f}, p={:.3f} < {:.3f} for {} evidence cycles)\n",
                                   id, L, 1.0f / (1.0f + std::exp(-L)), cfg.exist_removal_prob, cfg.exist_remove_frames), out);
        if (found)
        {
            // Retain the identity before deleting the node: a door that comes back at the same place is the SAME
            // door, and must resume its accumulated belief instead of being re-born as door_N+1 (see remember_ghost).
            world_->before_remove(id);
            const auto& s = it->second->model.state();
            record(K::Remove, id, s.cx, s.cy, std::format("logodds {:.2f}", L), it->second->node_name, out);
        }
        out.deletes.push_back({id, "existence"});
        world_->remove_door(id, L);
    }
}

}  // namespace rc::door::core
