// door_existence.cpp — see door_existence.h. The probes are DoorFitter's (door_fitter.cpp as of 8a1abeac) and
// DoorCore::update_existence is SpecificWorker::update_existence_beliefs (specificworker.cpp as of 8a1abeac), moved
// with their comments; only the sources of the values and the destinations of the outputs changed (door_existence.h).
#include "door_existence.h"
#include "door_core.h"
#include "door_visibility.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <locale>
#include <numbers>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgproc.hpp>
#include <Eigen/Eigenvalues>

#include "../../../common/instance_tracker/birth_evidence.h"   // rc::birth::observation_weight (SHARED): the ricoh channel's P(detect)
#include "../door_geometry.h"
#include "../door_leaf_visibility.h"                     // leaf_closedness / ricoh_detect_scale (Task D2)
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

// MaskCells, build_mask_cells and own_wall_edges live in door_visibility.{h,cpp} (shared with door_view).

}  // namespace

// ★Task 8 (spec §3.8): ONE ROW PER DOOR PER FRESH FRAME, every channel's contribution THIS cycle; `nan` = not computed this
// cycle (never a stale value). L = prior_band + prior_base + prior_size + carried + crossing + ledger. zed_dl / ric_dl / geo_dl /
// act_dl: what each channel changed the ledger sum by this cycle; view_id / ric_view_id: the ledger record the frame joined;
// zed_q / ric_q: P(detector reports the door from this view); zed_hit / ric_hit: this frame's association weight (0 = a miss);
// geo_x / geo_mu0 / geo_mu1 / geo_s: the contour statistic, the controls' mean, the present-hypothesis mean and the controls' sd.
const char* const kExistenceCsvHeader =
    "cycle,node,L,p_exists,cx,cy,role,delete_line,view_id,ric_view_id,"
    "prior_band,prior_base,prior_size,carried,crossing,ledger,n_views,"
    "zed_dl,ric_dl,geo_dl,act_dl,zed_q,ric_q,zed_hit,ric_hit,kappa,geo_x,geo_mu0,geo_mu1,geo_s\n";

// Ruling R8: the per-view rows tools/fit_detector_envelopes.py reads (its column names, exactly). Every field is
// THIS frame's (hit = its association weight, 0 = a miss); door = the node name or place_<key> of a pending candidate.
const char* const kViewsCsvHeader = "cycle,door,camera,view_id,q,fill_h,fill_v,vis,kappa,aspect,hit\n";

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
        ++rgb_gen_;
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

void ImageSource::stage_live(const cv::Mat& bgr, const cv::Mat& depth_m, std::uint64_t stamp_ms)
{
    if (not bgr.empty() and (stamp_ms == 0 or stamp_ms != last_stamp_)) ++rgb_gen_;
    last_stamp_ = stamp_ms;
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
    return {rc::card::term_kappa(var_b, range, static_cast<double>(half_diag_m)), static_cast<float>(sb * range)};
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
    const auto& door_cells     = cells.door;

    const auto& s = inst.model.state();
    const float hw = inst.leaf_pose.half_w;

    // The visibility rules (frustum, masks, walls, other doors, the leaf beyond the aperture) are door_visibility.cpp's ViewRig —
    // the SAME classification door_view() uses for any camera; this function keeps only the occupied/free bookkeeping.
    // (The long comment on the leaf beyond the aperture moved with the code: door_visibility.cpp, ViewRig::ViewRig.)
    const CameraSpec cam_spec = camera_spec(sc.cam);
    const CameraView view{cam_spec, Mopt.value(), {}, false};
    const ViewRig rig(view, sc, inst, cells);

    // Classify ONE LEAF-face sample (local x hinge→free edge, z absolute): ViewRig::at decides whether the camera sees it;
    // this tallies it.
    //
    // ★ The sample comes from door::leaf_point at the instance's CURRENT leaf pose (see ViewRig::at). With phi pinned at 0 (M0)
    // this yields the wall-plane rectangle; when phi is a fitted DOF it follows the leaf for free.
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
        const SampleView sv = rig.at(L, lx, lz);
        if (sv.fate == SampleFate::Unseen) return;                      // behind / out of the REAL frustum ⇒ not detectable
        if (sv.fate == SampleFate::Occluded) { ++out.n_occluded; return; }   // hidden ⇒ no vote
        const float col = sv.col, row = sv.row, w_room = sv.w_room;
        const double X = sv.X, Y = sv.Y, Z = sv.Z;
        const std::int64_t k = sv.key;
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
    constexpr int NX = kLeafGridNX, NZ = kLeafGridNZ;
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

// P(detect | present) of the APERTURE from this camera: the shared detector envelope (rc::detect) at the doorway's apparent
// size — the extent a virtual camera with the ZED's intrinsics aimed at the aperture centre would see, f·size/range per axis,
// the horizontal one foreshortened by the obliquity of the view to the wall. The doorway frame's detectability (Task 15): it
// does not depend on the leaf, so an open door keeps it; standing in or next to the opening overflows the envelope (→ 0), far
// away it falls under the min-fill shoulder. 0 without a camera pose or intrinsics.
float aperture_envelope(const std::optional<Mat4d>& room_T_zed, const CameraIntrinsics& cam, const rc::DoorFitState& inst,
                        const rc::detect::DetectorEnvelope& env)
{
    if (not room_T_zed.has_value() or not camera_usable(cam) or not inst.ai2_initialized)
        return 0.0f;
    const auto& ap = inst.aperture;
    const Eigen::Vector3f C(static_cast<float>(room_T_zed->coeff(0, 3)), static_cast<float>(room_T_zed->coeff(1, 3)),
                            static_cast<float>(room_T_zed->coeff(2, 3)));
    const Eigen::Vector2f c2 = ap.centre_xy();
    const Eigen::Vector3f r = Eigen::Vector3f(c2.x(), c2.y(), ap.centre_z()) - C;
    const float d = r.norm();
    if (not std::isfinite(d) or d < 1e-3f)
        return 0.0f;
    const Eigen::Vector3f fwd(static_cast<float>(room_T_zed->coeff(0, 1)), static_cast<float>(room_T_zed->coeff(1, 1)),
                              static_cast<float>(room_T_zed->coeff(2, 1)));   // the ZED's forward axis (+y) in the room
    if (not (r.dot(fwd) > 0.0f))
        return 0.0f;   // behind the camera: nothing to frame
    const Eigen::Vector2f rh(r.x(), r.y());
    const float oblq = rh.norm() > 1e-6f ? std::abs(rh.normalized().dot(ap.across_u())) : 1.0f;
    const float fill_h = cam.focal_x * ap.w * oblq / (d * static_cast<float>(cam.width));
    const float fill_v = cam.focal_y * ap.h / (d * static_cast<float>(cam.height));
    return rc::detect::p_detect(std::max(fill_h, fill_v), 1.0f, env);
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
    const Eigen::Vector2f r = inst.leaf_pose.centre_xy - cam;
    return rc::card::term_leaf_obliquity(r.x(), r.y(), inst.leaf_pose.ey.x(), inst.leaf_pose.ey.y());   // ey = the leaf's face normal
}

float aperture_band_offset(const std::vector<Eigen::Vector2f>& polygon, const rc::DoorFitState& inst)
{
    if (polygon.size() < 3)
        return std::numeric_limits<float>::quiet_NaN();
    const auto& ms = inst.model.state();
    return rc::door::signed_boundary_distance(polygon, Eigen::Vector2f(ms.ap_cx, ms.ap_cy));
}

// ═══ THE UPDATE (Task 8 of the door-existence model: rewritten, no branch of the old function ported) ═══════════════
// ONE belief per door place: L = prior + carried + crossing + Σ_views ℓ_v (place_belief.h). Per fresh frame (a new mask packet,
// a fresh ricoh frame in it, or a new RGB frame — ruling R3) every door's and every pending candidate's views receive this
// frame through the SAME likelihoods (existence_update.h), and the costed decision (door_decision.h) creates and deletes.

namespace
{
std::string fmt_or_dash(float v) { return std::isfinite(v) ? std::format("{:+.2f}", v) : std::string("—"); }
const char* role_name(Role r) { return r == Role::BeingCrossed ? "crossing" : r == Role::Route ? "route" : "ordinary"; }
}  // namespace

std::vector<FrameDetection> DoorCore::frame_detections() const
{
    std::vector<FrameDetection> v;
    v.reserve(dets_.size());
    for (std::size_t i = 0; i < dets_.size(); ++i)
        v.push_back({dets_[i].xy, dets_[i].xy_var, dets_[i].sensor, i < det_w_.size() ? det_w_[i] : 1.0f,
                     i < det_adm_.size() ? det_adm_[i] != 0 : true});
    return v;
}

void DoorCore::with_frame(const DoorCycleInput& in, bool zed_ran, bool use_rgb, const std::function<void(const ExistenceFrame&)>& f)
{
    const auto& cfg = cfg_.door;
    const auto& pkt = masks_.packet();
    const auto states = world_->fit_states();
    const std::optional<Mat4d> room_T_zed = CameraModel::room_T_zed(in.robot);
    const ExistenceScene scene{cfg, room_T_zed, in.camera, pkt, fit_.room_polygon(), fit_.room_band(), states};
    rc::edges::PreparedFrame rgb_prep;
    if (use_rgb and cfg.rgb_contour_check and not images_.rgb().empty())
    {
        rgb_prep = rc::edges::prepare_frame(images_.rgb());
        if (images_.frame_ref().has_value()) rgb_prep.frame_ref = images_.frame_ref().value();
    }
    if (sinks_.image_use) sinks_.image_use({rgb_prep.valid(), rgb_prep.frame_ref, false});
    const std::vector<FrameDetection> dets = frame_detections();
    const DetectorTable& det = detectors_;   // fixed within the session (fix round 3)
    ExistenceFrame fr{scene, cfg, det};
    if (camera_usable(in.camera)) fr.zed_spec = camera_spec(in.camera);
    fr.zed_ran = zed_ran and fr.zed_spec.has_value();
    if (zed_ran and in.ricoh.present and in.ricoh.width > 0 and in.ricoh.height > 0 and ricoh_fresh(pkt)
        and in.robot.room_T_body.has_value() and in.robot.body_T_ricoh.has_value())
    {
        fr.ricoh_ran = true;
        fr.ricoh_spec = camera_spec(in.ricoh);
        fr.room_T_ricoh = *in.robot.room_T_body * *in.robot.body_T_ricoh;
    }
    else if (zed_ran and ricoh_fresh(pkt) and in.robot.room_T_body.has_value())
    {
        fr.ricoh_hits_only = true;
        fr.room_T_ricoh = *in.robot.room_T_body;
    }
    fr.rgb = rgb_prep.valid() ? &rgb_prep : nullptr;
    fr.rt_cov = in.robot.rt_cov;
    fr.geometry_may_refute = cfg.exist_geometry_may_refute;
    // fr.geo_mu1_sd stays at its starting value (μ1 = μ0 + 2s): fixed within the session like the detector rates (fix round 3)
    fr.dets = dets;
    if (sinks_.image_roi)
        fr.note_poly = [this](const std::vector<cv::Point>& poly) { sinks_.image_roi({poly, {}, 0, true, false}); };
    f(fr);
}

// The per-frame view rows to etc/door_views_log.csv (one row per camera per frame joining a view; the fit groups them by view).
void DoorCore::log_view_rows(const std::string& place, const std::vector<ViewRow>& rows, DoorCycleOutput& out)
{
    for (const auto& r : rows)
    {
        std::ostringstream row;
        row.imbue(std::locale::classic());
        row << ex_dbg_ << ',' << place << ',' << (r.camera == 0 ? "zed" : "ricoh") << ',' << r.view_id << ','
            << (r.q_known ? r.q : std::numeric_limits<float>::quiet_NaN()) << ',' << r.fill_h << ',' << r.fill_v << ',' << r.vis << ','
            << r.kappa << ',' << r.aspect << ',' << r.hit << '\n';
        diag("door_views_log.csv", row.str(), out);
    }
}

// C_keep's samples (spec §8.2): an executing episode on one of a door's affordances that ends without the robot crossing it.
// An episode spans the cycles in which ANY of the door's aff_<door>_* affordances is executed; it is crossed when the door is
// backed by a crossing fact while it lasts, or the room changes under it.
void DoorCore::track_episodes(const DoorCycleInput& in)
{
    std::set<std::string> now;
    for (const auto& e : in.executing)
        for (const auto& a : in.affordances)
            if (a.node_id == e.to and a.name.starts_with("aff_door_"))
                if (const auto k = a.name.rfind('_'); k != std::string::npos and k > 4) now.insert(a.name.substr(4, k - 4));
    const auto room = world_->room_id();
    for (const auto& n : now)
        episodes_.try_emplace(n, Episode{in.t_wall_ms, false, room});
    for (auto& [name, ep] : episodes_)
    {
        if (ep.room != room) ep.crossed = true;
        for (const auto& [id, st] : world_->fit_states())
            if (st->node_name == name and std::ranges::contains(in.backed_faces, id)) ep.crossed = true;
    }
    for (auto it = episodes_.begin(); it != episodes_.end();)
        if (not now.contains(it->first))
        {
            if (not it->second.crossed and in.t_wall_ms > it->second.start_wall_ms)
                costs_.add(CostEstimator::Kind::Keep, static_cast<float>(in.t_wall_ms - it->second.start_wall_ms) * 1e-3f, "online");
            it = episodes_.erase(it);
        }
        else ++it;
}

// The candidates (spec §3.6, birth by comparison): every birthable detection that no live door's gate explains belongs to a
// pending place; each candidate in the room receives this frame like a door does, and its verdict is the birth evidence.
void DoorCore::prepare_candidates(const DoorCycleInput& in, const std::vector<rc::TrackView>& tracks, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    det_candidate_.assign(dets_.size(), -1);
    for (auto& d : dets_) d.birth_evidence = 0.0f;
    if (not info_.new_frame)
        return;
    const auto gated = [&](const Vec2f& xy, float var)
    {
        for (const auto& t : tracks)
        {
            const Vec2f e = xy - t.xy;
            if (t.has_cov)
            {
                const Eigen::Matrix2f S = t.cov + (var >= 0.0f ? var : cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m)
                                                  * Eigen::Matrix2f::Identity();
                if (const float m2 = e.dot(S.inverse() * e); std::isfinite(m2) and m2 <= cfg.tracker_gate_mahalanobis) return true;
            }
            else if (e.squaredNorm() <= cfg.tracker_gate_fallback_m * cfg.tracker_gate_fallback_m) return true;
        }
        return false;
    };
    // a candidate a live door now explains is that door (born, restored or carried meanwhile): it is no longer pending
    std::erase_if(candidates_, [&](const Candidate& c) { return c.room == room_name_ and gated(c.xy, c.var); });
    const auto& pkt = masks_.packet();
    const auto& polygon = fit_.room_polygon();
    const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    std::unordered_map<std::size_t, int> self_det;   // candidate index → the detection that is its own this frame
    for (std::size_t d = 0; d < dets_.size(); ++d)
    {
        const auto& dv = dets_[d];
        if (not dv.birthable or gated(dv.xy, dv.xy_var)) continue;
        const float var = dv.xy_var >= 0.0f ? dv.xy_var : rn2;
        int best = -1; float best_m2 = cfg.tracker_gate_mahalanobis;
        for (std::size_t c = 0; c < candidates_.size(); ++c)
        {
            if (candidates_[c].room != room_name_) continue;
            const float m2 = (candidates_[c].xy - dv.xy).squaredNorm() / std::max(1e-9f, candidates_[c].var + var);
            if (m2 <= best_m2) { best_m2 = m2; best = static_cast<int>(c); }
        }
        if (best < 0)
        {
            // A new place: the typical door at the blob, with the blob's width along its wall (its support's extent).
            const rc::DoorFitState probe = typical_door_at(dv.xy, cfg.door_prior_w_m, polygon);
            const Vec2f u = probe.aperture.wall_u;
            float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
            for (const auto& p : masks_.support(pkt.slices[static_cast<std::size_t>(dv.slice_index)]))
            { const float a = Vec2f(p.x(), p.y()).dot(u); lo = std::min(lo, a); hi = std::max(hi, a); }
            const float width = hi > lo ? hi - lo : cfg.door_prior_w_m;
            Candidate c;
            c.key = ++next_candidate_key_;
            c.room = room_name_;
            c.xy = dv.xy;
            c.var = var;
            c.theta = typical_door_at(dv.xy, width, polygon);
            // ★The memory of this place is the candidate's starting evidence (spec §3.7/§8.3): a matched row continues its
            //  ledger, aged by the change hazard; a REMOVED row ("no door here") pushes it down by exactly its aged L.
            if (const PlaceRow* row = places_.match(room_name_, dv.xy, Eigen::Matrix2f::Identity() * var, cfg.tracker_gate_mahalanobis))
            {
                c.belief.ledger = row->ledger;
                c.belief.carried = places_.aged_L(*row, in.t_wall_ms) - row->ledger.sum();
                c.seeded_from_removed_ms = row->removed_wall_ms;
            }
            candidates_.push_back(std::move(c));
            best = static_cast<int>(candidates_.size() - 1);
        }
        det_candidate_[d] = best;
        self_det.try_emplace(static_cast<std::size_t>(best), static_cast<int>(d));
    }
    // every pending candidate in this room receives the frame (hits from its own detection, misses where it should be seen)
    const auto groups = wall_groups(polygon, fit_.room_band().sigma_m);
    std::unordered_map<int, int> doors_on_wall;
    for (const auto& [id, st] : world_->fit_states())
        if (not st->is_bearing_hypothesis) ++doors_on_wall[wall_of(st->aperture.centre_xy(), polygon, groups)];
    const Costs costs = costs_.costs(cfg.exist_max_room_s);
    std::vector<char> exists(candidates_.size(), 0);
    with_frame(in, true, false, [&](const ExistenceFrame& fr)
    {
        for (std::size_t c = 0; c < candidates_.size(); ++c)
        {
            Candidate& cand = candidates_[c];
            if (cand.room != room_name_) continue;
            const int wall = wall_of(cand.theta.aperture.centre_xy(), polygon, groups);
            cand.belief.prior = prior_of(cand.theta, polygon, fit_.room_is_proto(), fit_.room_band(),
                                         wall >= 0 ? doors_on_wall[wall] : 0, cfg.tracker_detection_noise_m, 0.0f, true);
            const auto it = self_det.find(c);
            const int self_sensor = it == self_det.end() ? -1 : (dets_[static_cast<std::size_t>(it->second)].sensor != 0 ? 1 : 0);
            const float self_w = it == self_det.end() ? 0.0f : det_w_[static_cast<std::size_t>(it->second)];
            std::vector<ViewRow> rows;
            observe_place(cand.belief, cand.theta, fr, visit_, 0.0f, &rows, self_sensor, self_w);
            log_view_rows(std::format("place_{}", cand.key), rows, out);   // logged (no selection by birth), never learned from
            const Verdict v = decide(cand.belief.L(), Role::Ordinary, costs);
            exists[c] = v.exists ? 1 : 0;
            if (it != self_det.end() and (cand.belief.ledger.views().size() <= 1 or ex_dbg_ % 60 == 0))   // its first views, then 1 in 60
                diag("stdout", std::format("door_concept: [candidate] place_{} at ({:.2f},{:.2f}) w={:.2f} L={:+.2f} (band {:+.2f} base {:+.2f} "
                                           "size {:+.2f} carried {:+.2f} views {}) -> {}\n", cand.key, cand.xy.x(), cand.xy.y(),
                                           cand.theta.aperture.w, cand.belief.L(), cand.belief.prior.band, cand.belief.prior.base_rate,
                                           cand.belief.prior.size, cand.belief.carried, cand.belief.ledger.views().size(),
                                           v.exists ? "EXISTS" : "not born"), out);
            // ★A pending candidate below the line is simply NOT BORN (fix round 1, review I-3): it has no node to remove, so it
            //  writes no REMOVED row, books no C_delete and retires no false-alarm trials. Only a removed DOOR does those.
        }
    });
    // ruling R2: the model decides. ONE detection per candidate carries its verdict; the candidate's other detections this frame (the
    // other camera, a split mask) may not seed a second birth of the same place (the tracker's promoted candidates are not in its
    // same-frame duplicate test), so they are not birthable this frame — they still associate.
    for (std::size_t d = 0; d < dets_.size(); ++d)
        if (const int c = det_candidate_[d]; c >= 0)
        {
            const bool own = self_det.at(static_cast<std::size_t>(c)) == static_cast<int>(d);
            dets_[d].birth_evidence = own and exists[static_cast<std::size_t>(c)] ? 1.0f : 0.0f;
            if (not own) dets_[d].birthable = false;
        }
}

void DoorCore::update_existence(const DoorCycleInput& in, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    const auto& pkt = masks_.packet();
    const bool zed_fresh = pkt.valid and static_cast<int>(pkt.frame_id) != exist_last_mask_frame_;
    const bool rgb_fresh = cfg.rgb_contour_check and not images_.rgb().empty() and images_.rgb_generation() != last_rgb_gen_;
    if (not zed_fresh and not rgb_fresh)
        return;
    if (zed_fresh) exist_last_mask_frame_ = static_cast<int>(pkt.frame_id);
    last_rgb_gen_ = images_.rgb_generation();
    ++ex_dbg_;

    // The door being crossed (the controller's `executing` names its cross) or backed by a crossing fact: Role::BeingCrossed —
    // its evidence still accumulates, only its deletion is barred (spec §3.3; the decision rule carries the protection).
    std::set<std::string> being_crossed;
    for (const auto& e : in.executing)
        for (const auto& a : in.affordances)
            if (a.node_id == e.to and a.name.starts_with("aff_") and a.name.ends_with("_cross"))
                being_crossed.insert(a.name.substr(4, a.name.size() - 4 - 6));

    const auto states = world_->fit_states();
    const auto& polygon = fit_.room_polygon();
    const auto groups = wall_groups(polygon, fit_.room_band().sigma_m);
    for (const auto& [id, st] : states) door_seq_.try_emplace(id, next_door_seq_++);
    const Costs costs = costs_.costs(cfg.exist_max_room_s);
    std::vector<std::uint64_t> to_remove;
    std::vector<float> L_before(states.size());
    with_frame(in, zed_fresh, rgb_fresh, [&](const ExistenceFrame& fr)
    {
        std::vector<int> wall(states.size(), -1);
        for (std::size_t i = 0; i < states.size(); ++i)
            wall[i] = wall_of(states[i].second->aperture.centre_xy(), polygon, groups);
        for (std::size_t i = 0; i < states.size(); ++i)
        {
            const auto& [id, instp] = states[i];
            rc::DoorFitState& inst = *instp;
            inst.exist_trace = {};
            L_before[i] = inst.belief.L();
            if (inst.is_bearing_hypothesis) continue;   // no depth: no place to judge
            inst.role = door_role(inst.node_name, in.route_faces, being_crossed.contains(inst.node_name) or inst.fact_backed);
            // the structural prior, REPLACED (never accumulated); a newborn not yet fitted keeps its candidate's
            if (inst.ai2_initialized)
            {
                int before = 0;
                for (std::size_t j = 0; j < states.size(); ++j)
                    if (j != i and wall[j] == wall[i] and wall[i] >= 0 and not states[j].second->is_bearing_hypothesis
                        and door_seq_[states[j].first] < door_seq_[id])
                        ++before;
                const auto C = inst.ai2_belief.covariance();
                inst.belief.prior = prior_of(inst, polygon, fit_.room_is_proto(), fit_.room_band(), before,
                                             std::sqrt(std::max(0.0f, C(1, 1))), std::sqrt(std::max(0.0f, C(2, 2))), true);
            }
            const float act = std::exchange(inst.pending_actuation_llr, 0.0f);
            std::vector<ViewRow> rows;
            // ★A newborn's belief already holds its birth frame (its candidate observed it): it may not enter twice (fix round 1).
            ExistenceFrame fr_door = fr;
            if (const auto of = observed_frame_.find(id); of != observed_frame_.end() and of->second == pkt.frame_id)
                fr_door.zed_ran = fr_door.ricoh_ran = fr_door.ricoh_hits_only = false;
            inst.exist_trace = observe_place(inst.belief, inst, fr_door, visit_, act, &rows);
            if (act != 0.0f and not std::isfinite(inst.exist_trace.act_dl))
                inst.pending_actuation_llr += act;   // no view and no ZED pose this cycle: the evidence waits (fix round 1)
            const float L = inst.belief.L();
            const Verdict v = decide(L, inst.role, costs);
            inst.exist_trace.delete_line = v.delete_line;
            if (v.remove)
                to_remove.push_back(id);
            log_view_rows(inst.node_name, rows, out);
        }
    });

    // Decision log: every door's belief after this update (the input of the REMOVE below). a = L, b = P, c = x, d = y.
    for (const auto& [id, instp] : states)
        if (const auto& inst = *instp; not inst.is_bearing_hypothesis)
        {
            rc::door::DoorDecision d{out.cycle, out.t_ms, static_cast<std::uint16_t>(K::Existence), id, inst.node_name,
                                     inst.belief.L(), inst.belief.p(), inst.model.state().cx, inst.model.state().cy, {}};
            if (sinks_.decision) sinks_.decision(d);
            out.decisions.push_back(std::move(d));
        }
    // The readout: one CSV row per door per fresh frame (spec §3.8), stdout throttled (1 in 60, plus every cycle on which a
    // door moved by more than one nat or is removed) — the CSV is the record, stdout the operator's view.
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        const auto& [id, instp] = states[i];
        const auto& inst = *instp;
        if (inst.is_bearing_hypothesis) continue;
        const auto& b = inst.belief;
        const auto& t = inst.exist_trace;
        const auto& ms = inst.model.state();
        std::ostringstream ex_csv;
        ex_csv.imbue(std::locale::classic());
        ex_csv << ex_dbg_ << ',' << inst.node_name << ',' << b.L() << ',' << b.p() << ',' << ms.cx << ',' << ms.cy << ','
               << role_name(inst.role) << ',' << t.delete_line << ',' << t.zed_view << ',' << t.ric_view << ','
               << b.prior.band << ',' << b.prior.base_rate << ',' << b.prior.size << ',' << b.carried << ',' << b.crossing << ','
               << b.ledger.sum() << ',' << b.ledger.views().size() << ',' << t.zed_dl << ',' << t.ric_dl << ',' << t.geo_dl << ','
               << t.act_dl << ',' << t.zed_q << ',' << t.ric_q << ',' << t.zed_hit << ',' << t.ric_hit << ',' << t.kappa << ','
               << t.geo_x << ',' << t.geo_mu0 << ',' << t.geo_mu1 << ',' << t.geo_s << '\n';
        diag("door_existence_log.csv", ex_csv.str(), out);
        const bool removing = std::ranges::contains(to_remove, id);
        if (ex_dbg_ % 60 == 0 or removing or std::abs(b.L() - L_before[i]) > 1.0f)
            diag("stdout", std::format("door_concept: [existence] {} L={:+.2f} p={:.2f} role={} delete_line={} | prior band={:+.2f} "
                                       "base={:+.2f} size={:+.2f} carried={:+.2f} cross={:+.2f} | views={} | zed dl={} q={} hit={} "
                                       "| ric dl={} q={} hit={} | geo dl={} | act dl={} | kappa={}\n",
                                       inst.node_name, b.L(), b.p(), role_name(inst.role), fmt_or_dash(t.delete_line),
                                       b.prior.band, b.prior.base_rate, b.prior.size, b.carried, b.crossing,
                                       b.ledger.views().size(), fmt_or_dash(t.zed_dl), fmt_or_dash(t.zed_q), fmt_or_dash(t.zed_hit),
                                       fmt_or_dash(t.ric_dl), fmt_or_dash(t.ric_q), fmt_or_dash(t.ric_hit), fmt_or_dash(t.geo_dl),
                                       fmt_or_dash(t.act_dl), fmt_or_dash(t.kappa)), out);
    }
    if (ex_dbg_ % 600 == 0)
    {
        const auto& T = detectors_;
        const auto c = costs_.costs(cfg.exist_max_room_s);
        diag("stdout", std::format("door_concept: [existence] model (fixed this session): zed hit={:.3f} fa={:.4f} (n={:.0f}) | ricoh "
                                   "hit={:.3f} fa={:.4f} (n={:.0f}) | C_keep={:.1f}s (n={}) C_delete={:.1f}s (n={})\n",
                                   T.zed.hit_rate, T.zed.false_alarm, T.zed.n_hit_trials, T.ricoh.hit_rate, T.ricoh.false_alarm,
                                   T.ricoh.n_hit_trials, c.keep_s,
                                   costs_.n(CostEstimator::Kind::Keep), c.delete_ordinary_s, costs_.n(CostEstimator::Kind::DeleteOrdinary)), out);
    }

    // The removals: the decision is the core's; the graph half is the world's (before_remove / remove_door), in the order the
    // decision log has always had. The place's row is kept with its removal time: "no door here" (spec §3.7).
    for (const std::uint64_t id : to_remove)
    {
        const auto it = std::ranges::find(states, id, &std::pair<std::uint64_t, rc::DoorFitState*>::first);
        if (it == states.end()) continue;
        const rc::DoorFitState& inst = *it->second;
        const float L = inst.belief.L();
        diag("stdout", std::format("door_concept: [existence] REMOVE id={} {} (L={:.2f} < delete line {:.2f}, role {})\n", id,
                                   inst.node_name, L, inst.exist_trace.delete_line, role_name(inst.role)), out);
        PlaceRow r;
        r.room = room_name_; r.name = inst.node_name; r.xy = inst.aperture.centre_xy();
        const float sd2 = inst.ai2_initialized ? std::max(0.0f, inst.ai2_belief.covariance()(0, 0)) : cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
        r.cov = Eigen::Matrix2f::Identity() * sd2;
        r.yaw = inst.aperture.yaw(); r.width = inst.aperture.w; r.height = inst.aperture.h; r.depth = inst.aperture.thickness;
        r.L_evidence = inst.belief.L_evidence(); r.ledger = inst.belief.ledger; r.visit = visit_;
        r.last_seen_wall_ms = in.t_wall_ms; r.removed_wall_ms = in.t_wall_ms;
        places_.upsert(std::move(r));
        observed_frame_.erase(id);
        world_->before_remove(id);
        const auto& s = inst.model.state();
        record(K::Remove, id, s.cx, s.cy, std::format("logodds {:.2f}", L), inst.node_name, out);
        out.deletes.push_back({id, "existence"});
        world_->remove_door(id, L);
    }
}

}  // namespace rc::door::core
