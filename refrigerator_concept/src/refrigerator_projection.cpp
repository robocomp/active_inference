/*
 * refrigerator_projection.cpp — camera-projection unit for refrigerator_concept (extracted from RefrigeratorFitter).
 *
 * Implements the camera→room extrinsic (room_T_zed, room→body pinned to a capture stamp), the model-box →
 * normalised in-image ROI used by the controller's lock-on, and the PIXEL-LEVEL silhouette existence evidence
 * (project the solid box's camera-facing vertical faces, then over the predicted-visible pixels vote occupancy /
 * absence / occlusion against the current YOLO foreground). Owns the ZED CameraAPI (lazily bound to the "zed"
 * node) and uses the shared graph, InnerEigenAPI, and MaskIngestor packet source.
 */

#include "refrigerator_projection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_set>
#include <vector>

#include "../../common/occlusion/occlusion.h"   // rc::occlusion::walls_block — shared with door/table/cabinet

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rc {

namespace {
// Door-ness metric: AREA-NORMALISED AXIS-ALIGNED SEAM energy of an upright grayscale face patch — the mean of
// |Sobel_x| (vertical lines: door jamb, handle) PLUS |Sobel_y| (horizontal lines: the freezer/fridge door split,
// the gasket seam). A door face carries strong axis-aligned structure in AT LEAST ONE direction; a plain side
// panel or a bare wall is flat in both. Fixed warp size ⇒ the mean is comparable across faces.
//
// ★It used to be |Sobel_x| ONLY, i.e. blind to horizontal seams. On the apartment fridge the most salient (often
// the only) visible cue is the HORIZONTAL line separating the two doors, which scores exactly 0 under ∂/∂x — so
// the door face never beat a blank side, the margin stayed under FrontMinConfidence, and the cue was suppressed.
// With a square footprint (|w−h| ≈ 8 mm live) yaw is geometrically unobservable mod 90°, so losing this cue means
// the quarter-turn is never resolved and the fridge renders rotated 90°. See [[refrigerator-square-yaw-door-cue]].
float door_ness(const cv::Mat& gray)
{
    if (gray.empty())
        return 0.0f;
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);      // ∂/∂x → responds to VERTICAL lines
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);      // ∂/∂y → responds to HORIZONTAL lines (the door split)
    return static_cast<float>(cv::mean(cv::abs(gx))[0] + cv::mean(cv::abs(gy))[0]);
}

// DOOR SEAM detector: is there a door-separating line across THIS face? Returns a SELF-NORMALISING strength —
// the dominant horizontal ridge measured against the same patch's own typical row — so it carries an absolute
// meaning that mean gradient energy cannot. ~0 = a blank panel or a wall, ≫1 = a fridge front.
//
// ★Why this and not door_ness: mean |Sobel| has no absolute scale (it moves with lighting, albedo and texture),
// so it is only usable as a RELATIVE margin between two faces — which is why detect_front demanded ≥2 visible
// faces and emitted nothing for a head-on, wall-flush fridge (etc/fridge_1.png: `refrigerator 0.95`, one face,
// seam plainly visible, cue suppressed). A seam ratio is scale-free: dividing the peak row by the median row
// cancels illumination, so ONE face is enough to say "this face is the door" — and that, unlike a comparison
// between visible faces, resolves the 180° front/back ambiguity, because the back panel has no seam.
//
//   rowprof[r] = mean_x |∂I/∂y|                 — horizontal-line energy per row
//   contrast   = max_r rowprof / mean intensity — WEBER contrast: invariant to multiplicative lighting, which
//                                                 is the invariance a camera actually needs, and BOUNDED
//   isolation  = 1 − median/peak ∈ [0,1]        — →1 for one dominant ridge, →0 for general texture
//   coverage   = fraction of columns responding at the winning row — a real seam spans the door
//   score      = contrast · isolation · coverage
// ★NOT peak/median: on a smooth panel the median row energy is ~0, so that ratio explodes and a BLANK back
// panel scored as high as a door (verified — it is why this is written the way it is). Weber contrast has a
// real denominator, so "no seam" genuinely scores ~0.
float door_seam_score(const cv::Mat& gray)
{
    if (gray.empty() or gray.rows < 8 or gray.cols < 8)
        return 0.0f;
    cv::Mat gy;
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    const cv::Mat agy = cv::abs(gy);

    // Row profile, ignoring the outer border rows/cols: the warp's BORDER_REPLICATE edge and the box
    // silhouette boundary both produce strong artificial gradients that are not a seam.
    const int m = std::max(2, gray.rows / 12), mc = std::max(2, gray.cols / 12);
    std::vector<float> prof;
    prof.reserve(gray.rows);
    for (int r = m; r < gray.rows - m; ++r)
    {
        const cv::Mat row = agy.row(r).colRange(mc, gray.cols - mc);
        prof.push_back(static_cast<float>(cv::mean(row)[0]));
    }
    if (prof.size() < 8)
        return 0.0f;

    std::vector<float> sorted = prof;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const float med   = std::max(0.0f, sorted[sorted.size() / 2]);
    const auto  it    = std::max_element(prof.begin(), prof.end());
    const float peak  = *it;
    const int   rbest = m + static_cast<int>(std::distance(prof.begin(), it));
    if (peak <= 1e-6f)
        return 0.0f;
    // Weber contrast against the panel's own brightness ⇒ cancels lighting; bounded, unlike peak/median.
    const float mean_I   = std::max(1.0f, static_cast<float>(cv::mean(gray)[0]));
    const float contrast = peak / mean_I;
    const float isolation = std::clamp(1.0f - med / peak, 0.0f, 1.0f);   // one ridge, not general texture

    // Coverage: a genuine door seam runs across the panel, not a local blob (a handle, a sticker, a highlight).
    const cv::Mat brow = agy.row(rbest).colRange(mc, gray.cols - mc);
    int lit = 0;
    for (int c = 0; c < brow.cols; ++c)
        if (brow.at<float>(0, c) >= 0.5f * peak) ++lit;
    const float coverage = static_cast<float>(lit) / std::max(1, brow.cols);

    return contrast * isolation * coverage;
}
}  // namespace

// ─── Camera extrinsic (room_T_zed) ──────────────────────────────────────────────────────────────

// room_T_zed (camera→room): room→body pinned to pose_ts_ms (Nearest), the rigid body→zed mount at latest.
std::optional<Eigen::Matrix4d> RefrigeratorProjection::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin the moving room→body hop to the frame's capture stamp (Nearest); keep the rigid body→zed mount
    // at latest (it carries only a bootstrap stamp — a pinned query would fail). ts=0 → current pose.
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", pose_ts_ms);
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

// ─── Projected ROI ──────────────────────────────────────────────────────────────────────────────

// Project the model box through the camera extrinsic → normalised in-image ROI (centre offset + fill).
//
// Stored on the instance for the controller's centring/dwell lock-on. Degenerate projections (robot too
// close / a corner grazing the image plane → the bbox explodes) are marked invalid and clamped so consumers
// and logs never see garbage.
void RefrigeratorProjection::compute_projected_roi(RefrigeratorInstance& inst)
{
    inst.roi_valid = false;
    if (not inner_eigen_)
        return;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return;
    }

    const auto Mopt = room_T_zed_matrix();   // room_T_zed (camera→room)
    if (not Mopt.has_value())
        return;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();   // room point → camera frame

    const float fx = camera_api_->get_focal_x();
    const float fy = camera_api_->get_focal_y();
    const float W  = static_cast<float>(camera_api_->get_width());
    const float H  = static_cast<float>(camera_api_->get_height());
    if (fx <= 0.f or fy <= 0.f or W <= 0.f or H <= 0.f)
        return;
    const float cx_px = W * 0.5f, cy_px = H * 0.5f;

    // Project the 8 box corners (top slab + footprint at floor) into the image. Camera convention
    // matches the producer: X=right, Y=forward(depth), Z=up ⇒ col=cx+X/Y·fx, row=cy−Z/Y·fy.
    const auto& s = inst.model.state();
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const float hw = s.w * 0.5f, hh = s.h * 0.5f;
    float min_col = 1e9f, min_row = 1e9f, max_col = -1e9f, max_row = -1e9f;
    int in_front = 0;
    for (const int ix : {-1, 1})
        for (const int iy : {-1, 1})
            for (const float z : {0.0f, s.refrigerator_height})
            {
                const float lx = static_cast<float>(ix) * hw, ly = static_cast<float>(iy) * hh;
                const Eigen::Vector4d Pr(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, z, 1.0);
                const Eigen::Vector4d Pc = zed_T_room * Pr;
                const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
                if (Y <= 0.20) continue;   // skip corners at/near the image plane: X/Y explodes there
                ++in_front;
                const float col = cx_px + static_cast<float>(X / Y) * fx;
                const float row = cy_px - static_cast<float>(Z / Y) * fy;
                min_col = std::min(min_col, col); max_col = std::max(max_col, col);
                min_row = std::min(min_row, row); max_row = std::max(max_row, row);
            }

    // GEOMETRY gate (not a belief threshold): require ≥4 of the box's 8 corners in front of the image plane
    // before trusting the ROI. With <4 corners forward the projection is near-degenerate (X/Y blow up at the
    // plane) and the centre/fill estimate is meaningless — this rejects that geometry, it does not tune the fit.
    if (in_front < 4)   // need most of the box in front of the camera to trust the ROI
        return;

    const float roi_cx = 0.5f * (min_col + max_col);
    const float roi_cy = 0.5f * (min_row + max_row);
    const float off_x = (roi_cx - cx_px) / (0.5f * W);   // [-1,1], 0 = centred
    const float off_y = (roi_cy - cy_px) / (0.5f * H);
    const float fill  = std::max((max_col - min_col) / W, (max_row - min_row) / H);
    // Reject degenerate projections (robot too close / a corner grazing the image plane → the
    // bbox explodes to absurd offsets). Beyond a sane bound the ROI is unusable for centring:
    // mark invalid (the controller then keeps sweeping / treats framing as unknown) and clamp the
    // stored values so consumers/logs never see garbage.
    const bool sane = std::isfinite(off_x) and std::isfinite(off_y) and std::isfinite(fill) and std::abs(off_x) < 3.0f and std::abs(off_y) < 3.0f and fill < 4.0f;
    inst.roi_offset_x = std::clamp(off_x, -3.0f, 3.0f);
    inst.roi_offset_y = std::clamp(off_y, -3.0f, 3.0f);
    inst.roi_fill     = std::clamp(fill, 0.0f, 4.0f);
    inst.roi_valid    = sane;
}

// ─── Silhouette existence (pixel-level) ─────────────────────────────────────────────────────────

// PIXEL-LEVEL silhouette existence evidence (occupancy / absence / occlusion) — see the header / SilhouetteExistence.
//
// Projects the solid box's CAMERA-FACING vertical side faces (back-face culled; plus the top face only when the
// camera is above it) onto the image and, over the predicted-visible pixels, splits into: lit by a "refrigerator"
// mask (occupancy), lit by a non-refrigerator mask (occlusion → HOLD), or lit by nothing (absence — the KEY
// "gone" signal that fires even when NO YOLO mask does this frame).
SilhouetteExistence RefrigeratorProjection::compute_silhouette_existence(const RefrigeratorInstance& inst)
{
    SilhouetteExistence out;
    if (not inner_eigen_)
        return out;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return out;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return out;
    }
    const auto Mopt = room_T_zed_matrix();   // room→zed pinned to current pose
    if (not Mopt.has_value())
        return out;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();

    const float fx = camera_api_->get_focal_x(), fy = camera_api_->get_focal_y();
    const float W  = static_cast<float>(camera_api_->get_width());
    const float Himg = static_cast<float>(camera_api_->get_height());
    if (fx <= 0.f or fy <= 0.f or W <= 0.f or Himg <= 0.f)   // fx/fy: sanity only; projection uses CameraAPI
        return out;

    // Hashed pixel-cell coverage of the current YOLO foreground, split refrigerator (occupancy) vs other (occluder).
    // A CELL-px cell absorbs mask-boundary jitter and makes membership O(1). Key packs the two cell indices.
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid or pkt.mask_pixels.empty())
        return out;
    constexpr float CELL = 6.0f;   // spatial-hash cell size (px). A discretization/method constant, not a belief
                                   // gate: larger absorbs more mask-boundary jitter into membership, smaller is
                                   // sharper. Does not affect the AI fit; tune only if mask resolution changes.
    const auto key = [&](float col, float row) -> std::int64_t
    {
        const std::int64_t cx = static_cast<std::int64_t>(std::floor(col / CELL));
        const std::int64_t cy = static_cast<std::int64_t>(std::floor(row / CELL));
        return (cx << 32) ^ (cy & 0xffffffffLL);
    };
    std::unordered_set<std::int64_t> refrigerator_cells, occluder_cells;
    for (const auto& sl : pkt.slices)
    {
        const std::size_t b = std::min(sl.pixel_begin, pkt.mask_pixels.size());
        const std::size_t e = std::min(sl.pixel_end,   pkt.mask_pixels.size());
        auto& dst = (sl.label == "refrigerator") ? refrigerator_cells : occluder_cells;
        for (std::size_t i = b; i < e; ++i)
            dst.insert(key(pkt.mask_pixels[i].x(), pkt.mask_pixels[i].y()));
    }

    const auto& s = inst.ai2_belief.state();
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw), hw = 0.5f * s.w, hd = 0.5f * s.h;

    // ── LINE OF SIGHT against the room's own walls ────────────────────────────────────────────────────
    // "Predicted visible" was decided by the camera FRUSTUM alone, plus occlusion by whatever OTHER objects YOLO
    // happened to segment. A wall is not a YOLO class, so a wall could never occlude anything: from the next room
    // the fridge still projects into the image, lands on wall pixels that carry no fridge mask, and every sample
    // votes ABSENCE. Live kill (etc/ai2_log.csv cyc3590-3608): camera 7.0 m away through a wall, yet
    // in_fov_frac=1.00 and central_frac=1.00 — scored as the BEST possible verifying view — so p_detect=0.126 came
    // from range alone, sfree_eff≈80/cycle, ex_L −4 in three cycles, streak 15, DELETED. Exactly the
    // "it disappears when the robot is in the other room" report.
    //
    // A sample the camera cannot physically reach is not evidence of absence. Occlusion by known scene geometry
    // belongs in the observation model, and the code already has the concept — n_occluded ⇒ HOLD — it just had no
    // way to know about walls. Segment camera→sample vs every room-polygon edge; a strict interior crossing means
    // the sample is behind a wall. See [[refrigerator-silhouette-line-of-sight]].
    const Eigen::Vector3d cam3 = Mopt.value().block<3, 1>(0, 3);
    const Eigen::Vector2f cam_xy(static_cast<float>(cam3.x()), static_cast<float>(cam3.y()));
    // ★2026-07-31: this crossing test was hand-rolled here and is now the SHARED rc::occlusion::walls_block(),
    // the same call door_concept, table_concept and cabinet_concept make. One implementation, one place to fix.
    // NOTE the self-occlusion guard changed form: the local version used a ray-parameter slack (t < 1−1e-3, so
    // only a sample sitting essentially ON the wall line was exempt); walls_block uses a METRIC skip — a
    // segment closer than own_wall_skip_m to the sample never occludes it. For a wall-flush fridge both
    // protect the back face; the metric form additionally protects the whole carcass depth, which is the
    // intent. 0.70 m ≈ fridge depth. The validated failure this fixes (camera 7.0 m away through a wall,
    // in_fov_frac=1.00, ex_L −4 in three cycles → DELETED) is unaffected: that wall is metres from the
    // sample, far outside the skip. See [[refrigerator-silhouette-line-of-sight]].
    constexpr float kOwnWallSkipM = 0.70f;
    const auto wall_between = [&](const Eigen::Vector2f& q) -> bool
    { return rc::occlusion::walls_block(cam_xy, q, room_polygon_, kOwnWallSkipM); };

    // Classify ONE room-frame silhouette sample: project it, then vote occupancy / absence / occlusion. A
    // sample lit by a "refrigerator" mask is occupancy; lit by nothing is ABSENCE (the "gone" signal that fires even
    // with NO YOLO mask); lit by a non-refrigerator mask is OCCLUDED (a nearer object hides the point) ⇒ HOLD.
    double range_sum = 0.0;
    const auto classify = [&](float lx, float ly, float lz)
    {
        ++out.n_total;                                                 // one silhouette sample of the WHOLE object
        const Eigen::Vector4d Pr(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, lz, 1.0);
        const Eigen::Vector4d Pc = zed_T_room * Pr;
        const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
        if (Y <= 0.20) return;                                         // behind / at the image plane (near clip)
        // Real camera FRUSTUM: project through the DSR CameraAPI (true intrinsics/principal point, and the
        // node's projection model — pinhole for zed, equirectangular for a 360 sensor). A sample is "expected
        // visible" only if it lands inside the actual image, so out-of-FoV samples are NOT counted detectable.
        const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
        const float col = static_cast<float>(uv.x()), row = static_cast<float>(uv.y());
        if (col < 0.f or col >= W or row < 0.f or row >= Himg) return; // out of frustum ⇒ not detectable
        // A room wall between the camera and this sample hides it just as surely as a nearer object does — and
        // unlike an object, no YOLO mask will ever report it. Same verdict: OCCLUDED ⇒ excluded from
        // n_detectable ⇒ HOLD, never false absence.
        if (wall_between(Pr.head<2>().cast<float>()))
        { ++out.n_occluded; return; }
        const std::int64_t k = key(col, row);
        if (occluder_cells.contains(k) and not refrigerator_cells.contains(k))
        { ++out.n_occluded; return; }                                 // nearer object hides it ⇒ HOLD
        ++out.n_detectable;
        const float f = central_region_frac_, g = 1.0f - central_region_frac_;   // central box [f,1-f]²
        if (col > f * W and col < g * W and row > f * Himg and row < g * Himg)
            ++out.n_central;                                          // central image region ⇒ the robot is looking AT it
        range_sum += std::sqrt(X * X + Y * Y + Z * Z);                 // camera→sample distance (absence conf ∝ 1/range)
        if (refrigerator_cells.contains(k)) out.e_occ  += 1.0f;              // still there
        else                         out.e_free += 1.0f;              // predicted-but-absent
    };

    // A refrigerator is a SOLID floor-anchored cuboid (refrigerator_model.cpp), so its predicted silhouette is
    // the union of the VERTICAL side faces that face the camera — NOT the top face. Sampling the top face (the
    // table-concept geometry this unit was cloned from) is wrong here in a way that manufactures absence: the top
    // sits at H≈1.9 m, ABOVE the ZED and hidden behind the fridge's own body, so its samples project onto the
    // wall above the fridge, land outside the YOLO mask, and every one of them votes "predicted-visible but
    // ABSENT". Live evidence (etc/ai2_log.csv): a converged fit (0.63×0.61×1.91) at 2.78 m, dead centre of frame,
    // scored socc=0 / sfree=640 → L pinned at −4 → removed, then re-born. See [[refrigerator-table-geometry-churn]].
    //
    // Visibility is decided by the outward normal, evaluated in the box-LOCAL frame where the faces are
    // axis-aligned: the +x face is camera-facing iff the camera's local x exceeds +w/2, and so on. This is exact
    // back-face culling for a convex box (no threshold), and for a convex box the camera-facing faces never
    // occlude each other, so no further self-occlusion test is needed.
    const Eigen::Vector3d cam_room = Mopt.value().block<3, 1>(0, 3);   // camera origin in the room frame
    const float dxr = static_cast<float>(cam_room.x()) - s.cx, dyr = static_cast<float>(cam_room.y()) - s.cy;
    const float cam_lx =  c * dxr + sn * dyr;      // camera in box-local coords (inverse yaw rotation)
    const float cam_ly = -sn * dxr +  c * dyr;
    const float cam_lz = static_cast<float>(cam_room.z());

    // NU/NV are NUMERIC SAMPLING RESOLUTION for the detectability/occupancy count — a quadrature density, not a
    // decision threshold. More samples = smoother central_frac/e_occ estimate at linear cost. No config key.
    constexpr int NU = 16, NV = 20;   // across the face × up the height
    // Sample the side face whose outward normal is ±local-x (normal_along_x) or ±local-y, at sign `sgn`: the
    // face sits at that half-extent, the free lateral coordinate spans the OTHER half-extent, and the vertical
    // spans the full box height [0, H].
    const auto sample_face = [&](bool normal_along_x, float sgn)
    {
        const float half_lat = normal_along_x ? hd : hw;
        for (int iu = 0; iu < NU; ++iu)
        {
            const float t  = (-1.0f + 2.0f * (iu + 0.5f) / NU) * half_lat;
            const float lx = normal_along_x ? sgn * hw : t;
            const float ly = normal_along_x ? t        : sgn * hd;
            for (int iv = 0; iv < NV; ++iv)
                classify(lx, ly, s.H * (iv + 0.5f) / NV);
        }
    };
    if (cam_lx >  hw) sample_face(true,  +1.0f);
    if (cam_lx < -hw) sample_face(true,  -1.0f);
    if (cam_ly >  hd) sample_face(false, +1.0f);
    if (cam_ly < -hd) sample_face(false, -1.0f);
    // TOP face only when the camera is genuinely above it (it is not visible otherwise). Same quadrature grid.
    if (cam_lz > s.H)
        for (int iu = 0; iu < NU; ++iu)
            for (int iv = 0; iv < NV; ++iv)
                classify((-1.0f + 2.0f * (iu + 0.5f) / NU) * hw, (-1.0f + 2.0f * (iv + 0.5f) / NV) * hd, s.H);
    // No face qualified ⇒ the camera is inside the footprint (degenerate) ⇒ n_total==0 ⇒ the caller HOLDs.
    if (out.n_detectable > 0)
        out.mean_range_m = static_cast<float>(range_sum / out.n_detectable);
    return out;
}

// ─── Appearance-based FRONT (door) detection ──────────────────────────────────────────────────────

// Project the FITTED box's 4 vertical side faces into the live ZED RGB, warp each visible face to an upright
// patch, score its door-ness, and return the winning face's outward-normal bearing + a confidence margin. See
// the header. All geometry mirrors compute_projected_roi / compute_silhouette_existence (near-clip + CameraAPI).
std::optional<FrontCue> RefrigeratorProjection::detect_front(const RefrigeratorState& s, const cv::Mat& rgb,
                                                             std::uint64_t stamp_ms)
{
    if (rgb.empty() or not inner_eigen_)
        return std::nullopt;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return std::nullopt;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return std::nullopt;
    }
    const auto Mopt = room_T_zed_matrix(stamp_ms);   // room_T_zed pinned to the frame's capture time
    if (not Mopt.has_value())
        return std::nullopt;
    const Eigen::Matrix4d room_T_zed = Mopt.value();
    const Eigen::Matrix4d zed_T_room = room_T_zed.inverse();
    const Eigen::Vector3d cam_pos_room = room_T_zed.block<3, 1>(0, 3);   // camera centre in room frame

    const float W = static_cast<float>(camera_api_->get_width());
    const float H = static_cast<float>(camera_api_->get_height());
    if (W <= 0.f or H <= 0.f)
        return std::nullopt;
    // Guard against an intrinsics/frame size mismatch (the extrinsic is a separate stream): the projected pixel
    // coords are in the CameraAPI's intrinsic frame, so scale to the delivered image if they differ.
    const float sx = static_cast<float>(rgb.cols) / W;
    const float sy = static_cast<float>(rgb.rows) / H;

    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const float hw = 0.5f * s.w, hd = 0.5f * s.h, boxH = s.refrigerator_height;

    // Local→room for a footprint point (lx,ly) at height lz.
    const auto to_room = [&](float lx, float ly, float lz) -> Eigen::Vector3d
    { return {s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, lz}; };
    // Room→image via the camera. nullopt if the point is behind/at the image plane (near clip, matches silhouette).
    const auto project = [&](const Eigen::Vector3d& Pr) -> std::optional<cv::Point2f>
    {
        const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
        if (Pc.y() <= 0.20) return std::nullopt;                       // behind / at the image plane
        const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
        if (not std::isfinite(uv.x()) or not std::isfinite(uv.y())) return std::nullopt;
        return cv::Point2f(static_cast<float>(uv.x()) * sx, static_cast<float>(uv.y()) * sy);
    };

    // The 4 vertical side faces: outward LOCAL normal + the two footprint corners (bottom edge). Door = local −Y.
    struct Face { float nx, ny; float ax, ay; float bx, by; };
    const std::array<Face, 4> faces = {{
        { 1.f,  0.f,   hw, -hd,   hw,  hd },   // +X
        { 0.f,  1.f,   hw,  hd,  -hw,  hd },   // +Y
        {-1.f,  0.f,  -hw,  hd,  -hw, -hd },   // −X
        { 0.f, -1.f,  -hw, -hd,   hw, -hd },   // −Y (door)
    }};

    const float imgW = static_cast<float>(rgb.cols), imgH = static_cast<float>(rgb.rows);
    constexpr int DST_W = 96, DST_H = 160;   // fixed upright warp size → mean door-ness comparable across faces
    const cv::Point2f dst[4] = { {0, 0}, {DST_W - 1.f, 0}, {DST_W - 1.f, DST_H - 1.f}, {0, DST_H - 1.f} };

    float best = -1.0f, second = -1.0f, best_bearing = 0.0f, best_seam = 0.0f;
    int n_scored = 0;
    for (const auto& f : faces)
    {
        // Camera-facing: the outward normal must point toward the camera ⇒ nr·(cam − face_centre) > 0.
        const Eigen::Vector3d nr(c * f.nx - sn * f.ny, sn * f.nx + c * f.ny, 0.0);
        const Eigen::Vector3d fc = to_room(0.5f * (f.ax + f.bx), 0.5f * (f.ay + f.by), 0.5f * boxH);
        if (nr.dot(cam_pos_room - fc) <= 0.0)
            continue;                                          // back-facing → not visible

        // Project the 4 quad corners: bottom_a, bottom_b, top_b, top_a (consistent winding for the warp).
        const auto p0 = project(to_room(f.ax, f.ay, 0.0f));
        const auto p1 = project(to_room(f.bx, f.by, 0.0f));
        const auto p2 = project(to_room(f.bx, f.by, boxH));
        const auto p3 = project(to_room(f.ax, f.ay, boxH));
        if (not (p0 and p1 and p2 and p3))
            continue;                                          // any corner behind the image plane → skip
        const std::array<cv::Point2f, 4> quad = {*p0, *p1, *p2, *p3};

        // Require the face mostly in-image: ≥3 of 4 corners inside the frame (tolerates one edge-clipped corner).
        int inside = 0;
        for (const auto& q : quad)
            if (q.x >= 0.f and q.x < imgW and q.y >= 0.f and q.y < imgH) ++inside;
        if (inside < 3)
            continue;

        // Projected area (shoelace); reject faces too small to carry reliable edge statistics.
        const float area = 0.5f * std::abs(
            (quad[0].x * quad[1].y - quad[1].x * quad[0].y) +
            (quad[1].x * quad[2].y - quad[2].x * quad[1].y) +
            (quad[2].x * quad[3].y - quad[3].x * quad[2].y) +
            (quad[3].x * quad[0].y - quad[0].x * quad[3].y));
        if (area < front_min_face_area_px_)
            continue;

        // Warp the projected quad to an upright rectangle, grayscale, score door-ness (mean |Sobel_x|).
        const cv::Mat Mwarp = cv::getPerspectiveTransform(quad.data(), dst);
        cv::Mat patch;
        cv::warpPerspective(rgb, patch, Mwarp, cv::Size(DST_W, DST_H), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::Mat gray;
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
        // The SEAM is the discriminator (see door_seam_score); mean edge energy is kept only as a weak
        // tie-break between two faces that both lack a seam.
        const float seam  = door_seam_score(gray);
        const float score = seam + 1e-3f * door_ness(gray);
        if (seam > best_seam) best_seam = seam;
        ++n_scored;

        if (score > best)
        {
            second = best;
            best = score;
            best_bearing = std::atan2(nr.y(), nr.x());         // room-frame yaw the door (winning face) faces
        }
        else if (score > second)
            second = score;
    }

    if (n_scored < 1 or best <= 0.0f)
        return std::nullopt;

    // ★ONE visible face is enough when it carries a SEAM. The old rule needed ≥2 faces because the score was a
    // relative margin, so it emitted nothing for a fridge standing flush against a wall and viewed head-on —
    // the clearest possible view of the door (etc/fridge_1.png). A seam is absolute evidence about the face it
    // is on: "this face is the door" fixes the door bearing outright, and unlike any comparison between the
    // faces that happen to be visible it resolves the 180° front/back ambiguity, since the back has no seam.
    // The strength maps to confidence through the same saturating form used elsewhere: seam/(seam+1) — 1 unit
    // of seam ratio×coverage is half-confident. No new constant, and still suppressed by FrontMinConfidence
    // when the face is blank.
    const float confidence = (n_scored >= 2)
        ? (best - std::max(0.0f, second)) / (best + 1e-6f)      // two faces: which one is the door
        : best_seam / (best_seam + 1.0f);                       // one face: does it carry a seam at all
    if (confidence < front_min_confidence_)
        return std::nullopt;                                    // margin too weak → suppress (don't feed noise)
    return FrontCue{ best_bearing, std::clamp(confidence, 0.0f, 1.0f) };
}

// ─── Door-ness self-test (OpenCV) ─────────────────────────────────────────────────────────────────

bool RefrigeratorProjection::self_test()
{
    // A "handle + seams" patch: a mid-gray field with a few strong VERTICAL lines (what a fridge door shows).
    cv::Mat lined(160, 96, CV_8UC3, cv::Scalar(120, 120, 120));
    for (int x : {18, 30, 48, 66, 78})
        cv::line(lined, cv::Point(x, 6), cv::Point(x, 153), cv::Scalar(20, 20, 20), 2);
    cv::rectangle(lined, cv::Rect(60, 40, 6, 70), cv::Scalar(230, 230, 230), cv::FILLED);   // handle
    // A plain/flat side patch: a smooth field with a faint horizontal gradient (no vertical structure).
    cv::Mat plain(160, 96, CV_8UC3);
    for (int y = 0; y < plain.rows; ++y)
        plain.row(y).setTo(cv::Scalar(110 + y / 8, 110 + y / 8, 110 + y / 8));

    cv::Mat gl, gp;
    cv::cvtColor(lined, gl, cv::COLOR_BGR2GRAY);
    cv::cvtColor(plain, gp, cv::COLOR_BGR2GRAY);
    const float s_lined = door_ness(gl);
    const float s_plain = door_ness(gp);

    const bool ok = s_lined > 2.0f * std::max(1.0f, s_plain);   // door face clearly beats the plain side
    std::printf("RefrigeratorProjection::self_test [door-ness] %s  lined=%.2f plain=%.2f (ratio %.1f)\n",
                ok ? "PASS" : "FAIL", s_lined, s_plain, s_lined / std::max(0.01f, s_plain));

    // ── DOOR SEAM: the absolute, single-face discriminator (etc/fridge_1.png is exactly this view) ──────
    // A fridge FRONT: flat panel with one full-width horizontal separator + a handle. Against the BACK panel
    // (flat, no seam) and against a LIT panel (a bright vertical band — the specular/handle case that must NOT
    // read as a seam). Scale-free, so the score is compared to an absolute level, not to another face.
    const auto make = [&](bool seam, bool band, int base) {
        cv::Mat m(160, 96, CV_8UC3, cv::Scalar(base, base, base));
        for (int y = 0; y < m.rows; ++y)                                   // gentle shading, both patches
            m.row(y).setTo(cv::Scalar(base + y / 20, base + y / 20, base + y / 20));
        if (seam) cv::line(m, cv::Point(4, 88), cv::Point(91, 88), cv::Scalar(30, 30, 30), 2);   // door split
        if (band) cv::rectangle(m, cv::Rect(70, 10, 8, 140), cv::Scalar(240, 240, 240), cv::FILLED);
        return m;
    };
    cv::Mat gf, gb, gv;
    cv::cvtColor(make(true,  true,  120), gf, cv::COLOR_BGR2GRAY);   // front: seam + handle
    cv::cvtColor(make(false, false, 120), gb, cv::COLOR_BGR2GRAY);   // back: blank panel
    cv::cvtColor(make(false, true,  120), gv, cv::COLOR_BGR2GRAY);   // blank + a bright vertical band
    const float seam_front = door_seam_score(gf);
    const float seam_back  = door_seam_score(gb);
    const float seam_band  = door_seam_score(gv);
    // Confidence a SINGLE face would emit, seam/(seam+1), vs FrontMinConfidence's default 0.10.
    const float conf_front = seam_front / (seam_front + 1.0f);
    const float conf_back  = seam_back  / (seam_back  + 1.0f);
    const bool ok_seam = seam_front > 3.0f * std::max(0.05f, seam_back) and
                         seam_front > 3.0f * std::max(0.05f, seam_band) and
                         conf_front > 0.10f and conf_back < 0.10f;
    std::printf("RefrigeratorProjection::self_test [door-seam] %s  front=%.2f (conf %.2f) back=%.2f (conf %.2f) vert-band=%.2f\n",
                ok_seam ? "PASS" : "FAIL", seam_front, conf_front, seam_back, conf_back, seam_band);
    return ok and ok_seam;
}

}  // namespace rc
