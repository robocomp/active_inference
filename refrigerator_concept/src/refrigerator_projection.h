/*
 * refrigerator_projection.h — camera-projection unit for refrigerator_concept (extracted from RefrigeratorFitter).
 *
 * Owns the ZED CameraAPI (lazily bound to the "zed" node) and everything that projects the fitted refrigerator model
 * through the camera extrinsic/intrinsic: room_T_zed_matrix (the camera→room extrinsic, room→body pinned to a
 * capture stamp), compute_projected_roi (the model box → normalised in-image ROI for the controller's lock-on),
 * and compute_silhouette_existence (PIXEL-LEVEL silhouette existence evidence — occupancy / absence / occlusion
 * against the current YOLO foreground). Holds the graph, the InnerEigenAPI, and the mask-packet source
 * (MaskIngestor). Plain class (no Q_OBJECT).
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_camera_api.h>

#include "refrigerator_instance.h"      // rc::RefrigeratorInstance, RefrigeratorState, rc::FrontCue
#include "../../common/mask_ingestor/mask_ingestor.h"

namespace cv { class Mat; }             // forward-decl: OpenCV lives only in refrigerator_projection.cpp

namespace rc {

// PIXEL-LEVEL silhouette existence evidence (EXISTENCE_BELIEF_PLAN.md, mask channel). Projects the refrigeratortop
// TOP face + the 4 LEG axes onto the image and, over the predicted-VISIBLE pixels, counts how many are lit by a "refrigerator" YOLO
// mask (e_occ ⇒ still there) vs by nothing at all (e_free ⇒ predicted-visible-but-ABSENT ⇒ evidence it is
// gone, EVEN WITH NO YOLO MASK this frame). Pixels covered by a NON-refrigerator mask are OCCLUDED (a nearer object
// hides the refrigeratortop) and excluded from n_detectable → HOLD, never false absence. n_detectable==0 (out of
// FoV / fully occluded) ⇒ the caller HOLDs. Feeds rc::exist::mask_evidence. Called from update_existence.
struct SilhouetteExistence
{
    float e_occ = 0.0f, e_free = 0.0f;
    int   n_total      = 0;    // silhouette samples attempted (top face + legs) — the "whole object"
    int   n_detectable = 0;    // samples that land in the real camera FRUSTUM and are un-occluded (0 ⇒ HOLD)
    int   n_central    = 0;    // detectable samples in the CENTRAL image region (the ZED resolves those; a
                               // peripheral refrigerator is unreliable — the robot isn't really looking AT it)
    int   n_occluded   = 0;    // in-frustum samples hidden by a nearer (non-refrigerator) mask
    float mean_range_m = 0.0f; // mean camera→silhouette depth over the detectable samples (absence confidence ∝ 1/range)
    // "Should be visible" fraction: n_detectable / n_total. Absence is only evidence of removal in
    // proportion to how much of the object the sensor could actually have seen from here (real FoV).
    float in_fov_frac() const { return n_total > 0 ? static_cast<float>(n_detectable) / n_total : 0.0f; }
    // Fraction of detectable samples that fall CENTRALLY — how much the robot is actually LOOKING at the refrigerator
    // (vs it merely clipping the wide frustum edge). A verifying view is central; a peripheral one only warrants
    // an epistemic "go look", not a removal vote. See the verification-gated removal in refrigerator_existence.
    float central_frac() const { return n_detectable > 0 ? static_cast<float>(n_central) / n_detectable : 0.0f; }
};

class RefrigeratorProjection
{
public:
    RefrigeratorProjection(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::InnerEigenAPI* inner_eigen,
                    MaskIngestor* mask_ingestor)
        : G_(std::move(graph)), inner_eigen_(inner_eigen), mask_ingestor_(mask_ingestor) {}

    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(RefrigeratorInstance& inst);
    SilhouetteExistence compute_silhouette_existence(const RefrigeratorInstance& inst);

    // ── Appearance-based FRONT (door) detection ──────────────────────────────────────────────────────
    // Project the FITTED 3D box (from the belief, NOT YOLO's 2D mask) into the live ZED RGB and score
    // vertical-edge (door-ness) energy per visible vertical face; the door face (handle + seams = strong
    // vertical lines) wins over the plain sides. Because it uses the belief's own box it keeps working when
    // the robot is too close for YOLO to still detect the fridge. Returns a FrontCue {bearing_rad = room-frame
    // yaw of the winning face's OUTWARD normal (direction the door faces), confidence = (max−second)/(max+eps)}
    // or nullopt if <2 faces qualify / the RGB is empty / the extrinsic is unavailable / confidence < the gate.
    // stamp_ms pins the room→body extrinsic hop to the frame's capture time. Uses OpenCV (projection unit only).
    std::optional<FrontCue> detect_front(const RefrigeratorState& s, const cv::Mat& rgb, std::uint64_t stamp_ms);

    // Config knobs for detect_front: minimum projected face area (px²) to score, and the minimum door-ness
    // margin (confidence) below which the cue is suppressed (returned as nullopt). Set once from config.
    void set_front_params(float min_face_area_px, float min_confidence)
    { front_min_face_area_px_ = min_face_area_px; front_min_confidence_ = min_confidence; }

    // Door-ness self-test (OpenCV): a synthetic "handle+seams" patch (strong vertical lines) must score higher
    // than a flat/plain patch. Standalone; called alongside the pure-Eigen belief self_test at startup.
    static bool self_test();

    // Central-image box fraction: a detectable sample inside [f, 1-f]² of the image counts as "central"
    // (looking AT the refrigerator) → central_frac → p_detect → removal. Set once from config (default 0.25).
    void set_central_region_frac(float f) { central_region_frac_ = f; }

private:
    std::shared_ptr<DSR::DSRGraph>  G_;
    DSR::InnerEigenAPI*             inner_eigen_ = nullptr;
    MaskIngestor*                   mask_ingestor_ = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    float                           central_region_frac_ = 0.25f;
    float                           front_min_face_area_px_ = 900.0f;   // detect_front: min projected face area (px²)
    float                           front_min_confidence_   = 0.10f;    // detect_front: min door-ness margin to emit
};

}  // namespace rc
