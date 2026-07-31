/*
 * cabinet_projection.h — camera-projection unit for cabinet_concept (extracted from CabinetFitter).
 *
 * Owns the ZED CameraAPI (lazily bound to the "zed" node) and everything that projects the fitted cabinet model
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

#include "cabinet_instance.h"      // rc::CabinetInstance
#include "../../common/mask_ingestor/mask_ingestor.h"

namespace rc {

// PIXEL-LEVEL silhouette existence evidence (EXISTENCE_BELIEF_PLAN.md, mask channel). Projects the tabletop
// TOP face + the 4 LEG axes onto the image and, over the predicted-VISIBLE pixels, counts how many are lit by a "cabinet" YOLO
// mask (e_occ ⇒ still there) vs by nothing at all (e_free ⇒ predicted-visible-but-ABSENT ⇒ evidence it is
// gone, EVEN WITH NO YOLO MASK this frame). Pixels covered by a NON-cabinet mask are OCCLUDED (a nearer object
// hides the tabletop) and excluded from n_detectable → HOLD, never false absence. n_detectable==0 (out of
// FoV / fully occluded) ⇒ the caller HOLDs. Feeds rc::exist::mask_evidence. Called from update_existence.
struct SilhouetteExistence
{
    float e_occ = 0.0f, e_free = 0.0f;
    int   n_total      = 0;    // silhouette samples attempted (top face + legs) — the "whole object"
    int   n_detectable = 0;    // samples that land in the real camera FRUSTUM and are un-occluded (0 ⇒ HOLD)
    int   n_central    = 0;    // detectable samples in the CENTRAL image region (the ZED resolves those; a
                               // peripheral cabinet is unreliable — the robot isn't really looking AT it)
    int   n_occluded   = 0;    // in-frustum samples hidden by a nearer (non-cabinet) mask
    float mean_range_m = 0.0f; // mean camera→silhouette depth over the detectable samples (absence confidence ∝ 1/range)
    // "Should be visible" fraction: n_detectable / n_total. Absence is only evidence of removal in
    // proportion to how much of the object the sensor could actually have seen from here (real FoV).
    float in_fov_frac() const { return n_total > 0 ? static_cast<float>(n_detectable) / n_total : 0.0f; }
    // Fraction of detectable samples that fall CENTRALLY — how much the robot is actually LOOKING at the cabinet
    // (vs it merely clipping the wide frustum edge). A verifying view is central; a peripheral one only warrants
    // an epistemic "go look", not a removal vote. See the verification-gated removal in cabinet_existence.
    float central_frac() const { return n_detectable > 0 ? static_cast<float>(n_central) / n_detectable : 0.0f; }
};

class CabinetProjection
{
public:
    CabinetProjection(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::InnerEigenAPI* inner_eigen,
                    MaskIngestor* mask_ingestor)
        : G_(std::move(graph)), inner_eigen_(inner_eigen), mask_ingestor_(mask_ingestor) {}

    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(CabinetInstance& inst);
    SilhouetteExistence compute_silhouette_existence(const CabinetInstance& inst);

    // ── Room WALLS as occluders (silhouette line-of-sight) ─────────────────────────────────────────
    // The silhouette's only occluder evidence is the NON-cabinet YOLO masks, and YOLO never segments WALLS.
    // So a cabinet in the NEXT ROOM still projects into the frustum, lands on unmasked wall pixels, and every
    // sample votes ABSENCE at full strength — a removal verdict for an object with zero visibility. Same bug,
    // same fix and same shared helper (rc::occlusion::walls_block) as refrigerator_concept, door_concept and
    // table_concept. Fed from CabinetFitter::set_room_geometry, which already owns this polygon for the
    // wall-flush factor. Empty polygon ⇒ test inactive ⇒ historic behaviour.
    void set_room_polygon(std::vector<Eigen::Vector2f> poly) { room_polygon_ = std::move(poly); }

private:
    std::shared_ptr<DSR::DSRGraph>  G_;
    DSR::InnerEigenAPI*             inner_eigen_ = nullptr;
    MaskIngestor*                   mask_ingestor_ = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    std::vector<Eigen::Vector2f>    room_polygon_; // room walls, as OCCLUDERS for the line-of-sight test
};

}  // namespace rc
