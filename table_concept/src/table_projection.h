/*
 * table_projection.h — camera-projection unit for table_concept (extracted from TableFitter).
 *
 * Owns the ZED CameraAPI (lazily bound to the "zed" node) and everything that projects the fitted table model
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

#include "table_instance.h"      // rc::TableInstance
#include "../../common/mask_ingestor/mask_ingestor.h"

namespace rc {

// PIXEL-LEVEL silhouette existence evidence (EXISTENCE_BELIEF_PLAN.md, mask channel). Projects the tabletop
// TOP face + the 4 LEG axes onto the image and, over the predicted-VISIBLE pixels, counts how many are lit by a "table" YOLO
// mask (e_occ ⇒ still there) vs by nothing at all (e_free ⇒ predicted-visible-but-ABSENT ⇒ evidence it is
// gone, EVEN WITH NO YOLO MASK this frame). Pixels covered by a NON-table mask are OCCLUDED (a nearer object
// hides the tabletop) and excluded from n_detectable → HOLD, never false absence. n_detectable==0 (out of
// FoV / fully occluded) ⇒ the caller HOLDs. Feeds rc::exist::mask_evidence. Called from update_existence.
struct SilhouetteExistence
{
    float e_occ = 0.0f, e_free = 0.0f;
    int   n_total      = 0;    // silhouette samples attempted (top face + legs) — the "whole object"
    int   n_detectable = 0;    // samples that land in the real camera FRUSTUM and are un-occluded (0 ⇒ HOLD)
    int   n_central    = 0;    // detectable samples in the CENTRAL image region (the ZED resolves those; a
                               // peripheral table is unreliable — the robot isn't really looking AT it)
    int   n_occluded   = 0;    // in-frustum samples hidden by a nearer (non-table) mask
    float mean_range_m = 0.0f; // mean camera→silhouette depth over the detectable samples (absence confidence ∝ 1/range)
    // "Should be visible" fraction: n_detectable / n_total. Absence is only evidence of removal in
    // proportion to how much of the object the sensor could actually have seen from here (real FoV).
    float in_fov_frac() const { return n_total > 0 ? static_cast<float>(n_detectable) / n_total : 0.0f; }
    // Fraction of detectable samples that fall CENTRALLY — how much the robot is actually LOOKING at the table
    // (vs it merely clipping the wide frustum edge). A verifying view is central; a peripheral one only warrants
    // an epistemic "go look", not a removal vote. See the verification-gated removal in table_existence.
    float central_frac() const { return n_detectable > 0 ? static_cast<float>(n_central) / n_detectable : 0.0f; }
};

class TableProjection
{
public:
    TableProjection(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::InnerEigenAPI* inner_eigen,
                    MaskIngestor* mask_ingestor)
        : G_(std::move(graph)), inner_eigen_(inner_eigen), mask_ingestor_(mask_ingestor) {}

    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(TableInstance& inst);
    SilhouetteExistence compute_silhouette_existence(const TableInstance& inst);

    // Central-image box fraction: a detectable sample inside [f, 1-f]² of the image counts as "central"
    // (looking AT the table) → central_frac → p_detect → removal. Set once from config (default 0.25).
    void set_central_region_frac(float f) { central_region_frac_ = f; }

    // ── Room WALLS as occluders (the PRIMARY line-of-sight test) ───────────────────────────────────
    // The silhouette's only occluder evidence is the NON-table YOLO masks, which means it can see through
    // WALLS: YOLO segments objects, never walls, so a table in the NEXT ROOM projects into the frustum, lands
    // on unmasked wall pixels, and is scored predicted-visible-but-ABSENT — full-strength removal evidence for
    // an object with zero visibility. That is the "table removed from the other room" failure (live: table_1 at
    // 5.72 m and table_3 behind a wall, both n_detectable=640 with e_free=640 and streak=15 → deleted).
    // Identical bug and identical fix as refrigerator_concept (see [[refrigerator-silhouette-line-of-sight]])
    // and door_concept; both use the SHARED rc::occlusion::walls_block(), and so does this. Empty polygon ⇒
    // no room model ⇒ test disabled ⇒ historic behaviour.
    void set_room_polygon(std::vector<Eigen::Vector2f> poly) { room_polygon_ = std::move(poly); }

    // ── LiDAR line-of-sight oracle (OPTIONAL EXTRA, default OFF) ───────────────────────────────────
    // Complements the wall test above for occluders a room polygon cannot know about — furniture, people, an
    // open door leaf. Bins the room-frame sweep by azimuth about the sensor origin, keeps the NEAREST return
    // per bin, and treats a silhouette sample beyond that first hit (plus a margin) as OCCLUDED. Needs a fresh
    // sweep and is approximate (bin quantisation), so walls_block — exact, sweep-independent, and the
    // established pattern across the sibling agents — is the primary. ExistenceLosAzimBins = 0 disables.
    void set_lidar_los(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room,
                       float margin_m, int azim_bins);

private:
    std::shared_ptr<DSR::DSRGraph>  G_;
    DSR::InnerEigenAPI*             inner_eigen_ = nullptr;
    MaskIngestor*                   mask_ingestor_ = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    float                           central_region_frac_ = 0.25f;
    std::vector<Eigen::Vector2f>    room_polygon_;   // room walls, as OCCLUDERS for the line-of-sight test
    // LiDAR LoS oracle state: nearest-return range per azimuth bin about los_origin_ (room frame).
    std::vector<float>              los_min_range_;         // empty ⇒ oracle OFF
    Eigen::Vector3f                 los_origin_ = Eigen::Vector3f::Zero();
    float                           los_margin_m_ = 0.25f;  // a sample must be this far BEYOND the first hit
                                                            // to count as occluded (the table's own returns and
                                                            // registration error must not occlude the table)
    // True when something solid sits between los_origin_ and this room-frame point. False when the oracle is
    // off or the bearing was never probed (no return ⇒ unknown ⇒ do NOT claim occlusion).
    bool los_blocked(const Eigen::Vector3f& p_room) const;
};

}  // namespace rc
