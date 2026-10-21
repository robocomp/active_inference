/*
 * door_fitter.h
 *
 * The DSR-bound side of door_concept's fit (mirrors bottle_concept/bottle_fitter.h): the per-door instance map
 * (rc::DoorInstance), ensure_instance's read of the node, and the existence probes that still ask cortex for the
 * camera (silhouette, aperture frame, obliquity).
 *
 * ★The FIT itself — observation (the assigned mask slice → candidate / residual points), inference (support-bank
 * ingest + one recursive belief update), the leaf angle, the pre-belief geometry and read-back — moved into the
 * DSR-free DoorCore (src/core/door_fit.h, Task 4b), which runs it inside DoorCore::step(). SpecificWorker keeps the
 * DSR write-back (CoreWorld::after_fit → publish_door_cycle) and the post-fit epistemic / affordance / Qt steps.
 * Plain class (no Q_OBJECT).
 */

#pragma once

#include <algorithm>

#include <cmath>

#include "../../common/exclusion/exclusion.h"   // rc::exclusion::Claim (SHARED)
#include "../../common/contour_edge/contour_edge_check.h"  // rc::edges::PreparedFrame / contour_edge_support

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation
#include <dsr/api/dsr_camera_api.h>

#include "../../common/contour_edge/contour_edge_project.h"   // rc::edges::ContourSet — the shared silhouette + its null

#include "core/door_fit.h"    // rc::door::core::DoorFit — the fit, in the DSR-free DoorCore
#include "door_config.h"        // rc::DoorConfig
#include "door_instance.h"      // rc::DoorInstance, DoorState
#include "door_model.h"         // DoorModel / DoorModelParams
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "../../common/occlusion/occlusion.h"   // rc::occlusion::{cone_blocks, walls_block} — shared LoS occlusion
#include "door_scene_graph.h"
#include "door_semantic_field.h"   // rc::SemanticProbField (retina's graded posterior)
#include "door_frame_evidence.h"   // rc::door::FrameProbe — the aperture-frame existence probe
#include "door_room_band.h"        // rc::door::RoomBand — where a door sits relative to the room polygon
#include <opencv2/core.hpp>

namespace rc {

// Pixel-level silhouette existence evidence for ONE door (see DoorFitter::compute_silhouette_existence).
// Feeds rc::exist::mask_evidence — the shared channel table/chair use — so removal is a Bayesian decision on
// P(exists), never a miss-counter or a detectability heuristic.
struct DoorSilhouette
{
    float e_occ  = 0.0f;       // predicted samples LIT by a "door" mask     ⇒ still there
    float e_free = 0.0f;       // predicted samples lit by NOTHING           ⇒ predicted-but-absent ("gone")
    int   n_total      = 0;    // silhouette samples attempted (the whole panel face)
    int   n_detectable = 0;    // samples inside the REAL camera frustum and un-occluded (0 ⇒ not probed ⇒ HOLD)
    int   n_central    = 0;    // detectable samples in the central image region (the robot is LOOKING at it)
    // Silhouette centroid accumulators (image px) + the frame size, for the size-invariant central_frac
    // below. Summed over the DETECTABLE samples only, so an occluded or off-frustum part of the object
    // cannot drag the centroid toward a region the camera never saw.
    double sum_col = 0.0, sum_row = 0.0;
    int    img_w = 0, img_h = 0;
    // ★THE BELIEF'S OWN PROJECTED CONTOUR — the four corners of the leaf face as THIS agent projected
    // them, with THIS agent's camera and transform, plus the null it is scored against and the DEPTH it
    // predicted at each corner. Retained so the contour checks run on exactly the contour the existence
    // channel is defending. retina draws a similar rectangle for the human, but from the DSR node's
    // oriented box and a transform pinned to a different stamp — close, never identical, and a defence
    // measured on a not-quite-right contour is not a defence.
    //
    // ★THE CONTROLS ARE BUILT IN 3-D, ALONG THE LEAF'S OWN PLANE, then projected — not displaced in
    // image pixels. The contour check needs a null of the form "the same shape somewhere it could
    // equally have been". Displacing along the wall IN THE IMAGE is that null only while the leaf lies
    // in the wall; once it swings, the displaced copies stop sampling wall and start sampling the open
    // doorway and the room beyond, both edge-rich, so s_true < s_control and the channel votes to DELETE
    // the door precisely when it is open (measured: dL −1.4 to −2.6 per cycle at phi 35-45°). Displacing
    // along the leaf's OWN +x (hinge → free edge) and reprojecting is the same null at phi = 0 and stays
    // valid at every other angle, so the channel never has to stand down. That removes the abstention
    // rule, which was gating on a phi that is not reliable enough to gate on.
    //
    // Empty `contour.face` when any corner fell behind the camera ⇒ NOT MEASURED, never a refutation.
    // Construction is shared: common/contour_edge/contour_edge_project.h.
    rc::edges::ContourSet contour;
    int   n_occluded   = 0;    // in-frustum samples hidden behind a nearer NON-door mask
    // ── PROVENANCE OF THE OCC/FREE SPLIT ────────────────────────────────────────────────────────────
    // Whether e_occ/e_free above are the marginal over the phi posterior or a single-angle rendering,
    // and how concentrated that posterior was. Reported because a consumer must be able to tell an
    // absence the model is confident about from one it could only average over — and because a silent
    // switch between the two would make every earlier log incomparable with every later one.
    // ★THE MARGINAL COUNTS ARE KEPT SEPARATE FROM THE MODAL ONES ON PURPOSE. `n_detectable`,
    // `n_central`, `n_cells` and the centroid sums all describe ONE rendering — the leaf at phi_est —
    // and resolvability()/central_frac() divide by them. Overwriting n_detectable with a weighted
    // average across angles while leaving n_cells and the centroid at the modal pose would put a
    // numerator and a denominator from two different measurements into one ratio, which is the exact
    // shape of error that has produced three false findings in this project. So the marginal lives in
    // its own fields and is consulted only where absence is weighed.
    bool  phi_marginalised = false;
    int   phi_n_hyp        = 0;      // opening angles carrying weight
    float phi_w_max        = 0.0f;   // largest normalised weight; ~1/n_hyp = flat, ~1 = identified
    float n_det_marg   = -1.0f;      // E[detectable samples] over the phi posterior; <0 = not computed
    float n_total_marg = -1.0f;      // E[attempted samples]  over the phi posterior; <0 = not computed
    // Did the sensor look at this door AT ALL — under any opening angle the data still permits? The
    // HOLD branch turns on this and not on the modal count: a leaf whose fitted angle happens to point
    // it out of frame has not thereby become unobserved, and charging that as "not probed" would freeze
    // a phantom just as surely as charging it as absence would delete a real door.
    bool probed() const { return phi_marginalised ? n_det_marg > 0.5f : det_w > 0.0f; }
    // Σ over the detectable samples of 1 − P(the sample lies past the room-side face, i.e. is seen THROUGH
    // the opening) (door_room_band.h). Equal to n_detectable for a leaf that stays in the room or in its own
    // aperture plane; smaller for a leaf swung OUT through the aperture, a view the silhouette model does not
    // represent. e_occ/e_free are weighted the same way, and in_fov_frac() divides THIS by n_total.
    float det_w        = 0.0f;
    int   n_cells      = 0;    // DISTINCT pixel cells the detectable silhouette covers (see resolvability)
    float mean_range_m = 0.0f; // mean camera→sample distance over the detectable samples
    // ── POSTERIOR SAMPLED UNDER THE CONTOUR (retina's semantic_class_probs) ──────────────────────
    // Accumulated over the DETECTABLE samples only: a sample the camera could not have seen carries no
    // information about what the classifier thought there. `field_n` 0 ⇒ the field was unavailable
    // (ungraded model, or retina down) and every consumer must behave exactly as it did before.
    double field_sum = 0.0;    // Σ P(door) over detectable samples
    float  field_max = 0.0f;   // max P(door) over detectable samples
    int    field_n   = 0;
    float  field_bg  = 0.0f;   // mean P(door) over the WHOLE field this frame — the comparison population
    // Mean P(door) under the contour. Absolute value is NOT the signal (the classifier reads a plainly
    // visible door at 0.265 while calling it a wall at 0.676); the CONTRAST against field_bg is.
    float field_mean() const { return field_n > 0 ? static_cast<float>(field_sum / field_n) : 0.0f; }
    // How much of the door the sensor could actually have seen from here. Absence is evidence of removal only
    // in proportion to this — the rest is epistemic surprise ("I cannot resolve this from here"), not absence.
    float in_fov_frac() const
    {
        if (phi_marginalised and n_total_marg > 0.0f)
            return std::clamp(n_det_marg / n_total_marg, 0.0f, 1.0f);
        return n_total > 0 ? det_w / static_cast<float>(n_total) : 0.0f;
    }
    // ★★CENTRALITY IS ABOUT WHERE THE OBJECT IS, NOT HOW MUCH OF IT FITS IN A BOX.
    // The old form counted the FRACTION of silhouette samples landing inside a 50%x50% central box, which
    // is a SIZE measurement wearing an attention label: an object bigger than the box can never score well
    // from ANY position. Measured on hood 2026-08-17 (n=22 323 in-view cycles):
    //     roi_fill 0.00-0.25  central_frac med 0.441
    //     roi_fill 0.25-0.50  central_frac med 0.080
    //     roi_fill 0.50-0.75  central_frac med 0.000
    // and p_detect = envelope(fill) * in_fov_frac * central_frac, where the envelope PEAKS around
    // fill 0.3-0.6 — so the two factors pull against each other, and hardest exactly where the view is
    // best. Net effect: median p_detect 0.078 while the detector actually fired on 38% of those cycles.
    // Since absence AND confirmation are both scaled by p_detect, that silenced the camera in both
    // directions and left the LiDAR carve to decide alone (ex_p 0.959 -> 0.053 on hood_1).
    // ★The size-invariant question is where the silhouette's CENTROID sits relative to the principal
    // point. chair_concept already scored it this way (1 - centroid_radius); this brings the rest into
    // line with the one that had it right, rather than inventing a sixth variant.
    // ★NOTE central_region_frac_ no longer participates: a normalised radius has no box to size.
    float central_frac() const
    {
        if (n_detectable <= 0 or img_w <= 0 or img_h <= 0) return 0.0f;
        const float cx = static_cast<float>(sum_col / n_detectable);
        const float cy = static_cast<float>(sum_row / n_detectable);
        const float xn = (cx - 0.5f * img_w) / (0.5f * img_w);   // [-1,1], 0 = principal point
        const float yn = (cy - 0.5f * img_h) / (0.5f * img_h);
        return std::clamp(1.0f - std::hypot(xn, yn), 0.0f, 1.0f);
    }
    // RESOLVABILITY ∈ (0,1]: the fraction of the sampling budget that lands in DISTINCT image cells — i.e. how
    // many independent pixels the panel subtends per unit of model. Scale-free and reference-free: it is 1 when
    // every sample resolves separately (close, face-on) and falls as samples collapse together (far away, or
    // foreshortened edge-on). This one covariate replaces the old ZedRangeFull/ZedRangeRef falloff AND the
    // obliquity ramp, with no tuning constant and — unlike that ramp, which returned exactly 0 below cos 0.20
    // and made a grazing phantom permanently unjudgeable — it never reaches zero while the door is in view.
    float resolvability() const { return n_detectable > 0 ? static_cast<float>(n_cells) / n_detectable : 0.0f; }
};

class DoorFitter
{
public:
    // ★THE FIT MOVED INTO THE DoorCore (Task 4b; src/core/door_fit.h). What stays here is what needs the graph: the
    // instance map (rc::DoorInstance = the core's rc::DoorFitState + the DSR-bound affordance handles), the node read
    // of ensure_instance, and the existence probes that still query cortex for the camera (Task 5 moves those).
    using DoorObservation = rc::door::core::DoorObservation;

    DoorFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                DoorConfig& cfg,
                MaskIngestor* mask_ingestor,
                DoorSceneGraph* scene_graph,
                rc::door::core::DoorFit& fit);

    // Create the instance for a "door_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    // The node's attributes and its room→door RT are read here; the rest is DoorFit::init_instance.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    std::unordered_map<std::uint64_t, DoorInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // ── Room-containment pose prior ───────────────────────────────────────────────────────────────
    // The room polygon (room frame) is a trusted NOMINAL model (authored from the SVG layout, never fitted),
    // so it is a legitimate strong prior: P(a door's centre outside the walls) ≈ 0. Loaded from the room
    // node's delimiting_polygon_{x,y}. Used to suppress out-of-room births AND remove an instance that a
    // localization glitch placed outside (it can't be reached by the sensor to vacate it — behind a wall).
    // ★ONE copy: the DoorCore's fit holds it (the interior point was never read).
    void set_room_geometry(const Eigen::Vector2f& /*interior*/, std::vector<Eigen::Vector2f> polygon)
    { fit_.set_room_polygon(std::move(polygon)); }
    bool has_room_polygon() const { return fit_.has_room_polygon(); }
    // The walls, for anything that needs them as GEOMETRY rather than as a containment prior — notably the
    // NBV sight test, which must know the wall a door is set into is opaque. See rc::nbv::wall_obstacles.
    const std::vector<Eigen::Vector2f>& room_polygon() const { return fit_.room_polygon(); }
    // camera→room at the current pose (ts=0, main thread); nullopt while the chain does not resolve. Public read of
    // the transform the silhouette uses, for the remembered-door look's absence channel.
    std::optional<Eigen::Matrix4d> camera_in_room() const { return room_T_zed_matrix(); }

    // PIXEL-LEVEL silhouette existence evidence: project the panel face into the ZED and split the predicted
    // samples into occupancy / absence / occluded / out-of-frustum. See DoorSilhouette + DoorInstance::existence.
    // This is the ONLY channel that may remove a door — it is the only one that can tell "looked and found
    // nothing" from "never looked".
    // `field` may be null / invalid, in which case the silhouette's field_* accumulators stay 0 and the
    // existence channel is bit-for-bit what it was before this channel existed.
    DoorSilhouette compute_silhouette_existence(const DoorInstance& inst,
                                                const rc::SemanticProbField* field = nullptr);
    // The APERTURE FRAME (jambs + lintel), projected and visibility-weighted, ready to be scored against
    // the RGB / depth frame with rc::door::score_frame. The frame does not move when the leaf swings, so
    // this is the existence probe that still works when the leaf is open — above all when it has swung OUT
    // of the room the robot is in. See door_frame_evidence.h.
    rc::door::FrameProbe compute_frame_probe(const DoorInstance& inst);
    // The wall band this agent's existence model uses (door_room_band.h), from Existence.WallThicknessM /
    // Existence.BoundarySigmaM. One definition for the containment prior, the leaf-beyond weight and the
    // frame probe's seat on the room-side face.
    [[nodiscard]] rc::door::RoomBand room_band() const { return fit_.room_band(); }
    // Signed offset of the APERTURE centre from the room polygon (> 0 outside); NaN with no polygon.
    [[nodiscard]] float aperture_band_offset(const DoorInstance& inst) const;
    // Central-image box fraction: a detectable sample inside [f, 1-f]² of the image counts as "central" (the
    // robot is looking AT the door, not merely clipping the wide frustum edge). Set once from config.
    void set_central_region_frac(float f) { central_region_frac_ = f; }
    // View-obliquity onto the door FACE ∈ [0,1]: |cos| of the camera→door ray against the door's face normal
    // (= the wall normal, known exactly from the wall frame). 1 = viewed square-on, → 0 = grazing/edge-on.
    // DIAGNOSTIC ONLY (a log column). Obliquity is no longer a separate factor in the existence decision: the
    // silhouette's n_cells already measures the panel's subtended image area, which an edge-on view collapses
    // on its own. Keeping both would double-count the same geometry — and the standalone ramp was what froze a
    // grazing phantom at oblq=0.15 for 1200 frames.
    float door_view_obliquity(const DoorInstance& inst) const;
    // true if q is inside the polygon, or outside by no more than margin_m (tolerance for a wall-hugging door
    // whose centroid noise pokes through the wall). No polygon loaded ⇒ always true (unknown room → no prior).
    bool point_in_room(const Eigen::Vector2f& q, float margin_m = 0.0f) const;
    bool should_log(const DoorInstance& inst) const { return fit_.should_log(inst); }

private:
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DoorConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    DoorSceneGraph*               scene_graph_   = nullptr;
    rc::door::core::DoorFit&      fit_;            // the DoorCore's fit (init_instance, the room polygon)
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    float                           central_region_frac_ = 0.25f;   // central-image box [f,1-f]² (set from config)

    std::unordered_map<std::uint64_t, DoorInstance> instances_;
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
};

}  // namespace rc
