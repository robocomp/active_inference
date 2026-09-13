/*
 * door_fitter.h
 *
 * The active-inference core of door_concept (mirrors bottle_concept/bottle_fitter.h). Owns the
 * per-door instance map and runs the AI2 full-covariance belief update for each "door_*" node:
 *   - instance lifecycle (ensure_instance + the DoorModel factory),
 *   - observation: split the selected mask's support points into on-surface vs off-surface sets,
 *   - inference: support-bank ingest + one recursive belief update (DoorBelief) with the mask-motion
 *     channel as the observation precision R / bias gate, written back into inst.model,
 *   - the door-owned support-point memory (ownership gate + FNV cell keys).
 *
 * Collaborates with MaskIngestor (masks) and DoorSceneGraph. SpecificWorker keeps the orchestration
 * (process_door_node), the DSR write-back call, and the post-fit epistemic / affordance / Qt steps.
 * Plain class (no Q_OBJECT).
 */

#pragma once

#include <algorithm>

#include <cmath>

#include "../../common/exclusion/exclusion.h"   // rc::exclusion::Claim (SHARED)
#include "../../common/rgb_ingestor/rgb_ingestor.h"     // rc::RgbIngestor (leaf-tracker pixels)
#include "../../common/contour_edge/contour_edge_check.h"  // rc::edges::PreparedFrame / contour_edge_support

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation
#include <dsr/api/dsr_camera_api.h>

#include "../../common/contour_edge/contour_edge_project.h"   // rc::edges::ContourSet — the shared silhouette + its null

#include "door_config.h"        // rc::DoorConfig
#include "door_instance.h"      // rc::DoorInstance, DoorState
#include "door_model.h"         // DoorModel / DoorModelParams
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "../../common/occlusion/occlusion.h"   // rc::occlusion::{cone_blocks, walls_block} — shared LoS occlusion
#include "door_scene_graph.h"
#include "door_semantic_field.h"   // rc::SemanticProbField (retina's graded posterior)
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
    bool probed() const { return phi_marginalised ? n_det_marg > 0.5f : n_detectable > 0; }
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
        return n_total > 0 ? static_cast<float>(n_detectable) / n_total : 0.0f;
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
    // The other concepts' standing claims on space (common/exclusion). Set once per cycle by the
    // worker, alongside the existence channel's copy, so the FIT and the EXISTENCE belief judge the
    // same geometry. Null ⇒ the rule is inert and the fitter behaves exactly as before.
    void set_foreign_claims(const std::vector<rc::exclusion::Claim>* c) { foreign_claims_ = c; }

    struct DoorObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    DoorFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                DoorConfig& cfg,
                MaskIngestor* mask_ingestor,
                DoorSceneGraph* scene_graph);

    // Create the instance for a "door_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Read the "masks" node → candidate/residual split against the current SDF.
    DoorObservation observe(DoorInstance& inst, const DSR::Node& node);
    // One recursive full-covariance belief update (DoorBelief) on this frame's mask points, with the
    // mask-motion channel as the observation precision R / bias gate. Writes the result into inst.model
    // so all downstream publish/viewer code is unchanged. Returns the update free energy.
    float run_inference(DoorInstance& inst, const DoorObservation& observation);

    std::unordered_map<std::uint64_t, DoorInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // ── Room-containment pose prior ───────────────────────────────────────────────────────────────
    // The room polygon (room frame) is a trusted NOMINAL model (authored from the SVG layout, never fitted),
    // so it is a legitimate strong prior: P(a door's centre outside the walls) ≈ 0. Loaded from the room
    // node's delimiting_polygon_{x,y}. Used to suppress out-of-room births AND remove an instance that a
    // localization glitch placed outside (it can't be reached by the sensor to vacate it — behind a wall).
    void set_room_geometry(const Eigen::Vector2f& interior, std::vector<Eigen::Vector2f> polygon)
    { room_interior_ = interior; room_polygon_ = std::move(polygon); }
    bool has_room_polygon() const { return room_polygon_.size() >= 3; }
    // The walls, for anything that needs them as GEOMETRY rather than as a containment prior — notably the
    // NBV sight test, which must know the wall a door is set into is opaque. See rc::nbv::wall_obstacles.
    const std::vector<Eigen::Vector2f>& room_polygon() const { return room_polygon_; }

    // Robot/camera ego-motion (room frame), from the transform chain — producer-independent. Call once per
    // compute cycle; run_inference then gates the pose/shape update to STILLNESS ("be-still-to-update").
    void  update_ego_motion();
    float ego_lin_mps() const   { return ego_lin_mps_; }
    float ego_ang_radps() const { return ego_ang_radps_; }
    // True when this frame must be CONFIRMATION-ONLY (no pose/shape change): robot linear/angular speed above
    // the still-level, OR the mask's own ego-motion corruption (motion_dotd) above its still-level.
    // (A/B FALLBACK path only — the hard gate; used when cfg.ai2_motion_confirm_only is true.)
    // The SAME admissibility, evaluated on a RAW mask slice that has no instance yet: may this frame move
    // geometry? Birth is gated on it, so a frame the fit would REFUSE can never create an object.
    // ★A frame that may not MOVE an existing belief must not CREATE one — see
    // common/instance_tracker/birth_evidence.h rule 2. Mirrors this agent's own `gated` computation; the
    // instance-only parts of that gate (anything read off a fitted projection) cannot apply pre-birth.
    bool  frame_admissible(const rc::MaskIngestor::MaskSlice& sl) const;
    bool  confirm_only(const DoorInstance& inst) const;
    // PIXEL-LEVEL silhouette existence evidence: project the panel face into the ZED and split the predicted
    // samples into occupancy / absence / occluded / out-of-frustum. See DoorSilhouette + DoorInstance::existence.
    // This is the ONLY channel that may remove a door — it is the only one that can tell "looked and found
    // nothing" from "never looked".
    // `field` may be null / invalid, in which case the silhouette's field_* accumulators stay 0 and the
    // existence channel is bit-for-bit what it was before this channel existed.
    // ★M1 — PHI AS A REAL DOF. Estimated per cycle by scoring candidate leaf angles against the door
    // mask actually present in the image, with the agent's own actuation command as the prior. Returns
    // the chosen phi and writes phi_est / phi_support onto the instance. Must run BEFORE the leaf pose
    // is read, since every projection downstream is built from it.
    void estimate_phi(DoorInstance& inst);

    DoorSilhouette compute_silhouette_existence(const DoorInstance& inst,
                                                const rc::SemanticProbField* field = nullptr);
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
    float motion_magnitude(const DoorInstance& inst) const;   // combined ego-motion speed (m/s)
    float periphery_penalty(const DoorInstance& inst) const;  // off-axis penalty ∈ [0,1] (0 on-axis → 1 at periph_ref)
    // true if q is inside the polygon, or outside by no more than margin_m (tolerance for a wall-hugging door
    // whose centroid noise pokes through the wall). No polygon loaded ⇒ always true (unknown room → no prior).
    bool point_in_room(const Eigen::Vector2f& q, float margin_m = 0.0f) const;

    // Wall frame a door is anchored to: near corner O and along-wall unit u (room frame). A door lives IN a
    // wall, so its belief works in this frame (θ=[s,w,h]); the fitter associates the door to the nearest
    // room-polygon wall at birth and sets it on the belief. Mirrors cabinet_concept's build_wall_ref.
    struct WallFrame { bool ok = false; Eigen::Vector2f O{0.0f, 0.0f}, u{1.0f, 0.0f}; float len = 0.0f; };
    WallFrame nearest_wall(const Eigen::Vector2f& q) const;
    bool should_log(const DoorInstance& inst) const;
    // Part B (chain covariance): enable adding the localization/chain term J·Σ_chain·Jᵀ (measurement
    // frame → room, capture-stamp pinned) per instance, read by the scene-graph's RT-cov write.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled);

    // ★THE LEAF TRACKER NEEDS PIXELS, NOT JUST MASKS. estimate_phi scored each candidate angle by the
    // overlap between the predicted leaf and the semantic "door" mask — which works while the leaf fills
    // the aperture and fails completely once it swings out of it, because the mask then covers the FRAME
    // and the leaf is somewhere else entirely. Measured 2026-09-13 on a plainly open door: overlap 0.013
    // at EVERY angle, so the estimate drifted to where the residual noise was faintest and parked at
    // 10 deg on a door standing open past 90.
    // An open leaf is a large surface with strong boundaries and no useful label, so the evidence that
    // survives is GRADIENT. Giving the fitter the RGB lets the same contour statistic that already
    // audits the door's existence also DRIVE the angle — one measurement, two consumers, instead of a
    // channel that could only ever say "your angle is wrong" after the fact.
    // Non-owning; may be null, in which case the tracker falls back to mask overlap alone.
    void set_rgb_source(const rc::RgbIngestor* rgb) { rgb_src_ = rgb; }

    // The newest low-LiDAR sweep, already in the ROOM frame. Staged once per cycle by the worker; the
    // hinge branch selects from it per door. Copy-free: the ingestor owns the storage for the cycle.
    // ★THE ORIGIN IS NOT OPTIONAL. A LiDAR return is not a point, it is "the ray from HERE in this
    // direction stopped at range r". Without the origin the only question askable is "is this endpoint on
    // the door", and a ray that flew THROUGH an open doorway has its endpoint in the next room — from the
    // endpoint alone it is indistinguishable from any other far-away point. From the origin it is the
    // evidence: that ray traversed the aperture, and a closed leaf would have stopped it.
    void set_leaf_points(const std::vector<Eigen::Vector3f>& pts, const Eigen::Vector3f& origin)
    { leaf_pts_ = &pts; leaf_origin_ = origin; }
    // Room-frame XY a NEWLY born instance's model should cold-start at (from the tracker's detection).
    // The room→door RT written at birth is not reliably composable the same cycle, so without this the
    // model would start at 0,0; consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }

    // IDENTITY RE-ACQUISITION: hand a newly created node the CONVERGED belief of the door it is bringing back,
    // so it resumes that geometry (state + Σ, i.e. the accumulated evidence) instead of cold-starting at the
    // w,h template priors. Consumed once by the lazy belief init in run_inference. The existence log-odds is
    // deliberately NOT restored — the door was removed because absence evidence won, so its EXISTENCE is
    // re-earned from the birth prior even though its SHAPE is remembered.
    void note_reacquire(std::uint64_t id, const DoorBelief& belief) { restore_seeds_[id] = belief; }

    // Part C-birth: initialise `inst` as a bearing-only hypothesis — belief mean placed at `nominal_range`
    // along the ray from `robot_xy` at `azimuth`, with a broad along-ray / tight across-ray Σ (see
    // DoorBelief::seed_bearing). Sets ai2_initialized + is_bearing_hypothesis and writes the mean into the
    // model so the scene-graph/viewer show the hypothesis on the ray. No depth mask needed.
    void seed_bearing_hypothesis(DoorInstance& inst, const Eigen::Vector2f& robot_xy, float azimuth,
                                 float nominal_range, float along_std, float across_std, float yaw_std);

private:
    const std::vector<rc::exclusion::Claim>* foreign_claims_ = nullptr;
    mutable std::size_t n_explained_away_ = 0;   // points dropped this cycle (diagnostic)
    DoorBeliefParams make_belief_params() const;   // config → belief params (shared by init + hypothesis seed)
    // The SINGLE authoring point for a door's geometry: refresh the cached aperture / leaf / leaf_pose from
    // the belief and write the room-frame read-back into DoorState. Called from ensure_instance (pre-belief),
    // the bearing seed, and the per-cycle write-back — the only three places a door's shape can change.
    void refresh_geometry(DoorInstance& inst);
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(DoorInstance& inst);
    // Part B: localization/chain cov J·Σ_chain·Jᵀ at the door centre (measurement frame → room, zero
    // input cov), stored on the instance for the RT-cov write. No-op unless set_chain_cov_source enabled.
    void compute_chain_cov(DoorInstance& inst);

    void ingest_observation_support(DoorInstance& inst, const DoorObservation& observation);

    DoorModelParams  make_model_params() const;

    // Append one AI2 belief row (state + Σ diag + range/motion) to cfg_.ai2_csv_path. No-op if empty.
    void log_ai2_csv(const DoorInstance& inst, int npts, float R, bool gated, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    Eigen::Vector3f                leaf_origin_ = Eigen::Vector3f::Zero();   // bpearl centre, room frame
    const std::vector<Eigen::Vector3f>* leaf_pts_ = nullptr;   // bpearl sweep, room frame (set_leaf_points)
    const rc::RgbIngestor*         rgb_src_     = nullptr;   // leaf-tracker contour evidence (set_rgb_source)
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;
    bool                           chain_cov_enabled_ = false;
    DoorConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    DoorSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    float                           central_region_frac_ = 0.25f;   // central-image box [f,1-f]² (set from config)

    std::unordered_map<std::uint64_t, DoorInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (note_birth)
    std::unordered_map<std::uint64_t, DoorBelief> restore_seeds_;      // re-acquired identity's belief (note_reacquire)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::vector<Eigen::Vector2f>   room_polygon_;       // room-frame delimiting polygon (containment prior)
    Eigen::Vector2f                room_interior_ = Eigen::Vector2f::Zero();   // a known-interior point (centroid)
    // Ego-motion state (camera pose deltas → robot speed), updated once per cycle in update_ego_motion().
    float           ego_lin_mps_   = 0.0f;
    float           ego_ang_radps_ = 0.0f;
    Eigen::Vector3f prev_cam_pos_  = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_cam_fwd_  = Eigen::Vector3f::UnitY();
    bool            have_prev_cam_ = false;
    std::chrono::steady_clock::time_point prev_cam_tp_{};
    std::ofstream                  ai2_csv_;            // per-cycle AI2 belief log (optional)
    std::ofstream                  phi_curve_csv_;      // one row per HYPOTHESIS (leaf-tracker forensics)
    std::ofstream                  phi_points_csv_;     // one row per cycle: the selected cloud's geometry
};

}  // namespace rc
