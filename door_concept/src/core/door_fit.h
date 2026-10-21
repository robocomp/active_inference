// door_fit.h — the per-door FIT of door_concept, DSR-free: observe (the assigned mask slice → candidate / residual
// points), run_inference (one recursive full-covariance belief update, DoorBelief), the leaf angle (estimate_phi: the
// low LiDAR's rays through the doorway), the pre-belief geometry and the read-back into DoorState (refresh_geometry),
// the projected ROI and the localisation-chain covariance.
//
// ★MOVED, NOT REWRITTEN (Task 4b of .superpowers/sdd/2026-10-21-door-core-replay): the code and its ★ comments are
// DoorFitter's (door_fitter.cpp as of e902bf7f). What changed is ONLY where each value comes from:
//   · the mask packet, the low-LiDAR sweep, the foreign claims and the ego-motion: the DoorCore's views of the cycle's
//     input (core/sensing.h) instead of MaskIngestor / ConceptLidarIngestor / a graph walk;
//   · room ← zed and the zed intrinsics: the snapshot (RobotState / CameraIntrinsics, recorded at the top of the cycle)
//     instead of a ts=0 cortex query and DSR::CameraAPI mid-cycle;
//   · the chain covariance: cortex's InnerGaussianAPI::transform_point arithmetic, replayed over the RT chain the
//     snapshot recorded WITH the mask packet (MaskPacket::room_zed_chain, at the capture stamp) — chain_point_cov below;
//   · ★TIME: the cycle's t_steady_ns. The old fitter read steady_clock::now() inside estimate_phi / run_inference (the
//     leaf channel's dt, the along-wall registration's dt, the commanded-swing prior, the belief's age clock), so its dt
//     was measured between two MID-cycle reads that nobody recorded — a replay could never reproduce it. Now dt is the
//     interval between cycle tops: deterministic, and the same class of change as Task 4's ego-motion (one named
//     behaviour difference; task-4b-report.md quantifies it on the tapes);
//   · the stdout lines and the two CSV logs (etc/ai2_log.csv and its _phi twin): rows formatted here, through the
//     classic locale exactly as the old ofstream (global C++ locale "C") formatted them, and handed to a sink — the
//     adapter owns the files (a replay writes none).
// The legacy node-attribute observation path (last_sensing_frame / candidate_pts_att) is gone: no agent in this tree
// writes those attributes on a door node, so it could only ever answer "no fresh data" — which is what the core does.
#pragma once
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "door_cycle_io.h"
#include "sensing.h"
#include "../door_config.h"        // rc::DoorConfig (plain struct)
#include "../door_fit_state.h"     // rc::DoorFitState (the DSR-free half of rc::DoorInstance)
#include "../door_room_band.h"     // rc::door::RoomBand
#include "../../../common/exclusion/claims.h"   // rc::exclusion::Claim (SHARED, pure)

namespace rc::door::core
{

/// What observe() hands run_inference (was rc::DoorFitter::DoorObservation).
struct DoorObservation
{
    bool has_fresh_data = false;
    float explanation_ratio = 1.0f;
    std::vector<Eigen::Vector3f> candidate_pts;
    std::vector<Eigen::Vector3f> residual_pts;
};

/// The localisation/chain covariance J·Σ_chain·Jᵀ of a room-frame point, as cortex's InnerGaussianAPI computes it for
/// transform_point(room, {p_zed, 0}, zed, ts) — over the steps the snapshot recorded (room side first, inverted; then
/// down to the zed; cortex walks them in the REVERSE order). nullopt when the chain is empty (it did not resolve).
/// ★The arithmetic is cortex's, line for line (adjoint propagation, the pose-vector round trip, the 1e-6 central
/// differences), so the result is cortex's to the bit given the same steps; only the zed-frame point it is evaluated
/// at comes from this composition rather than InnerEigenAPI's (ruling: 1e-6 on recomputed transforms).
[[nodiscard]] std::optional<Eigen::Matrix3d> chain_point_cov(const std::vector<RtChainStep>& steps,
                                                             const Eigen::Vector3d& p_room);

/// Header lines of the two CSV logs whose rows DoorFit formats (the adapter writes them once, when it opens the file).
inline constexpr std::string_view kAi2CsvHeader =
    "cycle,node,npts,gated,energy,fe_baseline,fe_surprise,R,motion_var,depth_var,trunc_frac,range,clutter_frac,"
    "s,w,h,cx,cy,yaw,std_s,std_w,std_h,phi,"
    // ★APPENDED, never inserted: an existing reader keyed on column position keeps working.
    // ★phi_sigma IS THE COLUMN THAT MAKES phi READABLE. An angle without its width cannot be told apart from a guess,
    // and "off_plane" says whether the LiDAR branch had anything a CLOSED leaf could not explain: a shut leaf lies
    // exactly in the wall plane, so it is geometrically indistinguishable from the wall itself and LiDAR can only ever
    // see an OPEN one. A high off_plane with a low phi, or a low one with a high phi, is a contradiction the angle
    // alone cannot show.
    "leaf_pts,w_sdf,w_mask,w_edge,phi_sigma,phi_measured,off_plane,phi_cv,"
    "ray_hit0,ray_expl0,ray_signed0,"
    "phi_ninf,phi_beta,phi_lat,"
    "recess,std_recess\n";   // Task 5b: the leaf plane's depth beyond the wall line (DoorBeliefState::d) and its σ
inline constexpr std::string_view kPhiCsvHeader =
    // ★try96: s per cycle (it was logged only on the rare mask frames, so the slide had no trace) and the along-wall
    //   channel. loglik stays the LAST column (a ;-separated list), so the new ones go in front of it, by name.
    "t_ms,node,ox,oy,nrays,off_plane,ninf,beta,lat,prior_mean,prior_sigma,post_mean,post_sigma,measured,"
    "hit0,expl0,signed0,"
    "s,sd_s,reg_ok,reg_eps,reg_depth,reg_n,aw_runs,aw_z,aw_R,aw_nu,aw_beta,aw_pd,aw_b,aw_sd_b,"
    "hinge_llr,swing_llr,vp_n,vp_epoch,p_ok,pi_move,hazard,"   // try102: the leaf channel
    "loglik\n";
/// The _phi log's path: the ai2 path with "_phi" before the extension (as log_phi_csv derived it).
[[nodiscard]] std::string phi_csv_path(const std::string& ai2_csv_path);

class DoorFit
{
public:
    explicit DoorFit(const rc::DoorConfig& cfg) : cfg_(cfg) {}

    // ── the cycle's views (set by the DoorCore at the top of step(); the adapter's DoorFitter shares the polygon) ──
    void set_clock(std::int64_t t_steady_ns, std::int64_t t_wall_ms)
    {
        now_ = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(t_steady_ns));
        t_wall_ms_ = t_wall_ms;
    }
    void set_camera(const std::optional<Mat4d>& room_T_zed, const CameraIntrinsics& cam) { room_T_zed_ = room_T_zed; cam_ = cam; }
    void set_ego_motion(float lin_mps, float ang_radps) { ego_lin_mps_ = lin_mps; ego_ang_radps_ = ang_radps; }
    void set_sources(const MaskSource* masks, const LeafSweep* sweep, const std::vector<rc::exclusion::Claim>* claims)
    { masks_ = masks; sweep_ = sweep; claims_ = claims; }
    /// stdout lines and CSV rows: DiagRow{"stdout" | "ai2_log.csv" | "ai2_log_phi.csv", line}. Unset ⇒ nothing formatted.
    void set_diag(std::function<void(const DiagRow&)> d) { diag_ = std::move(d); }

    // ── the room-containment prior's polygon (room frame). ONE copy: DoorFitter::set_room_geometry writes it. ──
    void set_room_polygon(std::vector<Eigen::Vector2f> polygon) { room_polygon_ = std::move(polygon); }
    [[nodiscard]] const std::vector<Eigen::Vector2f>& room_polygon() const { return room_polygon_; }
    [[nodiscard]] bool has_room_polygon() const { return room_polygon_.size() >= 3; }
    [[nodiscard]] rc::door::RoomBand room_band() const { return {cfg_.exist_wall_thickness_m, cfg_.exist_boundary_sigma_m}; }

    // ── instance lifecycle ──
    /// The DSR-free half of DoorFitter::ensure_instance: `init_state` is what the node said (width_m/depth_m/height_m,
    /// the room→door RT); the birth seed noted for this id (consumed) overrides its centre, non-finite fields are reset,
    /// and the cached geometry is seeded from the box (refresh_geometry). Overwrites `inst`.
    void init_instance(rc::DoorFitState& inst, std::uint64_t id, std::string name, rc::DoorState init_state);
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }
    void note_reacquire(std::uint64_t id, const rc::DoorBelief& belief) { restore_seeds_[id] = belief; }
    void seed_bearing_hypothesis(rc::DoorFitState& inst, const Eigen::Vector2f& robot_xy, float azimuth,
                                 float nominal_range, float along_std, float across_std, float yaw_std);

    // ── the per-cycle fit ──
    DoorObservation observe(rc::DoorFitState& inst);
    float run_inference(rc::DoorFitState& inst, const DoorObservation& observation);
    void estimate_phi(rc::DoorFitState& inst);
    void refresh_geometry(rc::DoorFitState& inst);

    [[nodiscard]] bool  confirm_only(const rc::DoorFitState& inst) const;
    [[nodiscard]] float motion_magnitude(const rc::DoorFitState& inst) const;
    [[nodiscard]] float periphery_penalty(const rc::DoorFitState& inst) const;
    [[nodiscard]] bool  should_log(const rc::DoorFitState& inst) const;
    [[nodiscard]] float ego_lin_mps() const   { return ego_lin_mps_; }
    [[nodiscard]] float ego_ang_radps() const { return ego_ang_radps_; }

    struct WallFrame { bool ok = false; Eigen::Vector2f O{0.0f, 0.0f}, u{1.0f, 0.0f}; float len = 0.0f; };
    [[nodiscard]] WallFrame associate_wall(std::span<const Eigen::Vector2f> pts_xy,
                                           const std::optional<Eigen::Vector2f>& cam_xy) const;

private:
    [[nodiscard]] rc::DoorBeliefParams make_belief_params() const;
    [[nodiscard]] rc::DoorModelParams  make_model_params() const;
    void compute_projected_roi(rc::DoorFitState& inst);
    void compute_chain_cov(rc::DoorFitState& inst);
    void log_ai2_csv(const rc::DoorFitState& inst, int npts, float R, bool gated, float energy);
    void log_phi_csv(const rc::DoorFitState& inst, const rc::ai::LidarRays& rays,
                     const std::vector<std::pair<float, float>>& loglik,
                     float prior_mean, float prior_sigma, float post_mean, float post_sigma, bool measured);
    void say(std::string line) const { if (diag_) diag_({"stdout", std::move(line)}); }
    [[nodiscard]] bool leaf_sweep_present() const { return sweep_ != nullptr and sweep_->present(); }

    const rc::DoorConfig& cfg_;
    std::function<void(const DiagRow&)> diag_;
    const MaskSource* masks_ = nullptr;
    const LeafSweep*  sweep_ = nullptr;
    const std::vector<rc::exclusion::Claim>* claims_ = nullptr;
    std::chrono::steady_clock::time_point now_{};
    std::int64_t t_wall_ms_ = 0;
    std::optional<Mat4d> room_T_zed_;
    CameraIntrinsics cam_{};
    float ego_lin_mps_ = 0.0f, ego_ang_radps_ = 0.0f;
    std::vector<Eigen::Vector2f> room_polygon_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;    // tracker-provided birth XY (note_birth)
    std::unordered_map<std::uint64_t, rc::DoorBelief> restore_seeds_;   // re-acquired identity's belief (note_reacquire)
    std::size_t n_explained_away_ = 0;                                  // points dropped (diagnostic)
};

}  // namespace rc::door::core
