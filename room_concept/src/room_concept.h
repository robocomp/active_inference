#pragma once
#include <unordered_map>
#include <utility>   // std::exchange
#include <array>

#include <memory>
#include <vector>
#include <deque>
#include <map>
#include <limits>
#include <thread>
#include <mutex>

#include "door_apertures.h"
#include "scan_slam.h"
#include "layout_estimator.h"   // DoorAperture — the snapshot below, and no cycle: it knows nothing of us
#include <condition_variable>
#include <atomic>
#include <variant>
#include <optional>
#include <string>
#include <fstream>
#include <functional>
#include <random>


#include <Eigen/Dense>
#include "common_types.h"
#include "buffer_types.h"
#include "room_model.h"
#include "corner_detector.h"
#include "wall_map.h"
#include "room_boxes_channel.h"   // the box layout sidecar; it reaches into nothing here
#include "rerun_logger.h"
#include "object_anchor_types.h"
#include "object_anchor_factor.h"
#include "image_edge_types.h"
#include "image_edge_factor.h"
#include "se2_preintegration.h"
#include "motion_calibration.h"
#include "wheel_noise_learner.h"
#include "rest_density_learner.h"
#include "reloc_search.h"
#include "surprise.h"
#include "motion_noise_vc.h"
#include "motion_noise_innov.h"
#include "pose_field_bias.h"
#include "mount_factors.h"
#include "drift_monitor.h"
#include "stride_span.h"
#include "localizer_params.h"
#include "update_result.h"
#include "window_types.h"

namespace rc
{
/**
 * RoomConcept
 * =============
 * Implementación mínima para estimar (o refinar) el estado conjunto robot-habitación usando SDF.
 *
 * Estado (5): [width, length, x, y, phi]
 *  - width/length: dimensiones completas (m)
 *  - x,y,phi: pose del robot respecto al centro de la habitación (habitación centrada en (0,0))
 *
 * Flujo previsto en esta fase:
 *  - set_initial_state(...) con el GT / hipótesis inicial de la habitación y pose aproximada.
 *  - update(...) optimiza esos 5 parámetros minimizando mean(SDF^2) vía Adam.
 */

class RoomConcept
{
public:
    // moved to localizer_params.h (refactor step 3)
    using Params = loc::Params;


    // moved to update_result.h (refactor step 3)
    using UpdateResult = loc::UpdateResult;
    using OdometryPrior = loc::OdometryPrior;
    using MotionPriorSource = loc::MotionPriorSource;
    using MotionPriorSelection = loc::MotionPriorSelection;
    using MotionIngressDebug = loc::MotionIngressDebug;

    // ===== Threading / Run Context =====
    /// External buffers that the localization thread reads from directly.
    struct RunContext
    {
        HighLidarBuffer* high_lidar_buffer   = nullptr;  // lidar + GT pose
        VelocityBuffer* velocity_buffer = nullptr;  // joystick / controller commands
        OdometryBuffer* odometry_buffer = nullptr;  // measured odometry (encoders/IMU)
        ImuBuffer* imu_buffer = nullptr;            // inertial samples, ~10x the odometry rate
        const SimClockMap* sim_clock = nullptr;     // producer sim clock <- local wall clock
    };

    /// Thread-safe command variants (pushed from UI thread, drained in run loop)
    struct CmdSetPolygon  { std::vector<Eigen::Vector2f> vertices; };
    struct CmdSetPose     { float x; float y; float theta; };
    struct CmdGridSearch  { std::vector<Eigen::Vector3f> lidar_points; float grid_res; float angle_res; };
    using Command = std::variant<CmdSetPolygon, CmdSetPose, CmdGridSearch>;

    RoomConcept() = default;
    ~RoomConcept();

    /// Set the run context (buffer pointers).  Must be called before start().
    void set_run_context(const RunContext& ctx) { run_ctx_ = ctx; }

    /// Start the internal localization thread.  Requires set_run_context() first.
    void start();

    /// Request the localization thread to stop and join it.
    void stop();

    /// True while the localization thread is running.
    bool is_running() const { return loc_running_.load(); }

    /// True once the first successful UpdateResult has been published.
    bool is_loc_initialized() const { return loc_initialized_.load(); }

    /// Thread-safe: get the latest UpdateResult (nullopt if not yet available).
    std::optional<UpdateResult> get_last_result() const;

    /// Register a callback fired from the LOCALIZER THREAD the instant a fresh (ok) result is stored,
    /// so the owner can publish it to the DSR graph WITHOUT waiting for the next compute() tick. The
    /// callback must be cheap and thread-safe (SpecificWorker marshals to the main thread via a
    /// Qt::QueuedConnection); it must NOT touch the DSR graph directly. Removes ~one compute-period of
    /// lidar→RT-publish latency.
    void set_on_result_ready(std::function<void()> cb) { on_result_ready_ = std::move(cb); }

    /// Dead-reckon a room←robot pose forward by a constant body velocity (adv=forward, side=lateral,
    /// rot=CCW, robot frame) over dt seconds, using the same midpoint-SE2 convention as
    /// integrate_velocity_over_window. Pure/const — used for high-rate predict-publish between lidar
    /// corrections. Returns the input pose unchanged if dt <= 0.
    Eigen::Affine2f predict_pose_forward(const Eigen::Affine2f& pose,
                                         float adv, float side, float rot, float dt) const;

    /// Optimizer-timing telemetry accumulated on the localization thread since the last call.
    struct OptTiming
    {
        std::uint32_t count       = 0;   // updates that ran (new lidar frames processed)
        std::uint32_t early_exits = 0;   // of those, how many skipped Adam/LBFGS (prediction good)
        double        avg_update_ms = 0.0;
    };
    /// Thread-safe: read AND RESET the optimizer-timing accumulators (call ~1 Hz from the UI).
    OptTiming take_optimizer_timing();

    /// Thread-safe convenience: returns [half_w, half_h, x, y, theta] or zeros.
    Eigen::Matrix<float,5,1> get_loc_state() const;

    /// Thread-safe: record the latest command sample entering the motion pipeline.
    void record_command_ingress(const std::string& source,
                                float raw_adv,
                                float raw_rot,
                                float normalized_adv,
                                float normalized_rot,
                                std::int64_t ts_ms);

    /// What the RT edge ACTUALLY carried, recorded by the worker after the kinematic clamp has folded
    /// in its un-applied residual. The clamp runs downstream of this class's own debug write, so
    /// cov_xx in the log is the PRE-clamp value while every consumer sees the post-clamp one — a gap
    /// that made the published sigma unobservable from any log, and produced a claim about it that had
    /// to be retracted. `clamp_hit` makes the clamp's firing rate a measurement rather than a comment.
    /// ⚠ ONE-FRAME LAG: the caller runs after this class has already written its row, so these values
    /// appear on the NEXT row. Join them to the frame BEFORE the one they sit on.
    void note_published_covariance(float pub_cov_xx, float pub_cov_tt, bool clamp_hit)
    { last_pub_cov_xx_ = pub_cov_xx; last_pub_cov_tt_ = pub_cov_tt; last_clamp_hit_ = clamp_hit; }

    /// ── Exploration drive + affordance outcomes, for the CSV ────────────────────────────────────
    /// `belief_age_s` / `belief_decay` ARE the exploration drive (EpistemicPlanner::belief_decay):
    /// decay 1 = just refreshed, → 0 = long unexplored and the drive is maximal. `outcome` is the LAST
    /// completed affordance's outcome, `completions` counts them.
    /// ★ THE COUNTER IS WHAT MAKES THE OUTCOME USABLE. An outcome column alone is a LEVEL that
    /// persists for thousands of rows after its one event, so any per-row statistic over it is
    /// dominated by whichever outcome happened to last longest. A completion is an EVENT; segment on
    /// the counter CHANGING, never on the outcome value.
    ///   outcome: 0 none · 1 satisfied · 2 timeout · 3 refused · 4 abandoned
    /// ⚠ Same ONE-FRAME LAG as note_published_covariance: written after this class's row.
    /// `tgt_x/tgt_y` = the affordance target currently published (NaN when none). ★ Without it the
    /// log cannot answer the first question anyone asks of a repeating robot — "is it going back to
    /// the SAME cell, or working through different ones?" — which is the difference between a stuck
    /// planner and normal exploration, and is invisible in a completion count.
    /// `pub_x/pub_y` = the target actually WRITTEN to the affordance node, and `pub_ok` = whether the
    /// last publish re-armed it. ★ These differ from tgt_x/tgt_y, and the difference is the failure
    /// mode: publish_target declines silently when the node is Completed and the proposal is
    /// unchanged, so the planner can be picking a fresh cell every cycle while the CONTROLLER still
    /// holds the one the robot is standing on. Only the published pair explains the controller.
    void note_exploration_drive(float belief_age_s, float belief_decay, int outcome, long completions,
                                float tgt_x, float tgt_y, float pub_x, float pub_y, bool pub_ok)
    {
        last_pub_tx_ = pub_x; last_pub_ty_ = pub_y; last_pub_ok_ = pub_ok ? 1 : 0;
        last_tgt_x_ = tgt_x;
        last_tgt_y_ = tgt_y;
        last_belief_age_s_    = belief_age_s;
        last_belief_decay_    = belief_decay;
        last_aff_outcome_     = outcome;
        last_aff_completions_ = completions;
    }

    /// The odometry increment the PREDICTOR used this frame: the selected motion prior's delta_pose,
    /// [dx, dy, dtheta] in the room frame, integrated over the interval the prediction spans.
    /// This is the quantity the predicted pose is built from — not a velocity sample, and not the raw
    /// stream — so it is the one place a bad prediction can be traced to a bad increment rather than
    /// to the optimiser. Mirrored under the motion-ingress lock: written on the localiser thread,
    /// read by the viewer on the main one.
    Eigen::Vector3f get_predictor_delta() const;
    /// The MEASURED odometry increment used for the last fusion — [dx, dy, dtheta] in the body frame,
    /// with `valid`/`fresh` saying whether there was one at all. Exposed for the calibration channel,
    /// which regresses this against the localiser posterior over the same interval; it is the only
    /// consumer that needs the raw increment rather than the fused prior.
    [[nodiscard]] const OdometryPrior &last_measured_prior() const { return last_measured_prior_; }

    /// Thread-safe: record the latest measured odometry sample entering the motion pipeline.
    void record_odometry_ingress(const std::string& source,
                                 float raw_adv,
                                 float raw_rot,
                                 float normalized_adv,
                                 float normalized_rot,
                                 std::int64_t ts_ms);

    /// Thread-safe: push a command to be executed on the localization thread.
    /// A closed pivot, handed over from the scene-graph/main thread. NOT applied here: the estimator
    /// is touched only from the localiser thread, so this queues and service_calibration() drains it.
    /// Same pattern as push_command, and for the same reason.
    struct ClosureObs { double truth_rad, turned_rad, rate_rad_s, sigma_s; };
    void push_calibration_closure(const ClosureObs& c)
    {
        std::scoped_lock lk(pending_closures_mutex_);
        pending_closures_.push_back(c);
    }

    /// Forget every measurement and delete the state file. Queued: applied on the localiser thread.
    void request_calibration_reset() { calib_reset_pending_.store(true); }

    void push_command(Command cmd);

    /// Wake the localization thread when a newer lidar scan is available.
    void notify_new_lidar(std::int64_t lidar_timestamp_ms);

    // ----- Initialization configuration -----
    void configure_room_from_polygon(const std::vector<Eigen::Vector2f>& polygon_vertices);
    void configure_room_from_rect(float width, float length);
    /// Estimate mode: no layout. The first scan seeds a placeholder box (viewers only); the walls
    /// learnt from the LiDAR replace it as soon as they close.
    void configure_room_estimate();
    /// SEARCHING vs LOCALIZING is normally fixed by RoomShape.MapMode at load. It is NOT fixed for
    /// the run: crossing out of this room through one of its doors puts the agent back into
    /// estimation for the space it entered (begin_room_estimation). `params.map_mode` is never
    /// written after config load — the runtime half is a separate atomic — so no reader of params
    /// races with the transition.
    bool estimating() const
    { return runtime_estimate_.load(std::memory_order_relaxed) or params.map_mode == Params::MapMode::Estimate; }

    /// ── LOCALIZING (a given layout) → SEARCHING (learn the one we just walked into) ──────────────
    /// Called when the robot is judged to have crossed out of this room (room_scene_graph's
    /// step_proto_room). Every field it touches belongs to the LOCALISER thread, which is the thread
    /// that calls it: step_proto_room runs inside RoomSceneGraph::update, itself called from
    /// PosePublisher::maybe_publish_corrected_pose, which pose_publisher.h declares to be the
    /// localiser thread. So this is an inline transition, not a deferred one.
    ///
    /// ★ IT DOES NOT NEED TO SWITCH RECOVERY OFF — THAT FALLS OUT. Recovery, relocalisation, the
    /// symmetry check and the prediction early-exit all gate on map_guided_checks_allowed() →
    /// map_ready(), which in estimate mode is map_ready_, and configure_room_estimate() clears it.
    /// Entering SEARCHING *is* leaving the relocalisation branch; there is no second switch, and a
    /// second switch would be a place for the two halves to disagree.
    ///
    /// ★ THE GAUGE DOES NOT MOVE. The internal frame stays where it was — it is only a gauge, and the
    /// walls of the new room are estimated in it. What is published is the proto-room's frame, a
    /// constant transform of it, and reanchor_map_frame's one-shot move is absorbed into that
    /// constant (take_pending_reanchor) so no consumer is ever teleported.
    /// `why` is logged verbatim; pass what decided it, not a category.
    void begin_room_estimation(std::string_view why);

    /// ── THE RE-ANCHOR, FOR WHOEVER PUBLISHES A FRAME DERIVED FROM THE INTERNAL ONE ───────────────
    /// reanchor_map_frame moves the internal gauge once, when the learnt polygon first closes, and it
    /// rewrites every internal quantity to match. A frame published as a fixed transform of the
    /// internal one (the proto-room) would therefore JUMP unless it absorbs the same move — measured
    /// at 92 deg and 0.74 m in a single frame, which is a teleport to every consumer holding it.
    /// So the moves are accumulated here as T_new_old and handed over exactly once. Localiser thread.
    /// Returns false (and leaves `T` untouched) when nothing is pending.
    bool take_pending_reanchor(Eigen::Affine2f& T);

    /// ── THE TWO STATES THIS AGENT IS EVER IN ─────────────────────────────────────────────────────
    /// SEARCHING   the layout is unknown and is being estimated. The walls ARE the estimate: they are
    ///             landmarks in the solver, the derived polygon has no independent standing (scoring
    ///             a pose against it would score the walls against themselves), and the map frame
    ///             needs a gauge because it is otherwise unobservable.
    /// LOCALIZING  the layout is known and FROZEN. Nothing about it is optimised any more — no
    ///             births, deaths, merges, re-derivations, wall updates or theta0 motion — and the
    ///             robot localises against it exactly as it does against a layout loaded from file:
    ///             the room's SDF, the same terms, the same constants, the same stability test.
    ///
    /// The transition is one-way and happens the first time the polygon closes, is re-anchored and
    /// every corner passes the publish bar. A layout loaded from file starts in LOCALIZING.
    ///
    /// ⚠ TODO (deliberately not built yet): a SLOW REFINEMENT channel for the LOCALIZING state — a
    /// low-rate thread that accumulates residuals against the frozen layout and proposes small LOCAL
    /// amendments (a column that was never seen from the exploring trajectory, an alcove behind
    /// furniture, a wall whose extent was cut short by occlusion) without ever reopening the global
    /// estimate. The distinction that makes it safe is the one this state machine draws: a local
    /// amendment is evidence about one edge, whereas re-derivation replaces the whole cycle, and it
    /// was re-derivation storms that destroyed a correct map on 2026-09-12. Whatever that channel
    /// proposes must be adopted the way the level-2 operators are — priced against what it explains,
    /// on a frozen population — never accepted because a residual was large.
    /// The two states are named ONCE, in layout_estimator.h, so the estimation half and the
    /// localisation half cannot drift apart on what they mean. See that header for the split and for
    /// what still has to move.
    using LayoutState = LayoutEstimator::State;
    LayoutState layout_state() const
    { return (not estimating() or wall_frozen_) ? LayoutState::Localizing : LayoutState::Searching; }
    /// Set by RoomSceneGraph while a start-up PROTO-room owns the published frame: the one-shot internal
    /// re-anchor at map_ready is then skipped (internal frame == published frame until promotion).
    void set_freeze_internal_frame(bool v) { freeze_internal_frame_.store(v, std::memory_order_relaxed); }
    bool searching()  const { return layout_state() == LayoutState::Searching; }
    /// Plain SLAM owns the pose: Estimate mode, still learning the layout, and RoomShape.PlainSlam on.
    bool plain_slam_driving() const { return params.plain_slam and estimating() and searching(); }
    /// The layout as the SLAM map shows it (FreeSpacePolygon::manhattan_layout, INTERNAL frame), handed in by the
    /// scene graph. While plain SLAM drives it REPLACES the wall map's derived polygon — so the viewer, the published
    /// polygon, promotion and the wall nodes all see the same layout. Latched: the last one handed in stays.
    void set_map_layout(std::vector<Eigen::Vector2f> poly)
    { std::scoped_lock lk(wall_map_mutex_); map_layout_ = std::move(poly); }
    bool has_map_layout() const { std::scoped_lock lk(wall_map_mutex_); return map_layout_.size() >= 3; }
    /// ESTIMATING → LOCALIZING at promotion (2026-10-10): the map layout becomes the GIVEN room — the SDF model's
    /// polygon, the corner detector's model and the derived polygon — and wall_frozen_ puts the agent in its
    /// pre-existing localisation branch (the solve against the room's SDF). Plain SLAM stops driving (it runs only
    /// while searching). Pose, window and covariance carry over. Localiser thread (the scene graph's promotion).
    void adopt_map_layout_as_given();
    bool map_layout_adopted() const { return map_layout_adopted_; }
    /// An ACCEPTED refinement edit (layout_refiner.h): the given room becomes `L` (INTERNAL frame) — same frame,
    /// same gauge, the SDF the localiser solves against is rebuilt; nothing else changes. Localiser thread.
    void refine_given_layout(const std::vector<Eigen::Vector2f>& L);
    /// SHADOW (2026-10-10): after the switch plain SLAM keeps running on its own map; this is the distance (m) between
    /// its pose and the pose the SDF localiser published this frame. A MODEL-ERROR signal (where the given room does
    /// not explain the scans), not a pose input. NaN while plain SLAM is the pose provider.
    float slam_disagreement() const { return slam_disagree_.load(std::memory_order_relaxed); }
    bool localizing() const { return layout_state() == LayoutState::Localizing; }
    /// The LEARNT layout specifically has been frozen — i.e. this run reached LOCALIZING by
    /// estimating the room rather than by being given one. Prefer localizing() unless the difference
    /// actually matters to the caller.
    bool layout_frozen() const { return wall_frozen_; }
    /// Estimate mode: the polygon closed, was re-anchored and may be published. Given mode: always.
    bool map_ready() const { return not estimating() or map_ready_.load(); }
    /// Recovery, relocalisation, symmetry and the prediction early-exit all judge a pose against the
    /// MAP. Until there is one (Estimate mode before closure) they have nothing to judge against.
    bool map_guided_checks_allowed() const { return map_ready(); }
    const std::vector<Eigen::Vector2f>& polygon_vertices() const { return init_polygon_vertices_; }
    /// r2 mount factors (plan 2026-10-08 Task 5): what the INGEST thread's vertical factor needs from the
    /// localiser -- the wall map and the pose -- as one COPY taken under a lock (never read the map from the
    /// ingest thread). Written on the localizer thread every tracked cycle. vel = world-frame rate (per s) of
    /// the last cycle, so the reader can carry the pose to its own sweep's stamp.
    struct MountSnapshot
    {
        bool valid = false;
        std::int64_t ts_ms = 0;
        Eigen::Vector3f pose = Eigen::Vector3f::Zero();   ///< x, y, theta (room frame)
        Eigen::Vector3f vel  = Eigen::Vector3f::Zero();   ///< d(x, y, theta)/dt, per second
        std::vector<Eigen::Vector2f> polygon;             ///< the room polygon (room frame)
    };
    MountSnapshot mount_snapshot() const { std::scoped_lock lk(mount_snap_mutex_); return mount_snap_; }

    /// LOW OBSTACLES (2026-10-09): returns between the floor and the robot's top, robot frame (xy), with the
    /// sweep's capture stamp. Posted by the ingest thread (LidarIngestor::low_obstacle_step), taken by the
    /// localiser thread (RoomSceneGraph) to stamp MATTER into the start-up proto's free-space raster — helios
    /// scans ABOVE counters/tables/chairs, so its beams alone claim the floor under them as free.
    struct LowObstacles { std::int64_t stamp_ms = 0; std::vector<Eigen::Vector2f> xy; };
    void post_low_obstacles(LowObstacles o)
    {
        std::scoped_lock lk(low_obs_mutex_);
        low_obs_.push_back(std::move(o));
        while (low_obs_.size() > 16) low_obs_.pop_front();   // bounded: ~8 s at 2 Hz
    }
    std::deque<LowObstacles> take_low_obstacles()
    {
        std::scoped_lock lk(low_obs_mutex_);
        return std::exchange(low_obs_, {});
    }
    /// ── FOR THE LIVE LAYOUT EXPLORER (layout_explorer.h), LOCALISER THREAD ONLY ──────────────────────────
    /// The wall map has ONE owner, the localiser thread (the viewer gets a copy inside UpdateResult). The
    /// scene graph's update() runs on that same thread (pose_publisher.cpp, maybe_publish_corrected_pose),
    /// which is the only caller allowed: a reference valid for the calling cycle, never stored, never
    /// handed to another thread. The polygon is the one the PUBLISH TEST sees (carried + live window
    /// information) — the explorer predicts its gain through what the agent itself believes.
    const wallmap::WallMap& wall_map_localiser_only() const { return wall_map_; }
    wallmap::Polygon explorer_polygon_localiser_only() const
    { return wall_map_.build_polygon_with(window_wall_information()); }
    std::vector<Eigen::Vector2f> nominal_room_polygon() const
    {
        if (estimating())
        {
            // The DERIVED polygon (empty until the walls close). Read from the main thread by the
            // scene graph and the viewer while the localisation thread rewrites it: hence the lock.
            std::scoped_lock lk(wall_map_mutex_);
            return derived_polygon_;
        }
        if (init_use_polygon_ && init_polygon_vertices_.size() >= 3)
            return init_polygon_vertices_;

        const float half_w = init_room_width_ * 0.5f;
        const float half_l = init_room_length_ * 0.5f;
        return {
            {-half_w, -half_l},
            { half_w, -half_l},
            { half_w,  half_l},
            {-half_w,  half_l}
        };
    }
    void set_seed_pose_file(const std::string& pose_file_path);

    void set_initial_state(float width, float length, float x, float y, float phi);
    void set_polygon_room(const std::vector<Eigen::Vector2f>& polygon_vertices);
    void set_robot_pose(float x, float y, float theta, bool manual_reset = true);  // Set robot pose manually (e.g., from UI click)
    float evaluate_pose_fit(const std::vector<Eigen::Vector3f>& lidar_points,
                                     int max_samples = 300) const;
    // A Model that EXISTS is not the same as a Model that can be read: allocation and
    // init_from_state/init_from_polygon are two steps, and the seed-pose bootstrap can allocate
    // before any scan arrives. Callers use this to decide whether get_current_state() is safe, so
    // it has to mean exactly that -- reporting "initialised" for a model with undefined state made
    // the viewer throw out of a Qt timer handler every tick.
    bool is_initialized() const { return model_ != nullptr and model_->has_state(); }

    /// True while a global grid search is running, and for `hold_ms` after one finishes. The hold
    /// exists because the GUI polls at ~15-30 Hz from another thread: a search that completes between
    /// two polls would otherwise never be seen, and the operator would have no way to tell a search
    /// from a freeze. Safe to call from any thread (two atomics, no lock).
    /// "RELOCALISING" as the operator means it: a search is running, OR the localiser's own lost-belief filter no
    /// longer believes it is tracking (P(TRACKING) < ½ — the label names the more probable state). The search alone
    /// lasts ~80 ms; after a kidnap the pose can be wrong for tens of seconds between searches, and the window said
    /// ROOM STABLE throughout (2026-10-18, user). Safe from any thread.
    bool is_relocalising() const
    {
        return is_grid_searching() or p_track_display_.load(std::memory_order_relaxed) < 0.5f;
    }
    bool is_grid_searching(int hold_ms = 1500) const
    {
        if (grid_search_active_.load(std::memory_order_relaxed))
            return true;
        const auto ended = grid_search_end_ms_.load(std::memory_order_relaxed);
        if (ended == 0) return false;
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return (now - ended) < hold_ms;
    }

    // Grid search for initial pose (solves kidnapping problem)
    // Returns true if a good pose was found, false otherwise
    bool grid_search_initial_pose(const std::vector<Eigen::Vector3f>& lidar_points,
                                   float grid_resolution = 0.5f,  // meters
                                   float angle_resolution = M_PI_4);  // 45 degrees
    /// The rival pose modes the localiser is carrying (thread-safe copy). Empty ⇔ the pose is unambiguous.
    std::vector<rc::reloc::PoseHypothesis> pose_hypotheses() const
    {
        std::scoped_lock lk(rival_snapshot_mutex_);
        return rival_snapshot_;
    }

    Eigen::Matrix<float,5,1> get_current_state() const
    {
        if (model_) return model_->get_state();
        return Eigen::Matrix<float,5,1>::Zero();
    }

    // moved to window_types.h (refactor step 3)
    using WindowSlot = loc::WindowSlot;
    using BoundaryPrior = loc::BoundaryPrior;

    UpdateResult update(const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar,
                        const std::vector<rc::VelocityCommand> &velocity_history,
                        const std::vector<rc::OdometryReading> &odometry_history);

    Params params;

    /// Hand the latest validated object anchors (gathered on the MAIN thread from the DSR graph)
    /// to the localizer.  Copied into the newest window slot at the next update().  Thread-safe.
    /// Room's private landmark estimates (node id → room-frame position) — DISPLAY only. Empty unless
    /// object_anchor_optimize_landmark is on. Thread-safe copy under the anchor lock.
    std::map<std::uint64_t, Eigen::Vector2f> object_landmarks() const
    {
        std::scoped_lock lk(object_anchors_mutex_);
        std::map<std::uint64_t, Eigen::Vector2f> out;
        for (const auto& [id, e] : landmark_estimates_) out.emplace(id, e.p);
        return out;
    }

    /// The anchors the localizer is currently using — for DISPLAY only (the 2D canvas overlay).
    /// Thread-safe; returns a copy, so the caller never holds the lock while drawing.
    std::vector<ObjectAnchorObs> object_anchors() const
    {
        std::scoped_lock lk(object_anchors_mutex_);
        return latest_object_anchors_;
    }

    /// How well the RGB projection is agreeing with the map, per cycle — the readout the viewer
    /// plots. `chi2_per_dof` is the scale-free one and the only one with a meaningful target: it is
    /// the weighted residual per degree of freedom, so 1.0 means the residuals are exactly the size
    /// the per-sample sigmas claim. Above 1 the projection disagrees with the map by more than the
    /// sensor model admits — which is a statement about the MOUNT, the map or the matching, not about
    /// noise. `r_rms_px` is the same disagreement in raw pixels, kept because a chi2 can be pushed
    /// around by the sigmas while pixels cannot.
    struct ImageEdgeStats
    {
        bool  valid        = false;
        float chi2_per_dof = 0.f;
        float r_rms_px     = 0.f;
        float loss_img     = 0.f;
        int   n_used       = 0;
        int   n_segments   = 0;
        std::int64_t ts_ms = 0;
    };
    /// ── FACTOR B of the camera experiment, measured in the shadow ───────────────────────────────
    /// The same window solved under the mount as it is (calibrated) and as it would be with the
    /// self-calibration removed (nominal). Both poses are published RAW and never differenced here:
    /// the pose error each implies needs ground truth, which lives in the agent and not in this
    /// class, and a pre-combined delta is exactly the pairing that hides a mismatch.
    struct FactorB
    {
        bool  valid = false;
        std::int64_t ts_ms = 0;
        Eigen::Vector3f pose_calibrated = Eigen::Vector3f::Zero();
        Eigen::Vector3f pose_nominal    = Eigen::Vector3f::Zero();
        Eigen::Vector3f correction      = Eigen::Vector3f::Zero();
    };
    FactorB get_factor_b() const
    {
        std::scoped_lock lk(factor_b_mutex_);
        return factor_b_;
    }

    ImageEdgeStats get_image_edge_stats() const
    {
        std::scoped_lock lk(image_edge_stats_mutex_);
        return image_edge_stats_;
    }

    /// Latest RGB edge evidence (compute thread produces, solve consumes). Thread-safe copy.
    ImageEdgeObs image_edges() const
    {
        std::scoped_lock lk(image_edges_mutex_);
        return latest_image_edges_;
    }
    void set_image_edges(ImageEdgeObs obs)
    {
        std::scoped_lock lk(image_edges_mutex_);
        latest_triple_points_ = obs.triple_points;   // BEFORE the move
        latest_image_edges_   = std::move(obs);
    }
    /// Display copy of the newest triple points. Deliberately NOT read through image_edges(): that
    /// holder is emptied by take_image_edges() as soon as a slot consumes it, so a viewer peeking
    /// there would show corners only on the rare tick it happened to win the race. Kept separately
    /// because the drawing must not compete with the solver for the evidence.
    std::vector<TriplePoint> triple_points() const
    {
        std::scoped_lock lk(image_edges_mutex_);
        return latest_triple_points_;
    }
    /// TAKE, not peek: the extraction is moved out and the holder left empty, so one image cannot be
    /// attached to two slots. The rates are independent (images ~30 Hz, slots ~20 Hz and irregular),
    /// so a peek would silently re-use the same measurement whenever a slot arrived without a fresh
    /// frame — the same evidence counted twice, which is exactly what the Woodbury marginalisation
    /// downstream exists to prevent within a single frame. A slot with no fresh image simply carries
    /// none, which is the honest state and what `obs.empty()` already means everywhere.
    ImageEdgeObs take_image_edges()
    {
        std::scoped_lock lk(image_edges_mutex_);
        return std::exchange(latest_image_edges_, ImageEdgeObs{});
    }

    void set_object_anchors(std::vector<ObjectAnchorObs> anchors)
    {
        std::scoped_lock lk(object_anchors_mutex_);
        latest_object_anchors_ = std::move(anchors);
    }

    /// ── OPEN DOORWAYS, IN THE ROOM FRAME ────────────────────────────────────────────────────────
    /// Written by the worker (main thread, where the shared InnerEigenAPI lives) and read by the
    /// localisation thread once per scan. Same shape as the object anchors above and for the same
    /// reason: a small immutable snapshot crosses the boundary, never the graph itself.
    /// Empty is the normal state — every door shut, or none known — and costs the scan nothing.
    void set_door_apertures(std::vector<DoorAperture> doors)
    {
        std::scoped_lock lk(door_apertures_mutex_);
        latest_door_apertures_ = std::move(doors);
    }
    std::vector<DoorAperture> door_apertures() const
    {
        std::scoped_lock lk(door_apertures_mutex_);
        return latest_door_apertures_;
    }

    /// ── P(THE ROBOT IS OUTSIDE THIS ROOM, THROUGH A DOOR) ──────────────────────────────────────
    /// Written by the scene graph (door_crossing.h), read by recovery. The SDF misfit is evidence of a
    /// LOST pose only to the extent this room is expected to explain the scan; past a doorway it is not,
    /// and snapping the pose back inside would undo a real crossing. So recovery weighs the misfit by
    /// (1 - p). Both sides run on the LOCALISER thread today (step_proto_room writes it from inside
    /// maybe_publish_corrected_pose; recovery reads it in the same update). Atomic so that stays true
    /// if a writer ever moves. It is NOT a hand-off between threads — do not add a queue for it.
    void set_outside_prob(float p) { outside_prob_.store(std::clamp(p, 0.f, 1.f)); }
    float outside_prob() const { return outside_prob_.load(); }

    // Process noise covariance (diagonal [x, y, theta])
    Eigen::Vector3f process_noise = {0.01f, 0.01f, 0.01f};

    // Current covariance estimate [3x3]
    Eigen::Matrix3f current_covariance = Eigen::Matrix3f::Identity() * 0.1f;
    // Last pose the optimizer produced that was finite — the bottom rung of the divergence-recovery
    // cascade (corners -> lidar/SDF -> odometry -> this).
    Eigen::Vector3f last_good_pose_ = Eigen::Vector3f::Zero();
    bool            last_good_pose_valid_ = false;

private:
   // ===== Threading internals =====
   RunContext run_ctx_;
   std::thread loc_thread_;
   std::atomic<bool> stop_requested_{false};
   std::atomic<bool> loc_running_{false};
   std::atomic<bool> loc_initialized_{false};

   mutable std::mutex result_mutex_;
   std::optional<UpdateResult> last_result_;

   // Latest object anchors from the graph (set on main thread, consumed by the localizer thread).
   float zupt_pred_gain_tr_ = 1.f, zupt_pred_gain_ro_ = 1.f;
   float last_sdf_polish_mm_ = 0.f;
   /// Position variance accumulated since the last real solve, used ONLY to decide whether the
   /// early-exit gate may still be trusted. Never published, never handed to the solver.
   float unopt_pos_var_ = 0.f;
   bool  forced_solve_last_cycle_ = false;
   bool  sdf_polished_this_cycle_ = false;
   std::atomic<bool> calib_reset_pending_{false};
   bool calib_state_loaded_ = false;
   std::int64_t calib_state_last_save_ms_ = 0;
   std::mutex pending_closures_mutex_;
   std::vector<ClosureObs> pending_closures_;
   mutable std::mutex image_edge_stats_mutex_;
   ImageEdgeStats     image_edge_stats_{};
   mutable std::mutex factor_b_mutex_;
   FactorB            factor_b_{};
   mutable std::mutex image_edges_mutex_;
   ImageEdgeObs       latest_image_edges_;
   std::vector<TriplePoint> latest_triple_points_;   ///< display copy, see triple_points()
   mutable std::mutex door_apertures_mutex_;
   std::atomic<float> outside_prob_{0.f};
   /// Set by begin_room_estimation; never cleared for the life of the run (a room learnt is not
   /// un-learnt by walking back). Read by estimating() from any thread, hence atomic.
   std::atomic<bool>  runtime_estimate_{false};
   /// Accumulated, un-consumed re-anchor (T_new_old). Localiser thread only; see take_pending_reanchor.
   Eigen::Affine2f    pending_reanchor_ = Eigen::Affine2f::Identity();
   bool               has_pending_reanchor_ = false;
   std::vector<DoorAperture> latest_door_apertures_;          ///< room frame, from the worker
   /// The same apertures carried into the ROBOT frame, rebuilt once per scan on the localisation
   /// thread. Kept as a member rather than passed down through a dozen signatures: every consumer of
   /// the observation weights is inside this class's own update cycle.
   std::vector<DoorAperture> doors_robot_;
   std::unordered_map<std::uint64_t, float> door_recess_logged_;   ///< last logged seat δ per door (log-on-change only)
   /// The early-exit criterion as it would have been WITHOUT the door weighting, and the fraction of
   /// the scan's weight the doors removed. The gate decides on the weighted value; these two say what
   /// that cost, so a discount that is silently blinding the gate is visible instead of inferred.
   float ee_metric_unweighted_ = 0.f;
   float ee_door_discount_ = 0.f;
   /// Per-point door weight for a robot-frame scan, or an undefined tensor when no door is open —
   /// which makes every consumer fall back to the plain unweighted reduction, i.e. exactly the
   /// behaviour of a room without doors.
   Eigen::VectorXf door_point_weights(const ScanPoints& points_robot) const;

public:
   /// The scan with every through-an-open-door return REMOVED, robot frame in and out. The canvas
   /// draws this so the filter is visible: if a door is open and the points beyond it are still on
   /// screen, the filter is not working, and that is a stronger test than any log line.
   /// `removed` gets the count. No open door ⇒ the input is returned unchanged.
   std::vector<Eigen::Vector3f> filter_through_door(const std::vector<Eigen::Vector3f>& points_robot,
                                                    int* removed = nullptr) const;
private:
   mutable std::mutex object_anchors_mutex_;
   std::vector<ObjectAnchorObs> latest_object_anchors_;

   /// Room's PRIVATE landmark estimates, keyed by graph node id: room-frame position + the precision it
   /// has accumulated. Persist across frames — that accumulation over diverse viewing azimuths is the
   /// whole point (a position-only observation gives 2 constraints, so one viewpoint never determines a
   /// landmark; see ray_anisotropic_cov's bearing-only limit). Born from the producer's belief, then
   /// refined by room's own observations alone.
   struct LandmarkEstimate
   {
       Eigen::Vector2f p = Eigen::Vector2f::Zero();
       Eigen::Matrix2f information = Eigen::Matrix2f::Zero();
   };
   std::map<std::uint64_t, LandmarkEstimate> landmark_estimates_;
   std::function<void()> on_result_ready_;   // fired (localizer thread) after a fresh ok result is stored

   std::mutex cmd_mutex_;
   std::vector<Command> pending_commands_;
    std::atomic<bool> commands_pending_{false};
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
    std::int64_t latest_notified_lidar_ts_ = std::numeric_limits<std::int64_t>::min();

   /// The localization loop body (runs on loc_thread_)
   void run();

   std::shared_ptr<Model> model_;
   std::int64_t last_lidar_timestamp = 0;
   UpdateResult last_update_result;
   bool needs_orientation_search_ = true;  // Search for best orientation on first update
   // Set when bootstrap seeded the model from the saved pose before any sweep had arrived, so the
   // seed could not be checked against the room yet. The first new frame runs validate_seed_pose.
   // Localizer-thread only (bootstrap and run() are both on loc_thread_) — no atomic needed.
   bool pending_seed_validation_ = false;
   bool validate_seed_pose(const std::vector<Eigen::Vector3f>& pts);

   // Grid-search liveness for the UI (written on the localizer thread, read on the GUI thread).
   std::atomic<bool>         freeze_internal_frame_{false};   // see set_freeze_internal_frame
   std::atomic<bool>         grid_search_active_{false};
   std::atomic<std::int64_t> grid_search_end_ms_{0};

   // Manual pose reset - skip optimization for a few frames
   int manual_reset_frames_ = 0;  // Counter to skip optimization after manual reset

   // Smoothed pose to reduce jitter (legacy, used only when rfe_window_size == 1)
   Eigen::Vector3f smoothed_pose_ = Eigen::Vector3f::Zero();  // [x, y, theta]
   bool has_smoothed_pose_ = false;

   // Flip instrumentation: last OPTIMIZED pose, to detect a ~180° yaw jump and dump why (see [FLIP] log).
   float flip_prev_x_ = 0.f, flip_prev_y_ = 0.f, flip_prev_th_ = 0.f;
   bool  flip_prev_valid_ = false;

   friend struct RelocSelftestAccess;   // tools/reloc_selftest.cpp drives the belief and the rivals directly
   // ===== Relocalisation state (see Params, "Relocalisation") =====
   // Three-state HMM filter over {TRACKING, LOST, MISMATCH} and the online tracking emission of ln s.
   // LOST and MISMATCH (the view changed: a door opened, furniture moved, the map is off) emit s
   // identically, so s alone cannot separate them — the SEARCH is the experiment that does: after a kidnap
   // some pose explains the scan the way tracking does; in a changed view none can.
   enum BeliefState : std::size_t { kTrack = 0, kLost = 1, kMismatch = 2 };
   std::array<double, 3> belief_{1.0, 0.0, 0.0};
   bool  belief_init_       = false;
   std::atomic<float> p_track_display_{1.f};   ///< P(TRACKING) mirrored for the GUI thread (is_relocalising)
   float lost_logodds_      = std::numeric_limits<float>::quiet_NaN();   // logit P(LOST) — for the logs
   float track_mu_ln_s_     = std::numeric_limits<float>::quiet_NaN();   // NaN ⇒ seed from the params
   float track_var_ln_s_    = std::numeric_limits<float>::quiet_NaN();
   /// Filter one frame's s; returns P(LOST). `outside` (P past a doorway) mixes the evidence away.
   /// `view_change` ∈ [0,1]: how new this frame's view is (pose step in the likelihood's basin). It scales the
   /// MISMATCH → TRACKING return and the emission learning; 1 = the old per-frame behaviour.
   float update_lost_belief(float s, float outside, float view_change = 1.f);
   Eigen::Vector3f belief_prev_pose_ = Eigen::Vector3f::Zero();
   bool  belief_prev_pose_valid_ = false;
   float last_search_top_weight_ = 0.f;   ///< posterior weight of the mode the last search committed/confirmed
   void  reset_lost_belief();                       // → TRACKING (after a move, or at start)
   void  set_belief_mismatch();                     // → MISMATCH (a search found nothing better to move to)
   /// `extra_var` ADDS to the learnt spread of ln s (never replaces it): the steady-state filter's fit and a
   /// freshly polished single-scan pose's fit are different quantities, and the second carries pose error too.
   double log_tracking_emission(float s, double extra_var = 0.0);
   double log_broad_emission(float s) const;
   /// True when a FRESH pose (a search mode, a rival) whose median |SDF| is s is better explained as TRACKING
   /// than as a broad misfit. Its spread is the learnt one plus the prior population spread of ln s.
   bool  explains_like_tracking(float s)
   {
       const double prior_var = static_cast<double>(params.reloc_track_log_sigma) * params.reloc_track_log_sigma;
       return log_tracking_emission(s, prior_var) > log_broad_emission(s);
   }
   /// Fire the search from the lost belief and fold its outcome back into the belief.
   void  relocalise_from_belief(const std::vector<Eigen::Vector3f>& lidar_points, float s, float p_lost);
   bool  last_search_moved_ = false;
   /// Commit a relocalised pose as a DISCRETE JUMP: model tensors, the prediction base (last_update_result —
   /// the next frame predicts from it, not from the tensors), smoothing, last-good pose, window and stride
   /// state, and the publisher's clamp-release epoch. Every relocalisation that moves the pose goes through here.
   void  commit_relocalised_pose(const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov);
   /// Section 8 of the localisation loop, one frame: rival modes, the lost belief, and the search it fires.
   /// A method (not inline in run()) so tools/reloc_selftest can drive it through the real update().
   void  relocalisation_step(const UpdateResult& res, const std::vector<Eigen::Vector3f>& lidar_points);
   std::atomic<std::uint64_t> reloc_epoch_{0};   // see UpdateResult::reloc_epoch
   /// A carried rival mode: its pose and its log-likelihood ratio against the committed pose.
   struct RivalMode { Eigen::Vector3f pose; float llr = 0.f; };
   std::vector<RivalMode> rivals_;
   mutable std::mutex rival_snapshot_mutex_;
   std::vector<rc::reloc::PoseHypothesis> rival_snapshot_;   // copy for the main thread (scene graph → planner)
   void publish_rival_snapshot();
   Eigen::Vector3f rival_ref_pose_ = Eigen::Vector3f::Zero();   // committed pose rivals were last moved with
   bool            rival_ref_valid_ = false;
   /// Move rivals with the robot's motion, score them against the committed pose, switch when SPRT passes.
   void step_rival_modes(const std::vector<Eigen::Vector3f>& lidar_points);
   // Likelihood raster for the current room polygon, rebuilt when the polygon changes.
   rc::reloc::LikelihoodRaster reloc_raster_;
   std::vector<Eigen::Vector2f> reloc_raster_poly_;
   rc::reloc::Params reloc_params() const;
   std::vector<Eigen::Vector2f> current_room_polygon() const;
   bool  last_search_confirmed_incumbent_ = false;   // the last search's top mode WAS the pose it was handed
   float pending_episode_p_lost_ = std::numeric_limits<float>::quiet_NaN();   // for the episode row

   // ===== Loss/recovery episode log =====
   // One row per grid_search_initial_pose() call, written wherever it returns. The per-frame debug log
   // could only ever show recovery INDIRECTLY — I had to infer episodes from window_size resets and
   // proxy the early-exit metric through final_loss, which is exactly the kind of inference that has
   // been wrong twice in this work. This records what the search actually did, so the algorithm can be
   // improved from measurement instead of from a story about it.
   //
   // The load-bearing column is ESS (effective sample size of the softmax weights over the evaluated
   // poses): ~1 means one pose explains the scan and the fix is decisive; large means a flat valley of
   // near-equal candidates, i.e. the scan genuinely does not determine the pose and ANY winner drawn
   // from it is arbitrary. That distinction is invisible in a success/fail bit, and it is the one that
   // decides whether the answer is "search harder" or "go somewhere the geometry is informative".
   struct SearchEpisode
   {
       std::int64_t ts_ms = 0;
       const char*  trigger = "unknown";   // lost | mode_switch | seed | seed_validation | manual
       int          n_lidar = 0;
       float        incumbent_x = 0.f, incumbent_y = 0.f, incumbent_theta = 0.f;
       float        incumbent_loss = std::numeric_limits<float>::quiet_NaN();
       float        good_thr = 0.f;
       int          stage = -1;            // legacy: 0 yaw sweep, 1 local, 2 coarse, 3 fine · 10 = mixture search
       float        best_x = 0.f, best_y = 0.f, best_theta = 0.f;
       float        best_loss = std::numeric_limits<float>::quiet_NaN();
       float        topk_best = std::numeric_limits<float>::quiet_NaN();   // Stage-2 candidate spread:
       float        topk_worst = std::numeric_limits<float>::quiet_NaN();  // how flat is the valley
       float        ess = std::numeric_limits<float>::quiet_NaN();
       float        cov_xx = std::numeric_limits<float>::quiet_NaN();
       float        cov_yy = std::numeric_limits<float>::quiet_NaN();
       float        cov_tt = std::numeric_limits<float>::quiet_NaN();
       int          n_evals = 0;
       float        jump_m = 0.f;          // distance from the incumbent to the committed pose
       float        jump_rad = 0.f;
       bool         success = false;
       float        duration_ms = 0.f;
       float        beta = std::numeric_limits<float>::quiet_NaN();   // softmax temperature actually used
       int          n_points = 0;                                     // N behind the median's std error
       // ── mixture search (reloc_search.h); NaN / 0 on the legacy lattice ──
       int          n_modes = 0;
       float        mode_entropy = std::numeric_limits<float>::quiet_NaN();   // nats over mode weights
       float        top_weight = std::numeric_limits<float>::quiet_NaN();
       float        second_weight = std::numeric_limits<float>::quiet_NaN();
       int          n_yaws = 0;
       int          n_segments = 0;                                   // 0 ⇒ yaws came from the uniform sweep
       float        p_lost = std::numeric_limits<float>::quiet_NaN(); // belief that fired it (lost trigger)
   };
   // Why the next grid search was invoked. Set by the caller immediately before the call; the search
   // itself has no way to know, and "which trigger fires most and which of them actually resolve" is
   // the first question the episode log has to answer.
   const char*   search_trigger_ = "unknown";
   std::ofstream recovery_log_;
   std::string   recovery_log_path_;
   void write_search_episode(const SearchEpisode& e);

   // moved to window_types.h (refactor step 3)
   using WindowManager = loc::WindowManager;
   WindowManager window_mgr_;

   // ---- Strided-window state (see Params::window_stride_enabled) ----
   // Pose at which the last slot was ADMITTED, and the odometry delta + covariance accumulated since.
   // While the newest slot is being replaced rather than appended, the motion factor it carries has to
   // describe the whole interval back to the last admitted slot, not the last frame.
   Eigen::Vector3f stride_last_admitted_{0.f, 0.f, 0.f};
   bool            stride_has_admitted_ = false;
   // What the NEWEST slot's motion factor carries (delta, summed cov, chained preint): stride_span.h.
   rc::StrideSpan  stride_span_;
   bool preint_announced_ = false;   // one-shot "the propagated covariance is really in force" log

   // ---- Debug-log mirrors, set where the newest slot is BUILT so BOTH writer paths can emit them ----
   // The early-exit writer lives in a different function from the slot build, so anything it must log
   // has to be a member. Getting this wrong is how columns end up hardcoded to zero on one path only
   // (n_lidar still is), which then silently halves the usable population of every analysis.
   /// The appended tail of a debug-log row, emitted by BOTH writers so they cannot drift apart. One
   /// function rather than two copies precisely because two copies is how this file's columns got out
   /// of step before.
   void write_debug_tail();
   Eigen::Matrix3f last_slot_motion_cov_  = Eigen::Matrix3f::Zero();  // full 3x3, for the off-diagonals
   float           last_imu_cover_        = -1.f;  // fraction of segments whose dtheta came from the gyro
   // Running wheel/gyro heading ratio over IMU-covered segments. NaN until enough rotation has
   // accumulated to make the quotient mean anything (see the [ImuInject] guard) — a column that
   // reports a number for a parked robot is the defect that guard exists to prevent.
   float           last_wheel_gyro_ratio_ = std::numeric_limits<float>::quiet_NaN();
   float           last_pub_tx_ = std::numeric_limits<float>::quiet_NaN();  // what went ON THE NODE
   float           last_pub_ty_ = std::numeric_limits<float>::quiet_NaN();
   int             last_pub_ok_ = 0;                                        // did publish re-arm it?
   float           last_tgt_x_ = std::numeric_limits<float>::quiet_NaN();  // planner's internal choice
   float           last_tgt_y_ = std::numeric_limits<float>::quiet_NaN();
   float           last_belief_age_s_    = 0.f;   // s since the last refresh_belief()
   float           last_belief_decay_    = 1.f;   // exp(-age/belief_forget_time): the drive itself
   int             last_aff_outcome_     = 0;     // 0 none 1 satisfied 2 timeout 3 refused 4 abandoned
   long            last_aff_completions_ = 0;     // EVENT locator — segment on this changing
   float           last_pub_cov_xx_ = std::numeric_limits<float>::quiet_NaN();  // post-clamp, as published
   float           last_pub_cov_tt_ = std::numeric_limits<float>::quiet_NaN();
   bool            last_clamp_hit_  = false;
   int             last_preint_samples_   = 0;    // odometry samples summarised into this slot's factor
   float           last_preint_duration_s_ = 0.f; // and over how long — reveals strided chaining
   bool imu_injection_announced_ = false;  // one-shot "dtheta really is coming from the gyro" log
   // One-shot "the wheels' OWN stated variance really is reaching the preintegrator" log. Without it
   // the channel is indistinguishable from off: the quadrature combine can only ever loosen the
   // prior, so a producer that never publishes velCov and a consumer that never reads it produce
   // byte-identical behaviour and identical (empty) counters.
   bool odom_var_announced_ = false;
   // Rolling stats for the periodic [ImuInject] line. The one-shot above proves the path bound at
   // all; these say whether it is still binding, how much of each interval the IMU actually covers,
   // and -- the number that matters -- how much heading the gyro is taking OUT of the wheel estimate.
   // A silently degrading channel (IMU stalls, buffer too shallow, clock map lost) reads as normal in
   // every outcome metric until it has already cost accuracy.
   int          imu_seg_used_ = 0, imu_seg_total_ = 0;
   int          zupt_segs_ = 0;   // segments where the ZUPT gate blocked the gyro override this window
   double       imu_dtheta_sum_ = 0.0, wheel_dtheta_sum_ = 0.0;  // rad, over IMU-covered segments only
   bool         imu_stats_sim_clock_ = false;
   std::int64_t imu_stats_last_log_ms_ = 0;
   // Per-CYCLE twins of the rolling sums above, copied into UpdateResult. The rolling stats answer
   // "is the injection still binding"; these answer "which channel produced THIS cycle's heading",
   // which is the covariate an online scale estimate has to be regressed on.
   Eigen::Vector2f last_pred_pos_ = Eigen::Vector2f::Zero();  // odometry prediction, pre-optimizer
   float           last_pred_theta_ = 0.f;
   rc::calib::MotionCalibrator motion_calib_;
   float last_cycle_dt_s_ = 0.f;   // odometry window duration, for the calibrator's time covariate
   float cyc_dx_local_ = 0.f, cyc_dy_local_ = 0.f;   // wheel-claimed body-frame displacement
   float cyc_imu_dvx_ = 0.f, cyc_imu_dvy_ = 0.f, cyc_imu_dpx_ = 0.f, cyc_imu_dpy_ = 0.f;
   float cyc_wheel_dvx_ = 0.f, cyc_wheel_dvy_ = 0.f;
   int   cyc_imu_lin_segs_ = 0;
   float cyc_imu_dtheta_          = 0.f;
   float cyc_wheel_dtheta_        = 0.f;
   rc::calib::HeadingCovariates cyc_heading_cov_{};
   double cyc_gyro_w_dt_ = 0.0, cyc_heading_dt_ = 0.0;      // for the time-weighted mean gyro weight
   UpdateResult::HeadingDiag cyc_hdiag_{};
   rc::calib::WheelNoiseLearner wheel_noise_;
   rc::preint::RestMotionChannel rest_fwd_, rest_lat_, rest_rot_;   // is the body at rest? (body frame)
   bool rest_chan_init_ = false;
   double cyc_dens_w_dt_ = 0.0, cyc_dens_g_dt_ = 0.0;      // density * dt, for time-weighted means
   double gyro_w_dt_sum_ = 0.0, heading_dt_sum_ = 0.0;      // same, over the 5 s [ImuInject] window
   float cyc_wheel_shadow_dtheta_ = 0.f;
   int   cyc_imu_segs_ = 0, cyc_wheel_segs_ = 0;

   // Running second moment of the innovation, per axis (x, y, theta). See Params::adaptive_cov_enabled.
   Eigen::Vector3f innov_m2_ = Eigen::Vector3f::Zero();
   /// Raise the published covariance to at least the error the innovations demonstrate. Call ONLY on
   /// frames that actually had a prediction — a zero innovation otherwise means "no information", not
   /// "perfect agreement", and folding those in would drag the estimate back down.
   void feed_motion_calibrator(UpdateResult& res);
   /// Fill res.surprise. corrected = the estimate was moved by evidence this cycle (solve or polish).
   /// Must run after res.covariance and res.pred_* hold this cycle's values.
   void score_surprise(UpdateResult& res, bool corrected);
   /// The motion model's predictive covariance for the pose, accumulated from every cycle's selected-prior
   /// covariance since the last scored correction, then reset to that correction's posterior. Kept HERE and
   /// not read from current_covariance, because current_covariance does not propagate between solves
   /// while the polish is off (the growth step is gated on it) -- it would make every prediction look
   /// infinitely confident.
   Eigen::Matrix3f sur_P_pred_ = Eigen::Matrix3f::Zero();
   // The posterior at the last scored correction: sur_P_pred_ = sur_P_prev_ + the motion noise since.
   Eigen::Matrix3f sur_P_prev_ = Eigen::Matrix3f::Zero();
   // The motion-noise COMPONENTS accumulated over the open stretch (motion_noise_vc.h): each unit
   // covariance, and their coefficient-weighted sum as applied. sur_mixed_: a cycle without them.
   std::array<Eigen::Matrix3f, rc::preint::Interval::NC> sur_U_{Eigen::Matrix3f::Zero(), Eigen::Matrix3f::Zero(),
       Eigen::Matrix3f::Zero(), Eigen::Matrix3f::Zero(), Eigen::Matrix3f::Zero(), Eigen::Matrix3f::Zero()};
   Eigen::Matrix3f sur_QK_ = Eigen::Matrix3f::Zero();
   bool            sur_mixed_ = false;
   rc::DriftMonitor drift_mon_;                       // systematic drift per m / rad / s (drift_monitor.h)
   float sur_d_ = 0.f, sur_phi_ = 0.f, sur_T_ = 0.f;  // the open stretch's odometry motion
   // ── SCAN-TO-SCAN INNOVATION (motion_noise_innov.h, Fable window memo 2026-10-05) ──────────────
   // Each cycle: the scan-only pose z (the newest scan's SDF Gauss-Newton step alone) and its CLAIMED
   // covariance H_s^-1, the odometry increment, and this cycle's unit noise components. Consecutive
   // cycles give delta = z_n - z_{n-1} - odom_n, whose second moment is linear in the coefficients.
   rc::preint::MotionNoiseInnov noise_innov_;
   bool            noise_innov_init_ = false;
   std::array<Eigen::Matrix3f, rc::preint::Interval::NC> cyc_U_{};
   bool            cyc_U_ok_ = false;                         // a measured, preintegrated prior this cycle
   Eigen::Vector3f cyc_odom_ = Eigen::Vector3f::Zero();       // this cycle's odometry increment (world)
   float           cyc_dt_s_ = 0.f;                           // this cycle's duration, s
   long            innov_cycle_ = 0, innov_prev_cycle_ = -10;
   Eigen::Vector3f innov_z_ = Eigen::Vector3f::Zero(), innov_z_prev_ = Eigen::Vector3f::Zero();
   Eigen::Matrix3f innov_R_ = Eigen::Matrix3f::Zero(), innov_R_prev_ = Eigen::Matrix3f::Zero();
   bool            innov_ok_ = false, innov_prev_ok_ = false;
   int             last_scan_iters_ = 0;                      // GN iterations the last scan-only pose took
   std::ofstream   innov_csv_;
   // ── r2 KINEMATIC MOUNT FACTOR (mount_factors.h; plan 2026-10-08 Task 1/1b/5) ──────────────────────
   rc::mountf::KinematicMount kin_mount_;
   mutable std::mutex mount_snap_mutex_;                      // guards mount_snap_ (read by the ingest thread)
   std::mutex low_obs_mutex_;                                 // guards low_obs_ (ingest -> localiser)
   std::deque<LowObstacles> low_obs_;
   MountSnapshot   mount_snap_;
   std::ofstream   kin_csv_;                                  // tmp/mount_factors/kin_<ts>.csv: per fed cycle
   /// Feed kin_mount_ from this cycle's innovation (observe_innovation, only when `fed`).
   void observe_kinematic_mount(const Eigen::Vector3f &delta, const Eigen::Matrix3f &T,
                                const Eigen::Vector3f &odom_body, UpdateResult &res);
   /// The newest scan's SDF factor alone, linearised at (x, y, th): the same query, observation weights,
   /// IRLS Huber and Jacobian as the solver's SdfFactor (room_gn_solver.cpp). Shared by the polish.
   bool scan_linearize(const ScanPoints &pts, float x, float y, float th, Eigen::Matrix3f &H, Eigen::Vector3f &bb);
   /// z = x_lin - H^-1 b and R = H^-1 into innov_z_/innov_R_ (innov_ok_ on success).
   void scan_only_pose(const ScanPoints &pts, float x, float y, float th);
   void observe_innovation(UpdateResult &res);
   rc::PoseFieldBias field_bias_;
   bool              field_bias_init_ = false;
   void reset_noise_stretch();
   void fill_noise_vc(UpdateResult &res) const;
   bool            sur_init_ = false;
   int             sur_open_cycles_ = 0;   // cycles accumulated into sur_P_pred_ since the last scored correction   // first correction only seeds sur_P_pred_ (no pose prior before it)
   void apply_adaptive_covariance(UpdateResult& res);

   /// Drop the strided-window bookkeeping. MUST accompany every window_mgr_.clear(): after a recovery
   /// or relocalization the last-admitted pose refers to a trajectory that no longer exists, so
   /// comparing against it would suppress admissions until the robot happened to travel far from a
   /// stale reference.
   void reset_stride_state()
   {
       stride_has_admitted_ = false;
       stride_span_.reset();
   }

   // Prediction-based early exit tracking
   int tracking_step_count_ = 0;
   int prediction_early_exits_ = 0;
   /// Which condition sent a frame to the optimiser, so a gate's cost and its usefulness are
   /// separable. CornerVeto in particular exists to catch what the SDF cannot see (a 180 deg flip
   /// is SDF-ambiguous); if it never fires it is paying for a case that does not arise, and if it
   /// fires often the LiDAR corners are carrying the localiser rather than merely checking it.
   enum class EeReason { NotTracking = 0, SdfResidual, UnoptDrift, CornerVeto, Count };
   std::array<int, static_cast<int>(EeReason::Count)> ee_forced_{};
   int ee_taken_ = 0;
   std::int64_t ee_report_last_ms_ = 0;
   // Last mean |SDF| at the predicted pose evaluated by try_prediction_early_exit() this frame
   // (the early-exit decision variable). NaN when the gate's pre-conditions weren't met. Consumed
   // by update() to expose it on the Adam path too (the value that TRIGGERED optimization).
   float last_early_exit_metric_ = std::numeric_limits<float>::quiet_NaN();
   /// Median |SDF| at the SAME predicted pose and points as last_early_exit_metric_ (which is
   /// the mean). Kept so that recovery can compare like with like: its threshold is calibrated
   /// on medians (see RecoveryLossThreshold and compute_seed_error), and the mean runs ~15-20%
   /// higher. Logging both also makes the median/mean ratio measurable from any run.
   float last_pred_sdf_median_  = std::numeric_limits<float>::quiet_NaN();

   // Velocity-adaptive gradient weights [x, y, theta]
   Eigen::Vector3f current_velocity_weights_ = Eigen::Vector3f::Ones();

    // Corner detector
    CornerDetector corner_detector_;

    // ===== Wall-SLAM state (Estimate mode) =====
    wallmap::WallMap wall_map_;
    unsigned nis_log_tick_ = 0;      // row counter for tmp/corner_nis.csv
    std::ofstream nis_csv_;          // corner-NIS diagnostics (see the writer for the columns)
    std::ofstream resid_csv_;        // tmp/wall_residual.csv — per-association point-to-wall residual
    unsigned resid_tick_ = 0;        // its own row counter: the layout trace's stalls when the solve early-exits
    std::ofstream wallgeom_csv_;     // tmp/wall_geometry.csv — every wall's (phi, d, extent) every frame
    std::ofstream layout_csv_;       // tmp/layout_trace.csv — the PUBLISHED polygon + its per-corner
    unsigned layout_trace_tick_ = 0; // and per-edge sigma, one row per frame (see the writer).
    std::ofstream probe_csv_;        // tmp/corner_probe.csv — ONE ROW PER CANDIDATE (see CornerProbe).
    unsigned probe_rows_ = 0;        // capped: this is a diagnostic dump, not a permanent log.
    unsigned probe_frame_ = 0;       // its own frame counter (the NIS tick only advances when NIS does)
    std::mt19937     wall_rng_{12345};
    mutable std::mutex wall_map_mutex_;              // guards derived_polygon_ (read by the main thread)
    std::vector<Eigen::Vector2f> map_layout_;          // set_map_layout (guarded by wall_map_mutex_)
    bool map_layout_adopted_ = false;                  // adopt_map_layout_as_given has run: the room is GIVEN
    std::atomic<float> slam_disagree_{std::numeric_limits<float>::quiet_NaN()};
    std::ofstream shadow_csv_;                         // tmp/slam_shadow.csv
    std::int64_t shadow_flush_ms_ = 0;
    void record_slam_shadow(UpdateResult& r);         // plain SLAM vs the published pose, every frame
    std::vector<Eigen::Vector2f> derived_polygon_;   // the published polygon (map frame, CCW)
    bool projection_failed_logged_ = false;          // one warning per stretch of unclosed projections
    std::atomic<bool> map_ready_{false};
    // THE CEILING THE LIDAR MEASURED (m, body frame; 0 = not measured yet). Set from the ingestor's
    // startup/running ceiling check, which decides between a ring of ceiling returns and a wall-top
    // ring by which shape the measured radius fits. Read by the scene graph, which publishes it on
    // the room node, and by the image-edge module, which projects the wall-ceiling contour.
    std::atomic<float> measured_ceiling_m_{0.f};
    std::atomic<float> measured_ceiling_sigma_m_{0.f};
public:
    [[nodiscard]] float measured_ceiling() const noexcept { return measured_ceiling_m_.load(std::memory_order_relaxed); }
    void set_measured_ceiling(float z) noexcept { measured_ceiling_m_.store(z, std::memory_order_relaxed); }
    /// The ceiling measurement's own uncertainty, metres (the plane's spread, NOT a standard error —
    /// see LidarIngestor::update_ceiling_cap). 0 = never measured.
    ///
    /// ★ CARRIED SO THE HEIGHT CAN BE RE-MEASURED LATER, not for display. The camera's mount fit reads
    ///   the ceiling directly: solve it from floor corners alone and from ceiling corners alone and the
    ///   two disagree by exactly the ceiling's error, with a gain measured at -0.81 (ricoh) and -0.99
    ///   (zed) millimetres of apparent mount height per millimetre of ceiling (2026-09-12). So the
    ///   camera can refine this number, and a refinement needs a prior with a width — a value alone
    ///   cannot say how much it may move.
    [[nodiscard]] float measured_ceiling_sigma() const noexcept
    { return measured_ceiling_sigma_m_.load(std::memory_order_relaxed); }
    void set_measured_ceiling_sigma(float s) noexcept
    { measured_ceiling_sigma_m_.store(s, std::memory_order_relaxed); }
private:
    bool wall_reanchored_ = false;
    std::ofstream wall_input_rec_;                                   // RecordWallInput (diagnostic)
    /// Pose prior of window slot i for marginalisation: slot 0 = the solver's gauge (or the boundary prior once
    /// valid and the window holds > 1 slot — the solve's own rule); later slots = their recorded pose covariance.
    Eigen::Matrix3f slot_pose_precision(std::size_t i) const;
    /// Σ over live window slots of each wall's pose-marginalised statistical information — for the PUBLISH TEST
    /// only (build_polygon_with); never written back (the slots are absorbed as they leave the window).
    std::unordered_map<std::uint64_t, Eigen::Matrix2f> window_wall_information() const;
    bool          rec_stride_replace_ = false;                       // this frame's stride decision, for the recorder
    Eigen::Vector3f rec_odom_delta_ = Eigen::Vector3f::Zero();       // this frame's map-frame odometry delta
    rc::slam::ScanSlam plain_slam_;                                   // Estimate-mode pose (plain_slam_driving)
    bool plain_slam_seed_ = true;                                     // (re)seed from the window's pose on next step
    /// Set once, at the first publishable polygon, when WallMap::Params::freeze_when_publishable is
    /// on. From then on the layout is GIVEN: see the four guards that read it in room_concept.cpp.
    unsigned wall_mh_log_tick_ = 0;   // rate limit for the "still annealing" line
    bool wall_frozen_ = false;
    /// The published polygon as it stood at the instant of the freeze. LOCALIZING publishes THIS,
    /// not a fresh projection: manhattan_polygon() ends in decorate(), the level-2 stage, which
    /// re-decides every frame from the free-space grid and so kept changing the published SHAPE while
    /// the walls themselves were frozen — the "republished delimiting_polygon with 6 vertices (was
    /// 10)" flip-flop, and the notches seen on the canvas where a doorway leaves free space just
    /// outside a wall. A frozen layout that still changes shape is not frozen.
    std::optional<rc::wallmap::Polygon> frozen_pub_;
    std::vector<wallseg::WallSegment> last_wall_segments_;   // this frame's segments (viewer)
    wallmap::FrameResult last_wall_frame_;
    std::string last_wall_status_;                   // last polygon status logged (log on change only)
    int wall_stat_frames_ = 0, wall_stat_assoc_ = 0, wall_stat_segs_ = 0,
        wall_stat_twins_ = 0, wall_stat_births_ = 0, wall_stat_deaths_ = 0,
        wall_stat_contained_ = 0;   // rate-limited health counters
    int wall_frames_since_rederive_ = 0, wall_rejected_since_rederive_ = 0,
        wall_rederives_ = 0;        // global re-derivation cadence (WallMap::Params::rederive_*)
    // Wall-SLAM analysis CSVs (loc thread only; imbued classic — es_ES writes commas otherwise).
    // etc/wall_slam.csv: one row per frame. etc/wall_slam_events.csv: one row per birth/death,
    // carrying the Z DISTRIBUTION of the points the wall was built from — the column that separates
    // "a wall" from "the ceiling in the band" without a screenshot.
    std::ofstream wall_csv_, wall_events_csv_;
    std::vector<float> last_seg_z_;                  // per-segment mean z of its inliers, this frame
    float wall_z_mean_ = 0.f, wall_z_p95_ = 0.f, wall_z_max_ = 0.f;
    std::int64_t wall_frame_ts_ = 0;
    // ── THE BOX LAYOUT SIDECAR ───────────────────────────────────────────────────────────────
    // Runs beside the wall map on the same scan and the same pose, and publishes INSTEAD of it when
    // RoomShape.BoxLayout is true. Off by default: the shipped wall-SLAM layout holds a 6.00 x 4.00
    // rectangle to 2-3 mm while driving (d307c84) and is not being staked on an estimator whose
    // bench ladder still has open rows. Nothing else in this class reads these two members.
    rc::boxch::Channel box_channel_;
    /// The pose covariance the last solve produced, for the sidecar to weight its returns by. One
    /// frame stale on purpose — observe() runs before the solve, and a point's uncertainty is
    /// better described by the previous posterior than by an identity matrix.
    Eigen::Matrix3f box_pose_cov_ = Eigen::Matrix3f::Identity() * 1e-4f;

    /// Segment the newest slot's scan, associate to the walls, store the associations in the slot.
    void record_wall_input(const std::vector<Eigen::Vector3f>& points, const std::vector<Eigen::Vector2f>& pts,
                           const Eigen::VectorXf& weights, const Eigen::Vector3f& pose, std::int64_t timestamp_ms);
    void wall_slam_observe(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& pose,
                           std::int64_t timestamp_ms);
    /// After the solve: merge, derive the polygon, re-anchor once it is publishable, fill the view.
    void wall_slam_after_solve(UpdateResult& res);
    /// One-shot gauge change p' = R(−rot)(p − c) over everything that lives in the map frame.
    void reanchor_map_frame(const Eigen::Vector2f& c, float rot);

    // Startup initialization configuration
    bool init_use_polygon_ = false;
    std::vector<Eigen::Vector2f> init_polygon_vertices_;
    float init_room_width_ = 10.f;
    float init_room_length_ = 10.f;
    std::string seed_pose_file_path_;

   // ===== Debug Logging (localization thread only — no mutex needed) =====
   std::ofstream      debug_log_;
   // Lean per-update optimizer-timing CSV (loc-thread only; no lock). Captures the cost distribution
   // and CUDA warmup curve. Opened lazily on the first update; gated by params.optimizer_timing_csv.
   std::ofstream      opt_csv_;
   bool               opt_csv_open_attempted_ = false;

   std::ofstream      flip_csv_;   // one row per detected ~180° flip (tmp/sdf_localizer/flips_<ts>.csv)
   bool               flip_csv_open_attempted_ = false;
   // Hierarchical boundary-precision A/B trace (etc/hier_prec.csv): one row per converged frame while
   // params.hier_prec_boundary_enabled. Lazy-opened on first use; loc-thread only, no lock.
   std::ofstream      hier_prec_csv_;
   bool               hier_prec_csv_open_attempted_ = false;

   // ── Per-model-corner attribution (etc/corner_stats.csv) ────────────────────────────────────────
   // Accumulated over the run and rewritten whole every kCornerStatsRewritePeriod corner frames, so
   // the file is always a current snapshot rather than an append log to be re-aggregated. Answers the
   // question the aggregate rej_* counters cannot: WHICH vertices are worth keeping as landmarks.
   // Loc-thread only, no lock — same discipline as opt_csv_/hier_prec_csv_.
   struct CornerVertexStats
   {
       int    in_fov = 0;        // visible, detection attempted
       int    occluded = 0;      // in range but occluded — not the corner's own fault
       int    accepted = 0;      // survived to a match that entered the loss
       int    rival_n = 0;       // accepted samples that had a competing model corner in gate
       double assoc_prob_sum = 0.0;
       double runnerup_chi2_sum = 0.0;   // over rival_n only (INFEASIBLE rivals excluded)
       double resid_sum = 0.0;           // ‖detected − predicted‖ (m)
       double lambda_min_sum = 0.0;      // smallest eigenvalue of Λ_det — the degenerate direction
       double lambda_max_sum = 0.0;
       int    suppressed = 0;            // CUMULATIVE frames retired by the information-yield rule —
                                         // "has been retired at some point", NOT the current state
       bool   retired_now = false;       // state at the most recent match. Read THIS for "is it
                                         // retired right now"; with hysteresis the two differ often
       float  last_yield = 0.f;          // most recent running λ_min estimate behind that decision
   };
   static constexpr int kCornerStatsRewritePeriod = 200;
   std::map<int, CornerVertexStats> corner_vertex_stats_;
   int                corner_stats_frames_ = 0;
   std::ofstream      corner_stats_csv_;
   void accumulate_corner_stats(const CornerDetector::DetectionResult& det);
   void write_corner_stats_csv();
   RerunLogger        rerun_logger_;
   int                rerun_frame_counter_ = 0;
   bool               rerun_room_polygon_sent_ = false;
   std::vector<float> last_adam_losses_;    // per-iteration losses from last Adam/LBFGS run
   float              last_loss_init_  = 0.f;  // loss before first step
   float              prev_sdf_mse_    = 0.f;  // sdf_mse from previous frame (for boundary quality gate)
   // ===== Hierarchical boundary precision (HIERARCHICAL_PRECISION.md), persisted across frames =====
   float              u_b_             = 0.f;  // boundary log-precision posterior; exp(u_b_) = ⟨π⟩ = boundary_weight
   bool               u_b_init_        = false;// lazily seeded to g(v) on first use / after a recovery reset
   float              map_trust_v_     = 0.f;  // Option-A slow hyper-state v; g(v)=u0+g_gain·v predicts u_b_
   /// Stage-1 boundary-precision inference: one fast step on u_b_ (from the boundary residual r_b, using
   /// the post-optimization posterior covariance sigma_x for the tr(Λ_b Σ) term) plus one slow step on the
   /// map_trust hyper-state v. No-op unless params.hier_prec_boundary_enabled. Call once per frame after
   /// the window pose has converged.
   void               update_boundary_hyperprecision(const Eigen::Matrix3f &sigma_x);
   /// Append one row to etc/hier_prec.csv (lazy-opened). src ∈ {opt, ee, reloc}. Loc-thread only, no lock.
   void               log_hier_prec_row(const char *src, float r_b, float quad, float trace, bool reloc_fired);
   WindowManager::LossBreakdown last_loss_breakdown_;  // FE term breakdown after last Adam
   float              last_lbfgs_grad_norm_ = 0.f; // gradient norm at the end of the GN solve (0 on early-exit); name kept for the debug-CSV column
   // Per-frame timing — t_update_start_ set at entry of update(), shared with early-exit path
   std::chrono::high_resolution_clock::time_point t_update_start_;
   float              last_t_adam_ms_      = 0.f;
   float              last_t_cov_ms_       = 0.f;
   // Per-stage cost (2026-10-10), reset after each optimizer_timing.csv row: 0 = the stage did not run this frame.
   float              last_t_slam_ms_      = 0.f;   // plain SLAM step (driving or shadow)
   float              last_t_wall_ms_      = 0.f;   // wall-map estimator (wall_slam_observe)
   float              last_t_polish_ms_    = 0.f;   // SDF polish on an early-exit frame
   float              last_t_breakdown_ms_ = 0.f;
   // Optimizer-timing telemetry (written on loc_thread_, drained by the worker via take_optimizer_timing).
   std::atomic<std::uint32_t> opt_total_count_{0};
   std::atomic<std::uint32_t> opt_earlyexit_count_{0};
   std::atomic<std::uint64_t> opt_update_us_sum_{0};
   OdometryPrior      last_measured_prior_; // saved by apply_dual_prior_fusion for logging
    OdometryPrior      last_selected_prior_;
   Eigen::Vector3f    predictor_delta_ = Eigen::Vector3f::Zero();  // viewer mirror; see get_predictor_delta()
    MotionPriorSource  last_motion_prior_source_ = MotionPriorSource::None;
   Eigen::Matrix3f    last_cmd_cov_ = Eigen::Matrix3f::Identity();
    bool               last_command_prior_fresh_ = false;
    bool               last_measured_prior_fresh_ = false;
   std::int64_t       prev_lidar_ts_for_log_ = 0;
   std::string        slot_poses_pre_;    // packed slot poses before Adam  (debug log)
   std::string        slot_poses_post_;   // packed slot poses after Adam
   std::string        slot_sdf_mse_str_;  // packed sdf_mse_final per slot
   std::string        debug_log_path_;    // full path of the current log file (includes timestamp)
    mutable std::mutex motion_ingress_debug_mutex_;
    MotionIngressDebug motion_ingress_debug_;
   void init_debug_log();
   void init_recovery_log();   // always on, independent of DebugLog
    MotionIngressDebug get_motion_ingress_debug() const;


   // ===== Online motion model learned state =====
   // Initialised to -1 (sentinel = "warmup not done; use static params").
   Eigen::Vector3f learned_odom_bias_           = Eigen::Vector3f::Zero(); // [dx,dy,dtheta] bias
   // Robust buffer for trans-noise samples: collect N then take median for one EMA step.
   std::vector<float> trans_noise_sample_buf_;
   void            service_calibration();

   // Compute velocity-adaptive weights based on motion profile
   Eigen::Vector3f compute_velocity_adaptive_weights(const OdometryPrior& odometry_prior);
    float estimate_orientation_from_points(const std::vector<Eigen::Vector3f>& pts) const;
    void resolve_initial_yaw_ambiguity(const std::vector<Eigen::Vector3f>& lidar_points, float prior_phi);
    bool bootstrap_initialization_from_lidar();

    struct PredictionState
    {
        Eigen::Matrix3f propagated_cov = Eigen::Matrix3f::Zero();  // Predicted covariance
        bool have_propagated = false;   // Whether prediction was performed
        std::vector<float> previous_pose;  // Robot pose BEFORE prediction (for prior loss)
        std::vector<float> predicted_pose; // Robot pose after prediction
    };


public:
    /// PUBLIC so the offline bench can exercise the REAL fusion instead of imitating it. Both depend
    /// only on `params` and their arguments — no graph, no model_, no window — and both are called
    /// unchanged from the live path, so what the bench measures is what the agent does. The bench
    /// (tools/wall_slam_selftest.cpp, WS_TOUR) previously stood in a linear blend of intent and
    /// measurement, which showed the MECHANISM but could not be evidence about this code.
    Eigen::Matrix3f compute_motion_covariance(const OdometryPrior &odometry_prior,
                                              bool is_measured_odometry = false);

    /// Fuse command and measured priors into one Gaussian. Pure: it reads no member state, which is
    /// why it can be static and why the bench can call it without standing up an agent.
    static std::pair<Eigen::Vector3f, Eigen::Matrix3f> fuse_priors(
        const Eigen::Vector3f &pred_cmd, const Eigen::Matrix3f &cov_cmd,
        const Eigen::Vector3f &pred_odom, const Eigen::Matrix3f &cov_odom);
private:
    RoomConcept::OdometryPrior compute_odometry_prior(
                    const std::vector<VelocityCommand>& velocity_history,
                    const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar);
    /// `preint_out`, when non-null, additionally receives the interval PREINTEGRATED over the same
    /// samples — covariance and scale Jacobians propagated step by step (se2_preintegration.h). The
    /// returned mean is computed by the same lines either way, so passing it cannot change the mean.
    Eigen::Vector3f integrate_velocity_over_window(const Eigen::Affine2f &robot_pose,
                                                   const std::vector<VelocityCommand> &velocity_history,
                                                   const int64_t &t_start_ms, const int64_t &t_end_ms,
                                                   rc::preint::Interval *preint_out = nullptr);

    /// Integrate measured odometry (adv, side, rot) over the same time window.
    /// `imu_history` is optional; when it covers a segment, that segment's heading change comes from
    /// the gyro instead of the wheel-derived rot. Bounds are WALL epoch-ms and are converted
    /// internally to whatever clock the readings are measured in -- see the note at the definition.
    Eigen::Vector3f integrate_odometry_over_window(const Eigen::Affine2f &robot_pose,
                                                   const std::vector<OdometryReading> &odometry_history,
                                                   const int64_t &t_start_ms, const int64_t &t_end_ms,
                                                   rc::preint::Interval *preint_out = nullptr,
                                                   const std::vector<ImuReading> *imu_history = nullptr,
                                                   const SimClockMap *sim_clock = nullptr);
    // Its parts (room_concept_odometry_window.cpp): the interval's clock + per-segment sensor integrals,
    // one segment's wheel/gyro heading fusion, and the channel health logging.
    struct OdomSources;
    struct SegmentHeading { float dtheta, w_g, sig_om_fused, rot_eff; };
    SegmentHeading fuse_segment_heading(const OdometryReading &odom, float dt, float dy_local,
                                        std::int64_t effective_start_ms, std::int64_t effective_end_ms,
                                        const OdomSources &src, int &imu_segments, int &wheel_segments);
    void report_odometry_channels(const OdomSources &src, const std::vector<OdometryReading> &odometry_history,
                                  std::int64_t t_end_ms, int imu_segments, int wheel_segments, int odom_var_segments);

    /// Compute odometry prior from measured velocities (encoder/IMU feedback)
    OdometryPrior compute_measured_odometry_prior(
                    const std::vector<OdometryReading>& odometry_history,
                    const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar);


    PredictionState predict_step(std::shared_ptr<Model> &room,
                                  const OdometryPrior &odometry_prior,
                                  bool is_localized);

    // ===== update() helper methods =====
    static std::string_view motion_prior_source_name(MotionPriorSource source);

    // update()'s phases, in call order (room_concept_update*.cpp); UpdateCycle is in update_cycle.h.
    struct UpdateCycle;
    void update_predict(UpdateCycle& c);
    void update_build_slot(UpdateCycle& c);
    void update_carry_doors();
    void update_observe_walls(UpdateCycle& c);
    void update_observe_corners(UpdateCycle& c);
    void update_open_cycle(UpdateCycle& c);
    std::optional<UpdateResult> update_early_exit(UpdateCycle& c);
    void update_solve(UpdateCycle& c);
    void update_extract(UpdateCycle& c);
    void update_finalize(UpdateCycle& c);
    void update_write_debug_row(UpdateCycle& c);
    void update_publish_diagnostics(UpdateCycle& c);

    /// Compute command and measured priors, then select a single motion prior.
    MotionPriorSelection build_motion_prior_selection(
        const std::vector<VelocityCommand>& velocity_history,
        const std::vector<OdometryReading>& odometry_history,
        const std::pair<std::vector<Eigen::Vector3f>, std::int64_t>& lidar);

    /// Check if the predicted pose already has low SDF error and can skip Adam.
    /// Returns an UpdateResult if early exit is taken, nullopt otherwise.
    /// `measure_only` (2026-09-17): compute the prediction diagnostics ONLY — last_pred_sdf_median_ and
    /// last_early_exit_metric_, i.e. the "pred (SDF)" series — without growing the covariance and without
    /// ever deciding to skip the solve. Used while the layout is still being estimated, where the early
    /// exit must not fire (no fixed map) but the plot must still show the prediction's fit.
    std::optional<UpdateResult> try_prediction_early_exit(
        const ScanPoints& scan_pts,
        const Eigen::Vector3f& slot_odom_delta,
        const OdometryPrior& odometry_prior,
        std::int64_t lidar_timestamp_ms,
        bool measure_only = false);
    // Its parts (room_concept_early_exit.cpp): the corner consensus veto, the periodic taken/forced report,
    // the one damped SDF step, the early-exit UpdateResult, and that path's debug-log row.
    bool corner_veto_fires() const;
    void report_early_exit_split();
    void sdf_polish_step(const ScanPoints& scan_pts, const Eigen::Vector3f& pose0, const OdometryPrior& odometry_prior);
    UpdateResult build_early_exit_result(const ScanPoints& scan_pts, const Eigen::VectorXf& sdf_pred, float mean_sdf_pred,
                                         std::int64_t lidar_timestamp_ms, float x, float y, float phi);
    void write_early_exit_debug_row(const UpdateResult& res, const OdometryPrior& odometry_prior,
                                    std::int64_t lidar_timestamp_ms, float mean_sdf_pred, float x, float y, float phi);

    /// Run the Levenberg-Marquardt backend (room_gn_solver) over the sliding window, writing the
    /// solution back into the slot pose tensors. Returns {last_loss, iterations}. On failure the
    /// window is left exactly as it was and the loss is NaN.
    std::pair<float, int> run_gn_loop(const OdometryPrior& odometry_prior);
    /// The newest pose's precision as the WINDOW sees it, captured on the last optimised frame:
    /// Schur-complemented (the honest one) and the bare diagonal block (what dropping the
    /// cross-terms would claim). Held as plain matrices rather than gn::NewestMarginal because
    /// room_gn_solver.h includes THIS header — the dependency only runs one way.
    /// See Params::hessian_check for what the comparison is for.
    Eigen::Matrix3f last_marg_prec_  = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f last_block_prec_ = Eigen::Matrix3f::Zero();
    bool last_marg_ok_ = false;
    int  last_marg_dof_ = 0;
    int  last_gn_window_slots_ = 0;
    void log_hessian_check(const UpdateResult& res);
    std::int64_t hess_check_last_log_ms_ = 0;
    /// Wall-clock gap since the previous OPTIMISED frame. The recursion's missing
    /// prediction step makes the published sigma a function of this gap, so it is the
    /// covariate the whole comparison is read against.
    std::int64_t hess_prev_opt_ms_ = 0;
    /// The filter's posterior BEFORE apply_adaptive_covariance floors it, so the CSV can
    /// separate what the recursion computed from what the innovation floor raised it to.
    Eigen::Matrix3f hess_pre_adaptive_ = Eigen::Matrix3f::Zero();
    /// P_pred = F·P_prev·Fᵀ + Q, kept before the update consumes it: the quantity the process noise
    /// sets and the one the innovation must be consistent with.
    Eigen::Matrix3f hess_pred_cov_ = Eigen::Matrix3f::Zero();
    /// This slot's observation Hessian alone, so "Q too large" and "H too small" stay separable.
    Eigen::Matrix3f hess_obs_prec_ = Eigen::Matrix3f::Identity();
    /// Did the covariance update land, and if not which guard refused it.
    /// 1 = taken, -1 non-finite, -2 not positive definite, -3 condition number too high.
    int    hess_cov_accept_ = 0;
    double hess_cov_det_    = 0.0;
    double hess_cov_cond_   = 0.0;
    int          hess_check_rows_ = 0;

    /// Run the GN backend on the CURRENT window WITHOUT keeping its answer, and log it beside the
    /// authoritative backend's. Call after the authority has run; poses_before is the state both
    /// started from, poses_after the authority's solution (restored on return).
    /// RGB edge shadow: solve TWICE from the same start — with and without the image term — and log
    /// both answers. Two questions, kept in SEPARATE column groups because conflating them is how a
    /// term that carries no information gets shipped: "does it carry information?" (counts, sigmas,
    /// trace_raw vs trace_eff) and "who won?" (the pose difference and the covariance both ways).
    /// Also fits the systematic-residual monitor, which is what catches a mis-calibrated mount
    /// BEFORE it biases the pose.
    ///
    /// ★ TWO PASSES WITH VERY DIFFERENT COSTS, and `probe_pose` splits them. Pass 1 — evaluate the
    ///   residuals at the authority pose and fit r ~ b_const + b_invd·(fy/d) — is the MOUNT
    ///   measurement, and it is cheap: no solve, one 2x2 least squares. Pass 2 re-solves the window
    ///   with the term driving to answer "how far would it have moved the pose", and only means
    ///   anything on a cycle where the optimizer actually ran.
    ///   Passing false runs pass 1 alone, which is what lets the monitor sample EVERY cycle instead
    ///   of only optimizing ones. That mattered: the optimizer fires on ~0.25% of cycles here — the
    ///   better the prediction gets, the more starved this instrument became — so characterising the
    ///   mount was a matter of hours rather than minutes for no reason but where the call sat.
    void run_image_edge_shadow(const std::vector<Eigen::Vector3f>& poses_after,
                               float boundary_weight, std::int64_t timestamp_ms,
                               bool probe_pose);
    std::ofstream image_edge_csv_;
    void imgedge_health(std::int64_t timestamp_ms);
    int imgedge_rows_ = 0, imgedge_calls_ = 0, imgedge_no_window_ = 0,
        imgedge_no_obs_ = 0, imgedge_no_cam_ = 0;
    std::int64_t imgedge_health_last_ms_ = 0;
    /// Per-window mount fit. The accumulators are RESET after every solve, so each row is an
    /// independent measurement and the between-window scatter is visible. See mount_pooled_solve().
    void mount_pooled_solve(std::int64_t timestamp_ms);
    std::ofstream mount_csv_;
    std::ofstream triple_csv_;
    std::ofstream triple_pose_csv_;
    long          triple_rows_ = 0, triple_frames_ = 0;
    long          triple_floor_ = 0, triple_ceil_ = 0;
    // Running summary of the triple-point POSE factor's shadow. Reported, never applied.
    long          tps_frames_ = 0;
    double        tps_chi2_sum_ = 0.0;
    long          tps_chi2_n_ = 0;
    void          log_triple_points(const ImageEdgeObs& obs, std::int64_t timestamp_ms,
                                    const Eigen::Vector3f& pose);
    double mnt_S11_ = 0, mnt_S12_ = 0, mnt_S22_ = 0, mnt_Sy1_ = 0, mnt_Sy2_ = 0, mnt_Syy_ = 0;
    long   mnt_n_ = 0;
    std::int64_t mnt_win_start_ms_ = 0;
    float  last_mount_fy_ = 0.f;
    float  mnt_pose_x_ = 0.f, mnt_pose_y_ = 0.f, mnt_pose_th_ = 0.f;
    long   mnt_wins_ = 0;
    double mnt_pitch_sum_ = 0.0, mnt_pitch_sum2_ = 0.0;
    long   mnt_pitch_n_ = 0;   ///< windows monitor 1 could actually solve, which is not mnt_wins_
    // ── Monitor 2: rigid image translation,  r_k ~ tx*nx_k + ty*ny_k ─────────────────────────────
    // A basis the const + fy/d fit structurally cannot see. The residual is scalar along each
    // contour's own normal, so a rigid displacement of the prediction projects as t . n_hat and
    // averages toward zero in the constant column. Fed by EVERY contour class, because the near-
    // orthogonality of floor-junction (vertical) and wall-corner (horizontal) normals is what makes
    // this fit well conditioned while the fy/d one is degenerate at a standstill.
    double mnt_T11_ = 0, mnt_T12_ = 0, mnt_T22_ = 0, mnt_Tx_ = 0, mnt_Ty_ = 0, mnt_Tyy_ = 0;
    long   mnt_tn_ = 0;

    // ── Monitor 3 / STAGE 1 of online mount self-calibration ─────────────────────────────────────
    // The four mount nuisances (pitch, height, boresight yaw, image/LiDAR dt) as PARAMETERS rather
    // than as marginalised noise. No new geometry: ImageEdgeSample::h already holds
    // sigma_i * d(residual)/d(nuisance_i) per sample (image_edge_source.cpp:261-272), computed every
    // frame and, until now, used only to WIDEN the inlier variance and then thrown away. Summing
    // w*h*h' and w*h*r instead turns it into the exact normal-equation block.
    //
    // ★ h IS ALREADY SCALED BY THE PRIOR SIGMA, so the parameters are dimensionless — in units of
    //   their own prior — and the Gaussian prior P0^-1 is exactly the IDENTITY. That also makes
    //   `informed` fall out with no new convention: posterior sigma < 0.9 in these units IS the
    //   existing rule (calibration_estimator.h), because the prior sigma is 1 by construction.
    //
    // ★ THIS CANNOT SEPARATE BORESIGHT YAW FROM ROBOT HEADING WITHIN ONE WINDOW — they are perfectly
    //   collinear in a single view, which is why measuring it needed the Webots supervisor. Pooling
    //   ACROSS windows at different poses separates them only under the assumption that the
    //   localiser's heading error is zero-mean. That assumption is what stage 2 (RGB<->LiDAR corner
    //   coupling) removes. Until then the between-window scatter, not the within-window sigma, is
    //   the honest uncertainty on yaw — hence the pooled accumulators below.
    Eigen::Matrix4d mnt_H_ = Eigen::Matrix4d::Zero();
    Eigen::Vector4d mnt_b_ = Eigen::Vector4d::Zero();
    long            mnt_hn_ = 0;
    Eigen::Vector4d mnt_p_sum_  = Eigen::Vector4d::Zero();
    Eigen::Vector4d mnt_p_sum2_ = Eigen::Vector4d::Zero();
    long            mnt_p_n_ = 0;

    /// The boundary-prior precision scale for this frame (hierarchical π=exp(u_b) when enabled, else
    /// the legacy quality gate). One definition, so every backend minimises the same objective.
    float boundary_weight_now() const;

    /// Slot poses as plain [x, y, θ], newest last — the exchange format with the GN backend.
    std::vector<Eigen::Vector3f> read_window_poses() const;
    void write_window_poses(const std::vector<Eigen::Vector3f>& poses);


    /// Posterior covariance of the newest pose: the GN window marginal when the solve produced one,
    /// otherwise the Gauss-Newton information of this scan + the motion prior.
    /// Updates current_covariance and returns {covariance, condition_number}.
    std::pair<Eigen::Matrix3f, float> compute_posterior_covariance(const ScanPoints& scan_pts);

    // Find best initial orientation by testing multiple candidates (0°, 90°, 180°, 270°)
    float find_best_initial_orientation(const std::vector<Eigen::Vector3f>& lidar_points,
                                        float x, float y, float base_phi);

    /// THE fit metric for this class: median |SDF| in METRES, robust to the outliers that furniture
    /// (absent from the polygon) always contributes. Every pose-quality number must be reduced with
    /// this one function so they stay comparable — recovery's trigger, the grid search's success bar,
    /// the symmetry check's candidate scores and the stability gate are all judged against each other,
    /// and each was previously computing its own variant (mean-squared m², sqrt-of-a-length) while
    /// being compared as though the units matched. See UpdateResult::sdf_mse.
    /// The LOWER median for an even count, as torch::median returned — every reported sdf_mse
    /// series (and every threshold calibrated against it) was produced that way.
    static float median_abs_sdf(const Eigen::VectorXf &sdf_vals)
    {
        if (sdf_vals.size() == 0) return std::numeric_limits<float>::quiet_NaN();
        std::vector<float> a(static_cast<std::size_t>(sdf_vals.size()));
        for (Eigen::Index i = 0; i < sdf_vals.size(); ++i) a[static_cast<std::size_t>(i)] = std::abs(sdf_vals(i));
        const auto mid = a.begin() + static_cast<std::ptrdiff_t>((a.size() - 1) / 2);
        std::nth_element(a.begin(), mid, a.end());
        return *mid;
    }

    /// ── THE SAME MEDIAN, WITH THE DOOR WEIGHTS ──────────────────────────────────────────────────
    /// A beam that went through an open doorway measured the next room; it must not inflate the
    /// number we report for THIS room. The fit already discounts those returns (see
    /// weights_from_normals), but the reported sdf_mse is a separate reduction and would keep
    /// counting them — which is the whole complaint: opening a door made the SDF rise.
    /// This is the ordinary weighted median — the value at which cumulative weight crosses half the
    /// total — so a point behind a certainly-open door contributes NOTHING, one behind a half-open
    /// door contributes half, and nothing anywhere decides whether a door is "open enough".
    /// Empty or mismatched weights fall back to the plain median, so a room with no open door
    /// reports precisely what it reported before.
    static float weighted_median_abs_sdf(const Eigen::VectorXf &sdf_vals, const Eigen::VectorXf &weights)
    {
        if (weights.size() == 0 or weights.size() != sdf_vals.size())
            return median_abs_sdf(sdf_vals);
        std::vector<std::pair<float, float>> v;
        v.reserve(static_cast<size_t>(sdf_vals.size()));
        double total = 0.0;
        for (Eigen::Index i = 0; i < sdf_vals.size(); ++i)
        {
            const float aa = std::abs(sdf_vals(i)), wa = weights(i);
            if (wa > 0.f and std::isfinite(aa)) { v.emplace_back(aa, wa); total += wa; }
        }
        if (v.empty() or total <= 0.0) return median_abs_sdf(sdf_vals);   // every return discounted
        std::sort(v.begin(), v.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
        double acc = 0.0;
        for (const auto &[val, wt] : v)
        {
            acc += wt;
            if (acc >= 0.5 * total) return val;
        }
        return v.back().first;
    }

    // Fit metric at the model's CURRENT pose.
    static float compute_sdf_median_abs(const ScanPoints &points_xyz, const Model &m)
    {
        return median_abs_sdf(m.sdf(points_xyz));
    }
};

} // namespace rc

