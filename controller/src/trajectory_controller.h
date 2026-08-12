#pragma once

#include <vector>
#include <limits>
#include <optional>
#include <string>
#include <Eigen/Dense>

#include "../../common/robot_footprint/robot_footprint.h"

#include "lidar_buffer_types.h"

#include "controller_types.h"
#include "trackers/plain_tracker.h"
#include "trackers/pd_tracker.h"
#include "trackers/mppi_tracker.h"

namespace rc { class RouteSpline; }   // set_route: non-owning, the session's RouteFollower owns it

namespace rc
{
/**
 * TrajectoryController — MPPI-based local controller with ESDF.
 *
 * Proper MPPI: warm-start + Gaussian perturbations + AR(1) noise.
 * All K samples are perturbations of the previous optimal sequence.
 * Weighted average over the FULL T-step sequence (not just first step).
 */
class TrajectoryController : public PathWorld, public FieldWorld
{
public:
    // PLAIN = the route-following tracker in trackers/plain_tracker.h. Curvature feedforward previewed
    // by the identified actuator lag, plus critically-damped feedback on the Frenet error pair
    // (e_y, e_psi). It performs NO avoidance — that belongs to the band and the planner.
    enum class ControlMode { MPPI, PD, PLAIN };

    // ── Params and ControlOutput now live in controller_types.h at namespace scope ──────────
    // so a tracker can name them without including this header. These aliases keep every existing
    // TrajectoryController::Params / ::ControlOutput reference compiling unchanged.
    using Params        = ::rc::TrackerParams;
    using ControlOutput = ::rc::ControlOutput;

    TrajectoryController() = default;

    void set_path(const std::vector<Eigen::Vector2f>& path_room);
    /// As set_path, but WITHOUT relax_path/smooth_path_spline. For a path that is already smooth and
    /// already footprint-checked (RouteSpline): re-running the elastic band would push samples around
    /// with its own clearance heuristic and could undo the feasibility pass, and re-smoothing a C2 curve
    /// with a C1 spline would put back the curvature steps the whole exercise removes.
    void set_path_presmoothed(const std::vector<Eigen::Vector2f>& path_room);

    /// Swap the path GEOMETRY without resetting the follower — for a continuously deformed route
    /// (the local elastic band), which hands over a slightly different curve every cycle.
    ///
    /// set_path/set_path_presmoothed both go through reset_mppi_state, which is correct for a NEW route
    /// and catastrophic at control rate: it clears prev_optimal_ (the warm start the whole sampler leans
    /// on), the adaptive lambda/K/T, smoothed_vel_, last_cmd_valid_ and carrot_seg_hint_. Calling it
    /// every cycle would restart the optimisation from scratch 10 times a second.
    ///
    /// SAFE ONLY FOR A DEFORMATION, not a re-route: the caller must guarantee the new curve is the same
    /// route, still starts behind the robot, and has not moved under it — which is what freezing the
    /// control points before the robot buys. carrot_seg_hint_ is kept and merely clamped, because with a
    /// frozen prefix the arc length behind the robot is unchanged, so the index still means what it did.
    void update_path_geometry(const std::vector<Eigen::Vector2f>& path_room);
    // Optional room-frame heading (atan2(dy,dx) direction) the robot's forward axis
    // should face once the goal position is reached. Empty = legacy behaviour:
    // declare the goal reached immediately on arrival, no final rotation.
    void set_goal_facing_yaw(std::optional<float> yaw_rad);

    /// Where ARRIVAL is judged, when that is not the end of the path.
    ///
    /// The follower shapes speed against the end of its path: the nominal seed scales adv down inside
    /// `nominal_goal_dist_scale`, the straight-speed override switches off inside
    /// `straight_speed_min_goal_dist`, and `effective_d_safe_for_goal_dist` relaxes the standoff. All of
    /// that is correct behaviour for ARRIVING and wrong for PASSING THROUGH. Handing the follower a path
    /// that ends at the next waypoint therefore made every waypoint a deceleration — on a 30-waypoint
    /// tour with 1.28 m legs, the robot was inside the arrival regime for essentially the whole run and
    /// never reached cruise (measured mean 0.30 m/s against a 0.7 m/s limit).
    ///
    /// So the two questions are separated: the path may run several waypoints ahead, which is what the
    /// SPEED terms see, while arrival is judged against this near point. Unset ⇒ the path end, i.e. the
    /// previous behaviour exactly.
    /// `outgoing_dir_room` is the direction the route CONTINUES in after this point. With it, arrival
    /// also fires on PASSAGE — once the robot is past the plane through the point normal to that
    /// direction — not only on proximity.
    ///
    /// This is not optional once the path runs past the arrival point. The carrot sits
    /// `carrot_lookahead` (2 m) ahead ALONG the path, so with a 4 m horizon it aims well beyond the
    /// waypoint and the robot deliberately cuts the corner — often never entering the 0.25 m arrival
    /// radius at all. Measured: the robot drove 6.03 m on a 0.41 m leg, ended 1.57 m away, and the leg
    /// never completed; the follower then ran out of path, went inactive, the session re-issued the same
    /// path and it turned around and drove it again. Cutting the corner is the POINT of the horizon, so
    /// arrival has to be a question about passing, not about touching.
    void set_arrival_point(std::optional<Eigen::Vector2f> room_pos,
                           std::optional<Eigen::Vector2f> outgoing_dir_room = std::nullopt)
    { arrival_point_room_ = room_pos; arrival_outgoing_room_ = outgoing_dir_room; }

    /// WHO OWNS ARRIVAL. The endpoint test above is EUCLIDEAN: "am I near path_room_.back()?". That is a
    /// correct question for a path that goes from here to somewhere else, and a WRONG one for a closed
    /// route, where the end IS the beginning: at s=0 the robot stands within the goal radius of the
    /// endpoint it has not driven a single metre toward, so arrival fires on the first cycle and the tour
    /// never starts. Proximity alone cannot distinguish "not yet departed" from "returned" — only arc
    /// length can, and the RouteFollower already carries it (forward-only `progress()`), so a continuous
    /// route judges its own completion and switches this test OFF. Two arrival definitions for one motion
    /// is the defect; this leaves exactly one in force at a time.
    void set_endpoint_arrival(bool on) { endpoint_arrival_ = on; }

    // CURVATURE-LIMITED SPEED. A ceiling on max_adv for this cycle, supplied by whoever owns the
    // reference curve. The MPPI cannot derive it: its horizon is ~1.4 s of travel and its nominal seed
    // scales speed by carrot DISTANCE and alignment, neither of which knows how sharp the turn ahead is.
    // The route does know — it is C2, so kappa(s) is continuous — and v = sqrt(a_lat_max / kappa) is the
    // physical relation between lateral acceleration and turn radius, not a tuning threshold.
    //
    // It must be a CEILING ON THE PLAN, not a clamp on the output: clamping the command after the fact
    // would leave the MPPI planning trajectories it is not allowed to execute, and the resulting
    // commanded-vs-measured gap is exactly what the wedge detector reads as being stuck.
    // nullopt clears it (a click target has no curve, so it keeps the full speed envelope).
    void set_speed_limit(std::optional<float> v_max_mps) { speed_limit_ = v_max_mps; }

    // Seed the carrot's forward-only anchor when a path is (re)installed mid-route — after a repair the
    // robot is somewhere in the middle, and starting the search from segment 0 would aim it back at the
    // route's beginning. set_path resets this to 0, which is correct for a path that starts at the robot.
    // The curve the ROUTE tracker follows. Non-owning: the session's RouteFollower owns it, and the
    // elastic band deforms it in place on this same thread just before compute — so the pointer stays
    // valid and the tracker automatically reads the deformed curve. nullptr = fall back to PD.
    // ★Handing over a DIFFERENT spline invalidates any arc length measured against the previous one, so
    // the trackers' projections are dropped here too. Same defect as in stop(): s_hint_ is monotone
    // within one traversal and means nothing across two.
    // The arc length the PLAIN control law actually used this cycle. Empty in any other mode, and
    // before the first cycle of a traversal. NOT RouteFollower::progress() — a different projection
    // with a different window, which is why both are logged.
    [[nodiscard]] std::optional<float> tracker_arc_length() const { return plain_tracker_.arc_length_hint(); }

    // `force_reset` = "this is a NEW curve", even if it arrives in the SAME object. A mission route is
    // built once and its address identifies it, but every point target — click or affordance — is
    // refitted into the session's one `plan_spline_` member, whose address never changes. Comparing
    // pointers alone would then carry the previous traversal's arc-length hint into a different curve,
    // which is the tracker steering at the old route's geometry from the new route's start.
    void set_route(const RouteSpline *spline, bool force_reset = false)
    {
        if (force_reset or spline != route_spline_) { plain_tracker_.reset(); pd_tracker_.reset(); }
        route_spline_ = spline;
    }

    void set_carrot_hint(int segment_index) { carrot_seg_hint_ = std::max(0, segment_index); }

    ControlOutput compute(const Eigen::Affine2f& robot_pose);
    /// Same cycle, with the cloud supplied instead of read from the buffer — the replay path.
    ControlOutput compute(const Eigen::Affine2f& robot_pose,
                          const std::vector<Eigen::Vector3f>& lidar_points);

    /// Seed the sampler. The default seeds from std::random_device, so two runs with IDENTICAL inputs
    /// produce DIFFERENT commands — which is correct for a robot and useless for a comparison: it puts
    /// sampling noise inside every A/B, on top of the 14.5% the world already contributes. Setting a seed
    /// makes a cycle reproducible, which is what tools/mppi_bench needs to answer "what would this cost
    /// change have done" without driving.
    void set_seed(std::uint32_t seed) { mppi_tracker_.set_seed(seed); }

    /// Write everything the NEXT compute() consumes to `path`: pose, the lidar points it actually used,
    /// the obstacle and boundary polygons, the path, and the parameters. Replaying it rebuilds the ESDF
    /// with the SAME build_esdf on the SAME inputs, so nothing is re-derived by a second implementation
    /// (the trap route_bench avoids by recording the planner's raster).
    /// ★Snapshots the INPUTS, not the ESDF: here the producer and the replayer are the same function, so
    /// the inputs are the smaller, more honest artefact.
    void request_snapshot(std::string path) { snapshot_path_ = std::move(path); }

    /// Restore a snapshot: parameters, path, obstacle and boundary points are set on THIS controller,
    /// and the cycle's pose and cloud are handed back so the caller can replay it with
    /// compute(pose, lidar). Returns false on a malformed stream.
    bool load_snapshot(std::istream &is, Eigen::Affine2f &pose_out,
                       std::vector<Eigen::Vector3f> &lidar_out);
    void stop();
    void set_lidar_buffer(LidarPointBuffer *buffer) { lidar_buffer_ = buffer; }

    // ── PathWorld / FieldWorld: what a tracker is allowed to see (trackers/tracker.h) ──────────
    const RouteSpline* route_spline() const override { return route_spline_; }
    float body_extent(const Eigen::Vector2f& dir, float heading) const override
    { return footprint_.support_radius(-heading, dir); }
    float body_extent_max() const override { return footprint_.circumscribed_radius(); }
    float esdf_at(float rx, float ry) const override { return query_esdf(rx, ry); }
    Eigen::Vector2f esdf_gradient_at(float rx, float ry) const override { return query_esdf_gradient(rx, ry); }
    float body_extent_toward_obstacle(float rx, float ry, float theta) const override;
    float body_extent_here() const override;

    void set_control_mode(ControlMode mode) { control_mode_ = mode; }
    ControlMode control_mode() const { return control_mode_; }

    /// Set static obstacle polygons (furniture). Their edges are sampled into points
    /// and injected into every ESDF build, transformed to robot frame.
    void set_static_obstacles(const std::vector<std::vector<Eigen::Vector2f>>& obstacles_room,
                              float sample_spacing = 0.05f);

    /// Set room boundary polygon. Edges are sampled into points used by relax_path
    /// to push waypoints away from walls toward the center of free space.
    void set_room_boundary(const std::vector<Eigen::Vector2f>& polygon,
                           float sample_spacing = 0.05f);

    bool is_active() const { return active_ && !path_room_.empty(); }
    int current_waypoint_index() const { return wp_index_; }
    const std::vector<Eigen::Vector2f>& get_path() const { return path_room_; }
    std::optional<Eigen::Vector2f> current_waypoint_room() const;

    /// Signed-distance-to-obstacle (m) at a point given in the ROBOT frame, sampled from
    /// the ESDF built during the most recent compute(). Used by recovery to probe side /
    /// rear clearance. Returns the unknown-distance sentinel if no ESDF exists yet.
    ///
    /// FRAME: the same one the rollouts integrate in — **+Y is FORWARD, +X is RIGHT**
    /// (`x += adv·sin θ`, `y += adv·cos θ`). This is NOT the OmniRobot command frame
    /// (adv/side/rot, +side = left). So a LEFT probe is (−d, 0), RIGHT is (+d, 0),
    /// FRONT is (0, +d) and REAR is (0, −d).
    ///
    /// ⚠ ONLY VALID ON A CYCLE WHERE compute() RAN. The grid is rebuilt by build_esdf()
    /// inside compute(); nothing else refreshes it. During an escape/recovery manoeuvre
    /// compute() is not called at all, so this keeps returning the field as it was FROZEN at
    /// the moment the manoeuvre began — still expressed in robot-frame coordinates — while
    /// the robot is reversing and rotating out from under it. Treat readings taken during
    /// such a manoeuvre as a snapshot of the situation at its START, not as live clearance.
    float clearance_at(float rx, float ry) const { return query_esdf(rx, ry); }
    /// Gradient of the same field (robot frame), for a caller that wants to DESCEND it rather than
    /// merely read it — the local elastic band. Outside the grid clearance_at returns
    /// esdf_unknown_distance (100 m), so a one-sided clearance term sees no deficit and therefore no
    /// force: the band cannot be pulled by geometry the live field does not cover.
    Eigen::Vector2f clearance_gradient_at(float rx, float ry) const { return query_esdf_gradient(rx, ry); }

    // What the room-boundary injection did on the last build_esdf: how many cells it contributed, and
    // whether it was rejected as implausible. Recorded rather than printed so it can be compared per run.
    int  esdf_boundary_cells() const { return esdf_boundary_cells_; }
    /// Room-frame model points dropped because the lidar already reported an obstacle there. A large
    /// number is not a fault — it means the model and the measurement agree, which is the normal case.
    /// It rising sharply means they have started to DISAGREE, i.e. the pose has moved under the model.
    int  esdf_model_points_dropped() const { return esdf_model_points_dropped_; }
    bool esdf_boundary_rejected() const { return esdf_boundary_rejected_; }

    /// Body reach toward whatever is nearest RIGHT NOW (robot frame, heading 0) — the number to
    /// subtract from clearance_at(0,0) to get the true gap between the body and the world.

    /// The robot's real silhouette. Exposed so callers ask THIS object what the body reaches instead of
    /// keeping their own radius — that habit is what produced six independent margins across three agents.
    const RobotFootprint& footprint() const { return footprint_; }

    Params params;

private:
    Params active_params_;

    bool active_ = false;
    ControlMode control_mode_ = ControlMode::MPPI;
    PlainTracker plain_tracker_{*this};
    bool plain_no_curve_logged_ = false;   // the PD-fallback notice is worth saying once, not at 20 Hz
    // How long the PD fallback has been driving. A latch alone made "one cycle" and "the whole session"
    // print the same single line, so the fallback could not be measured at all — which is how a silent
    // tracker swap survives a validation run.
    int  plain_no_curve_cycles_ = 0;
    PdTracker    pd_tracker_{*this, *this};
    MppiTracker  mppi_tracker_{*this, *this};
    // Fills the fields every tracker reads. One place, so a new tracker cannot be handed a stale carrot.
    TrackerInput make_tracker_input(const Eigen::Affine2f& robot_pose,
                                    const Eigen::Vector2f& carrot_robot,
                                    const Eigen::Vector2f& goal_robot) const;
    // Where the robot projects onto the path, from the last compute_carrot. Kept because compute_carrot
    // already computes it and the tracker needs it; recomputing would risk a second, disagreeing answer.
    Eigen::Vector2f last_carrot_room_ = Eigen::Vector2f::Zero();
    bool  has_last_carrot_ = false;      // reset with the path: a new route legitimately moves the carrot
    Eigen::Vector2f last_path_proj_room_ = Eigen::Vector2f::Zero();
    bool last_path_proj_valid_ = false;
    std::vector<Eigen::Vector2f> path_room_;
    int wp_index_ = 0;
    std::optional<float> goal_facing_yaw_;
    std::optional<float> speed_limit_;   // see set_speed_limit
    const RouteSpline *route_spline_ = nullptr;   // see set_route
    int carrot_seg_hint_ = 0;            // forward-only anchor for compute_carrot; see the comment there
    std::optional<Eigen::Vector2f> arrival_point_room_;
    std::optional<Eigen::Vector2f> arrival_outgoing_room_;  // desired room-frame facing dir after arrival
    bool endpoint_arrival_ = true;   // see set_endpoint_arrival — off while a continuous route owns arrival
    // ── PASSED-THE-GOAL WATCH (see the recession test in compute) ────────────────────────────────
    // Closest approach on this traversal, the largest per-cycle change in distance seen (which is what
    // a slow cycle could have jumped), and how many consecutive cycles the robot has been receding.
    // Reset on every new path/route and on arrival, or the next target inherits a closest of ~0 and
    // "recedes" from it immediately, stopping the robot before it has gone anywhere.
    float closest_to_goal_ = std::numeric_limits<float>::infinity();
    float last_dist_to_goal_ = std::numeric_limits<float>::infinity();
    float max_goal_step_ = 0.f;
    int   receding_cycles_ = 0;
    void reset_arrival_watch()
    {
        closest_to_goal_ = std::numeric_limits<float>::infinity();
        last_dist_to_goal_ = std::numeric_limits<float>::infinity();
        max_goal_step_ = 0.f;
        receding_cycles_ = 0;
    }
    // The robot's real shape — the SAME polygon the global planner collides (common/robot_footprint), so the
    // two layers cannot disagree about what fits. Previously the MPPI used a 0.25 m disc while the planner
    // used the footprint, which meant the planner could route through a 0.46 m gap the MPPI would then refuse.
    RobotFootprint footprint_ = RobotFootprint::shadow();
    bool aligning_ = false;                 // true while doing the final in-place rotation
    // Worst-case time the base may keep executing one alignment command before we can revise it. Used to bound
    // the in-place rotation so it cannot overshoot the tolerance band (see the arrival block). Measured p99 of
    // the compute cadence is ~1 s with a tail to 1.5 s, so this is deliberately pessimistic — being slow to
    // finish an alignment is harmless; overshooting it means never finishing at all.
    // ── THE ALIGN BOUND'S IDEA OF A CYCLE — AND WHY IT IS NOT THE MEASURED ONE ───────────────────
    // The terminal rotation is capped at |yaw_err| / this, so one cycle can never sweep more than the
    // error that remains: overshoot becomes impossible without a damping term. 1.0 s is far longer than
    // the measured 50 ms loop, so on paper it is twentyfold too conservative.
    // ★TRIED AND REVERTED (2026-08-08): feeding the MEASURED cycle here. It is wrong, and the reason is
    // that this bound is not only a safety backstop — it is also the APPROACH PROFILE. At 1.0 s the cap
    // is |yaw_err| rad/s, which sits below align_kp (1.5) everywhere and keeps the turn off its ceiling
    // until 0.8 rad of error. Set it to the real cycle and the cap goes slack, align_kp takes over, and
    // the rotation saturates at max_rot (0.8 rad/s ~ 45 deg/s) for any error past 0.53 rad — measured
    // live: 65% of align cycles above 0.7 rad/s, a base swinging at full speed right in front of the
    // object it is trying to photograph, and NO benefit (episodes 1.46 s vs 1.50 s before).
    // So the slow turn is not this number's fault, and speeding it up here costs the profile. If the
    // terminal rotation should be quicker, the honest lever is align_kp with a deceleration envelope
    // that lands inside the contract's own .still() omega — the rate a clean look actually permits —
    // not the removal of the bound that happens to be shaping the approach.
    float align_worst_cycle_s_ = 1.0f;

    // Pending snapshot request (see request_snapshot). Written at the END of compute(), when the cycle's
    // inputs and its outcome are both known.
    std::string snapshot_path_;
    void write_snapshot(const Eigen::Affine2f &robot_pose,
                        const std::vector<Eigen::Vector3f> &lidar_points) const;

    // ---- ESDF ----
    std::vector<float> esdf_data_;
    int esdf_N_ = 0;
    LidarPointBuffer *lidar_buffer_ = nullptr;

    // Static obstacles (furniture) — pre-sampled points in room frame
    std::vector<Eigen::Vector2f> static_obstacle_points_room_;

    // Room boundary walls — pre-sampled points in room frame (for path relaxation)
    std::vector<Eigen::Vector2f> room_boundary_points_room_;
    // Cells the boundary pass turned on this cycle — kept so a rejected boundary can be rolled back
    // without erasing obstacles the LiDAR or furniture had already marked at the same cells.
    std::vector<int> boundary_cells_;
    int  esdf_model_points_dropped_ = 0;   // model points deferred to the lidar this build
    int  esdf_boundary_cells_ = 0;
    bool esdf_boundary_rejected_ = false;

    bool  carrot_curve_active_ = false; // hold reduced carrot lookahead while inside a curve

    // Path blockage detection state
    int   blockage_streak_ = 0;          // consecutive cycles with blockage detected
    int   blockage_cooldown_ = 0;        // cycles remaining before next replan allowed

    void refresh_active_params();

    // Installs `path_room` and resets ALL per-path controller state (MPPI warm start, adaptive
    // K/T/λ/σ, ESS, smoothing, blockage, alignment) WITHOUT any geometry conditioning. Both
    // set_path and set_path_presmoothed go through this; only set_path then relaxes + splines.
    // Does NOT set wp_index_ — the two callers legitimately differ, so each sets its own.
    void reset_mppi_state(const std::vector<Eigen::Vector2f>& path_room);
    // Path-blockage detector. Called from BOTH control modes (the PD branch returns before step 15).
    void detect_path_blockage(ControlOutput& out, const Eigen::Affine2f& robot_pose);

    // Elastic-band path relaxation: push waypoints toward center of free space
    void relax_path(int iterations = 20);

    // Catmull-Rom spline smoothing: replace polyline with smooth curve
    void smooth_path_spline();

    // ---- Methods ----
    void build_esdf(const std::vector<Eigen::Vector3f>& lidar_points,
                    const Eigen::Affine2f& robot_pose);
    std::vector<Eigen::Vector3f> read_lidar_points_robot(const Eigen::Affine2f& robot_pose) const;
    float query_esdf(float rx, float ry) const;
    Eigen::Vector2f query_esdf_gradient(float rx, float ry) const;

    Eigen::Vector2f compute_carrot(const Eigen::Affine2f& robot_pose);
    void advance_waypoints(const Eigen::Affine2f& robot_pose);

    // Robot extent toward the nearest obstacle at a pose — the footprint's support function along the
    // ESDF's negative gradient. Replaces the constant robot_radius disc in the obstacle terms.
    // Robot extent toward a KNOWN direction, for a body at heading `heading`. Use this wherever the thing
    // being tested has a bearing — a LiDAR return, a nearest obstacle point, a path tangent — because then
    // the exact answer is available and a disc is pure pessimism. `dir` and `heading` must be in the SAME
    // frame; which frame does not matter, only that they agree.
    //
    // ONE negation, in ONE place: this controller integrates x += adv·sin θ, y += adv·cos θ, so its heading
    // is measured from +y toward +x — CLOCKWISE — while RobotFootprint rotates counter-clockwise. The Shadow
    // hull is very nearly x-symmetric, so mixing the two costs only millimetres, which is precisely why it
    // would never be caught by watching the robot drive. Keep the conversion here and it cannot drift.
    // (public override above; this overload kept for the heading-defaulted call sites)
    float body_extent(const Eigen::Vector2f& dir) const { return body_extent(dir, 0.f); }
    // Worst-case extent over all bearings. The ONLY legitimate fallback: score normalisations and scale
    // constants, where no bearing exists and being conservative costs nothing but a slightly wider ramp.
    // Can the robot get from HERE to `carrot_robot` by heading straight at it? The MPPI pulls toward the
    // carrot in a straight line, so a carrot the robot cannot cut to directly is not a target, it is a
    // trap. Samples the chord against the live ESDF with the body's real reach at the chord heading.
    bool chord_admits_body(const Eigen::Vector2f& carrot_robot) const;
    // The furthest point along the chord that IS admissible, so the carrot can be pulled back to it.
    Eigen::Vector2f clip_carrot_to_reachable(const Eigen::Vector2f& carrot_robot) const;

    // PD carrot-follower (alternative to MPPI)

    static float clamp01(float x);
    static float smoothstep01(float x);
    static Eigen::Vector2f room_to_robot(const Eigen::Vector2f& p_room,
                                          const Eigen::Affine2f& robot_pose);
};

} // namespace rc

