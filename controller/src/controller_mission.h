/*
 * controller_mission.h — high-level missions: a named, replayable sequence of navigation targets.
 *
 * WHY THIS EXISTS
 * Every controller change so far has been judged by watching the robot. That cannot distinguish "better"
 * from "different", and it cannot detect a regression that only shows up in a passage the robot happens not
 * to have taken today. A mission fixes the STIMULUS — same waypoints, same order, same room — so that what
 * changes between two runs is the controller and nothing else. The metrics below are then comparable by
 * construction, which is the whole point: they are not interesting in absolute terms, they are interesting
 * as a diff against the previous run.
 *
 * COEXISTENCE WITH THE EPISTEMIC AFFORDANCE PLANNER — see DriveMode.
 * In MissionOnly a running mission is the sole target source (below a manual mouse click, above the
 * affordance planner), and the affordance planner resumes automatically when the mission ends or is
 * stopped. This is deliberate: if affordance selection could preempt a benchmark run, two runs would differ
 * whenever PERCEPTION differed, and the measurement would be worthless for judging the controller. Missions
 * are a measuring instrument, so the instrument must not be part of the experiment.
 * MissionWithAffordances is the third, UNIMPLEMENTED mode — the open design question, stated in the .cpp.
 *
 * WAYPOINTS ARE RECORDED, NOT COMPUTED
 * You author a tour by clicking it in the 2D view, and it is saved to a TOML file. A tour derived from room
 * geometry would silently change whenever the room model changed — which is precisely the property a
 * repeatability baseline must not have.
 *
 * Pure Eigen/STL + the controller's own types; no Qt, no DSR → unit-testable in isolation (self_test()).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "controller_runtime_types.h"

namespace rc
{

// What is allowed to drive the robot. This is the EXPERIMENTAL CONDITION of a run, so it is named
// explicitly and written into the metrics CSV — "which mode produced this row" must never have to be
// reconstructed from memory when two runs are compared weeks apart.
enum class DriveMode
{
    AffordancesOnly,        // no mission; the epistemic planner has the base (the default)
    MissionOnly,            // the mission is the sole target source — the clean benchmark condition
    MissionWithAffordances, // the mission is the backbone; affordances are INSERTED BETWEEN LEGS
    Target                  // driving to a single clicked point; entered automatically by clicking one
};

const char *to_string(DriveMode m);
// Combo-box order, in one place so the widget and the worker cannot disagree about what index 2 means.
int       to_index(DriveMode m);
DriveMode from_index(int index);
// Does this mode drive from a mission? False for AffordancesOnly and Target, where the mission widgets
// have nothing to act on.
bool      uses_mission(DriveMode m);

struct MissionWaypoint
{
    Eigen::Vector2f pos = Eigen::Vector2f::Zero();
    // Facing yaw to hold on arrival. Absent ⇒ arrive and move on without rotating, which is what you want
    // for a tour: forcing a facing at every corner would measure the alignment controller, not the follower.
    std::optional<float> yaw_rad;
};

struct Mission
{
    std::string name;
    std::vector<MissionWaypoint> waypoints;
    int loops = 1;
};

// CONTINUOUS CHARACTERISATION OF ONE TRAJECTORY.
//
// Legs are gone. They were waypoint-to-waypoint segments, and a segment can only say where it started
// and where it ended — not how the robot got there. Worse, the segmentation was an artefact of how the
// route USED to be driven; once the route is one curve, cutting it at the old waypoints measures nothing
// physical. Everything below is defined at every instant of the run and aggregated over it.
//
// The two quantities legs could never express, and which now carry most of the signal:
//   CROSS-TRACK ERROR — how far the driven path sat from the reference curve. This is "did it follow the
//   route", continuously. A leg knew only its endpoints, so a leg that bulged two metres wide and came
//   back scored the same as one driven straight.
//   LATERAL ACCELERATION v^2*kappa — how hard the trajectory is driven through its own curvature. It is
//   what makes motion feel abrupt or natural, it is bounded in human and vehicle motion, and it is
//   meaningless without a continuous curve to measure curvature on.
struct TrajectoryStats
{
    float duration_s = 0.f;
    float distance_m = 0.f;          // integrated |dpos| — what was actually driven
    float route_length_m = 0.f;      // arc length of the reference curve
    float progress_m = 0.f;          // arc length reached
    int   laps_completed = 0;

    float mean_speed_mps = 0.f;
    float max_speed_mps = 0.f;

    // Tracking — driven path vs reference curve
    float cross_track_rms_m = 0.f;
    float cross_track_max_m = 0.f;
    float heading_err_rms_rad = 0.f;

    // Smoothness. rot_* and lin_* are COMMANDED (what the controller decided, noise-free);
    // cross-track, clearance and speed are MEASURED.
    float rot_effort_rad = 0.f;      // integral |omega| dt
    float rot_energy = 0.f;          // integral omega^2 dt
    int   rot_reversals = 0;         // commanded-omega sign flips: the hunting signature
    float lin_accel_effort = 0.f;    // sum |dv|  — total variation of speed, sampling-rate independent
    float lin_accel_max = 0.f;
    float lin_jerk_effort = 0.f;     // sum |d2v| — total variation of acceleration
    float lin_jerk_max = 0.f;
    float lat_accel_rms = 0.f;
    float lat_accel_max = 0.f;

    // Safety — reported as a CONSTRAINT, never folded into an objective.
    float min_clearance_m = 0.f;
    float p05_clearance_m = 0.f;     // a minimum is extreme-value noisy (measured 44% spread); a low
                                     // percentile says the same thing without the jitter
    int   safety_guard_cycles = 0;
    int   escapes = 0;
    int   replans = 0;

    // ── GEOMETRIC REPEATABILITY (multi-lap runs only; NaN below two laps) ──
    // How closely each later lap retraces the FIRST one, in space. This is the metric a multi-lap run
    // exists to produce: the robot is far more repeatable in space than in time (measured 2.5 cm mean /
    // 27 cm max against a 2.0% spread in lap time and 14.5% in reversals), so lap-to-lap geometry is the
    // most sensitive signal available for whether a change altered where the robot actually goes.
    // Measured as the distance from each sampled pose of lap N to the NEAREST point of lap 1's trace —
    // a curve-to-curve distance, not a same-index comparison, which would confound geometry with speed.
    float lap_repeat_mean_m = std::numeric_limits<float>::quiet_NaN();
    float lap_repeat_max_m  = std::numeric_limits<float>::quiet_NaN();
};

// One row of the actuation stream, sampled in the VELOCITY-OUTPUT thread rather than in compute().
// Two reasons, both load-bearing for smoothness analysis:
//  1. RATE. compute() runs at 10 Hz (Nyquist 5 Hz), and the stutter this work exists to remove is a
//     ~5 Hz command reversal. At the compute rate that defect sits exactly AT Nyquist and cannot be
//     resolved. The output thread runs at 50 ms (20 Hz).
//  2. UNIFORMITY. A spectral measure needs even sampling. compute()'s cadence is ragged (median 108 ms,
//     p99 1.26 s); the output thread is fixed-rate by construction.
struct MissionProfileSample
{
    std::uint64_t t_ms = 0;
    float adv_mps = 0.f, side_mps = 0.f, rot_rps = 0.f;
    float freshness = 1.f;      // command-age attenuation actually applied to this tick
    float v_meas_mps = 0.f;     // latest MEASURED speed
    bool  v_meas_fresh = false; // false => held from a previous row; not signal
    int   lap = 0;              // which lap this instant belongs to
    // WHERE ON THE ROUTE this instant happened, in metres of arc length, cumulative across laps.
    // Without it the actuation stream can say a reversal HAPPENED but never WHERE, so no claim about
    // which piece of geometry causes them can be tested — and one such claim (that the ~100 reversals
    // per lap come from the sub-inscribed-radius pivots) survived unexamined until the optimiser removed
    // three of the four pivots and the count did not move. -1 when the run is not driving a continuous
    // route (a click target has no arc length), which is a fact about the run, not a missing value.
    float route_s_m = -1.f;
};

// Everything about the AGENT that a metric computed later needs in order to know whether two runs are
// comparable at all. A run recorded without it is a number with no units: a 12 % improvement means
// nothing if MaxAdvSpeed or the comfort standoff changed in between.
struct MissionRunContext
{
    std::string build;
    float max_adv_mps = 0.f;
    float max_rot_rps = 0.f;
    float comfort_standoff_m = 0.f;
    float footprint_safety_margin_m = 0.f;
    float planner_cell_size_m = 0.f;
    float body_inscribed_m = 0.f;
    float body_circumscribed_m = 0.f;
};

class MissionRunner
{
public:
    enum class State { Idle, Recording, Running };

    // ── Library ────────────────────────────────────────────────────────────────────────
    // Missing file is NOT an error: a fresh install has no missions until you record one.
    bool load(const std::string &path);
    bool save(const std::string &path) const;
    std::vector<std::string> names() const;
    bool select(const std::string &name);
    const std::string &selected_name() const { return selected_; }
    const Mission *selected_mission() const;
    bool remove(const std::string &name);

    // Smooth the SELECTED mission's waypoints in place, keeping every one feasible.
    // `is_free(pos, heading)` is supplied by the caller — in the agent it is the grid planner's exact
    // footprint test — so this stays DSR-free and testable, and so that "smooth" can never produce a
    // route the planner would then refuse to drive. Endpoints are held: they are where the tour starts
    // and ends, and moving them would silently redefine it.
    // `max_shift_m` bounds how far a point may travel from where it was AUTHORED. That is a limit on how
    // much "smooth" may reinterpret the user's intent, not a physics gate — without it, repeated
    // smoothing walks a deliberate detour into a straight line.
    // Returns the number of waypoints actually moved.
    int smooth_selected(const std::function<bool(const Eigen::Vector2f &, float)> &is_free,
                        int iterations = 40, float alpha = 0.30f, float max_shift_m = 0.50f);

    // ── Recording / editing ────────────────────────────────────────────────────────────
    void start_recording();
    // Move waypoint `index` of the mission currently being shown (the recording buffer while recording,
    // otherwise the selected mission). Returns false if the index does not exist or a run is in progress —
    // editing the route under a running measurement would invalidate it silently.
    bool move_waypoint(int index, const Eigen::Vector2f &pos);
    void add_point(const Eigen::Vector2f &room_pos);
    void undo_point();
    // Commits the recorded points under `name`, replacing any mission of that name. Fewer than 2 points is
    // rejected: a one-point "tour" is a manual target with extra steps.
    bool finish_recording(const std::string &name);
    void cancel_recording();
    const std::vector<MissionWaypoint> &recorded() const { return recording_; }

    // ── Drive mode ─────────────────────────────────────────────────────────────────────
    void set_mode(DriveMode m);
    DriveMode mode() const { return mode_; }
    // MissionWithAffordances is DELIBERATELY NOT IMPLEMENTED YET — see the note at the top of the .cpp.
    // It is selectable so the choice is visible, and start() refuses it, because a mode that silently ran
    // as MissionOnly would write rows labelled "MissionWithAffordances" and quietly poison the dataset.
    bool mode_implemented() const { return mode_ != DriveMode::MissionWithAffordances; }

    // ── Running ────────────────────────────────────────────────────────────────────────
    bool start(int loops, std::uint64_t now_ms);
    // Ends the run and closes out the record. `reason` is written to the CSV so an aborted run is never
    // mistaken for a completed one when the numbers are compared later.
    void stop(const std::string &reason, std::uint64_t now_ms);

    // One-shot: true once, after a run ends BY ITSELF (all laps done, or nothing left to drive to). Not
    // set when the user stops it, changes mode, or supersedes it with a click — those already express an
    // intent about driving, whereas a tour finishing on its own does not mean "keep going".
    bool consume_completed();

    State state() const { return state_; }
    bool  running() const { return state_ == State::Running; }
    bool  recording() const { return state_ == State::Recording; }

    // ── Click target ───────────────────────────────────────────────────────────────────
    // A mouse click supersedes whatever the mission was doing, and is shown on the canvas and in the status
    // line like a one-point tour. It is NOT recorded as a run: a single click has no legs, no laps and no
    // repeatability, so a row in the metrics CSV would answer none of the questions that file exists for —
    // the same reason MissionWithAffordances refuses rather than pretending.
    void set_click_target(const std::optional<Eigen::Vector2f> &pos);
    bool has_click_target() const { return click_target_.has_value(); }

    // ── Target supply ──────────────────────────────────────────────────────────────────

    // Progress along the reference curve, fed by the route follower. Laps and completion are read off
    // arc length instead of counting waypoint arrivals.
    // `laps_done` is laps FINISHED, not the lap index. Passing the index and subtracting one is what
    // made a completed single-lap run report 0 of 1 — see RouteFollower::laps_completed_at.
    void note_progress(float progress_m, float route_length_m, int laps_done);
private:
    // One decimated position trace per lap, for lap_repeat_*. Decimated at kLapTraceStepM so the
    // end-of-run comparison stays O(n^2) on a few hundred points rather than a few thousand.
    static constexpr float kLapTraceStepM = 0.10f;
    std::vector<std::vector<Eigen::Vector2f>> lap_traces_;
    void compute_lap_repeat();
public:

    // ── Per-cycle instrumentation ──────────────────────────────────────────────────────
    // One instant of the trajectory. `cross_track_m` and `heading_err_rad` are the deviation from the
    // reference curve, `ref_curvature` its curvature there — the three things a leg could never carry.
    // `body_clearance_m` = ESDF at the robot origin MINUS the body's reach toward the obstacle, i.e. the
    // real gap; pass a negative value when unknown and it is ignored.
    void sample(const Eigen::Vector2f &pos, float rot_rps, float speed_mps,
                float body_clearance_m, bool safety_guard,
                float cross_track_m, float heading_err_rad, float ref_curvature,
                std::uint64_t now_ms);
    void note_replan();
    void note_escape();

    // ── Readout ────────────────────────────────────────────────────────────────────────
    std::string status_text() const;                 // one line for the UI
    const std::vector<Eigen::Vector2f> &display_waypoints() const { return display_wps_; }
    int  current_lap() const { return lap_; }
    // Laps still to drive, INCLUDING the one in progress — a countdown reads as "how much is left",
    // so the lap you are on has not been done yet. 0 when nothing is running.
    int  laps_remaining() const { return running() ? std::max(0, loops_ - lap_) : 0; }
    const TrajectoryStats &stats() const { return stats_; }
    TrajectoryStats summary() const;
    // Append the finished run (legs + summary) to a CSV. Header is written once per file.
    bool write_csv(const std::string &path) const;

    // ── Per-run record ─────────────────────────────────────────────────────────────────────────────
    // ONE FILE PER RUN, under <dir>/<mission>/. Not one growing file: the CSV already demonstrated
    // what happens when a schema changes under an appending log — old and new rows silently disagree
    // and the whole file becomes unreadable. A file per run cannot suffer that; adding a field only
    // makes newer files richer.
    // WRITE-ONLY from C++. Nothing here is ever read back to drive behaviour, so a hand-edited or
    // half-written record can change an analysis but can never change what the robot does.
    void set_run_context(const MissionRunContext &ctx) { run_ctx_ = ctx; }
    void set_run_dir(std::string dir) { run_dir_ = std::move(dir); }

    // Called from the VELOCITY-OUTPUT thread (see MissionProfileSample). Buffered in memory and written
    // once at the end of the run: file I/O on the fixed-rate actuation thread would be the one place in
    // this agent where a slow disk could delay a command to the base.
    void add_profile_sample(std::uint64_t t_ms, float adv, float side, float rot, float freshness);
    // Latest measured speed, pushed from the control thread; picked up by the next profile row.
    void note_measured_speed(float mps);
    // Extra per-cycle diagnostic files to ARCHIVE beside the run's JSON and profile when it ends. They
    // are written live to a fixed path (the writer cannot know the run's timestamp until it stops), so
    // without this each run silently destroys the previous one's raw data — which is exactly what
    // happened to the MPPI baseline the first time two runs were compared. The summary survived in a
    // notebook; the rows did not.
    void archive_on_stop(std::string path) { archive_.push_back(std::move(path)); }
    void set_csv_path(std::string path) { csv_path_ = std::move(path); }
    const std::string &csv_path() const { return csv_path_; }

    static bool self_test();

private:
    void  refresh_display_waypoints();
    Mission *selected_writable();

    std::vector<Mission> library_;
    std::string selected_;
    std::string csv_path_ = "mission_metrics.csv";
    std::string run_dir_ = "etc/runs";
    std::vector<std::string> archive_;   // live diagnostic files copied into the run folder at stop
    MissionRunContext run_ctx_;

    // Profile buffer. Touched by the output thread (append) and the control thread (measured speed,
    // start/stop, write), so it carries its own mutex — the rest of MissionRunner is single-threaded.
    mutable std::mutex profile_mutex_;
    std::vector<MissionProfileSample> profile_;
    float pending_v_meas_ = 0.f;
    bool  pending_v_meas_fresh_ = false;
    // Route arc length, written by the CONTROL thread in note_progress and read by the OUTPUT thread on
    // the next profile row. Atomic rather than under profile_mutex_: it is one scalar on the control
    // thread's hot path, and the actuation thread must never wait on that thread to emit a row.
    // Deliberately NOT marked fresh/stale like v_meas: position along a route is a continuous quantity
    // and the last known value is the best estimate of it, whereas a held SPEED reading would be a
    // repeated measurement masquerading as a new one.
    std::atomic<float> pending_route_s_{-1.f};
    // Bound: ~2.8 h at 20 Hz. A run left going overnight must not consume memory without limit.
    static constexpr std::size_t kMaxProfileRows = 200000;
    bool profile_truncated_ = false;
    bool write_run_json(const std::string &dir, const std::string &stamp, const std::string &iso) const;
    bool write_profile_csv(const std::string &dir, const std::string &stamp) const;
    State state_ = State::Idle;
    DriveMode mode_ = DriveMode::AffordancesOnly;

    std::vector<MissionWaypoint> recording_;
    std::vector<Eigen::Vector2f> display_wps_;
    std::optional<Eigen::Vector2f> click_target_;

    // Run state
    Mission active_;
    int lap_ = 0;              // 0-based
    int loops_ = 1;
    std::uint64_t run_start_ms_ = 0;
    std::string stop_reason_;
    bool completed_event_ = false;

    // Continuous accumulators — no segmentation.
    TrajectoryStats stats_;
    std::uint64_t run_first_sample_ms_ = 0;
    std::uint64_t last_sample_ms_ = 0;
    double ct_sq_sum_ = 0.0, hd_sq_sum_ = 0.0, lat_sq_sum_ = 0.0;
    long   ct_n_ = 0;
    std::vector<float> clearances_;
    std::optional<float> prev_cmd_speed_;   // for sum|dv|
    std::optional<float> prev_cmd_dv_;      // for sum|d2v|
    int prev_rot_sign_ = 0;                 // for the reversal count
    std::optional<Eigen::Vector2f> last_pos_;

};

}  // namespace rc
