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

// One leg = one waypoint-to-waypoint segment. These are the numbers a controller change actually moves.
struct MissionLegMetrics
{
    int   lap = 0;
    int   leg = 0;
    float duration_s = 0.f;
    float path_length_m = 0.f;     // integrated |Δpos| — what the robot actually drove
    float straight_line_m = 0.f;   // waypoint-to-waypoint distance
    float detour_ratio = 0.f;      // path_length / straight_line; 1.0 = perfect, >1 = wandering
    float min_body_clearance_m = 0.f;   // closest the BODY came to anything (ESDF minus the body's reach)
    float mean_speed_mps = 0.f;
    float rot_effort = 0.f;        // ∫|ω| dt (rad) — total turning done; a smoothness/economy proxy
    float rot_energy = 0.f;        // ∫ω² dt — penalises fast reversals specifically (the stutter signature)
    float arrival_error_m = 0.f;   // distance from the waypoint where the leg was declared reached
    int   replans = 0;
    int   escapes = 0;
    int   safety_guard_cycles = 0;
    bool  completed = false;       // false ⇒ the leg was abandoned (mission stopped mid-leg)
};

// Whole-run roll-up, including the one number that answers "is my controller repeatable".
struct MissionRunSummary
{
    std::string mission_name;
    int   laps_completed = 0;
    int   legs_completed = 0;
    float total_duration_s = 0.f;
    float total_path_length_m = 0.f;
    float mean_detour_ratio = 0.f;
    float min_body_clearance_m = 0.f;
    float total_rot_effort = 0.f;
    int   total_replans = 0;
    int   total_escapes = 0;
    // Lap-to-lap repeatability: mean and max distance from each sampled pose of laps 2..N to the polyline
    // traced on lap 1. Zero laps>1 ⇒ not measured (reported as NaN, never as 0 — a missing measurement and
    // a perfect score must not print the same).
    float lap_repeat_mean_m = 0.f;
    float lap_repeat_max_m = 0.f;
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
    // The current waypoint as a controller target. nullopt ⇒ the mission is finished; the caller should
    // stop() and let the affordance planner resume.
    std::optional<ControllerTargetInfo> current_target() const;
    // Call when the current target has been reached. Advances to the next waypoint / lap / end.
    void advance(const Eigen::Vector2f &arrived_at, std::uint64_t now_ms);

    // ── Per-cycle instrumentation ──────────────────────────────────────────────────────
    // `body_clearance_m` = ESDF at the robot origin MINUS the body's own reach, i.e. the real gap between
    // the robot and the nearest obstacle. Pass a negative value when unknown and it is ignored.
    void sample(const Eigen::Vector2f &pos, float rot_rps, float speed_mps,
                float body_clearance_m, bool safety_guard, std::uint64_t now_ms);
    void note_replan();
    void note_escape();

    // ── Readout ────────────────────────────────────────────────────────────────────────
    std::string status_text() const;                 // one line for the UI
    const std::vector<Eigen::Vector2f> &display_waypoints() const { return display_wps_; }
    int  current_waypoint_index() const { return wp_index_; }
    int  current_lap() const { return lap_; }
    const std::vector<MissionLegMetrics> &legs() const { return legs_; }
    MissionRunSummary summary() const;
    // Append the finished run (legs + summary) to a CSV. Header is written once per file.
    bool write_csv(const std::string &path) const;
    void set_csv_path(std::string path) { csv_path_ = std::move(path); }
    const std::string &csv_path() const { return csv_path_; }

    static bool self_test();

private:
    void  begin_leg(std::uint64_t now_ms);
    void  close_leg(const Eigen::Vector2f &arrived_at, std::uint64_t now_ms, bool completed);
    void  refresh_display_waypoints();
    Mission *selected_writable();
    float distance_to_lap1(const Eigen::Vector2f &p) const;

    std::vector<Mission> library_;
    std::string selected_;
    std::string csv_path_ = "mission_metrics.csv";
    State state_ = State::Idle;
    DriveMode mode_ = DriveMode::AffordancesOnly;

    std::vector<MissionWaypoint> recording_;
    std::vector<Eigen::Vector2f> display_wps_;
    std::optional<Eigen::Vector2f> click_target_;

    // Run state
    Mission active_;
    int lap_ = 0;              // 0-based
    int wp_index_ = 0;
    int loops_ = 1;
    std::uint64_t run_start_ms_ = 0;
    std::string stop_reason_;

    // Leg accumulators
    MissionLegMetrics leg_{};
    // Is a leg currently in flight? stop() must close an ABANDONED leg but must not re-close the one
    // advance() just finished — without this, completing the final waypoint appended a phantom zero-length
    // leg marked incomplete, which made every clean run look like it had been aborted.
    bool leg_open_ = false;
    std::uint64_t leg_start_ms_ = 0;
    std::uint64_t last_sample_ms_ = 0;
    std::optional<Eigen::Vector2f> last_pos_;
    Eigen::Vector2f leg_origin_ = Eigen::Vector2f::Zero();
    std::vector<MissionLegMetrics> legs_;

    // Repeatability: the pose trace of lap 1, and the running deviation of later laps from it.
    std::vector<Eigen::Vector2f> lap1_trace_;
    double repeat_sum_ = 0.0;
    long   repeat_n_ = 0;
    float  repeat_max_ = 0.f;
};

}  // namespace rc
