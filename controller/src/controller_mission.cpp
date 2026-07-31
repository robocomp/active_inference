/*
 * controller_mission.cpp — see controller_mission.h
 *
 * ★ OPEN: DriveMode::MissionWithAffordances IS NOT IMPLEMENTED.
 * The mode exists in the enum and in the UI so the three conditions are nameable and the choice is visible,
 * but start() refuses it. Making it "work" is a few lines — serve an affordance at each waypoint arrival,
 * then resume — and that is exactly why it is not done: the easy version answers none of the questions that
 * decide whether the result means anything.
 *
 *   1. WHOSE OBJECTIVE IS IT? The epistemic planner selects by EFE, G = λ·dist − ΔH, over affordances it
 *      believes reachable. A tour imposes an exogenous route the planner never chose. Either the tour's
 *      next waypoint enters the EFE as one more option (in which case the planner may simply never pick it,
 *      and the "mission" quietly stops being a mission), or it does not (in which case the two are not
 *      arbitrating at all — they are taking turns, and calling it active inference is a stretch).
 *   2. WHAT DOES λ·dist MEAN MID-TOUR? Navigation cost is measured from the robot. Standing at a waypoint,
 *      the affordance the planner likes is cheap now but may be far off the remaining route. Without a term
 *      for "distance back to the tour", interleaving will bias every run toward affordances near wherever
 *      the tour happens to pass — which is a property of the tour, not of the world.
 *   3. WHAT HAPPENS TO REPEATABILITY? The lap-to-lap number assumes the stimulus is fixed. Once affordances
 *      are inserted, lap 2 differs from lap 1 because the robot LEARNED something on lap 1 — beliefs
 *      sharpen, ΔH drops, and the same affordance stops being selected. That is the interesting science and
 *      it is fatal to using the number as a controller regression test. The two purposes need separating
 *      before the mode exists, not after it has produced a folder of ambiguous CSVs.
 *
 * Until that is settled, MissionOnly stays the clean instrument and AffordancesOnly stays normal operation.
 */

#include "controller_mission.h"

#include <algorithm>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>

namespace rc
{

namespace
{
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

std::string trim(std::string s)
{
    const auto not_space = [](unsigned char c) { return not std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::string unquote(std::string s)
{
    s = trim(std::move(s));
    if (s.size() >= 2 and (s.front() == '"' or s.front() == '\'') and s.back() == s.front())
        return s.substr(1, s.size() - 2);
    return s;
}

// Parses `[a, b, c]`. Non-numeric entries (we write "nan" for "no facing yaw") come back as NaN, which is
// exactly how the caller distinguishes an absent yaw from a yaw of zero.
//
// ★ LOCALE. This MUST NOT use stof/strtod/atof. Those honour the C locale's LC_NUMERIC, and QApplication
// calls setlocale(LC_ALL, "") at startup — so on a machine with a comma decimal separator (es_ES, de_DE,
// fr_FR, …) stof("-1.9114") parses "-1", stops at the '.', and returns -1.0. The file was written with
// dots by a C++ stream (whose locale is the CLASSIC one unless std::locale::global is called), so every
// load silently TRUNCATED each coordinate toward zero, and the next save wrote the truncated value back —
// a tour degraded a little more every time the agent restarted. std::from_chars is locale-independent by
// definition, which is the whole reason it exists.
std::vector<float> parse_array(const std::string &value)
{
    std::vector<float> out;
    const auto lb = value.find('[');
    const auto rb = value.rfind(']');
    if (lb == std::string::npos or rb == std::string::npos or rb < lb) return out;
    std::stringstream ss(value.substr(lb + 1, rb - lb - 1));
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item = trim(item);
        if (item.empty()) continue;
        const char *first = item.data();
        const char *last = item.data() + item.size();
        if (*first == '+') ++first;              // from_chars rejects a leading '+'; hand-edits may have one
        float v = kNaN;
        const auto [ptr, ec] = std::from_chars(first, last, v);
        // Require the WHOLE token to parse. A partial parse is exactly the failure this replaced: it looks
        // like a number and is off by the fractional part.
        out.push_back(ec == std::errc{} and ptr == last ? v : kNaN);
    }
    return out;
}
}  // namespace

const char *to_string(DriveMode m)
{
    switch (m)
    {
        case DriveMode::MissionOnly:            return "mission_only";
        case DriveMode::MissionWithAffordances: return "mission_with_affordances";
        case DriveMode::Target:                 return "target";
        case DriveMode::AffordancesOnly:
        default:                                return "affordances_only";
    }
}

int to_index(DriveMode m)
{
    switch (m)
    {
        case DriveMode::MissionOnly:            return 1;
        case DriveMode::MissionWithAffordances: return 2;
        case DriveMode::Target:                 return 3;
        case DriveMode::AffordancesOnly:
        default:                                return 0;
    }
}

DriveMode from_index(int index)
{
    switch (index)
    {
        case 1:  return DriveMode::MissionOnly;
        case 2:  return DriveMode::MissionWithAffordances;
        case 3:  return DriveMode::Target;
        default: return DriveMode::AffordancesOnly;
    }
}

bool uses_mission(DriveMode m)
{
    return m == DriveMode::MissionOnly or m == DriveMode::MissionWithAffordances;
}

void MissionRunner::set_mode(DriveMode m)
{
    if (m == mode_) return;
    mode_ = m;
    // Changing the condition mid-run would produce a single CSV row set spanning two conditions, which is
    // worse than either: end the run instead, and say so.
    if (state_ == State::Running)
        stop("mode changed", last_sample_ms_);
    // Leaving Target abandons the clicked point: the mode selector says what is driving, so it must not
    // claim "Mission" while a stale click marker is still the thing on the canvas.
    if (m != DriveMode::Target)
        click_target_.reset();
    // In "affordances only" there is no tour in play either, so leaving one drawn is stale decoration on a
    // canvas whose whole job is to show what the robot is actually doing.
    refresh_display_waypoints();
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Library I/O
// ─────────────────────────────────────────────────────────────────────────────────────────

bool MissionRunner::load(const std::string &path)
{
    std::ifstream in(path);
    if (not in) return false;   // no file yet is the normal state before the first recording

    std::vector<Mission> loaded;
    Mission cur;
    std::vector<float> xs, ys, yaws;

    const auto flush = [&]()
    {
        if (cur.name.empty()) return;
        const std::size_t n = std::min(xs.size(), ys.size());
        cur.waypoints.clear();
        for (std::size_t i = 0; i < n; ++i)
        {
            MissionWaypoint w;
            w.pos = {xs[i], ys[i]};
            if (i < yaws.size() and std::isfinite(yaws[i])) w.yaw_rad = yaws[i];
            cur.waypoints.push_back(w);
        }
        if (cur.waypoints.size() >= 2) loaded.push_back(cur);
        cur = Mission{};
        xs.clear(); ys.clear(); yaws.clear();
    };

    std::string line;
    while (std::getline(in, line))
    {
        line = trim(line);
        if (line.empty() or line[0] == '#') continue;
        if (line.rfind("[[mission]]", 0) == 0) { flush(); continue; }
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key == "name")       cur.name = unquote(value);
        else if (key == "loops") { try { cur.loops = std::max(1, std::stoi(value)); } catch (...) {} }
        else if (key == "x")     xs = parse_array(value);
        else if (key == "y")     ys = parse_array(value);
        else if (key == "yaw")   yaws = parse_array(value);
    }
    flush();

    library_ = std::move(loaded);
    if (not library_.empty() and selected_.empty()) selected_ = library_.front().name;
    // Absolute path, because the configured one is RELATIVE and therefore depends on the launch directory —
    // "my edits vanished" and "I am reading a different file than I wrote" look identical without it.
    std::error_code ec;
    std::printf("[mission] loaded %zu mission(s) from %s\n", library_.size(),
                std::filesystem::absolute(path, ec).c_str());
    for (const auto &m : library_)
        std::printf("[mission]   '%s': %zu waypoints\n", m.name.c_str(), m.waypoints.size());
    std::fflush(stdout);
    return true;
}

bool MissionRunner::save(const std::string &path) const
{
    if (const auto parent = std::filesystem::path(path).parent_path(); not parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream out(path, std::ios::trunc);
    if (not out) return false;
    // Write dots regardless of the process locale, and keep it explicit rather than relying on the global
    // C++ locale still being the classic one. Reading is locale-independent (see parse_array), so pinning
    // the writer as well means the file cannot become machine-specific.
    out.imbue(std::locale::classic());

    out << "# controller missions — recorded by clicking in the 2D view. Room frame, metres.\n"
        << "# A mission is a REPLAYABLE stimulus: keep these coordinates stable and two runs differ only by\n"
        << "# what you changed in the controller. yaw = nan means \"arrive, do not rotate\".\n";
    out.setf(std::ios::fixed);
    out.precision(4);
    for (const auto &m : library_)
    {
        out << "\n[[mission]]\nname = \"" << m.name << "\"\nloops = " << m.loops << "\n";
        const auto emit = [&](const char *key, auto fn)
        {
            out << key << " = [";
            for (std::size_t i = 0; i < m.waypoints.size(); ++i)
            {
                if (i) out << ", ";
                fn(m.waypoints[i]);
            }
            out << "]\n";
        };
        emit("x", [&](const MissionWaypoint &w) { out << w.pos.x(); });
        emit("y", [&](const MissionWaypoint &w) { out << w.pos.y(); });
        emit("yaw", [&](const MissionWaypoint &w) { if (w.yaw_rad) out << *w.yaw_rad; else out << "nan"; });
    }
    out.flush();
    const bool ok = out.good();
    std::error_code ec;
    std::printf("[mission] saved %zu mission(s) to %s%s\n", library_.size(),
                std::filesystem::absolute(path, ec).c_str(), ok ? "" : "  ** WRITE FAILED **");
    std::fflush(stdout);
    return ok;
}

std::vector<std::string> MissionRunner::names() const
{
    std::vector<std::string> out;
    out.reserve(library_.size());
    for (const auto &m : library_) out.push_back(m.name);
    return out;
}

bool MissionRunner::select(const std::string &name)
{
    const auto it = std::ranges::find(library_, name, &Mission::name);
    if (it == library_.end()) return false;
    selected_ = name;
    if (state_ == State::Idle) { display_wps_.clear(); for (const auto &w : it->waypoints) display_wps_.push_back(w.pos); }
    return true;
}

const Mission *MissionRunner::selected_mission() const
{
    const auto it = std::ranges::find(library_, selected_, &Mission::name);
    return it == library_.end() ? nullptr : &*it;
}

Mission *MissionRunner::selected_writable()
{
    const auto it = std::ranges::find(library_, selected_, &Mission::name);
    return it == library_.end() ? nullptr : &*it;
}

bool MissionRunner::remove(const std::string &name)
{
    const auto it = std::ranges::find(library_, name, &Mission::name);
    if (it == library_.end()) return false;
    // Deleting the mission that is currently driving would leave a run whose route no longer exists; end
    // the run first so the CSV rows it already produced are closed out properly.
    if (state_ == State::Running and active_.name == name)
        stop("mission deleted", last_sample_ms_);
    library_.erase(it);
    if (selected_ == name)
        selected_ = library_.empty() ? std::string{} : library_.front().name;
    refresh_display_waypoints();
    return true;
}

void MissionRunner::set_click_target(const std::optional<Eigen::Vector2f> &pos)
{
    click_target_ = pos;
    if (state_ == State::Idle)
        refresh_display_waypoints();
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Recording
// ─────────────────────────────────────────────────────────────────────────────────────────

void MissionRunner::start_recording()
{
    recording_.clear();
    display_wps_.clear();
    state_ = State::Recording;
}

bool MissionRunner::move_waypoint(int index, const Eigen::Vector2f &pos)
{
    if (state_ == State::Running) return false;
    auto &points = state_ == State::Recording ? recording_
                                              : (selected_writable() != nullptr ? selected_writable()->waypoints
                                                                                : recording_);
    if (index < 0 or index >= static_cast<int>(points.size())) return false;
    points[index].pos = pos;
    refresh_display_waypoints();
    return true;
}

void MissionRunner::add_point(const Eigen::Vector2f &room_pos)
{
    if (state_ != State::Recording) return;
    recording_.push_back(MissionWaypoint{.pos = room_pos, .yaw_rad = std::nullopt});
    display_wps_.push_back(room_pos);
}

void MissionRunner::undo_point()
{
    if (state_ != State::Recording or recording_.empty()) return;
    recording_.pop_back();
    display_wps_.pop_back();
}

bool MissionRunner::finish_recording(const std::string &name)
{
    if (state_ != State::Recording or recording_.size() < 2 or name.empty())
    {
        cancel_recording();
        return false;
    }
    Mission m;
    m.name = name;
    m.waypoints = recording_;
    m.loops = 1;
    if (const auto it = std::ranges::find(library_, name, &Mission::name); it != library_.end())
        *it = m;
    else
        library_.push_back(m);
    selected_ = name;
    recording_.clear();
    state_ = State::Idle;
    refresh_display_waypoints();
    return true;
}

void MissionRunner::cancel_recording()
{
    recording_.clear();
    state_ = State::Idle;
    refresh_display_waypoints();
}

void MissionRunner::refresh_display_waypoints()
{
    display_wps_.clear();
    // A click supersedes the mission, so it also supersedes it on the canvas: showing the abandoned tour's
    // waypoints next to the target the robot is actually driving to is exactly the clutter that makes a
    // display untrustworthy.
    if (click_target_.has_value() and state_ != State::Recording)
    {
        display_wps_.push_back(*click_target_);
        return;
    }
    if (not uses_mission(mode_))
        return;   // no mission is in play; see set_mode()
    if (state_ == State::Recording)
    {
        for (const auto &w : recording_) display_wps_.push_back(w.pos);
        return;
    }
    if (state_ == State::Running)
    {
        for (const auto &w : active_.waypoints) display_wps_.push_back(w.pos);
        return;
    }
    if (const auto *m = selected_mission())
        for (const auto &w : m->waypoints) display_wps_.push_back(w.pos);
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Running
// ─────────────────────────────────────────────────────────────────────────────────────────

bool MissionRunner::start(int loops, std::uint64_t now_ms)
{
    if (not uses_mission(mode_))
    {
        std::printf("[mission] cannot run a mission in '%s' mode. Select 'Mission' first.\n", to_string(mode_));
        std::fflush(stdout);
        return false;
    }
    if (not mode_implemented())
    {
        // Refusing beats running as MissionOnly under a MissionWithAffordances label: the CSV would record
        // a condition that never happened, and nothing downstream could tell the difference.
        std::printf("[mission] 'mission + affordances' is NOT IMPLEMENTED — how the epistemic planner should\n"
                    "          arbitrate against an imposed route is still an open question (see the design\n"
                    "          note in controller_mission.cpp). Use 'mission only' for a clean benchmark.\n");
        std::fflush(stdout);
        return false;
    }

    const auto *m = selected_mission();
    if (m == nullptr or m->waypoints.size() < 2) return false;

    active_ = *m;
    loops_ = std::max(1, loops);
    lap_ = 0;
    wp_index_ = 0;
    run_start_ms_ = now_ms;
    stop_reason_.clear();
    legs_.clear();
    lap1_trace_.clear();
    repeat_sum_ = 0.0; repeat_n_ = 0; repeat_max_ = 0.f;
    last_pos_.reset();
    state_ = State::Running;
    completed_event_ = false;   // a fresh run must not inherit the previous run's completion
    refresh_display_waypoints();
    begin_leg(now_ms);
    std::printf("[mission] START '%s' — %zu waypoints x %d lap(s)\n",
                active_.name.c_str(), active_.waypoints.size(), loops_);
    std::fflush(stdout);
    return true;
}

void MissionRunner::stop(const std::string &reason, std::uint64_t now_ms)
{
    if (state_ != State::Running) { state_ = State::Idle; return; }
    // Close the in-flight leg as INCOMPLETE rather than dropping it. A run that was abandoned two thirds of
    // the way through a leg is a different measurement from one that never started it, and the CSV has to
    // say which — otherwise a stopped run silently looks like a shorter clean one. close_leg is a no-op when
    // advance() already closed the last leg (the normal completion path).
    close_leg(last_pos_.value_or(Eigen::Vector2f::Zero()), now_ms, false);
    stop_reason_ = reason;
    completed_event_ = reason == "completed" or reason == "exhausted";
    state_ = State::Idle;
    refresh_display_waypoints();   // a cancelled/finished tour stops being drawn
    const auto s = summary();
    std::printf("[mission] STOP '%s' (%s) — %d/%d laps, %d legs, %.1f s, %.2f m driven, detour x%.2f, "
                "min body clearance %.2f m, %d replans, %d escapes\n",
                active_.name.c_str(), reason.c_str(), s.laps_completed, loops_, s.legs_completed,
                s.total_duration_s, s.total_path_length_m, s.mean_detour_ratio, s.min_body_clearance_m,
                s.total_replans, s.total_escapes);
    if (std::isfinite(s.lap_repeat_mean_m))
        std::printf("[mission] lap repeatability vs lap 1: mean %.3f m, max %.3f m\n",
                    s.lap_repeat_mean_m, s.lap_repeat_max_m);
    std::fflush(stdout);
    if (not csv_path_.empty() and not legs_.empty())
        write_csv(csv_path_);
}

void MissionRunner::begin_leg(std::uint64_t now_ms)
{
    leg_ = MissionLegMetrics{};
    leg_.lap = lap_ + 1;
    leg_.leg = wp_index_ + 1;
    leg_.min_body_clearance_m = std::numeric_limits<float>::max();
    leg_start_ms_ = now_ms;
    last_sample_ms_ = now_ms;
    leg_origin_ = last_pos_.value_or(Eigen::Vector2f::Zero());
    leg_open_ = true;
}

void MissionRunner::close_leg(const Eigen::Vector2f &arrived_at, std::uint64_t now_ms, bool completed)
{
    if (not leg_open_) return;
    leg_open_ = false;
    if (wp_index_ >= static_cast<int>(active_.waypoints.size())) return;
    const Eigen::Vector2f goal = active_.waypoints[wp_index_].pos;
    leg_.duration_s = static_cast<float>(now_ms - leg_start_ms_) / 1000.f;
    leg_.straight_line_m = (goal - leg_origin_).norm();
    leg_.detour_ratio = leg_.straight_line_m > 1e-3f ? leg_.path_length_m / leg_.straight_line_m : kNaN;
    leg_.mean_speed_mps = leg_.duration_s > 1e-3f ? leg_.path_length_m / leg_.duration_s : 0.f;
    leg_.arrival_error_m = (arrived_at - goal).norm();
    leg_.completed = completed;
    if (leg_.min_body_clearance_m == std::numeric_limits<float>::max())
        leg_.min_body_clearance_m = kNaN;
    legs_.push_back(leg_);
}

bool MissionRunner::consume_completed()
{
    const bool e = completed_event_;
    completed_event_ = false;
    return e;
}

std::optional<ControllerTargetInfo> MissionRunner::current_target() const
{
    if (state_ != State::Running) return std::nullopt;
    if (wp_index_ < 0 or wp_index_ >= static_cast<int>(active_.waypoints.size())) return std::nullopt;

    const auto &w = active_.waypoints[wp_index_];
    ControllerTargetInfo t;
    t.node_id = 0;
    // The name carries lap and index, so consecutive waypoints are DISTINCT targets even if a tour revisits
    // the same point — the session's target-changed test compares position, and a repeated point would
    // otherwise look like "no new target" and never replan.
    t.node_name = "mission:" + active_.name + "#" + std::to_string(lap_ + 1) + "." + std::to_string(wp_index_ + 1);
    t.room_pos = w.pos;
    t.yaw_rad = w.yaw_rad.value_or(0.f);
    // from_affordance drives the arrival behaviour: true = rotate to the facing yaw before declaring the goal
    // reached. A tour waypoint only asks for a facing when one was recorded.
    t.from_affordance = w.yaw_rad.has_value();
    return t;
}

void MissionRunner::advance(const Eigen::Vector2f &arrived_at, std::uint64_t now_ms)
{
    if (state_ != State::Running) return;

    close_leg(arrived_at, now_ms, true);
    ++wp_index_;
    if (wp_index_ >= static_cast<int>(active_.waypoints.size()))
    {
        wp_index_ = 0;
        ++lap_;
        if (lap_ >= loops_)
        {
            stop("completed", now_ms);
            return;
        }
        std::printf("[mission] lap %d/%d complete\n", lap_, loops_);
        std::fflush(stdout);
    }
    last_pos_ = arrived_at;   // begin_leg takes the leg origin from here
    begin_leg(now_ms);
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Instrumentation
// ─────────────────────────────────────────────────────────────────────────────────────────

void MissionRunner::sample(const Eigen::Vector2f &pos, float rot_rps, float speed_mps,
                           float body_clearance_m, bool safety_guard, std::uint64_t now_ms)
{
    if (state_ != State::Running) { last_pos_ = pos; return; }

    const float dt = static_cast<float>(now_ms - last_sample_ms_) / 1000.f;
    last_sample_ms_ = now_ms;
    if (last_pos_.has_value())
        leg_.path_length_m += (pos - *last_pos_).norm();
    else
        leg_origin_ = pos;   // first sample of the run establishes the leg origin

    // dt can be large after a stall; integrating a stale ω over it would fabricate turning that never
    // happened. Bound it at a plausible cycle time — this is an integration guard, not a behaviour gate.
    const float dt_int = std::clamp(dt, 0.f, 0.5f);
    leg_.rot_effort += std::abs(rot_rps) * dt_int;
    leg_.rot_energy += rot_rps * rot_rps * dt_int;
    if (body_clearance_m >= 0.f)
        leg_.min_body_clearance_m = std::min(leg_.min_body_clearance_m, body_clearance_m);
    if (safety_guard) ++leg_.safety_guard_cycles;
    (void)speed_mps;   // speed is derived from path_length/duration, which is what actually happened

    // Repeatability trace. Lap 1 IS the reference, so it is recorded; later laps are scored against it.
    if (lap_ == 0)
    {
        if (lap1_trace_.empty() or (pos - lap1_trace_.back()).norm() > 0.05f)
            lap1_trace_.push_back(pos);
    }
    else if (not lap1_trace_.empty())
    {
        const float d = distance_to_lap1(pos);
        repeat_sum_ += d;
        ++repeat_n_;
        repeat_max_ = std::max(repeat_max_, d);
    }

    last_pos_ = pos;
}

float MissionRunner::distance_to_lap1(const Eigen::Vector2f &p) const
{
    // Distance to the lap-1 POLYLINE, not to its nearest sample: with a 5 cm sampling step, nearest-vertex
    // would report up to 2.5 cm of pure discretisation as if it were deviation.
    float best = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i + 1 < lap1_trace_.size(); ++i)
    {
        const Eigen::Vector2f a = lap1_trace_[i], b = lap1_trace_[i + 1];
        const Eigen::Vector2f ab = b - a;
        const float len2 = ab.squaredNorm();
        const float t = len2 > 1e-9f ? std::clamp((p - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
        best = std::min(best, (p - (a + t * ab)).norm());
    }
    return best == std::numeric_limits<float>::max() ? kNaN : best;
}

void MissionRunner::note_replan() { if (state_ == State::Running) ++leg_.replans; }
void MissionRunner::note_escape() { if (state_ == State::Running) ++leg_.escapes; }

// ─────────────────────────────────────────────────────────────────────────────────────────
// Readout
// ─────────────────────────────────────────────────────────────────────────────────────────

MissionRunSummary MissionRunner::summary() const
{
    MissionRunSummary s;
    s.mission_name = active_.name.empty() ? selected_ : active_.name;
    s.laps_completed = lap_;
    float detour_sum = 0.f;
    int   detour_n = 0;
    s.min_body_clearance_m = std::numeric_limits<float>::max();
    for (const auto &l : legs_)
    {
        if (l.completed) ++s.legs_completed;
        s.total_duration_s += l.duration_s;
        s.total_path_length_m += l.path_length_m;
        s.total_rot_effort += l.rot_effort;
        s.total_replans += l.replans;
        s.total_escapes += l.escapes;
        if (std::isfinite(l.detour_ratio)) { detour_sum += l.detour_ratio; ++detour_n; }
        if (std::isfinite(l.min_body_clearance_m))
            s.min_body_clearance_m = std::min(s.min_body_clearance_m, l.min_body_clearance_m);
    }
    s.mean_detour_ratio = detour_n ? detour_sum / static_cast<float>(detour_n) : kNaN;
    if (s.min_body_clearance_m == std::numeric_limits<float>::max()) s.min_body_clearance_m = kNaN;
    // NaN, not 0: "we never ran a second lap" and "the second lap was perfect" are opposite findings.
    s.lap_repeat_mean_m = repeat_n_ ? static_cast<float>(repeat_sum_ / static_cast<double>(repeat_n_)) : kNaN;
    s.lap_repeat_max_m = repeat_n_ ? repeat_max_ : kNaN;
    return s;
}

std::string MissionRunner::status_text() const
{
    switch (state_)
    {
        case State::Recording:
            return "recording — click to add waypoints (" + std::to_string(recording_.size()) + ")";
        case State::Running:
        {
            const int n = static_cast<int>(active_.waypoints.size());
            return active_.name + "  lap " + std::to_string(lap_ + 1) + "/" + std::to_string(loops_)
                 + "  wp " + std::to_string(wp_index_ + 1) + "/" + std::to_string(n);
        }
        case State::Idle:
        default:
            if (mode_ == DriveMode::Target)
                return click_target_.has_value()
                     ? "target — driving to the clicked point (not recorded)"
                     : "target — click a point in the view";
            if (mode_ == DriveMode::AffordancesOnly)
                return "affordances — the epistemic planner has the base";
            if (not mode_implemented())
                return "mission + affordances — NOT IMPLEMENTED (see controller_mission.cpp)";
            if (const auto *m = selected_mission(); m != nullptr)
                return m->name + " — " + std::to_string(m->waypoints.size()) + " waypoints (idle)";
            return "no mission selected";
    }
}

bool MissionRunner::write_csv(const std::string &path) const
{
    const bool exists = std::filesystem::exists(path);
    std::ofstream out(path, std::ios::app);
    if (not out) return false;
    // A comma decimal separator in a COMMA-SEPARATED file would be unreadable, so this is pinned too.
    out.imbue(std::locale::classic());
    if (not exists)
        out << "mission,mode,run_start_ms,stop_reason,lap,leg,completed,duration_s,path_length_m,straight_line_m,"
               "detour_ratio,min_body_clearance_m,mean_speed_mps,rot_effort_rad,rot_energy,arrival_error_m,"
               "replans,escapes,safety_guard_cycles,lap_repeat_mean_m,lap_repeat_max_m\n";

    out.setf(std::ios::fixed);
    out.precision(4);
    const auto s = summary();
    for (const auto &l : legs_)
        out << active_.name << ',' << to_string(mode_) << ',' << run_start_ms_ << ','
            << (stop_reason_.empty() ? "?" : stop_reason_)
            << ',' << l.lap << ',' << l.leg << ',' << (l.completed ? 1 : 0) << ','
            << l.duration_s << ',' << l.path_length_m << ',' << l.straight_line_m << ','
            << l.detour_ratio << ',' << l.min_body_clearance_m << ',' << l.mean_speed_mps << ','
            << l.rot_effort << ',' << l.rot_energy << ',' << l.arrival_error_m << ','
            << l.replans << ',' << l.escapes << ',' << l.safety_guard_cycles << ','
            << s.lap_repeat_mean_m << ',' << s.lap_repeat_max_m << '\n';
    return out.good();
}

// ─────────────────────────────────────────────────────────────────────────────────────────

bool MissionRunner::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char *m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // (1) Record → select → run supplies each waypoint in order, then finishes.
    {
        MissionRunner r;
        r.set_csv_path("");   // no file I/O in the test
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording();
        r.add_point({0.f, 0.f});
        r.add_point({1.f, 0.f});
        r.add_point({1.f, 1.f});
        check(r.finish_recording("t"), "a 3-point recording must commit");
        check(r.names().size() == 1, "the library must hold the recorded mission");

        check(r.start(1, 0), "start must accept a 2+ waypoint mission");
        std::uint64_t t = 0;
        for (int i = 0; i < 3; ++i)
        {
            const auto tgt = r.current_target();
            check(tgt.has_value(), "a running mission must supply a target");
            if (not tgt.has_value()) break;
            r.sample(tgt->room_pos, 0.f, 0.f, 0.5f, false, t += 100);
            r.advance(tgt->room_pos, t);
        }
        check(not r.running(), "the mission must end after its last waypoint of the last lap");
        // A tour that ends BY ITSELF must announce it, so the UI can drop back to "Run" instead of leaving
        // the robot armed with nothing to drive to.
        check(r.consume_completed(), "finishing all laps must raise the completed event");
        check(not r.consume_completed(), "the completed event is ONE-SHOT");
        check(r.legs().size() == 3, "one leg per waypoint");
        check(r.legs().back().completed, "legs reached in order must be marked completed");
    }

    // (2) Two laps: the second lap is scored against the first, and a mission stopped mid-leg records that
    //     leg as INCOMPLETE — the distinction the CSV depends on.
    {
        MissionRunner r;
        r.set_csv_path("");
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording();
        r.add_point({0.f, 0.f});
        r.add_point({2.f, 0.f});
        r.finish_recording("t2");
        r.start(2, 0);
        std::uint64_t t = 0;
        r.sample({0.f, 0.f}, 0.f, 0.f, 1.f, false, t += 100);
        r.advance({0.f, 0.f}, t);
        r.sample({2.f, 0.f}, 0.f, 0.f, 1.f, false, t += 100);
        r.advance({2.f, 0.f}, t);              // lap 1 done
        check(r.current_lap() == 1, "the second lap must start after the first completes");
        r.sample({0.f, 0.f}, 0.f, 0.f, 1.f, false, t += 100);
        r.stop("aborted", t);
        check(not r.running(), "stop must end the run");
        check(not r.consume_completed(),
              "a run the USER stopped must NOT look completed — that would silently halt driving they asked for");
        const auto legs = r.legs();
        check(not legs.empty() and not legs.back().completed,
              "a leg abandoned by stop() must be recorded as incomplete, not dropped");
    }

    // (3) Path length is what was DRIVEN, not the straight line — the detour ratio is the whole point.
    {
        MissionRunner r;
        r.set_csv_path("");
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording();
        r.add_point({0.f, 0.f});
        r.add_point({2.f, 0.f});
        r.finish_recording("t3");
        r.start(1, 0);
        std::uint64_t t = 0;
        r.sample({0.f, 0.f}, 0.f, 0.f, 1.f, false, t += 100);
        r.advance({0.f, 0.f}, t);
        // Detour: out to (1,1) and back down to (2,0) — 2*sqrt(2) driven for 2.0 straight.
        r.sample({1.f, 1.f}, 0.f, 0.f, 1.f, false, t += 100);
        r.sample({2.f, 0.f}, 0.f, 0.f, 1.f, false, t += 100);
        r.advance({2.f, 0.f}, t);
        const auto &l = r.legs().back();
        std::printf("  detour leg: drove %.3f m for a %.3f m straight line → ratio %.3f\n",
                    l.path_length_m, l.straight_line_m, l.detour_ratio);
        check(std::abs(l.path_length_m - 2.f * std::sqrt(2.f)) < 1e-3f, "path length must integrate the trace");
        check(l.detour_ratio > 1.41f and l.detour_ratio < 1.42f, "detour ratio must be driven/straight");
    }

    // (4) TOML round-trip, including the absent-yaw case — a yaw of 0 and "no yaw" must not collapse.
    {
        MissionRunner a;
        a.start_recording();
        a.add_point({1.25f, -2.5f});
        a.add_point({3.f, 4.f});
        a.finish_recording("rt");
        const std::string path = std::filesystem::temp_directory_path() / "rc_mission_selftest.toml";
        check(a.save(path), "save must succeed");
        MissionRunner b;
        check(b.load(path), "load must succeed");
        const auto *m = b.selected_mission();
        check(m != nullptr and m->waypoints.size() == 2, "the round trip must preserve the waypoints");
        if (m != nullptr and m->waypoints.size() == 2)
        {
            check(std::abs(m->waypoints[0].pos.x() - 1.25f) < 1e-3f, "x must survive the round trip");
            check(std::abs(m->waypoints[1].pos.y() - 4.f) < 1e-3f, "y must survive the round trip");
            check(not m->waypoints[0].yaw_rad.has_value(), "an absent yaw must stay absent, not become 0");
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    // (4b) LOCALE INDEPENDENCE. This is the regression test for the bug that silently destroyed tours: the
    //      file is written with '.' but was read with stof(), which honours LC_NUMERIC. Under a
    //      comma-decimal locale every coordinate was truncated toward zero on load and the truncated value
    //      was written back on the next save. Nothing in the UI could show this — the numbers looked
    //      plausible, just wrong — so it is asserted here, under an actual comma locale.
    {
        const char *saved = std::setlocale(LC_NUMERIC, nullptr);
        const char *applied = nullptr;
        for (const char *cand : {"es_ES.UTF-8", "de_DE.UTF-8", "fr_FR.UTF-8", "es_ES", "de_DE"})
            if ((applied = std::setlocale(LC_NUMERIC, cand)) != nullptr) break;
        std::printf("  locale round trip under LC_NUMERIC=%s\n", applied ? applied : "(none available)");

        MissionRunner a;
        a.set_mode(DriveMode::MissionOnly);
        a.start_recording();
        a.add_point({-1.9114f, 3.6829f});
        a.add_point({0.5361f, -2.7738f});
        a.finish_recording("loc");
        const std::string path = std::filesystem::temp_directory_path() / "rc_mission_locale.toml";
        a.save(path);

        MissionRunner b;
        b.load(path);
        const auto *m = b.selected_mission();
        check(m != nullptr and m->waypoints.size() == 2, "the locale round trip must preserve the waypoints");
        if (m != nullptr and m->waypoints.size() == 2)
        {
            std::printf("  read back (%.4f, %.4f) and (%.4f, %.4f)\n",
                        m->waypoints[0].pos.x(), m->waypoints[0].pos.y(),
                        m->waypoints[1].pos.x(), m->waypoints[1].pos.y());
            check(std::abs(m->waypoints[0].pos.x() + 1.9114f) < 1e-4f,
                  "a negative fractional coordinate must survive a comma-decimal locale (was truncated to -1)");
            check(std::abs(m->waypoints[0].pos.y() - 3.6829f) < 1e-4f, "fraction must survive on y too");
            check(std::abs(m->waypoints[1].pos.x() - 0.5361f) < 1e-4f, "a sub-1 coordinate must not become 0");
            check(std::abs(m->waypoints[1].pos.y() + 2.7738f) < 1e-4f, "and must not round instead of parse");
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (saved != nullptr) std::setlocale(LC_NUMERIC, saved);
    }

    // (5) Degenerate inputs must be refused rather than half-accepted.
    {
        MissionRunner r;
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording();
        r.add_point({0.f, 0.f});
        check(not r.finish_recording("one"), "a single-point mission must be rejected");
        check(not r.start(1, 0), "starting with no selected mission must fail, not crash");
    }

    // (6) DRIVE MODE GATING. The failure this guards against is silent: an unimplemented mode that ran as
    //     MissionOnly would produce CSV rows labelled with a condition that never existed, and nothing
    //     downstream could detect it. So the refusal is asserted, not assumed.
    {
        MissionRunner r;
        r.set_csv_path("");
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording();
        r.add_point({0.f, 0.f});
        r.add_point({1.f, 0.f});
        r.finish_recording("m");

        r.set_mode(DriveMode::AffordancesOnly);
        check(not r.start(1, 0), "'affordances' must refuse to run a mission");
        r.set_mode(DriveMode::Target);
        check(not r.start(1, 0), "'target' must refuse to run a mission");
        r.set_mode(DriveMode::MissionWithAffordances);
        check(not r.mode_implemented(), "'mission + affordances' must report itself unimplemented");
        check(not r.start(1, 0), "an UNIMPLEMENTED mode must refuse to run, never fall back to another");
        r.set_mode(DriveMode::MissionOnly);
        check(r.start(1, 0), "'mission only' must run");

        // Switching condition mid-run must END the run, not silently relabel it.
        r.set_mode(DriveMode::AffordancesOnly);
        check(not r.running(), "changing the drive mode mid-run must stop the run");
    }

    // (7) EDIT / DELETE / CLICK. All three change what the canvas shows, and a stale overlay is the whole
    //     complaint these features exist to fix — so assert the overlay, not just the library.
    {
        MissionRunner r;
        r.set_csv_path("");
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording();
        r.add_point({0.f, 0.f});
        r.add_point({1.f, 0.f});
        r.finish_recording("tour");
        check(r.display_waypoints().size() == 2, "an idle selected mission must be drawn");

        // Editing IS dragging: move_waypoint edits the SELECTED mission in place. The failure mode guarded
        // here is an "edit" that writes somewhere other than the mission you were looking at.
        check(r.move_waypoint(1, {4.f, 4.f}), "dragging a waypoint of the selected mission must succeed");
        const auto *m = r.selected_mission();
        check(m != nullptr and m->waypoints.size() == 2, "a drag must MOVE a point, never add one");
        check(m != nullptr and std::abs(m->waypoints[1].pos.x() - 4.f) < 1e-4f,
              "the drag must land on the waypoint that was dragged");
        check(r.display_waypoints().size() == 2 and std::abs(r.display_waypoints()[1].x() - 4.f) < 1e-4f,
              "the overlay must follow the drag");
        check(not r.move_waypoint(7, {0.f, 0.f}), "dragging a non-existent index must fail, not crash");

        // A click supersedes the tour on the canvas too.
        r.set_click_target(Eigen::Vector2f{5.f, 5.f});
        check(r.display_waypoints().size() == 1, "a click target must REPLACE the tour overlay, not add to it");
        r.set_click_target(std::nullopt);
        check(r.display_waypoints().size() == 2, "clearing the click must restore the tour overlay");

        // A finished/cancelled run must stop being drawn.
        r.start(1, 0);
        r.sample({0.f, 0.f}, 0.f, 0.f, 1.f, false, 100);
        check(not r.display_waypoints().empty(), "a running tour must be drawn");
        r.stop("user", 200);
        check(r.display_waypoints().size() == 2, "after stopping, the overlay falls back to the selection");
        // A run is a measurement: editing its route mid-flight would invalidate it silently.
        r.start(1, 0);
        check(not r.move_waypoint(0, {9.f, 9.f}), "a waypoint must NOT be draggable while a run is measuring");
        r.stop("user", 300);

        // Switching to a non-mission mode must clear the tour from the canvas, not just disable the buttons.
        r.set_mode(DriveMode::AffordancesOnly);
        check(r.display_waypoints().empty(), "'affordances' must leave NO mission on the canvas");
        r.set_mode(DriveMode::MissionOnly);
        check(r.display_waypoints().size() == 2, "returning to a mission mode must restore the overlay");

        // TARGET mode: a click owns the canvas, and leaving Target must not leave the click marker behind
        // while the selector claims something else is driving.
        r.set_mode(DriveMode::Target);
        r.set_click_target(Eigen::Vector2f{7.f, 7.f});
        check(r.display_waypoints().size() == 1, "in Target mode only the clicked point is drawn");
        r.set_mode(DriveMode::MissionOnly);
        check(not r.has_click_target(), "leaving Target must drop the click target");
        check(r.display_waypoints().size() == 2, "leaving Target must restore the mission overlay");

        // Index mapping is shared by the widget and the worker; a mismatch would silently pick a mode the
        // user did not choose.
        for (const auto m : {DriveMode::AffordancesOnly, DriveMode::MissionOnly,
                             DriveMode::MissionWithAffordances, DriveMode::Target})
            check(from_index(to_index(m)) == m, "drive-mode index mapping must round-trip");

        check(r.remove("tour"), "delete must remove the mission");
        check(r.names().empty(), "the library must be empty after deleting its only mission");
        check(r.display_waypoints().empty(), "a deleted mission must leave NOTHING on the canvas");
        check(not r.remove("tour"), "deleting a mission twice must fail, not crash");
    }

    std::printf("MissionRunner::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
