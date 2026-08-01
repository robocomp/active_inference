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
#include <format>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <chrono>
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
// JSON has NO representation for NaN or infinity — emitting a bare `nan` produces a file that a strict
// parser rejects outright. And every metric here can legitimately be undefined: detour_ratio when the
// straight line is zero, min_body_clearance when the leg never saw the ESDF. `null` says "not
// measured", which is exactly the distinction the rest of this file works to preserve.
std::string jnum(float v)
{
    if (not std::isfinite(v)) return "null";
    // ★ std::to_chars, NOT snprintf. snprintf honours the C locale's LC_NUMERIC, which QApplication sets
    // from the environment — so on a comma-decimal machine it writes 0,7000 and the JSON is invalid.
    // Imbuing the STREAM with the classic locale does not help: snprintf never touches the stream. This
    // is the same fault as the mission-file truncation, entering by the other door, and it is why the
    // rule is "no locale-sensitive C formatting on file data" rather than "remember to imbue".
    char buf[48];
    const auto [p, ec] = std::to_chars(buf, buf + sizeof buf, static_cast<double>(v),
                                       std::chars_format::fixed, 4);
    return ec == std::errc{} ? std::string(buf, p) : std::string("null");
}

// Mission names come from a free-text dialog, so they can contain anything.
std::string jstr(const std::string &in)
{
    std::string out = "\"";
    for (const char c : in)
        switch (c)
        {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                {
                    char esc[8];
                    std::snprintf(esc, sizeof esc, "\\u%04x", c);
                    out += esc;
                }
                else out += c;
        }
    return out + "\"";
}

std::string iso_now(std::string *stamp_out)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    ::localtime_r(&t, &tm);
    char iso[40], stamp[24];
    std::strftime(iso, sizeof iso, "%Y-%m-%dT%H:%M:%S%z", &tm);
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
    if (stamp_out) *stamp_out = stamp;
    return iso;
}

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
    if (it == library_.end())
    {
        // A selection that names nothing is not a no-op to be swallowed: the combo and the library
        // have diverged, and every later "why is it still the old route" follows from it.
        std::printf("[mission] SELECT FAILED: '%s' is not in the library (%zu loaded).\n",
                    name.c_str(), library_.size());
        std::fflush(stdout);
        return false;
    }
    if (selected_ == name) return true;
    selected_ = name;
    // Use the ONE canonical refresh. This used to rebuild display_wps_ inline, which bypassed the
    // drive-mode gate and the click-target precedence that refresh_display_waypoints() applies — so
    // selecting a mission could leave the canvas showing a route the mode says is not driving, or a
    // stale click marker sitting on top of it. Two code paths for "what is drawn" is one too many.
    refresh_display_waypoints();
    std::printf("[mission] selected '%s' (%zu waypoints)\n", name.c_str(), it->waypoints.size());
    std::fflush(stdout);
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

void MissionRunner::add_profile_sample(std::uint64_t t_ms, float adv, float side, float rot, float freshness)
{
    if (state_ != State::Running) return;   // only a measured run produces a profile
    const std::lock_guard<std::mutex> lock(profile_mutex_);
    if (profile_.size() >= kMaxProfileRows) { profile_truncated_ = true; return; }
    profile_.push_back(MissionProfileSample{.t_ms = t_ms, .adv_mps = adv, .side_mps = side, .rot_rps = rot,
                                            .freshness = freshness, .v_meas_mps = pending_v_meas_,
                                            .v_meas_fresh = pending_v_meas_fresh_,
                                            .lap = lap_ + 1});
    pending_v_meas_fresh_ = false;   // consumed: the NEXT row must not claim this reading again
}

void MissionRunner::note_measured_speed(float mps)
{
    const std::lock_guard<std::mutex> lock(profile_mutex_);
    pending_v_meas_ = mps;
    pending_v_meas_fresh_ = true;
}

bool MissionRunner::write_profile_csv(const std::string &dir, const std::string &stamp) const
{
    std::vector<MissionProfileSample> rows;
    bool truncated = false;
    {
        const std::lock_guard<std::mutex> lock(profile_mutex_);
        rows = profile_;
        truncated = profile_truncated_;
    }
    if (rows.empty()) return false;

    std::error_code ec;
    const std::filesystem::path folder = std::filesystem::path(dir) / active_.name;
    std::filesystem::create_directories(folder, ec);
    const std::filesystem::path path = folder / (stamp + "_profile.csv");
    std::ofstream o(path, std::ios::trunc);
    if (not o) return false;
    o.imbue(std::locale::classic());   // a comma decimal separator in a CSV would be unreadable
    o.setf(std::ios::fixed);
    o.precision(5);
    o << "# actuation stream, sampled in the velocity-output thread (fixed rate, see MissionProfileSample)\n"
      << "# v_meas_fresh=0 means v_meas_mps is HELD from an earlier row — do not treat it as signal\n";
    if (truncated)
        o << "# ** TRUNCATED at " << kMaxProfileRows << " rows — the run outlasted the buffer **\n";
    o << "t_ms,lap,adv_mps,side_mps,rot_rps,freshness,v_meas_mps,v_meas_fresh\n";
    for (const auto &r : rows)
        o << r.t_ms << ',' << r.lap << ','
          << r.adv_mps << ',' << r.side_mps << ',' << r.rot_rps << ','
          << r.freshness << ',' << r.v_meas_mps << ',' << (r.v_meas_fresh ? 1 : 0) << '\n';
    const bool ok = o.good();
    std::printf("[mission] profile (%zu rows @ output rate) -> %s%s\n", rows.size(), path.c_str(),
                ok ? "" : "  ** WRITE FAILED **");
    std::fflush(stdout);
    return ok;
}

int MissionRunner::smooth_selected(const std::function<bool(const Eigen::Vector2f &, float)> &is_free,
                                   int iterations, float alpha, float max_shift_m)
{
    if (state_ == State::Running) return 0;   // never rewrite the route under a measurement
    Mission *m = selected_writable();
    if (m == nullptr or m->waypoints.size() < 3 or not is_free) return 0;

    auto &wp = m->waypoints;
    const std::vector<MissionWaypoint> original = wp;
    const int n = static_cast<int>(wp.size());

    for (int it = 0; it < std::max(1, iterations); ++it)
        for (int i = 1; i < n - 1; ++i)
        {
            // Pull toward the midpoint of the neighbours — the standard elastic-band relaxation, the
            // same shape as TrajectoryController::relax_path, but gated on the EXACT footprint test
            // instead of a clearance radius.
            const Eigen::Vector2f mid = 0.5f * (wp[i - 1].pos + wp[i + 1].pos);
            Eigen::Vector2f cand = wp[i].pos + alpha * (mid - wp[i].pos);

            if ((cand - original[i].pos).norm() > max_shift_m) continue;   // keep the route recognisable

            // The heading the robot will actually present here is its direction of travel — the same
            // model the grid planner searches under. Testing at an arbitrary heading would accept
            // poses the planner then rejects.
            const Eigen::Vector2f tangent = wp[i + 1].pos - wp[i - 1].pos;
            const float heading = tangent.squaredNorm() > 1e-12f
                                ? std::atan2(tangent.y(), tangent.x()) : 0.f;
            if (is_free(cand, heading))
                wp[i].pos = cand;
        }

    int moved = 0;
    for (int i = 0; i < n; ++i)
        if ((wp[i].pos - original[i].pos).norm() > 1e-4f) ++moved;
    refresh_display_waypoints();
    std::printf("[mission] smoothed '%s': %d of %d waypoints moved\n", m->name.c_str(), moved, n);
    std::fflush(stdout);
    return moved;
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

    // The ONLY failure path here that used to be silent — and silence is the worst possible answer,
    // because on_run only enables driving when start() succeeds. A mission that cannot start therefore
    // left the robot halted with nothing printed, which is indistinguishable from "the button did not
    // work". Every refusal says why.
    const auto *m = selected_mission();
    if (m == nullptr)
    {
        std::printf("[mission] cannot run: no mission named '%s' in the library (%zu loaded). "
                    "Pick one in the Mission box.\n", selected_.c_str(), library_.size());
        std::fflush(stdout);
        return false;
    }
    if (m->waypoints.size() < 2)
    {
        std::printf("[mission] cannot run '%s': it has %zu waypoint(s), at least 2 are needed.\n",
                    m->name.c_str(), m->waypoints.size());
        std::fflush(stdout);
        return false;
    }

    active_ = *m;
    loops_ = std::max(1, loops);
    lap_ = 0;
    run_start_ms_ = now_ms;
    stop_reason_.clear();
    stats_ = TrajectoryStats{};
    run_first_sample_ms_ = 0;
    lap_traces_.clear();
    ct_sq_sum_ = hd_sq_sum_ = lat_sq_sum_ = 0.0;
    ct_n_ = 0;
    clearances_.clear();
    prev_cmd_speed_.reset();
    prev_cmd_dv_.reset();
    prev_rot_sign_ = 0;
    last_pos_.reset();
    state_ = State::Running;
    completed_event_ = false;   // a fresh run must not inherit the previous run's completion
    refresh_display_waypoints();
    last_sample_ms_ = now_ms;
    std::printf("[mission] START '%s' — %zu waypoints x %d lap(s)\n",
                active_.name.c_str(), active_.waypoints.size(), loops_);
    std::fflush(stdout);
    return true;
}

void MissionRunner::compute_lap_repeat()
{
    stats_.lap_repeat_mean_m = std::numeric_limits<float>::quiet_NaN();
    stats_.lap_repeat_max_m = std::numeric_limits<float>::quiet_NaN();
    if (lap_traces_.size() < 2 or lap_traces_[0].size() < 2) return;   // one lap has nothing to repeat

    const auto &ref = lap_traces_[0];
    double sum = 0.0;
    std::size_t n = 0;
    float worst = 0.f;
    for (std::size_t L = 1; L < lap_traces_.size(); ++L)
        for (const auto &p : lap_traces_[L])
        {
            float best = std::numeric_limits<float>::max();
            for (const auto &q : ref) best = std::min(best, (p - q).norm());
            sum += best;
            worst = std::max(worst, best);
            ++n;
        }
    if (n == 0) return;
    stats_.lap_repeat_mean_m = static_cast<float>(sum / static_cast<double>(n));
    stats_.lap_repeat_max_m = worst;
}

void MissionRunner::stop(const std::string &reason, std::uint64_t now_ms)
{
    compute_lap_repeat();
    if (state_ != State::Running) { state_ = State::Idle; return; }
    stop_reason_ = reason;
    completed_event_ = reason == "completed" or reason == "exhausted";
    state_ = State::Idle;
    refresh_display_waypoints();   // a cancelled/finished tour stops being drawn
    const auto s2 = summary();
    std::printf("[mission] STOP '%s' (%s) - %d/%d laps, %.1f m of %.1f m route in %.1f s\n"
                "[mission]   track: cross %.3f m rms / %.3f m max | heading %.3f rad rms\n"
                "[mission]   smooth: yaw %.1f rad in %d reversals | dv %.1f (peak %.2f m/s2) | "
                "d2v %.1f (peak %.1f m/s3) | a_lat %.2f rms %.2f max\n"
                "[mission]   safety: clearance %.3f min / %.3f p05 | %d guard, %d escapes, %d replans\n"
                "[mission]   repeat: lap-to-lap %.3f m mean / %.3f m max (NaN = single lap)\n",
                active_.name.c_str(), reason.c_str(), s2.laps_completed, loops_,
                s2.progress_m, s2.route_length_m, s2.duration_s,
                s2.cross_track_rms_m, s2.cross_track_max_m, s2.heading_err_rms_rad,
                s2.rot_effort_rad, s2.rot_reversals, s2.lin_accel_effort, s2.lin_accel_max,
                s2.lin_jerk_effort, s2.lin_jerk_max, s2.lat_accel_rms, s2.lat_accel_max,
                s2.min_clearance_m, s2.p05_clearance_m,
                s2.safety_guard_cycles, s2.escapes, s2.replans,
                s2.lap_repeat_mean_m, s2.lap_repeat_max_m);
    std::fflush(stdout);
    if (not csv_path_.empty() and stats_.duration_s > 0.f)
        write_csv(csv_path_);
    if (not run_dir_.empty())
    {
        // ONE timestamp for both artefacts, so the JSON and its profile share a name and can never be
        // paired up wrongly by a millisecond landing either side of a second boundary.
        std::string stamp;
        const std::string iso = iso_now(&stamp);
        write_run_json(run_dir_, stamp, iso);
        write_profile_csv(run_dir_, stamp);
    }
}

bool MissionRunner::consume_completed()
{
    const bool e = completed_event_;
    completed_event_ = false;
    return e;
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// Instrumentation
// ─────────────────────────────────────────────────────────────────────────────────────────

void MissionRunner::note_progress(float progress_m, float route_length_m, int laps_done)
{
    if (state_ != State::Running) return;
    stats_.progress_m = progress_m;
    stats_.route_length_m = route_length_m;
    stats_.laps_completed = std::max(0, laps_done);
    lap_ = std::max(0, laps_done);
}

void MissionRunner::sample(const Eigen::Vector2f &pos, float rot_rps, float speed_mps,
                           float body_clearance_m, bool safety_guard,
                           float cross_track_m, float heading_err_rad, float ref_curvature,
                           std::uint64_t now_ms)
{
    if (state_ != State::Running) { last_pos_ = pos; return; }
    if (run_first_sample_ms_ == 0) run_first_sample_ms_ = now_ms;

    const float dt = static_cast<float>(now_ms - last_sample_ms_) / 1000.f;
    last_sample_ms_ = now_ms;
    stats_.duration_s = static_cast<float>(now_ms - run_first_sample_ms_) / 1000.f;

    if (last_pos_.has_value())
    {
        const float step = (pos - *last_pos_).norm();
        stats_.distance_m += step;
        // Guarded on dt: a short or zero dt turns pose jitter into an enormous fictitious speed, and one
        // such sample would dominate the maximum for the whole run.
        if (dt > 0.02f) stats_.max_speed_mps = std::max(stats_.max_speed_mps, step / dt);
    }
    if (stats_.duration_s > 1e-3f) stats_.mean_speed_mps = stats_.distance_m / stats_.duration_s;

    // Per-lap trace for geometric repeatability. lap_ is laps COMPLETED, so it indexes the lap in
    // progress; note_progress runs before sample() each cycle, so it is current.
    {
        const std::size_t idx = static_cast<std::size_t>(std::max(0, lap_));
        if (lap_traces_.size() <= idx) lap_traces_.resize(idx + 1);
        auto &tr = lap_traces_[idx];
        if (tr.empty() or (pos - tr.back()).norm() >= kLapTraceStepM) tr.push_back(pos);
    }

    // dt can be large after a stall; integrating a stale omega over it would fabricate turning that never
    // happened. Bound it at a plausible cycle time — an integration guard, not a behaviour gate.
    const float dt_int = std::clamp(dt, 0.f, 0.5f);
    stats_.rot_effort_rad += std::abs(rot_rps) * dt_int;
    stats_.rot_energy += rot_rps * rot_rps * dt_int;

    // Speed variation and its own variation. Sum|dv| is sampling-rate independent, so runs at different
    // cycle times stay comparable; sum|d2v| is jerk, which sum|dv| is blind to (a smooth ramp and a ramp
    // made of steps reach the same speed with the same total variation).
    if (prev_cmd_speed_.has_value())
    {
        const float dv = speed_mps - *prev_cmd_speed_;
        stats_.lin_accel_effort += std::abs(dv);
        if (dt > 0.02f) stats_.lin_accel_max = std::max(stats_.lin_accel_max, std::abs(dv) / dt);
        if (prev_cmd_dv_.has_value())
        {
            const float d2v = std::abs(dv - *prev_cmd_dv_);
            stats_.lin_jerk_effort += d2v;
            if (dt > 0.02f) stats_.lin_jerk_max = std::max(stats_.lin_jerk_max, d2v / (dt * dt));
        }
        prev_cmd_dv_ = dv;
    }
    prev_cmd_speed_ = speed_mps;

    // Yaw reversals. Integral omega^2 cannot distinguish a steady turn from an alternating one at the
    // cycle rate, and only the second is the stutter this controller has been chased over. Deadbanded so
    // a command dithering about zero is not read as a sequence of reversals.
    constexpr float kRotDeadband = 0.05f;
    const int rot_sign = rot_rps > kRotDeadband ? 1 : (rot_rps < -kRotDeadband ? -1 : 0);
    if (rot_sign != 0)
    {
        if (prev_rot_sign_ != 0 and rot_sign != prev_rot_sign_) ++stats_.rot_reversals;
        prev_rot_sign_ = rot_sign;
    }

    // TRACKING and LATERAL ACCELERATION — the continuous quantities legs could not express.
    if (std::isfinite(cross_track_m))
    {
        const float e = std::abs(cross_track_m);
        ct_sq_sum_ += static_cast<double>(e) * e;
        hd_sq_sum_ += static_cast<double>(heading_err_rad) * heading_err_rad;
        const float a_lat = speed_mps * speed_mps * ref_curvature;
        lat_sq_sum_ += static_cast<double>(a_lat) * a_lat;
        ++ct_n_;
        stats_.cross_track_max_m = std::max(stats_.cross_track_max_m, e);
        stats_.lat_accel_max = std::max(stats_.lat_accel_max, std::abs(a_lat));
    }

    if (body_clearance_m >= 0.f) clearances_.push_back(body_clearance_m);
    if (safety_guard) ++stats_.safety_guard_cycles;

    note_measured_speed(dt > 0.02f and last_pos_.has_value() ? (pos - *last_pos_).norm() / dt : 0.f);
    last_pos_ = pos;
}

TrajectoryStats MissionRunner::summary() const
{
    TrajectoryStats t = stats_;
    if (ct_n_ > 0)
    {
        t.cross_track_rms_m = static_cast<float>(std::sqrt(ct_sq_sum_ / static_cast<double>(ct_n_)));
        t.heading_err_rms_rad = static_cast<float>(std::sqrt(hd_sq_sum_ / static_cast<double>(ct_n_)));
        t.lat_accel_rms = static_cast<float>(std::sqrt(lat_sq_sum_ / static_cast<double>(ct_n_)));
    }
    if (not clearances_.empty())
    {
        std::vector<float> c = clearances_;
        std::sort(c.begin(), c.end());
        t.min_clearance_m = c.front();
        t.p05_clearance_m = c[static_cast<std::size_t>(0.05 * (c.size() - 1))];
    }
    else { t.min_clearance_m = kNaN; t.p05_clearance_m = kNaN; }
    return t;
}

void MissionRunner::note_replan() { if (state_ == State::Running) ++stats_.replans; }
void MissionRunner::note_escape() { if (state_ == State::Running) ++stats_.escapes; }

// ─────────────────────────────────────────────────────────────────────────────────────────
// Readout
// ─────────────────────────────────────────────────────────────────────────────────────────

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
                 + "  " + std::to_string(static_cast<int>(stats_.progress_m)) + "/"
                 + std::to_string(static_cast<int>(stats_.route_length_m)) + " m";
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
    // ONE ROW PER RUN. There is no segmentation left to make rows out of, and inventing one would put
    // back the artefact this change removed. The per-instant detail lives in the profile CSV.
    static constexpr const char *kHeader =
        "mission,mode,run_start_ms,stop_reason,laps,duration_s,distance_m,route_length_m,progress_m,"
        "mean_speed_mps,max_speed_mps,cross_track_rms_m,cross_track_max_m,heading_err_rms_rad,"
        "rot_effort_rad,rot_energy,rot_reversals,lin_accel_effort,lin_accel_max,lin_jerk_effort,"
        "lin_jerk_max,lat_accel_rms,lat_accel_max,min_clearance_m,p05_clearance_m,"
        "safety_guard_cycles,escapes,replans,lap_repeat_mean_m,lap_repeat_max_m";

    // ★ THE HEADER MUST MATCH THE FILE, OR THE FILE IS NOT DATA.
    // The header used to be written only when the file was ABSENT, so adding a column to an existing
    // log appended wider rows under a narrower header. That is exactly what happened when `mode` was
    // added: mission_metrics.csv ended up with a 20-field header and 21-field rows, silently shifting
    // every column index for the newer half of the file. Nothing complained, and the file cannot be
    // analysed at all — two attempts to read the same column from it disagreed, and the disagreement
    // WAS the corruption. A metrics file that is quietly wrong is worse than no metrics file, so an
    // append into a mismatched header now REFUSES and says what to do about it.
    const bool exists = std::filesystem::exists(path);
    if (exists)
    {
        std::ifstream probe(path);
        std::string first;
        std::getline(probe, first);
        if (not first.empty() and first.back() == '\r') first.pop_back();
        if (first != kHeader)
        {
            std::printf("[mission] REFUSING to append to '%s': its header does not match this build.\n"
                        "          on disk: %s\n"
                        "          expected: %s\n"
                        "          Rename or delete the old file — appending would misalign every column.\n",
                        path.c_str(), first.c_str(), kHeader);
            std::fflush(stdout);
            return false;
        }
    }

    std::ofstream out(path, std::ios::app);
    if (not out) return false;
    // A comma decimal separator in a COMMA-SEPARATED file would be unreadable, so this is pinned too.
    out.imbue(std::locale::classic());
    if (not exists)
        out << kHeader << '\n';

    out.setf(std::ios::fixed);
    out.precision(4);
    const auto t = summary();
    out << active_.name << ',' << to_string(mode_) << ',' << run_start_ms_ << ','
        << (stop_reason_.empty() ? "?" : stop_reason_) << ',' << t.laps_completed << ','
        << t.duration_s << ',' << t.distance_m << ',' << t.route_length_m << ',' << t.progress_m << ','
        << t.mean_speed_mps << ',' << t.max_speed_mps << ','
        << t.cross_track_rms_m << ',' << t.cross_track_max_m << ',' << t.heading_err_rms_rad << ','
        << t.rot_effort_rad << ',' << t.rot_energy << ',' << t.rot_reversals << ','
        << t.lin_accel_effort << ',' << t.lin_accel_max << ',' << t.lin_jerk_effort << ','
        << t.lin_jerk_max << ',' << t.lat_accel_rms << ',' << t.lat_accel_max << ','
        << t.min_clearance_m << ',' << t.p05_clearance_m << ','
        << t.safety_guard_cycles << ',' << t.escapes << ',' << t.replans << ','
        << t.lap_repeat_mean_m << ',' << t.lap_repeat_max_m << '\n';
    return out.good();
}

bool MissionRunner::write_run_json(const std::string &dir, const std::string &stamp,
                                   const std::string &iso) const
{
    if (dir.empty() or stats_.duration_s <= 0.f) return false;

    std::error_code ec;
    const std::filesystem::path folder = std::filesystem::path(dir) / active_.name;
    std::filesystem::create_directories(folder, ec);
    const std::filesystem::path path = folder / (stamp + ".json");

    std::ofstream o(path, std::ios::trunc);
    if (not o) return false;
    // Same reason as everywhere else in this file: a comma decimal separator would produce invalid
    // JSON, silently, on any machine with a non-English locale.
    o.imbue(std::locale::classic());
    o.setf(std::ios::fixed);
    o.precision(4);

    const auto t = summary();
    o << "{\n  \"schema\": 2,\n";
    o << "  \"mission\": " << jstr(active_.name) << ",\n";
    o << "  \"date\": " << jstr(iso) << ",\n";
    o << "  \"run_start_ms\": " << run_start_ms_ << ",\n";
    o << "  \"laps_requested\": " << loops_ << ",\n";
    o << "  \"laps_completed\": " << t.laps_completed << ",\n";
    o << "  \"stop_reason\": " << jstr(stop_reason_.empty() ? "?" : stop_reason_) << ",\n";
    o << "  \"drive_mode\": " << jstr(to_string(mode_)) << ",\n";
    o << "  \"waypoints\": " << active_.waypoints.size() << ",\n";

    o << "  \"params\": {\n"
      << "    \"build\": " << jstr(run_ctx_.build) << ",\n"
      << "    \"max_adv_mps\": " << jnum(run_ctx_.max_adv_mps) << ",\n"
      << "    \"max_rot_rps\": " << jnum(run_ctx_.max_rot_rps) << ",\n"
      << "    \"comfort_standoff_m\": " << jnum(run_ctx_.comfort_standoff_m) << ",\n"
      << "    \"footprint_safety_margin_m\": " << jnum(run_ctx_.footprint_safety_margin_m) << ",\n"
      << "    \"planner_cell_size_m\": " << jnum(run_ctx_.planner_cell_size_m) << ",\n"
      << "    \"body_inscribed_m\": " << jnum(run_ctx_.body_inscribed_m) << ",\n"
      << "    \"body_circumscribed_m\": " << jnum(run_ctx_.body_circumscribed_m) << "\n  },\n";


    o << "  \"trajectory\": {\n"
      << "    \"duration_s\": "          << jnum(t.duration_s)          << ",\n"
      << "    \"distance_m\": "          << jnum(t.distance_m)          << ",\n"
      << "    \"route_length_m\": "      << jnum(t.route_length_m)      << ",\n"
      << "    \"progress_m\": "          << jnum(t.progress_m)          << ",\n"
      << "    \"mean_speed_mps\": "      << jnum(t.mean_speed_mps)      << ",\n"
      << "    \"max_speed_mps\": "       << jnum(t.max_speed_mps)       << ",\n"
      << "    \"cross_track_rms_m\": "   << jnum(t.cross_track_rms_m)   << ",\n"
      << "    \"cross_track_max_m\": "   << jnum(t.cross_track_max_m)   << ",\n"
      << "    \"heading_err_rms_rad\": " << jnum(t.heading_err_rms_rad) << ",\n"
      << "    \"rot_effort_rad\": "      << jnum(t.rot_effort_rad)      << ",\n"
      << "    \"rot_energy\": "          << jnum(t.rot_energy)          << ",\n"
      << "    \"rot_reversals\": "       << t.rot_reversals             << ",\n"
      << "    \"lin_accel_effort\": "    << jnum(t.lin_accel_effort)    << ",\n"
      << "    \"lin_accel_max\": "       << jnum(t.lin_accel_max)       << ",\n"
      << "    \"lin_jerk_effort\": "     << jnum(t.lin_jerk_effort)     << ",\n"
      << "    \"lin_jerk_max\": "        << jnum(t.lin_jerk_max)        << ",\n"
      << "    \"lat_accel_rms\": "       << jnum(t.lat_accel_rms)       << ",\n"
      << "    \"lat_accel_max\": "       << jnum(t.lat_accel_max)       << ",\n"
      << "    \"min_clearance_m\": "     << jnum(t.min_clearance_m)     << ",\n"
      << "    \"p05_clearance_m\": "     << jnum(t.p05_clearance_m)     << ",\n"
      << "    \"safety_guard_cycles\": " << t.safety_guard_cycles       << ",\n"
      << "    \"escapes\": "             << t.escapes                   << ",\n"
      << "    \"replans\": "             << t.replans                   << ",\n";
    // NaN is not valid JSON. A single-lap run has no repeatability to report, and null says that
    // honestly where 0.0 would read as "perfectly repeatable".
    const auto json_or_null = [](float v) { return std::isfinite(v) ? std::format("{:.4f}", v) : std::string("null"); };
    o << "    \"lap_repeat_mean_m\": " << json_or_null(t.lap_repeat_mean_m) << ",\n"
      << "    \"lap_repeat_max_m\": "  << json_or_null(t.lap_repeat_max_m)  << "\n  }\n}\n";

    const bool ok = o.good();
    std::printf("[mission] run record -> %s%s\n", path.c_str(), ok ? "" : "  ** WRITE FAILED **");
    std::fflush(stdout);
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────────────────

bool MissionRunner::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char *m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // (1) Record -> select -> run -> continuous stats. No legs: the run is characterised as a whole.
    {
        MissionRunner r;
        r.set_csv_path(""); r.set_run_dir("");
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording();
        r.add_point({0.f, 0.f}); r.add_point({3.f, 0.f}); r.add_point({3.f, 3.f});
        check(r.finish_recording("t"), "a 3-point recording must commit");
        check(r.start(1, 0), "'mission only' must run");

        std::uint64_t t = 0;
        // Drive 3 m at 0.5 m/s, 1 m off the reference the whole way, on a curve of kappa = 0.5.
        for (int i = 0; i <= 6; ++i)
            r.sample({static_cast<float>(i) * 0.5f, 0.f}, 0.f, 0.5f, 0.42f, false,
                     1.0f, 0.10f, 0.5f, t += 100);
        r.note_progress(3.f, 6.f, 0);
        const auto s1 = r.summary();
        std::printf("  continuous: dist %.2f m, cross rms %.3f m, a_lat rms %.3f, clear %.3f\n",
                    s1.distance_m, s1.cross_track_rms_m, s1.lat_accel_rms, s1.min_clearance_m);
        check(std::abs(s1.distance_m - 3.f) < 1e-3f, "distance must integrate the driven path");
        check(std::abs(s1.cross_track_rms_m - 1.f) < 1e-3f, "cross-track rms must be the deviation held");
        // a_lat = v^2*kappa = 0.25*0.5 = 0.125
        check(std::abs(s1.lat_accel_rms - 0.125f) < 1e-3f, "lateral acceleration must be v^2 * curvature");
        check(std::abs(s1.min_clearance_m - 0.42f) < 1e-3f, "clearance must be the minimum seen");
        check(s1.laps_completed == 0, "a lap in progress is not a lap completed");
        r.stop("user", t);
    }

    // (2) SMOOTHNESS accumulators, hand-computed. These are the numbers a controller change moves.
    {
        MissionRunner r;
        r.set_csv_path(""); r.set_run_dir("");
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording(); r.add_point({0.f, 0.f}); r.add_point({1.f, 0.f});
        r.finish_recording("s"); r.start(1, 0);
        std::uint64_t t = 0;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        // commanded speed 0.5 -> 0.2 -> 0.2 : sum|dv| = 0.3 ; omega +,-,+ : 2 reversals
        r.sample({0.f, 0.f}, +0.4f, 0.5f, 1.f, false, nan, 0.f, 0.f, t += 100);
        r.sample({1.f, 0.f}, -0.4f, 0.2f, 1.f, false, nan, 0.f, 0.f, t += 100);
        r.sample({1.f, 0.f}, +0.4f, 0.2f, 1.f, true,  nan, 0.f, 0.f, t += 100);
        const auto s2 = r.summary();
        std::printf("  smoothness: sum|dv| %.3f, reversals %d, guard %d, cross-track n/a -> rms %.3f\n",
                    s2.lin_accel_effort, s2.rot_reversals, s2.safety_guard_cycles, s2.cross_track_rms_m);
        check(std::abs(s2.lin_accel_effort - 0.3f) < 1e-3f, "sum|dv| must be the total variation of speed");
        check(s2.rot_reversals == 2, "a +,-,+ command sequence is TWO reversals");
        check(s2.safety_guard_cycles == 1, "guard cycles must be counted");
        check(s2.cross_track_rms_m == 0.f,
              "with NO reference, cross-track must stay unset rather than be invented as zero-error");
        r.stop("user", t);
    }

    // (3) Library round trip under a comma-decimal locale — the bug that silently truncated every route.
    {
        const char *saved = std::setlocale(LC_NUMERIC, nullptr);
        const char *applied = nullptr;
        for (const char *cand : {"es_ES.UTF-8", "de_DE.UTF-8", "fr_FR.UTF-8"})
            if ((applied = std::setlocale(LC_NUMERIC, cand)) != nullptr) break;
        std::printf("  locale round trip under LC_NUMERIC=%s\n", applied ? applied : "(none available)");
        MissionRunner a;
        a.set_mode(DriveMode::MissionOnly);
        a.start_recording(); a.add_point({-1.9114f, 3.6829f}); a.add_point({0.5361f, -2.7738f});
        a.finish_recording("loc");
        const std::string path = std::filesystem::temp_directory_path() / "rc_mission_locale.toml";
        a.save(path);
        MissionRunner b; b.load(path);
        const auto *m = b.selected_mission();
        check(m != nullptr and m->waypoints.size() == 2, "the locale round trip must preserve waypoints");
        if (m != nullptr and m->waypoints.size() == 2)
            check(std::abs(m->waypoints[0].pos.x() + 1.9114f) < 1e-4f,
                  "a fractional coordinate must survive a comma-decimal locale (was truncated to -1)");
        std::error_code ec; std::filesystem::remove(path, ec);
        if (saved != nullptr) std::setlocale(LC_NUMERIC, saved);
    }

    // (4) Editing and deletion still behave (drag-to-edit, delete, click target precedence).
    {
        MissionRunner r;
        r.set_csv_path(""); r.set_run_dir("");
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording(); r.add_point({0.f, 0.f}); r.add_point({1.f, 0.f});
        r.finish_recording("e");
        check(r.move_waypoint(1, {4.f, 4.f}), "dragging a waypoint must succeed");
        check(not r.move_waypoint(7, {0.f, 0.f}), "an out-of-range drag must fail, not crash");
        r.set_click_target(Eigen::Vector2f{7.f, 7.f});
        check(r.display_waypoints().size() == 1, "a click target must REPLACE the tour overlay");
        r.set_mode(DriveMode::AffordancesOnly);
        check(r.display_waypoints().empty(), "'affordances' must leave NO mission on the canvas");
        r.set_mode(DriveMode::MissionOnly);
        check(r.remove("e"), "delete must remove the mission");
        check(r.display_waypoints().empty(), "a deleted mission must leave nothing on the canvas");
    }

    // (5) Degenerate inputs refused rather than half-accepted.
    {
        MissionRunner r;
        r.set_mode(DriveMode::MissionOnly);
        r.start_recording(); r.add_point({0.f, 0.f});
        check(not r.finish_recording("one"), "a single-point mission must be rejected");
        check(not r.start(1, 0), "starting with no selected mission must fail, not crash");
        r.set_mode(DriveMode::MissionWithAffordances);
        check(not r.mode_implemented() and not r.start(1, 0),
              "an UNIMPLEMENTED mode must refuse to run, never fall back to another");
    }

    std::printf("MissionRunner::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
