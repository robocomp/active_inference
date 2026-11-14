#pragma once
/*
 * whereabouts.h — where the robot has been, time-stamped: the PAST a command may refer to
 * (docs/superpowers/specs/2026-11-14-tense-in-commands-design.md §3).
 *
 * Fed from every digest refresh (~500 ms) with the current room and the robot pose in its frame. A sample is stored
 * when the room changes, otherwise at most one per `period_ms`; samples older than `keep_ms` are dropped. In memory
 * only (a restart starts empty; the episodic memory keeps the long record).
 *
 *   at(t)           the last sample at or before t — "where were you 10 minutes ago"
 *   visits()        per room, the latest interval the robot was in it — "have you been in room 3?"
 *   state_line()    the STATE line: every room once, latest visit first
 *   where_was(...)  the where_was tool's answer
 *
 * ★ A past pose is where the robot BELIEVED it was then, in that room's frame. ltsm recalls a room in the same frame,
 *   so the pose still means the same spot after leaving and re-entering; the answer says how long ago.
 *
 * Pure: no graph, no Qt. Times are wall-clock epoch ms; clock text is local time.
 */

#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace rc::chat
{

struct WhereSample
{
    std::int64_t t_ms = 0;
    std::string room;      ///< "" = no current room known then
    bool has_pose = false;
    double x = 0, y = 0, yaw_deg = 0;
};

struct Visit
{
    std::string room;
    std::int64_t from_ms = 0, to_ms = 0;   ///< to_ms = the last sample in it
    bool now = false;                      ///< the robot is in it at the latest sample
};

/// "HH:MM" of an epoch ms, local time.
[[nodiscard]] inline std::string clock_hhmm(std::int64_t t_ms)
{
    const std::time_t t = static_cast<std::time_t>(t_ms / 1000);
    std::tm lt{};
    localtime_r(&t, &lt);
    return std::format("{:02}:{:02}", lt.tm_hour, lt.tm_min);
}

/// "12 min ago" / "40 s ago" / "2 h 5 min ago".
[[nodiscard]] inline std::string ago_text(std::int64_t dt_ms)
{
    const std::int64_t s = dt_ms < 0 ? 0 : dt_ms / 1000;
    if (s < 60) return std::format("{} s ago", s);
    const std::int64_t m = (s + 30) / 60;
    if (m < 60) return std::format("{} min ago", m);
    return m % 60 == 0 ? std::format("{} h ago", m / 60) : std::format("{} h {} min ago", m / 60, m % 60);
}

class Whereabouts
{
public:
    explicit Whereabouts(std::int64_t period_ms = 5'000, std::int64_t keep_ms = 24LL * 3600 * 1000)
        : period_ms_(period_ms), keep_ms_(keep_ms) {}

    /// One digest refresh. Returns true when it was stored (the room changed, or period_ms passed since the last
    /// stored one). Every refresh is also kept as the LATEST, so "now" and the end of the current visit are fresh.
    bool add(const WhereSample& s)
    {
        if (latest_.has_value() and s.t_ms < latest_->t_ms) return false;   // the clock went back: keep the order
        latest_ = s;
        const bool room_changed = trace_.empty() or trace_.back().room != s.room;
        if (not room_changed and s.t_ms - trace_.back().t_ms < period_ms_) return false;
        trace_.push_back(s);
        while (not trace_.empty() and s.t_ms - trace_.front().t_ms > keep_ms_) trace_.pop_front();
        return true;
    }

    [[nodiscard]] bool empty() const { return trace_.empty(); }
    [[nodiscard]] std::size_t size() const { return trace_.size(); }
    [[nodiscard]] std::optional<std::int64_t> first_ms() const
    {
        return trace_.empty() ? std::nullopt : std::optional<std::int64_t>(trace_.front().t_ms);
    }

    /// The last sample at or before t; nullopt before the first one.
    [[nodiscard]] std::optional<WhereSample> at(std::int64_t t_ms) const
    {
        std::optional<WhereSample> out;
        for (const auto& s : trace_)
        {
            if (s.t_ms > t_ms) break;
            out = s;
        }
        return out;
    }

    /// Per room, its LATEST visit (consecutive samples in one room merged), latest first. Samples with no room skipped.
    [[nodiscard]] std::vector<Visit> visits() const
    {
        std::vector<Visit> runs;
        for (const auto& s : all())
        {
            if (not runs.empty() and runs.back().room == s.room) runs.back().to_ms = s.t_ms;
            else runs.push_back({.room = s.room, .from_ms = s.t_ms, .to_ms = s.t_ms});
        }
        if (not runs.empty()) runs.back().now = true;
        std::vector<Visit> out;
        for (auto it = runs.rbegin(); it != runs.rend(); ++it)
        {
            if (it->room.empty()) continue;
            bool seen = false;
            for (const auto& v : out) seen = seen or v.room == it->room;
            if (not seen) out.push_back(*it);
        }
        return out;
    }

    /// "where you have been (since 00:12): room_1 now, since 00:28; room_2 00:22-00:27 (4 min ago); …"
    [[nodiscard]] std::string state_line(std::int64_t now_ms, std::size_t max_rooms = 6) const
    {
        if (trace_.empty()) return {};
        const auto v = visits();
        std::string s = std::format("where you have been (since {}, when this chat started):", clock_hhmm(trace_.front().t_ms));
        if (v.empty()) return s + " no room known yet\n";
        std::size_t n = 0;
        for (const auto& x : v)
        {
            if (n++ == max_rooms) { s += " …"; break; }
            s += (n == 1 ? " " : "; ") + x.room;
            if (x.now) s += std::format(" now, since {}", clock_hhmm(x.from_ms));
            else s += std::format(" {}-{} ({})", clock_hhmm(x.from_ms), clock_hhmm(x.to_ms), ago_text(now_ms - x.to_ms));
        }
        return s + "\n";
    }

    /// The where_was tool's answer for time `t_ms`. `ways` = the STATE's leg text to the sample's room when it is not
    /// the current one ("" = none known). `span_ms` = how much time the asked time names: "13:38" is the whole minute
    /// [13:38:00, 13:39:00), so a record that starts inside it answers it (the chat started at 13:38:05).
    [[nodiscard]] std::string where_was(std::int64_t t_ms, std::int64_t now_ms, const std::string& current_room,
                                        const std::string& ways = {}, std::int64_t span_ms = 0) const
    {
        if (trace_.empty()) return "no record yet: the robot's whereabouts are kept from when this chat starts";
        if (t_ms > now_ms) return "that time is in the future";
        auto s = at(t_ms);
        if (not s.has_value() and trace_.front().t_ms < t_ms + span_ms) s = trace_.front();
        if (not s.has_value())
            return std::format("no record before {} ({}), when this chat started", clock_hhmm(trace_.front().t_ms),
                               ago_text(now_ms - trace_.front().t_ms));
        std::string out = std::format("at {} ({}) the robot was ", clock_hhmm(t_ms), ago_text(now_ms - t_ms));
        if (s->room.empty()) return out + "in no known room (the map was still being made)";
        out += "in " + s->room;
        if (s->has_pose)
            out += std::format(" at pose:{}({:.2f}, {:.2f}, {:.0f}deg)", s->room, s->x, s->y, s->yaw_deg);
        if (s->room == current_room) out += "; that is the room it is in now";
        else
        {
            out += "; it is in " + (current_room.empty() ? std::string("an unknown room") : current_room) + " now";
            if (not ways.empty()) out += "; way there: " + ways;
        }
        if (s->has_pose)
            out += std::format(". To go back there, call plan_mission with a request that names pose:{}({:.2f}, {:.2f})",
                               s->room, s->x, s->y);
        return out;
    }

private:
    /// The stored samples, then the latest refresh when it was not stored.
    [[nodiscard]] std::vector<WhereSample> all() const
    {
        std::vector<WhereSample> v(trace_.begin(), trace_.end());
        if (latest_.has_value() and (v.empty() or latest_->t_ms > v.back().t_ms)) v.push_back(*latest_);
        return v;
    }

    std::deque<WhereSample> trace_;
    std::optional<WhereSample> latest_;
    std::int64_t period_ms_, keep_ms_;
};

}   // namespace rc::chat
