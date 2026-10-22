#pragma once
/*
 * fleet_control.h — the chat starts, stops and restarts fleet agents (user, 2026-10-19), the way the tour scripts do
 * (room_concept/estimation_scripts/scripts/run_crossing.sh): start = `cd <agent> && setsid bin/<agent> etc/config.toml`
 * detached, stop = SIGTERM to the process with the agent's exact name (`pkill -TERM -x`). Never SIGKILL: an agent that
 * ignores SIGTERM is reported, not killed.
 *
 * The rules (user, 2026-10-19):
 *   - stopping or restarting the controller or retina needs the person's yes (Policy::confirm): the robot stops, or
 *     goes blind. The yes is the same request made again after a NEW message from the person (Confirmations), so
 *     the model cannot confirm on its own.
 *   - refused while a tour runs: the tour scripts hold ai_fleet_tour.lock with their PID (tour_running).
 *   - chat_agent never stops itself, and robot_concept is not touched: every other agent depends on it.
 *
 * Pure (std + filesystem): the session (chat_session.cpp) does the launching, the signals and the timers.
 */

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "system_facts.h"

namespace rc::chat::fleet
{

enum class Action { start, stop, restart };

/// A stopped agent has this long to exit after SIGTERM. Agents tear down in ~1 s (DDS deletes + 300 ms flush).
inline constexpr std::int64_t kStopWaitMs = 10000;
/// A started agent has this long to publish its first heartbeat: it may wait for its required peers and for DDS
/// discovery first, a few seconds usually; 30 s covers a cold start.
inline constexpr std::int64_t kStartWaitMs = 30000;
/// The person's yes must come within one exchange of the question.
inline constexpr std::int64_t kConfirmWindowMs = 60000;

[[nodiscard]] inline std::string lower_trim(std::string_view s)
{
    while (not s.empty() and std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (not s.empty() and std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

[[nodiscard]] inline std::optional<Action> parse_action(std::string_view s)
{
    const std::string a = lower_trim(s);
    if (a == "start") return Action::start;
    if (a == "stop") return Action::stop;
    if (a == "restart") return Action::restart;
    return std::nullopt;
}

[[nodiscard]] inline std::string_view to_string(Action a)
{
    return a == Action::start ? "start" : a == Action::stop ? "stop" : "restart";
}

/// "a, b ,,c" → {a, b, c}
[[nodiscard]] inline std::vector<std::string> split_names(std::string_view csv)
{
    std::vector<std::string> out;
    std::size_t a = 0;
    while (a <= csv.size())
    {
        const auto b = std::min(csv.find(',', a), csv.size());
        if (std::string n = lower_trim(csv.substr(a, b - a)); not n.empty()) out.push_back(std::move(n));
        a = b + 1;
    }
    return out;
}

/// The agent the words name: its exact name, or the short one the model tends to use ("chair" → chair_concept).
[[nodiscard]] inline std::expected<std::string, std::string> resolve_agent(std::string_view words,
                                                                         const std::vector<std::string>& known)
{
    std::string w = lower_trim(words);
    for (std::string_view article : {"the ", "el ", "la "})
        if (w.starts_with(article)) w.erase(0, article.size());
    std::ranges::replace(w, ' ', '_');
    for (const std::string& cand : {w, w + "_concept"})
        if (std::ranges::find(known, cand) != known.end()) return cand;
    if (w == "chat_agent" or w == "robot_concept") return w;   // named so decide() can say why not
    std::string list;
    for (const auto& k : known) list += (list.empty() ? "" : ", ") + k;
    return std::unexpected("no agent called '" + std::string(words) + "'; the agents I can start or stop are: " + list);
}

/// The name the kernel keeps for a process: its first 15 characters (TASK_COMM_LEN − 1).
[[nodiscard]] inline std::string process_name(std::string_view agent) { return std::string(agent.substr(0, 15)); }

/// The PIDs of the processes running as `agent` in a /proc tree (`pkill -x`'s match).
[[nodiscard]] inline std::vector<int> find_pids(const std::filesystem::path& proc_root, std::string_view agent)
{
    std::vector<int> pids;
    const std::string want = process_name(agent);
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(proc_root, ec))
    {
        const std::string d = e.path().filename().string();
        int pid = 0;
        if (const auto r = std::from_chars(d.data(), d.data() + d.size(), pid); r.ec != std::errc{} or
                                                                              r.ptr != d.data() + d.size())
            continue;
        std::ifstream f(e.path() / "comm");
        std::string comm;
        if (std::getline(f, comm) and comm == want) pids.push_back(pid);
    }
    std::ranges::sort(pids);
    return pids;
}

/// A tour holds the lock while the PID written in it is alive (a crashed script's lock does not count).
[[nodiscard]] inline bool tour_running(const std::filesystem::path& lock, const std::filesystem::path& proc_root)
{
    std::ifstream f(lock);
    std::string line;
    if (not std::getline(f, line)) return false;
    const std::string t = lower_trim(line);
    int pid = 0;
    if (const auto r = std::from_chars(t.data(), t.data() + t.size(), pid); r.ec != std::errc{} or
                                                                          r.ptr != t.data() + t.size())
        return false;
    std::error_code ec;
    return std::filesystem::exists(proc_root / std::to_string(pid), ec);
}

struct Policy
{
    std::vector<std::string> agents;    ///< what may be started or stopped
    std::vector<std::string> confirm;   ///< stopping these needs the person's yes
};

struct Decision
{
    enum class Kind { refuse, ask, go } kind;
    std::string text;   ///< why not / the question to put to the person
};

[[nodiscard]] inline Decision decide(Action action, const std::string& agent, const Policy& p, bool tour, bool running,
                                     bool confirmed)
{
    using K = Decision::Kind;
    if (agent == "chat_agent") return {K::refuse, "I cannot stop or restart myself (chat_agent)."};
    if (agent == "robot_concept")
        return {K::refuse, "robot_concept holds the robot's body and every other agent depends on it; start or stop "
                           "it from a terminal, not from the chat."};
    if (std::ranges::find(p.agents, agent) == p.agents.end())
        return {K::refuse, "'" + agent + "' is not one of the agents I can start or stop."};
    if (tour) return {K::refuse, "A door-crossing tour is running; agents are not started or stopped until it ends."};
    if (action == Action::start and running) return {K::refuse, agent + " is already running."};
    if (action == Action::stop and not running) return {K::refuse, agent + " is not running."};
    const bool stops = running and action != Action::start;
    if (stops and not confirmed and std::ranges::find(p.confirm, agent) != p.confirm.end())
        return {K::ask, std::string(action == Action::stop ? "Stop " : "Restart ") + agent + "? " +
                            (agent == "controller" ? "The robot stops moving until it is back."
                                                   : "The robot is blind until it is back.")};
    return {K::go, {}};
}

/// A message that is nothing but a yes: it carries out the pending request without the model, which answered "yes"
/// by saying the agent was stopped without calling the tool (eval 2026-10-19).
[[nodiscard]] inline bool is_yes(std::string_view text)
{
    std::string t = lower_trim(text);
    while (not t.empty() and (t.back() == '!' or t.back() == '.' or t.back() == ',')) t.pop_back();
    static const std::vector<std::string> yes{"yes", "yes please", "yeah", "yep", "ok", "okay", "do it", "go ahead",
                                              "confirm", "sí", "si", "sí por favor", "vale", "hazlo", "adelante"};
    return std::ranges::find(yes, t) != yes.end();
}

/// The yes: the same action on the same agent asked again after a later message from the person, within
/// kConfirmWindowMs. Used once.
class Confirmations
{
public:
    void ask(Action a, std::string agent, std::uint64_t user_turn, std::int64_t now_ms)
    {
        pending_ = Pending{a, std::move(agent), user_turn, now_ms};
    }
    struct Pending
    {
        Action action;
        std::string agent;
        std::uint64_t user_turn;
        std::int64_t t_ms;
    };
    /// The request waiting for the person's yes, if any.
    [[nodiscard]] const std::optional<Pending>& pending() const { return pending_; }

    [[nodiscard]] bool confirmed(Action a, const std::string& agent, std::uint64_t user_turn, std::int64_t now_ms)
    {
        if (not pending_.has_value()) return false;
        const Pending& p = *pending_;
        if (p.action != a or p.agent != agent or user_turn <= p.user_turn) return false;
        const bool in_time = now_ms - p.t_ms <= kConfirmWindowMs;
        pending_.reset();
        return in_time;
    }

private:
    std::optional<Pending> pending_;
};

/// One start / stop / restart in progress. tick() with whether a process of that name is alive and whether the
/// NEW instance is up (started_since); it says when to launch and, at the end, what to tell the person.
class Op
{
public:
    Op(Action a, std::string agent, std::int64_t now_ms)
        : action_(a), agent_(std::move(agent)), t0_(now_ms), state_(a == Action::start ? State::launch : State::stopping)
    {}

    struct Step
    {
        bool launch = false;
        bool done = false;
        std::string report;
    };

    Step tick(std::int64_t now_ms, bool alive, bool up)
    {
        Step s;
        switch (state_)
        {
            case State::stopping:
                if (not alive)
                {
                    if (action_ == Action::stop) return finish(s, agent_ + " has stopped.");
                    state_ = State::launch;
                    [[fallthrough]];
                }
                else
                {
                    if (now_ms - t0_ > kStopWaitMs)
                        return finish(s, agent_ + " is still running " + std::to_string(kStopWaitMs / 1000) +
                                             " s after SIGTERM; it was not force-killed.");
                    return s;
                }
            case State::launch:
                s.launch = true;
                launched_ = now_ms;
                state_ = State::starting;
                return s;
            case State::starting:
                if (up) return finish(s, agent_ + " is up" + (action_ == Action::restart ? " again." : "."));
                if (now_ms - launched_ > kStartWaitMs)
                    return finish(s, alive ? agent_ + " is running but sent no heartbeat in " +
                                                 std::to_string(kStartWaitMs / 1000) + " s."
                                           : agent_ + " exited soon after starting.");
                return s;
            case State::done: return s;
        }
        return s;
    }

    [[nodiscard]] const std::string& agent() const { return agent_; }
    [[nodiscard]] Action action() const { return action_; }
    [[nodiscard]] std::int64_t launched_ms() const { return launched_; }

private:
    enum class State { stopping, launch, starting, done };
    Step& finish(Step& s, std::string report)
    {
        state_ = State::done;
        s.done = true;
        s.report = std::move(report);
        return s;
    }
    Action action_;
    std::string agent_;
    std::int64_t t0_, launched_ = 0;
    State state_;
};

/// The agent's heartbeat comes from the process started `elapsed_ms` ago, not from the old one's last breath.
[[nodiscard]] inline bool started_since(const AgentFact& a, std::int64_t elapsed_ms)
{
    // uptime is whole seconds: one second of slack
    return agent_running(a) and a.uptime_s.has_value() and
           static_cast<std::int64_t>(*a.uptime_s) * 1000 <= elapsed_ms + 1000;
}

}   // namespace rc::chat::fleet
