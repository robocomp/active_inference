#pragma once
/*
 * controller_relative_motion.h — "move <metres>" / "turn <degrees>": short ROBOT-RELATIVE motions (2026-10-12, robot
 * chat phase 2). One grammar for the Mission text box, remote_cmd.txt and chat_agent's intention node
 * (SpecificWorker::run_mission_command).
 *
 *   move <m>                 drive straight along the current heading; + forward, − backward (reverse)
 *   turn <deg>               rotate in place; + = left (counter-clockwise), − = right
 *   turn <deg> [then|and] move <m>   (and any sequence of up to kMaxSteps of the two) — executed in the order given
 *   A number may carry its unit glued or separated: "move 0.5m", "turn -90 deg".
 *
 * ★ WHY NOT THE CLICK-TARGET PATH (manual target → GridPlanner::plan → TrajectoryController). It was the obvious
 *   reuse and it does not fit these requests: (1) its arrival test is goal_threshold = 0.25 m, so a 0.3 m step is
 *   "reached" after ~5 cm; (2) the planner drives FORWARD only, so "move -0.5" would turn round, drive, and end
 *   facing the wrong way; (3) a clicked target has no final facing (wants_final_facing() is false for it), so
 *   "turn 90" cannot be expressed at all. A relative step is a straight segment or a pivot, and is executed as one.
 *
 * ★ SAFETY IS THE PLANNER'S, NOT A NEW MODEL. Every pose is tested with the SAME predicate the footprint planner
 *   plans with — GridPlanner::pose_free: the robot's real footprint (plus FootprintSafetyMargin) against the
 *   rasterised room polygon, obstacle polygons and residual cells, rebuilt every cycle in sync_world_state.
 *     · before starting: the whole chain is simulated — every 5 cm of each segment at its heading, every ~8.6° of
 *       each pivot's swept footprint, and each segment's end must lie inside the room polygon. Any failure ⇒ the
 *       command is REFUSED with the distance that is free, and nothing moves.
 *     · while driving: the next 0.35 m ahead (or behind, reversing) is re-tested every cycle; a pose that became
 *       occupied (a person stepped in, an obstacle was born) STOPS the robot and fails the command with a reason.
 *     · while turning: the remaining 30° of the sweep is re-tested every cycle.
 *   Speeds are capped below the controller's own limits (≤ 0.25 m/s forward, ≤ 0.15 m/s reversing — there is no
 *   rear camera, so reversing is the blind direction — ≤ 0.6 rad/s), and a time budget (1.5× the nominal duration
 *   + 10 s) ends a motion that is not making progress.
 *   The completion tolerances (3 cm, 2°) and the minimum speeds that overcome base friction are CONTROL constants
 *   of a terminal manoeuvre, not model gates; they are flagged here as such.
 *
 * Pure: Eigen + std, no graph, no Qt — the caller passes the pose and the two predicates. Control-thread only.
 */

#include <Eigen/Dense>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rc
{

class RelativeMotion
{
public:
    static constexpr std::size_t kMaxSteps = 4;
    static constexpr float kMaxMoveM = 3.0f;      ///< per segment; farther is a navigation target, not a "step"
    static constexpr float kMaxTurnDeg = 360.0f;

    struct Step
    {
        enum class Kind { Turn, Move } kind = Kind::Move;
        float value = 0.f;   ///< Turn: radians (+ = left/CCW); Move: metres (+ = forward)
    };

    /// Parse the grammar. Error = a sentence a person can act on. Locale-proof (std::from_chars, CLAUDE.md).
    static std::expected<std::vector<Step>, std::string> parse(std::string_view text)
    {
        std::vector<std::string_view> tok;
        for (std::size_t i = 0; i < text.size();)
        {
            while (i < text.size() and text[i] == ' ') ++i;
            const std::size_t j = text.find(' ', i);
            const std::size_t e = j == std::string_view::npos ? text.size() : j;
            if (e > i) tok.push_back(text.substr(i, e - i));
            i = e;
        }
        std::vector<Step> steps;
        for (std::size_t i = 0; i < tok.size(); ++i)
        {
            const auto w = tok[i];
            if (w == "then" or w == "and" or w == ",") continue;
            const bool is_turn = w == "turn" or w == "rotate";
            const bool is_move = w == "move" or w == "advance";
            if (not is_turn and not is_move)
                return std::unexpected(std::format("'{}' not understood (expected: move <metres> | turn <degrees>)", w));
            if (i + 1 >= tok.size()) return std::unexpected(std::format("'{}' needs a number", w));
            std::string_view num = tok[++i];
            // A unit glued to the number ("0.5m", "90deg") or as the next word ("0.5 m", "90 deg").
            const char* first = num.data();
            const char* last = num.data() + num.size();
            if (first != last and *first == '+') ++first;
            float v = 0.f;
            const auto [ptr, ec] = std::from_chars(first, last, v);
            if (ec != std::errc{} or ptr == first)
                return std::unexpected(std::format("'{} {}': not a number", w, num));
            std::string_view unit(ptr, static_cast<std::size_t>(last - ptr));
            if (unit.empty() and i + 1 < tok.size())
            {
                const auto u = tok[i + 1];
                if (u == "m" or u == "metres" or u == "meters" or u == "deg" or u == "degrees" or u == "º" or u == "°")
                { unit = u; ++i; }
            }
            if (not std::isfinite(v)) return std::unexpected(std::format("'{} {}': not a finite number", w, num));
            if (is_move)
            {
                if (not unit.empty() and unit != "m" and unit != "metres" and unit != "meters")
                    return std::unexpected(std::format("move takes metres, not '{}'", unit));
                if (std::abs(v) > kMaxMoveM)
                    return std::unexpected(std::format("move {:.2f} m is longer than a relative step (max {:.1f} m)", v, kMaxMoveM));
                if (v != 0.f) steps.push_back({Step::Kind::Move, v});
            }
            else
            {
                if (not unit.empty() and unit != "deg" and unit != "degrees" and unit != "º" and unit != "°")
                    return std::unexpected(std::format("turn takes degrees, not '{}'", unit));
                if (std::abs(v) > kMaxTurnDeg)
                    return std::unexpected(std::format("turn {:.0f} deg is more than a full turn", v));
                if (v != 0.f) steps.push_back({Step::Kind::Turn, v * std::numbers::pi_v<float> / 180.f});
            }
            if (steps.size() > kMaxSteps) return std::unexpected(std::format("at most {} steps per command", kMaxSteps));
        }
        if (steps.empty()) return std::unexpected(std::string("nothing to do (zero distance and zero angle)"));
        return steps;
    }

    enum class Phase { Idle, Turning, Moving, Done, Failed, Cancelled };
    static const char* phase_name(Phase p)
    {
        switch (p)
        {
            case Phase::Idle: return "idle";
            case Phase::Turning: return "turning";
            case Phase::Moving: return "moving";
            case Phase::Done: return "done";
            case Phase::Failed: return "failed";
            case Phase::Cancelled: return "cancelled";
        }
        return "?";
    }

    struct Params
    {
        float max_fwd_mps = 0.25f;
        float max_back_mps = 0.15f;      ///< reversing is blind (no rear camera)
        float max_rot_rps = 0.6f;
        float k_adv = 1.2f;              ///< 1/s on the remaining distance
        float k_rot = 1.5f;              ///< 1/s on the heading error
        float min_adv_mps = 0.04f;       ///< CONTROL constant: below this the base does not creep
        float min_rot_rps = 0.10f;       ///< CONTROL constant: idem for the pivot
        float pos_tol_m = 0.03f;         ///< CONTROL constant: terminal tolerance of a segment
        float yaw_tol_rad = 0.035f;      ///< CONTROL constant: terminal tolerance of a pivot (2°)
        float sample_m = 0.05f;          ///< footprint test spacing along a segment (≈ 2 grid cells)
        float sweep_rad = 0.15f;         ///< footprint test spacing over a pivot
        float lookahead_m = 0.35f;       ///< re-tested every cycle while driving
        float lookahead_rad = 0.52f;     ///< re-tested every cycle while turning
        float timeout_slack_s = 10.f;
    };
    Params params;

    /// pos/theta: the controller's room-frame robot pose (theta = ControllerRobotPose::theta; the robot FACES
    /// theta + π/2 — the GridPlanner convention). free(p, facing_yaw) = GridPlanner::pose_free; inside(p) = in room.
    using FreeFn = std::function<bool(const Eigen::Vector2f&, float)>;
    using InsideFn = std::function<bool(const Eigen::Vector2f&)>;

    struct Command { float adv = 0.f, rot = 0.f; };
    struct View
    {
        std::uint64_t id = 0;     ///< increments per start(); 0 = never started
        Phase phase = Phase::Idle;
        std::string text;         ///< the command as understood, e.g. "turn 90 move 0.30"
        std::string last_event;   ///< progress while running (changes every cycle), the outcome/reason at the end
        std::string current;      ///< the step running, e.g. "turn 90 (1 of 2)" — changes only between steps
    };

    /// Check the whole chain and start it. Returns the refusal reason, or nullopt when started.
    std::optional<std::string> start(std::vector<Step> steps, const Eigen::Vector2f& pos, float theta,
                                     std::uint64_t now_ms, const FreeFn& free, const InsideFn& inside)
    {
        const std::string text = describe(steps);
        ++id_;
        text_ = text;
        if (const auto why = check_chain(steps, pos, theta, free, inside); why.has_value())
        {
            phase_ = Phase::Failed;
            last_event_ = *why;
            return why;
        }
        steps_ = std::move(steps);
        index_ = 0;
        float nominal_s = 0.f;
        for (const auto& s : steps_)
            nominal_s += s.kind == Step::Kind::Turn ? std::abs(s.value) / params.max_rot_rps
                                                    : std::abs(s.value) / (s.value > 0 ? params.max_fwd_mps : params.max_back_mps);
        budget_ms_ = static_cast<std::uint64_t>((1.5f * nominal_s + params.timeout_slack_s) * 1000.f);
        active_ms_ = 0;
        last_step_ms_ = now_ms;
        begin_step(pos, theta);
        return std::nullopt;
    }

    /// One control cycle. Returns the base command (zero when not active). Ends the motion on arrival, on a newly
    /// occupied pose ahead, or when the time budget is spent.
    Command step(const Eigen::Vector2f& pos, float theta, std::uint64_t now_ms, const FreeFn& free)
    {
        if (not active()) return {};
        // Only time spent actually stepping counts: a pause (Run/Pause) must not eat the budget.
        active_ms_ += std::min<std::uint64_t>(now_ms >= last_step_ms_ ? now_ms - last_step_ms_ : 0, 250);
        last_step_ms_ = now_ms;
        if (active_ms_ > budget_ms_) { fail(std::format("timed out after {:.0f} s ({})", active_ms_ / 1000.0, progress(pos))); return {}; }

        const Step& s = steps_[index_];
        if (s.kind == Step::Kind::Turn)
        {
            // Accumulated, not wrap(goal − theta): a 270° turn left must not become 90° right, nor 360° nothing.
            turned_ += wrap(theta - prev_theta_);
            prev_theta_ = theta;
            const float err = s.value - turned_;
            if (std::abs(err) < params.yaw_tol_rad) { next_step(pos, theta); return {}; }
            // The rest of the sweep (up to lookahead_rad) must still be clear for the body.
            const float span = std::min(std::abs(err), params.lookahead_rad);
            const int n = std::max(1, static_cast<int>(std::ceil(span / params.sweep_rad)));
            for (int k = 1; k <= n; ++k)
            {
                const float h = theta + std::copysign(span * k / n, err);
                if (not free(pos, h + std::numbers::pi_v<float> / 2))
                { fail(std::format("stopped: turning further would hit something ({})", progress(pos))); return {}; }
            }
            float rot = std::clamp(params.k_rot * err, -params.max_rot_rps, params.max_rot_rps);
            if (std::abs(rot) < params.min_rot_rps) rot = std::copysign(params.min_rot_rps, err);
            last_event_ = progress(pos);
            return {0.f, rot};
        }
        // Move: the remaining distance along the segment's direction (signed by the requested sense).
        const float dir = s.value > 0 ? 1.f : -1.f;
        const Eigen::Vector2f fwd = forward(seg_theta_);
        const float remaining = dir * (goal_pos_ - pos).dot(fwd);
        if (remaining < params.pos_tol_m) { next_step(pos, theta); return {}; }
        // Re-test the next stretch of the segment at the segment heading, every cycle.
        const float span = std::min(remaining, params.lookahead_m);
        const int n = std::max(1, static_cast<int>(std::ceil(span / params.sample_m)));
        for (int k = 1; k <= n; ++k)
        {
            const Eigen::Vector2f p = pos + dir * (span * k / n) * fwd;
            if (not free(p, seg_theta_ + std::numbers::pi_v<float> / 2))
            {
                fail(std::format("stopped: something is now {:.2f} m {} ({})", span * (k - 1) / n,
                                 dir > 0 ? "ahead" : "behind", progress(pos)));
                return {};
            }
        }
        const float vmax = dir > 0 ? params.max_fwd_mps : params.max_back_mps;
        const float adv = dir * std::clamp(params.k_adv * remaining, params.min_adv_mps, vmax);
        // Hold the segment heading (a differential base drifts a little on a straight line).
        const float rot = std::clamp(params.k_rot * wrap(seg_theta_ - theta), -0.3f, 0.3f);
        last_event_ = progress(pos);
        return {adv, rot};
    }

    // ── Task 8: WITH THE LOCAL POLICY DRIVING, a relative step is a GOAL, not a control law. ─────────────────────────────
    // The step's goal pose (the controller's theta convention, the robot faces theta + pi/2):
    //   Move — the segment's end at the segment heading (a Place with that facing: reversing keeps the heading);
    //   Turn — a heading at the start position. The ACCUMULATED turn is preserved: while more than a quarter turn
    //          remains, the goal is the next quarter-turn mark from the step's start heading (start + k·pi/2 toward the
    //          sense asked), so a 270-degree left turn is not taken as 90 degrees right by a heading goal that wraps.
    // Completion stays this class's own (yaw_tol_rad / pos_tol_m); failure is the policy's verdict (policy_step).
    struct PolicyGoal { bool turn = false; Eigen::Vector2f pos = Eigen::Vector2f::Zero(); float theta = 0.f; };
    [[nodiscard]] std::optional<PolicyGoal> policy_goal() const
    {
        if (not active()) return std::nullopt;
        const Step& s = steps_[index_];
        if (s.kind == Step::Kind::Move) return PolicyGoal{false, goal_pos_, seg_theta_};
        const float start = goal_theta_ - s.value;
        const float quarter = std::numbers::pi_v<float> / 2.f;
        const float remaining = s.value - turned_;
        if (std::abs(remaining) <= quarter) return PolicyGoal{true, start_pos_, goal_theta_};
        const float done_q = std::floor(std::abs(turned_) / quarter + 1e-3f) * (turned_ * s.value >= 0.f ? 1.f : 0.f);
        return PolicyGoal{true, start_pos_, start + std::copysign((done_q + 1.f) * quarter, s.value)};
    }
    // One cycle of a policy-driven step: the budget, the accumulated turn, this class's own completion test, and — when the
    // policy cannot reach the goal (`failed`: Stuck / Unreachable) — the failure, with the policy's reason.
    void policy_step(const Eigen::Vector2f& pos, float theta, std::uint64_t now_ms, const std::optional<std::string>& failed)
    {
        if (not active()) return;
        active_ms_ += std::min<std::uint64_t>(now_ms >= last_step_ms_ ? now_ms - last_step_ms_ : 0, 250);
        last_step_ms_ = now_ms;
        if (active_ms_ > budget_ms_) { fail(std::format("timed out after {:.0f} s ({})", active_ms_ / 1000.0, progress(pos))); return; }
        const Step& s = steps_[index_];
        bool done = false;
        if (s.kind == Step::Kind::Turn)
        {
            turned_ += wrap(theta - prev_theta_);
            prev_theta_ = theta;
            done = std::abs(s.value - turned_) < params.yaw_tol_rad;
        }
        else
        {
            const float dir = s.value > 0 ? 1.f : -1.f;
            done = dir * (goal_pos_ - pos).dot(forward(seg_theta_)) < params.pos_tol_m;
        }
        if (done) { next_step(pos, theta); return; }
        if (failed.has_value()) { fail(std::format("stopped: {} ({})", *failed, progress(pos))); return; }
        last_event_ = progress(pos);
    }

    void cancel(const std::string& why)
    {
        if (not active()) return;
        phase_ = Phase::Cancelled;
        last_event_ = why;
    }

    [[nodiscard]] bool active() const { return phase_ == Phase::Turning or phase_ == Phase::Moving; }
    [[nodiscard]] View view() const
    {
        std::string cur;
        if (active())
            cur = std::format("{}{}", describe({steps_[index_]}),
                              steps_.size() > 1 ? std::format(" ({} of {})", index_ + 1, steps_.size()) : std::string{});
        return {.id = id_, .phase = phase_, .text = text_, .last_event = last_event_, .current = cur};
    }

    static std::string describe(const std::vector<Step>& steps)
    {
        std::string out;
        for (const auto& s : steps)
            out += (out.empty() ? "" : " ") + (s.kind == Step::Kind::Turn
                                                   ? std::format("turn {:.0f}", s.value * 180.f / std::numbers::pi_v<float>)
                                                   : std::format("move {:.2f}", s.value));
        return out;
    }

    static float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }
    /// Unit vector the robot faces at pose angle theta (it faces theta + π/2).
    static Eigen::Vector2f forward(float theta) { return {-std::sin(theta), std::cos(theta)}; }

private:
    std::optional<std::string> check_chain(const std::vector<Step>& steps, Eigen::Vector2f pos, float theta,
                                           const FreeFn& free, const InsideFn& inside) const
    {
        if (steps.empty()) return std::string("nothing to do");
        for (const auto& s : steps)
        {
            if (s.kind == Step::Kind::Turn)
            {
                const int n = std::max(1, static_cast<int>(std::ceil(std::abs(s.value) / params.sweep_rad)));
                for (int k = 1; k <= n; ++k)
                {
                    const float h = theta + s.value * k / n;
                    if (not free(pos, h + std::numbers::pi_v<float> / 2))
                        return std::format("turning {:.0f} deg would bring the body into a wall or obstacle at about "
                                           "{:.0f} deg", s.value * 180.f / std::numbers::pi_v<float>,
                                           s.value * k / n * 180.f / std::numbers::pi_v<float>);
                }
                theta += s.value;
            }
            else
            {
                const float dir = s.value > 0 ? 1.f : -1.f;
                const Eigen::Vector2f fwd = forward(theta);
                const Eigen::Vector2f end = pos + s.value * fwd;
                if (not inside(end))
                    return std::format("{:.2f} m {} would leave the room", std::abs(s.value), dir > 0 ? "forward" : "backward");
                const float len = std::abs(s.value);
                const int n = std::max(1, static_cast<int>(std::ceil(len / params.sample_m)));
                for (int k = 1; k <= n; ++k)
                {
                    const Eigen::Vector2f p = pos + dir * (len * k / n) * fwd;
                    if (not free(p, theta + std::numbers::pi_v<float> / 2))
                        return std::format("a wall or obstacle blocks the way {}: only {:.2f} m of the {:.2f} m are clear",
                                           dir > 0 ? "ahead" : "behind", len * (k - 1) / n, len);
                }
                pos = end;
            }
        }
        return std::nullopt;
    }

    void begin_step(const Eigen::Vector2f& pos, float theta)
    {
        const Step& s = steps_[index_];
        // Goals are chained from the PLANNED poses, not the achieved ones, so tolerances do not accumulate
        // into the heading of the next segment; the position of a move starts from where the robot is.
        if (s.kind == Step::Kind::Turn)
        {
            goal_theta_ = (index_ == 0 ? theta : goal_theta_) + s.value;
            phase_ = Phase::Turning;
        }
        else
        {
            seg_theta_ = index_ == 0 ? theta : goal_theta_;
            goal_theta_ = seg_theta_;
            goal_pos_ = pos + s.value * forward(seg_theta_);
            phase_ = Phase::Moving;
        }
        start_pos_ = pos;
        prev_theta_ = theta;
        turned_ = 0.f;
        last_event_ = std::format("{} started", describe({s}));
    }

    void next_step(const Eigen::Vector2f& pos, float theta)
    {
        if (++index_ >= steps_.size())
        {
            phase_ = Phase::Done;
            last_event_ = std::format("done: {}", describe(steps_));
            return;
        }
        begin_step(pos, theta);
    }

    void fail(std::string why)
    {
        phase_ = Phase::Failed;
        last_event_ = std::move(why);
    }

    std::string progress(const Eigen::Vector2f& pos) const
    {
        const Step& s = steps_[index_];
        if (s.kind == Step::Kind::Turn)
            return std::format("turned {:.0f} of {:.0f} deg", std::abs(turned_) * 180.f / std::numbers::pi_v<float>,
                               std::abs(s.value) * 180.f / std::numbers::pi_v<float>);
        return std::format("moved {:.2f} of {:.2f} m", std::abs((pos - start_pos_).dot(forward(seg_theta_))), std::abs(s.value));
    }

    std::uint64_t id_ = 0;
    Phase phase_ = Phase::Idle;
    std::string text_, last_event_;
    std::vector<Step> steps_;
    std::size_t index_ = 0;
    Eigen::Vector2f goal_pos_ = Eigen::Vector2f::Zero(), start_pos_ = Eigen::Vector2f::Zero();
    float goal_theta_ = 0.f, seg_theta_ = 0.f, prev_theta_ = 0.f, turned_ = 0.f;
    std::uint64_t budget_ms_ = 0, active_ms_ = 0, last_step_ms_ = 0;
};

}   // namespace rc
