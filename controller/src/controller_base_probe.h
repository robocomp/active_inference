#pragma once
/*
 * controller_base_probe.h — `probe <v> <w> <seconds>`: an OPEN-LOOP measurement command (Task 7, local-motion policy).
 *
 * Holds the base at (v m/s, w rad/s) for <seconds>, then stops. Nothing closes the loop on purpose: the question it
 * answers is "does the base move at all under this command" (controller/tools/measure_deadband.py steps it through
 * small speeds and reads the motion back from the pose and room_concept's GT log). A closed loop would hide exactly
 * that — it raises the command until something moves.
 *
 * ★ Not a navigation command. Limits are the PROBE's (|v| <= 0.15 m/s, |w| <= 0.6 rad/s, 0 < hold <= 4 s), so the
 *   whole sweep is at most 0.6 m / 2.4 rad. The caller checks the predicted sweep (predicted_poses) against the
 *   planner's footprint predicate BEFORE starting and refuses if any pose is not free; while it runs, Stop / driving
 *   off / no pose stop it (SpecificWorker::step_base_probe). These limits are a measuring instrument's range, flagged.
 * ★ It never ARMS the base: refused unless the output is already armed, driving on and not paused (refusal()).
 *
 * Conventions (= RelativeMotion): the robot faces theta + pi/2, forward = (-sin theta, cos theta); + w = CCW (left).
 * Pure: Eigen + std. Control-thread only.
 */

#include <Eigen/Dense>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rc
{

class BaseProbe
{
public:
    static constexpr float kMaxV = 0.15f, kMaxW = 0.6f, kMaxHoldS = 4.f;

    struct Spec { float v = 0.f, w = 0.f, hold_s = 0.f; };
    struct Command { float adv = 0.f, rot = 0.f; };
    struct View { int id = 0; Spec spec; std::uint64_t t_start_ms = 0, t_end_ms = 0; bool active = false; };

    /// "probe <v> <w> <seconds>". Locale-proof (std::from_chars): "0,03" is refused, not read as 0.
    static std::expected<Spec, std::string> parse(std::string_view text)
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
        if (tok.size() != 4 or tok[0] != "probe")
            return std::unexpected(std::string("expected: probe <v m/s> <w rad/s> <seconds>"));
        float n[3];
        for (int k = 0; k < 3; ++k)
        {
            const auto s = tok[k + 1];
            const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), n[k]);
            if (ec != std::errc{} or p != s.data() + s.size() or not std::isfinite(n[k]))
                return std::unexpected(std::format("'{}' is not a number", s));
        }
        const Spec sp{n[0], n[1], n[2]};
        if (std::abs(sp.v) > kMaxV) return std::unexpected(std::format("|v| {:.3f} > probe limit {:.2f} m/s", sp.v, kMaxV));
        if (std::abs(sp.w) > kMaxW) return std::unexpected(std::format("|w| {:.3f} > probe limit {:.2f} rad/s", sp.w, kMaxW));
        if (not (sp.hold_s > 0.f) or sp.hold_s > kMaxHoldS)
            return std::unexpected(std::format("hold {:.2f} s outside (0, {:.0f}] s", sp.hold_s, kMaxHoldS));
        return sp;
    }

    /// The poses the command would sweep if the base executed it exactly (unicycle), every dt seconds incl. the end.
    static std::vector<std::pair<Eigen::Vector2f, float>> predicted_poses(const Spec &s, Eigen::Vector2f pos, float theta, float dt)
    {
        std::vector<std::pair<Eigen::Vector2f, float>> out;
        const int n = std::max(1, static_cast<int>(std::ceil(s.hold_s / dt)));
        const float h = s.hold_s / static_cast<float>(n);
        out.emplace_back(pos, theta);
        for (int k = 0; k < n; ++k)
        {
            const float mid = theta + 0.5f * s.w * h;   // midpoint heading: exact for the straight and pivot cases
            pos += s.v * h * Eigen::Vector2f(-std::sin(mid), std::cos(mid));
            theta += s.w * h;
            out.emplace_back(pos, theta);
        }
        return out;
    }

    /// ★ A PROBE NEVER ARMS THE BASE (review fix round 1): the operator's Stop always wins. It runs only when the base
    /// output is already armed, driving is on and nothing is paused — i.e. after the operator (or the script) sent
    /// `run`. Returns the refusal, or nullopt when the probe may start.
    static std::optional<std::string> refusal(bool output_armed, bool driving_enabled, bool paused)
    {
        if (not output_armed) return std::string("base output is DISARMED (Stop) — send `run` first; a probe never re-arms");
        if (not driving_enabled) return std::string("driving is off — send `run` first; a probe never re-arms");
        if (paused) return std::string("PAUSED — resume first");
        return std::nullopt;
    }

    void start(const Spec &s, std::uint64_t now_ms)
    {
        view_ = View{.id = view_.id + 1, .spec = s, .t_start_ms = now_ms,
                     .t_end_ms = now_ms + static_cast<std::uint64_t>(std::llround(s.hold_s * 1000.f)), .active = true};
    }
    /// The command for this cycle; at/after the end it stops (and the probe ends).
    Command step(std::uint64_t now_ms)
    {
        if (not view_.active) return {};
        if (now_ms >= view_.t_end_ms) { view_.active = false; return {}; }
        return {view_.spec.v, view_.spec.w};
    }
    void cancel() { view_.active = false; }
    bool active() const { return view_.active; }
    const View &view() const { return view_; }

private:
    View view_;
};

}  // namespace rc
