#pragma once
/*
 * imagined_world.h — the world a subjunctive question is asked about ("would I fit through door_1 if it were half
 * open?"). It is a COPY of the room facts (SeeScene, can_see.h) plus the robot's own uncertainties; an intervention
 * edits the copy and appends a plain-words note, so the live graph can never be written from here (pure values: no DSR,
 * no Qt in this header's own code). Interventions take the world BY VALUE and return the new one.
 */

#include <Eigen/Geometry>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <expected>
#include <format>
#include <map>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "can_see.h"
#include "topology.h"
#include "../../common/episodic/em_schema.h"

namespace rc::chat::imagine
{

struct ImaginedWorld
{
    SeeScene scene;                                   ///< the room facts (can_see.h)
    std::optional<float> body_half_width;             ///< m, from the robot mesh footprint; nullopt = "no body model"
    std::optional<float> robot_sigma_m;               ///< robot localisation 1-sigma (m); nullopt = unknown
    std::map<std::string, float> door_sigma_m;        ///< per door: 1-sigma of its centre along the wall
    std::map<std::string, float> object_sigma_m;      ///< per object instance: 1-sigma of its position
    Topology topology;
    std::vector<rc::em::Record> records;              ///< today's EM records
    std::vector<std::string> notes;                   ///< assumptions made by interventions, in plain words
};

// ── opening words ──
struct Opening { std::optional<float> phi_rad; std::string words; };   ///< nullopt phi = "as it is now"

namespace imagine_detail
{
inline std::string trim_lower(std::string_view s)
{
    while (not s.empty() and std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (not s.empty() and std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return lower_ascii(std::string(s));
}
inline Opening angle_opening(float deg, std::string_view word)
{
    const float rad = deg * std::numbers::pi_v<float> / 180.0f;
    return {rad, word.empty() ? std::format("{:.0f} degrees", deg) : std::format("{} (taken as {:.0f}°)", word, deg)};
}
}   // namespace imagine_detail

/// shut|closed, ajar, half[ open], wide[ open]|open|fully open, as_is|"", or degrees ("30", "30 deg", "30°", clamped to
/// [0, 90]: pi/2 is the leaf's physical travel, Openable.PhiMaxRad).
[[nodiscard]] inline std::expected<Opening, std::string> parse_opening(std::string_view s)
{
    using imagine_detail::angle_opening;
    const std::string t = imagine_detail::trim_lower(s);
    if (t.empty() or t == "as_is") return Opening{std::nullopt, "as it is now"};
    if (t == "shut" or t == "closed") return angle_opening(0.0f, t);
    if (t == "ajar") return angle_opening(15.0f, t);
    if (t == "half" or t == "half open") return angle_opening(45.0f, t);
    if (t == "wide" or t == "wide open" or t == "open" or t == "fully open") return angle_opening(90.0f, t);

    std::string_view n = t;
    for (const std::string_view suffix : {"degrees", "degree", "deg", "°"})
        if (n.ends_with(suffix)) { n.remove_suffix(suffix.size()); break; }
    while (not n.empty() and std::isspace(static_cast<unsigned char>(n.back()))) n.remove_suffix(1);
    float deg = 0;
    const auto [p, ec] = std::from_chars(n.data(), n.data() + n.size(), deg);
    if (not n.empty() and ec == std::errc{} and p == n.data() + n.size() and std::isfinite(deg))
    {
        const float c = std::clamp(deg, 0.0f, 90.0f);
        Opening o = angle_opening(c, "");
        if (c != deg) o.words = std::format("{:.0f} degrees (the leaf only opens to 90°, taken as 90°)", deg);
        return o;
    }
    return std::unexpected(std::format("I don't know how open '{}' is; say shut, ajar, half, wide or an angle in degrees", std::string(s)));
}

// ── interventions ──
using Intervened = std::expected<ImaginedWorld, std::string>;

[[nodiscard]] inline Intervened with_door_phi(ImaginedWorld w, std::string_view door, float phi_rad)
{
    const std::string key = lower_ascii(std::string(door));
    const auto it = std::ranges::find_if(w.scene.doors, [&](const SeeDoor& d) { return lower_ascii(d.name) == key; });
    if (it == w.scene.doors.end()) return std::unexpected(std::format("I don't know a door called {}", door));
    it->phi_rad = phi_rad;
    it->p_open.reset();   // the evaluator then uses phi, not the old P(open)
    w.notes.push_back(std::format("imagined: {} opened to {:.0f}° (instead of how it is now)", it->name,
                                  phi_rad * 180.0f / std::numbers::pi_v<float>));
    return w;
}

[[nodiscard]] inline Intervened with_robot_at(ImaginedWorld w, Eigen::Vector2f xy, float yaw)
{
    w.scene.robot_xy = xy;
    w.scene.robot_yaw = yaw;
    w.notes.push_back(std::format("imagined: the robot at ({:.2f}, {:.2f}) m, heading {:.0f}°", xy.x(), xy.y(),
                                  yaw * 180.0f / std::numbers::pi_v<float>));
    return w;
}

[[nodiscard]] inline Intervened with_object_moved(ImaginedWorld w, std::string_view object, float dx, float dy)
{
    const std::string key = lower_ascii(std::string(object));
    const auto it = std::ranges::find_if(w.scene.objects, [&](const SeeObject& o) { return lower_ascii(o.name) == key; });
    if (it == w.scene.objects.end()) return std::unexpected(std::format("I don't know an object called {}", object));
    it->x += dx;
    it->y += dy;
    if (it->standoff.has_value()) *it->standoff += Eigen::Vector2f(dx, dy);
    w.notes.push_back(std::format("imagined: {} moved by ({:+.2f}, {:+.2f}) m", it->name, dx, dy));
    return w;
}

// ── "there" ──
struct Standpoint { Eigen::Vector2f xy; std::optional<float> yaw; std::string words; };

[[nodiscard]] inline std::expected<Standpoint, std::string> resolve_standpoint(const ImaginedWorld& w, std::string_view from)
{
    const SeeScene& s = w.scene;
    const std::string f = imagine_detail::trim_lower(from);
    if (f.empty() or f == "here" or f == "me" or f == "the robot" or f == lower_ascii(s.room))
    {
        if (not s.robot_xy.has_value()) return std::unexpected("I don't know where I am");
        return Standpoint{*s.robot_xy, s.robot_yaw + std::numbers::pi_v<float> / 2.0f, "where I am"};
    }
    for (const auto& d : s.doors)
        if (lower_ascii(d.name) == f) return Standpoint{d.xy, std::nullopt, std::format("the doorway of {}", d.name)};

    const SeeObject* pick = nullptr;
    for (const auto& o : s.objects)
        if (lower_ascii(o.name) == f) { pick = &o; break; }
    if (pick == nullptr)
    {
        const std::string cls = canonical_concept(f);
        std::vector<const SeeObject*> same;
        for (const auto& o : s.objects)
            if (lower_ascii(o.subtype) == cls) same.push_back(&o);
        if (same.size() == 1) pick = same.front();
    }
    if (pick == nullptr) return std::unexpected("ASK: which place do you mean by 'there'?");

    const Eigen::Vector2f c(pick->x, pick->y);
    const bool sized = pick->w.has_value() and pick->d.has_value();
    const float obj_r = sized ? 0.5f * std::hypot(*pick->w, *pick->d) : 0.0f;
    const float body_r = w.body_half_width.value_or(kRobotRadiusM);
    // toward the robot (the side it is on); with no robot pose, along +x
    Eigen::Vector2f dir = s.robot_xy.has_value() ? Eigen::Vector2f(*s.robot_xy - c) : Eigen::Vector2f(1, 0);
    dir = dir.norm() > 1e-6f ? Eigen::Vector2f(dir.normalized()) : Eigen::Vector2f(1, 0);
    const Eigen::Vector2f xy = c + dir * (obj_r + body_r);
    const float yaw = std::atan2(-dir.y(), -dir.x());   // facing the object
    std::string words = std::format("next to {}, {:.2f} m from its centre", pick->name, obj_r + body_r);
    if (not sized) words += " (its size is unknown, so I took it as a point)";
    if (not w.body_half_width.has_value()) words += std::format(" (no body model, so I used a {:.2f} m robot radius)", kRobotRadiusM);
    return Standpoint{xy, yaw, std::move(words)};
}

}   // namespace rc::chat::imagine
