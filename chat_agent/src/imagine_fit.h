#pragma once
/*
 * imagine_fit.h — "would I fit through door_X if it were half open?" and "do I fit between A and B?", answered on the
 * CONTROLLER's own fit model (common/interaction_anchor/anchor_fit.h: anchor_p_fit, anchor_clear_gap) over an
 * ImaginedWorld. Pure values, no DSR, no Qt. There is no second fit formula here.
 */

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "imagined_world.h"
#include "../../common/interaction_anchor/anchor_fit.h"
#include "../../common/interaction_anchor/anchor_leaf.h"

namespace rc::chat::imagine
{

struct FitAnswer { std::optional<float> p; float slack_m = 0; std::optional<float> phi_needed_rad; std::string text; };

/// Shadow's published height (m): the robot body occupies the vertical band [0, kRobotHeightM].
inline constexpr float kRobotHeightM = 1.25f;

namespace fit_detail
{
inline constexpr float kPi = std::numbers::pi_v<float>;
inline constexpr float kDeg = 180.0f / kPi;
/// Anchor-frame y of the hinge pin. It shifts the leaf across the wall only; the gap ALONG the wall (all the fit model
/// reads) does not depend on it.
inline constexpr float kHingeFaceY = -0.05f;

/// The four hinge (left/right) x swing (toward/away) leaves at the given opening, p = 0.25 each (hinge and swing are not
/// published on the door node; a published hinge belief replaces the weights later). anchor_clear_gap reads only the
/// leaf's footprint ALONG THE WALL, which a swing to the other side leaves unchanged, so the swing terms agree here and
/// differ only through the model's leaf thickness; the mixture is kept whole on purpose.
inline std::vector<rc::interaction::AnchorLeaf> leaves_at(float width, float phi)
{
    return rc::interaction::anchor_leaf_hypotheses(0.5f * width, kHingeFaceY, -1.f, -1.f, width, phi, 0.f, 0.f, 0.f);
}

struct DoorEval { float p = 0, mean_gap = 0; };
inline DoorEval eval_door(float width, float phi, float body_half, float sigma)
{
    const auto leaves = leaves_at(width, phi);
    DoorEval e;
    for (const auto& l : leaves)
    {
        const auto g = rc::anchor_clear_gap(width, &l);
        e.p += l.p * rc::anchor_p_fit(g.width, body_half, 0.f, sigma);
        e.mean_gap += l.p * g.width;
    }
    e.p = std::clamp(e.p, 0.f, 1.f);
    return e;
}

inline std::string deg1(float rad) { return std::format("{:.0f}", rad * kDeg); }
}   // namespace fit_detail

inline constexpr std::string_view kDoorCaveat = "an upper bound: the leaf's depth across the doorway is not modelled";

[[nodiscard]] inline FitAnswer answer_fit_door(const ImaginedWorld& w, std::string_view door, const Opening& opening)
{
    using namespace fit_detail;
    FitAnswer a;
    const std::string key = imagine_detail::norm_name(door);
    const auto it = std::ranges::find_if(w.scene.doors, [&](const SeeDoor& d) { return imagine_detail::norm_name(d.name) == key; });
    if (it == w.scene.doors.end()) { a.text = std::format("I don't know a door called {}", door); return a; }
    if (not w.body_half_width.has_value()) { a.text = "I have no model of my body width"; return a; }
    const float bh = *w.body_half_width;
    const float width = it->width_m > 0.f ? it->width_m : kNominalDoorWidthM;

    // σ_lat: the door's and the robot's, whichever are known.
    std::optional<float> sd, sr = w.robot_sigma_m;
    if (const auto s = w.door_sigma_m.find(it->name); s != w.door_sigma_m.end()) sd = s->second;
    const bool exact = not sd.has_value() and not sr.has_value();
    const float sigma = std::hypot(sd.value_or(0.f), sr.value_or(0.f));   // anchor_p_fit takes sigma 0 as a step

    // φ*: the opening at which the model's clear gap equals the body's width (P = 0.5), by bisection on the model's gap
    // (monotone beyond a leaf-thickness-sized dip near 0, where the gap is clamped at 0 anyway).
    const float target = 2.f * bh;
    const auto gap_at = [&](float phi) { return eval_door(width, phi, bh, sigma).mean_gap; };
    std::string star;
    if (gap_at(kPi / 2) < target)
        star = "even fully open I would not fit";
    else if (gap_at(0.f) >= target)
        star = "I fit at any opening";
    else
    {
        float lo = 0.f, hi = kPi / 2;
        for (int i = 0; i < 50; ++i) { const float m = 0.5f * (lo + hi); (gap_at(m) < target ? lo : hi) = m; }
        a.phi_needed_rad = 0.5f * (lo + hi);
        star = std::format("I would need it open to about {}°", deg1(*a.phi_needed_rad));
    }

    std::string sig_note;
    if (exact) sig_note = " (treating the geometry as exact)";
    else if (not sd.has_value()) sig_note = std::format(" ({}'s position uncertainty is unknown; counted as exact)", it->name);
    else if (not sr.has_value()) sig_note = " (my own position uncertainty is unknown; counted as exact)";
    if (not (it->width_m > 0.f)) sig_note += std::format(" (its width is not measured; taken as {:.2f} m)", kNominalDoorWidthM);
    const auto line = [&](const DoorEval& e, std::string_view angle)
    {
        return std::format("P(fit) = {:.2f} with the door {}, slack {:+.0f} cm", e.p, angle, (e.mean_gap - target) * 100.f);
    };

    std::string head;
    if (opening.phi_rad.has_value())
    {
        const auto e = eval_door(width, *opening.phi_rad, bh, sigma);
        a.p = e.p;
        a.slack_m = e.mean_gap - target;
        head = line(e, opening.words);
    }
    else if (it->phi_rad.has_value() and *it->phi_rad >= 0.f)
    {
        const auto e = eval_door(width, *it->phi_rad, bh, sigma);
        a.p = e.p;
        a.slack_m = e.mean_gap - target;
        head = line(e, std::format("as it is now ({}°)", deg1(*it->phi_rad)));
    }
    else
    {
        const auto shut = eval_door(width, 0.f, bh, sigma), wide = eval_door(width, kPi / 2, bh, sigma);
        head = std::format("I don't know how open {} is now; shut: P(fit) = {:.2f}, slack {:+.0f} cm; wide open: P(fit) = {:.2f}, slack {:+.0f} cm",
                           it->name, shut.p, (shut.mean_gap - target) * 100.f, wide.p, (wide.mean_gap - target) * 100.f);
    }
    a.text = std::format("{}{}. For P = 0.5, {}. Hinge side and swing are not known, so I averaged the four combinations. Caveat: {}.",
                         head, sig_note, star, kDoorCaveat);
    return a;
}

// ── between two referents ──
namespace fit_detail
{
using Pt = Eigen::Vector2f;

/// A convex footprint: a box (4 corners) or a wall segment (2 points).
struct Referent
{
    std::string name;
    std::vector<Pt> pts;
    bool is_wall = false;
    std::optional<float> sigma;
};

inline float dist_point_segment(const Pt& p, const Pt& a, const Pt& b)
{
    const Pt ab = b - a;
    const float L2 = ab.squaredNorm();
    const float t = L2 > 0.f ? std::clamp((p - a).dot(ab) / L2, 0.f, 1.f) : 0.f;
    return (p - (a + t * ab)).norm();
}
inline bool segments_cross(const Pt& a, const Pt& b, const Pt& c, const Pt& d)
{
    const auto cr = [](const Pt& u, const Pt& v) { return u.x() * v.y() - u.y() * v.x(); };
    const float d1 = cr(b - a, c - a), d2 = cr(b - a, d - a), d3 = cr(d - c, a - c), d4 = cr(d - c, b - c);
    return ((d1 > 0.f) != (d2 > 0.f)) and ((d3 > 0.f) != (d4 > 0.f)) and d1 != d2 and d3 != d4;
}
inline float dist_segments(const Pt& a, const Pt& b, const Pt& c, const Pt& d)
{
    if (segments_cross(a, b, c, d)) return 0.f;
    return std::min({dist_point_segment(a, c, d), dist_point_segment(b, c, d), dist_point_segment(c, a, b), dist_point_segment(d, a, b)});
}
inline bool inside_convex(const Pt& p, const std::vector<Pt>& poly)
{
    if (poly.size() < 3) return false;
    bool pos = false, neg = false;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const Pt e = poly[(i + 1) % poly.size()] - poly[i], v = p - poly[i];
        const float c = e.x() * v.y() - e.y() * v.x();
        pos = pos or c > 0.f;
        neg = neg or c < 0.f;
    }
    return (pos or neg) and not (pos and neg);   // a footprint with no area (all crosses 0) contains nothing
}
inline std::vector<std::pair<Pt, Pt>> edges_of(const std::vector<Pt>& p)
{
    std::vector<std::pair<Pt, Pt>> e;
    if (p.size() == 1) e.emplace_back(p[0], p[0]);
    else if (p.size() == 2) e.emplace_back(p[0], p[1]);
    else for (std::size_t i = 0; i < p.size(); ++i) e.emplace_back(p[i], p[(i + 1) % p.size()]);
    return e;
}
/// Minimum distance between two convex footprints (0 when they overlap).
[[nodiscard]] inline float footprint_distance(const std::vector<Pt>& a, const std::vector<Pt>& b)
{
    if (not a.empty() and inside_convex(a[0], b)) return 0.f;
    if (not b.empty() and inside_convex(b[0], a)) return 0.f;
    float best = std::numeric_limits<float>::infinity();
    for (const auto& [p, q] : edges_of(a))
        for (const auto& [r, s] : edges_of(b)) best = std::min(best, dist_segments(p, q, r, s));
    return best;
}
/// An oriented box (x, y, yaw, w along local x, d along local y), the can_see.h obstacle convention. Missing sizes are 0.
inline std::vector<Pt> box_corners(float x, float y, float yaw, float w, float d)
{
    const Pt c(x, y), ux(std::cos(yaw), std::sin(yaw)), uy(-std::sin(yaw), std::cos(yaw));
    return {c + 0.5f * w * ux + 0.5f * d * uy, c - 0.5f * w * ux + 0.5f * d * uy,
            c - 0.5f * w * ux - 0.5f * d * uy, c + 0.5f * w * ux - 0.5f * d * uy};
}
}   // namespace fit_detail

[[nodiscard]] inline FitAnswer answer_fit_between(const ImaginedWorld& w, std::string_view a_name, std::string_view b_name)
{
    using namespace fit_detail;
    FitAnswer ans;
    constexpr std::string_view kCaveat = "a prediction from what I believe; while driving my local controller decides from the live LiDAR "
                                         "and may meet legs, cables or clutter I have not modelled";
    if (not w.body_half_width.has_value()) { ans.text = "I have no model of my body width"; return ans; }
    const float bh = *w.body_half_width;

    struct Side { std::optional<Referent> ref; std::string label, note; bool ignored = false; std::string err; };
    const auto resolve = [&](std::string_view raw) -> Side
    {
        Side s;
        const std::string n = imagine_detail::norm_name(raw);
        s.label = std::string(raw);
        if (n == "wall" or n == "pared" or n == "the_wall") { s.ref = Referent{"the wall", {}, true, w.robot_sigma_m}; return s; }
        const SeeObject* pick = nullptr;
        for (const auto& o : w.scene.objects)
            if (imagine_detail::norm_name(o.name) == n) { pick = &o; break; }
        if (pick == nullptr)
        {
            const std::string cls = canonical_concept(n);
            std::vector<const SeeObject*> same;
            for (const auto& o : w.scene.objects) if (lower_ascii(o.subtype) == cls) same.push_back(&o);
            if (same.size() == 1) pick = same.front();
            else if (same.size() > 1) { s.err = std::format("there is more than one {}; which one do you mean?", raw); return s; }
        }
        if (pick == nullptr) { s.err = std::format("I don't know an object called {}", raw); return s; }
        s.label = pick->name;
        if (pick->h.has_value())
        {
            const float base = object_base_z(*pick, *pick->h), top = base + *pick->h;
            if (not (base < kRobotHeightM and top > 0.f)) { s.ignored = true; }
        }
        else s.note = std::format("{}'s height is unknown, so I treated it as blocking", pick->name);
        if (not pick->w.has_value() and not pick->d.has_value())
            s.note += std::string(s.note.empty() ? "" : "; ") + std::format("{}'s size is unknown, so I treated it as a point at its centre", pick->name);
        else if (not pick->w.has_value() or not pick->d.has_value())
            s.note += std::string(s.note.empty() ? "" : "; ") + std::format("{}'s size is only partly known, so I used what is known", pick->name);
        std::optional<float> sg;
        if (const auto f = w.object_sigma_m.find(pick->name); f != w.object_sigma_m.end()) sg = f->second;
        s.ref = Referent{pick->name, box_corners(pick->x, pick->y, pick->yaw, pick->w.value_or(0.f), pick->d.value_or(0.f)), false, sg};
        return s;
    };

    Side A = resolve(a_name), B = resolve(b_name);
    if (not A.err.empty()) { ans.text = A.err; return ans; }
    if (not B.err.empty()) { ans.text = B.err; return ans; }
    if (A.ref->is_wall and B.ref->is_wall) { ans.text = "between two walls is the room, ask me about a door"; return ans; }

    // the wall side: the room-polygon segment nearest the other referent
    for (auto [self, other] : {std::pair{&A, &B}, std::pair{&B, &A}})
        if (self->ref->is_wall)
        {
            const auto& poly = w.scene.polygon;
            if (poly.size() < 2) { ans.text = "I have no room outline, so I cannot place the wall"; return ans; }
            float best = std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < poly.size(); ++i)
            {
                const Pt p = poly[i], q = poly[(i + 1) % poly.size()];
                const float d = footprint_distance({p, q}, other->ref->pts);
                if (d < best) { best = d; self->ref->pts = {p, q}; }
            }
        }

    const float gap = footprint_distance(A.ref->pts, B.ref->pts);
    if (A.ignored or B.ignored)
    {
        const Side& up = A.ignored ? A : B;
        const Side& other = A.ignored ? B : A;
        ans.p = 1.f;
        ans.slack_m = gap - 2.f * bh;
        std::string t = std::format("{} sits entirely outside my body's height band (0 to {:.2f} m), so it does not block. ", up.label, kRobotHeightM);
        if (other.ignored) t += std::format("Neither side is at my height: nothing at my height blocks there. P(fit) = 1.00.");
        else t += std::format("Their footprints are {:.0f} cm apart (I am {:.0f} cm wide), but nothing at my height blocks there. P(fit) = 1.00.", gap * 100.f, 2.f * bh * 100.f);
        for (const Side* sd : {&A, &B}) if (not sd->note.empty()) t += " Note: " + sd->note + ".";
        ans.text = t + std::format(" Caveat: {}.", kCaveat);
        return ans;
    }
    std::optional<float> sg;
    if (A.ref->sigma or B.ref->sigma) sg = std::hypot(A.ref->sigma.value_or(0.f), B.ref->sigma.value_or(0.f));
    const float sigma = sg.value_or(0.f);
    ans.p = rc::anchor_p_fit(gap, bh, 0.f, sigma);
    ans.slack_m = gap - 2.f * bh;

    // who would move: a wall cannot; between two objects the one with the larger sigma, else the first
    const Referent* mover = A.ref->is_wall ? &*B.ref : B.ref->is_wall ? &*A.ref
                          : (B.ref->sigma.value_or(0.f) > A.ref->sigma.value_or(0.f) ? &*B.ref : &*A.ref);
    std::string text = std::format("The gap between {} and {} is {:.0f} cm and I am {:.0f} cm wide: slack {:+.0f} cm, P(fit) = {:.2f}.",
                                   A.label, B.label, gap * 100.f, 2.f * bh * 100.f, ans.slack_m * 100.f, *ans.p);
    if (not sg.has_value()) text += " I treated the geometry as exact (no position uncertainties known).";
    if (*ans.p < 0.5f)
        text += std::format(" To make it 50/50, {} would have to move away by about {:.0f} cm.", mover->name, -ans.slack_m * 100.f);
    for (const Side* s : {&A, &B}) if (not s->note.empty()) text += " Note: " + s->note + ".";
    text += std::format(" Caveat: {}.", kCaveat);
    ans.text = std::move(text);
    return ans;
}

}   // namespace rc::chat::imagine
