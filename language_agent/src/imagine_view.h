#pragma once
/*
 * imagine_view.h — "show me how you imagine a chair": sample a concept from its card's generative prior, render each
 * sample through the ZED and the two LiDAR models, average the evaluator over the prior (Monte-Carlo P(detect)), write
 * the canvas data (imagined_io.h) and word the answer. Also the Monte-Carlo can_see for a HYPOTHETICAL target.
 *
 * Pure like can_see.h (Eigen + the shared card code; imagined_io.h adds Qt Gui for the PNG). The graph is read by
 * perception_world.h on the main thread; everything here works on the SeeScene.
 *
 * Rulings (task 8 brief):
 *   · `where` sets the Context: a door or object name = `at` it, the word "wall"/"pared" forces against_wall, empty =
 *     the card's own mixture. `from` is as in can_see (resolve_from).
 *   · The target must not also be an obstacle: the named same-class instance at the `at` place (and any same-class
 *     instance sitting on that point) is left out of the World, and the answer says so.
 *   · The robot's own body (kRobotRadiusM, as can_see) at the viewpoint is an obstacle of the SAMPLER only (not of the
 *     renderers): the robot cannot stand inside the thing it imagines. A viewpoint that lies in the 2-D footprint of a
 *     drawn sample's parts is reported ("the viewpoint is inside the imagined <concept>").
 *   · n samples are drawn for display; the P(detect) is a Monte-Carlo over max(n, kMcSamples) draws of the SAME
 *     distribution as the pictures (sizes AND place AND heading, physical rejection), fixed place or not.
 *   · Doors sit at their most probable state (R12, world_for's default), and the answer names each assumption.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <map>
#include <string>
#include <vector>

#include "can_see.h"
#include "imagined_io.h"

namespace rc::chat
{

inline constexpr int kMcSamples = 32;      ///< draws behind a P(detect) mean ± sd
inline constexpr int kDisplaySamples = 8;  ///< default samples drawn on the canvas
inline constexpr int kMaxDisplaySamples = 16;
inline constexpr std::uint64_t kHypotheticalSeed = 1;   ///< can_see is a question, not a picture: a fixed seed, so the same question gets the same answer

struct ImagineRequest
{
    std::string concept_, where, from, sensor = "both";   ///< sensor: zed | lidar | both
    int n = kDisplaySamples;
};

/// Where the imagined thing stands: the sampler's Context, the words for it, and the graph instance that IS this place.
struct ImaginePlace
{
    rc::card::Context ctx;
    std::string words;                 ///< "against a wall", "at door_1", "by table_1", "wherever such things usually stand"
    std::vector<std::string> exclude;  ///< graph instances left out of the World (they would occlude their own imagined self)
};

[[nodiscard]] inline bool names_wall(const std::string& lower)
{
    return lower.find("wall") != std::string::npos or lower.find("pared") != std::string::npos;
}

/// `where` → a place. A door or object name is `at` it (a door: one robot radius inside, as can_see's "from door"; an
/// object of ANOTHER class: its standoff; an object of the SAME class: its own spot, and that instance is excluded).
/// "wall"/"pared" forces against_wall; empty is the card's mixture; anything else asks.
[[nodiscard]] inline std::expected<ImaginePlace, SeeError> resolve_place(const SeeScene& s, const std::string& cls,
                                                                              const std::string& where)
{
    ImaginePlace pl;
    const std::string w = lower_ascii(where);
    // "here" / "in this room": the room-based mixture (free or against a wall). An EMPTY `where` is the canonical view
    // (imagine_view handles it before this): the robot's idea of the thing, not a thing placed in its room.
    if (w.empty() or w == "here" or w == "in this room" or w == "in the room" or w == "aqui" or w == "aqu\u00ed" or w == "en esta habitacion")
    { pl.words = "wherever such things usually stand"; return pl; }
    for (const auto& d : s.doors)
        if (lower_ascii(d.name) == w)
        {
            pl.ctx.at = see_detail::door_viewpoint(s, d);
            pl.words = "at " + d.name;
            return pl;
        }
    for (const auto& o : s.objects)
        if (lower_ascii(o.name) == w)
        {
            if (canonical_concept(o.subtype) == cls)
            {
                pl.ctx.at = Eigen::Vector2f(o.x, o.y);
                pl.exclude.push_back(o.name);
                pl.words = "where " + o.name + " is";
            }
            else
            {
                pl.ctx.at = o.standoff.value_or(Eigen::Vector2f(o.x, o.y));
                pl.words = "by " + o.name;
            }
            return pl;
        }
    if (names_wall(w)) { pl.ctx.force_context = "against_wall"; pl.words = "against a wall"; return pl; }
    return std::unexpected(see_ask("ASK: which place do you mean? Name a door or an object, or say against the wall."));
}

/// A hypothetical target: no instance of the concept exists and `where` names a door, or an object of a KNOWN, different
/// class. False whenever `where` names an object whose class is empty or the concept's own (the old can_see path keeps
/// resolving named instances), and false while an instance of the concept exists (the old path then evaluates the named
/// instance by its own class).
[[nodiscard]] inline bool is_hypothetical(const SeeScene& s, const std::string& what, const std::string& where)
{
    if (what.empty() or where.empty()) return false;
    const std::string cls = canonical_concept(what), w = lower_ascii(where);
    for (const auto& o : s.objects)
        if (canonical_concept(o.subtype) == cls or lower_ascii(o.name) == cls) return false;
    if (std::ranges::any_of(s.doors, [&](const SeeDoor& d) { return lower_ascii(d.name) == w; })) return true;
    return std::ranges::any_of(s.objects, [&](const SeeObject& o)
                               { return lower_ascii(o.name) == w and not o.subtype.empty(); });
}

/// Does the place lie inside the instance's own oriented footprint (its w, d, yaw; the card's geometry when unpublished)?
[[nodiscard]] inline bool footprint_contains(const SeeObject& o, const rc::card::Card& card, const Eigen::Vector2f& p)
{
    const float w = o.w.value_or(card.geometry.w), d = o.d.value_or(card.geometry.d);
    const Eigen::Vector2f r = p - Eigen::Vector2f(o.x, o.y);
    const float c = std::cos(o.yaw), sn = std::sin(o.yaw);
    return std::abs(c * r.x() + sn * r.y()) <= 0.5f * w and std::abs(-sn * r.x() + c * r.y()) <= 0.5f * d;
}

/// The World the sampler and the renderers share: the scene without the instances in `exclude` (plus, for a fixed
/// place, any same-class instance whose own footprint contains it: the same thing seen twice).
[[nodiscard]] inline rc::card::World imagine_world(const SeeScene& s, const rc::card::Card& card, const std::string& cls,
                                                   const ImaginePlace& pl, std::vector<std::string>& excluded)
{
    excluded = pl.exclude;
    if (pl.ctx.at)
        for (const auto& o : s.objects)
            if (canonical_concept(o.subtype) == cls and footprint_contains(o, card, *pl.ctx.at) and
                std::ranges::find(excluded, o.name) == excluded.end())
                excluded.push_back(o.name);
    SeeScene t = s;
    std::erase_if(t.objects, [&](const SeeObject& o) { return std::ranges::find(excluded, o.name) != excluded.end(); });
    return world_for(t, "");
}

/// The viewpoint lies in the 2-D footprint of one of the sample's parts (any height: a seat below the camera counts).
[[nodiscard]] inline bool viewpoint_in_footprint(const rc::card::Viewpoint& vp, const rc::card::Placed& pl)
{
    return std::ranges::any_of(pl.boxes, [&](const rc::nbv::Obstacle& b)
    {
        const float c = std::cos(b.yaw), sn = std::sin(b.yaw);
        const float rx = vp.xy.x() - b.cx, ry = vp.xy.y() - b.cy;
        return std::abs(c * rx + sn * ry) <= 0.5f * b.w and std::abs(-sn * rx + c * ry) <= 0.5f * b.h;
    });
}

/// The sampler's World: the render World plus the robot's body at the viewpoint as a floor-to-ceiling box of the robot's
/// radius (kRobotRadiusM, sourced as can_see's). Not given to the renderers: the robot does not occlude its own sensors.
[[nodiscard]] inline rc::card::World with_robot_body(rc::card::World w, const rc::card::Viewpoint& vp)
{
    rc::nbv::Obstacle ob;
    ob.cx = vp.xy.x(); ob.cy = vp.xy.y(); ob.w = ob.h = 2.0f * kRobotRadiusM; ob.yaw = 0.0f; ob.z0 = 0.0f; ob.z1 = 0.0f;
    w.obstacles.emplace_back("robot", ob);
    return w;
}

/// The evaluator over the drawn places, headings AND sizes (fixed place or not). Same aggregate as evaluate_mc: mean, population sd, modal limit (ties to the lower enum value).
[[nodiscard]] inline rc::card::McResult mc_over_samples(const rc::card::Card& card, const rc::card::World& world,
                                                        const std::string& cls, const std::vector<rc::card::Sample>& samples,
                                                        const rc::card::Viewpoint& vp)
{
    rc::card::McResult out;
    std::vector<rc::card::Result> rs;
    for (const auto& sm : samples)
    {
        if (sm.parts.boxes.empty()) continue;
        const auto t = rc::card::detail::target_of(sm.parts);
        rc::card::Hypothesis h;
        h.concept_ = cls; h.x = t.cx; h.y = t.cy; h.yaw = t.yaw;
        h.size = rc::card::GeometryPrior{t.w, t.h, t.z1 - t.z0, t.z0, t.z1};
        rs.push_back(rc::card::evaluate(card, world, h, vp));
    }
    if (rs.empty()) return out;
    out.n = static_cast<int>(rs.size());
    const double m = out.n;
    double sn = 0, sr = 0, vn = 0, vr = 0;
    for (const auto& r : rs) { sn += r.p_notice; sr += r.p_recognise; }
    sn /= m; sr /= m;
    for (const auto& r : rs) { vn += (r.p_notice - sn) * (r.p_notice - sn); vr += (r.p_recognise - sr) * (r.p_recognise - sr); }
    out.p_notice_mean = float(sn); out.p_recognise_mean = float(sr);
    out.p_notice_sd = float(std::sqrt(vn / m)); out.p_recognise_sd = float(std::sqrt(vr / m));
    std::map<rc::card::Limit, int> cnt;
    for (const auto& r : rs) ++cnt[r.limit];
    int best = -1;
    for (const auto& [l, c] : cnt) if (c > best) { best = c; out.modal_limit = l; }
    return out;
}

/// How far in front of the robot the canonical instance stands: "how I would best see one" = the distance that maximises
/// the card's own mean P(recognise) over the canonical scene (empty space, the robot at the origin heading +y, the same
/// Monte-Carlo as the answer but kSearchSamples draws with a fixed seed: deterministic and cheap). Grid 0.5..8 m, step
/// 0.1 m; the NEAREST distance within one MC standard error of the maximum is taken. When the card cannot be evaluated anywhere (no camera,
/// no draw placed): kFreeSpaceDistanceM, flagged.
/// How far in front the canonical instance stands when nothing better is known (the one constant for it).
inline constexpr float kFreeSpaceDistanceM = 2.0f;
struct CanonicalDistance
{
    enum class Kind { Searched, NoSensor, NeverRecognised };
    float m = kFreeSpaceDistanceM;
    Kind kind = Kind::NoSensor;
    rc::card::Limit limit = rc::card::Limit::None;   ///< NeverRecognised: what most often stops it, over the whole search
    [[nodiscard]] bool searched() const { return kind == Kind::Searched; }
};
inline constexpr int kSearchSamples = 8;
inline constexpr std::uint64_t kSearchSeed = 1;
[[nodiscard]] inline CanonicalDistance canonical_distance(const rc::card::Card& card, const SeeScene& s, const std::string& cls)
{
    const auto vp = resolve_from(s, "");
    // No sensor at all (neither the camera nor a LiDAR rig) or no generative model: nothing to evaluate. A LiDAR-only card
    // is searched like any other: mc_over_samples evaluates whatever channels the card models.
    if (not vp.has_value() or not vp->yaw.has_value() or not card.generative
        or not (s.zed.has_value() or s.helios.has_value() or s.bpearl.has_value())) return {};
    ImaginePlace place;
    std::vector<std::string> excluded;
    const auto world = imagine_world(s, card, cls, place, excluded);
    const Eigen::Vector2f dir(std::cos(*vp->yaw), std::sin(*vp->yaw));
    struct At { float d, p; };
    std::vector<At> curve;
    std::map<rc::card::Limit, int> limits;
    float best_p = -1.0f, best_se = 0.0f;
    for (int i = 5; i <= 80; ++i)
    {
        const float d = static_cast<float>(i) * 0.1f;
        rc::card::Context ctx;
        ctx.at = vp->xy + d * dir;
        const auto set = rc::card::sample_concept(card, with_robot_body(world, *vp), ctx, kSearchSamples, kSearchSeed);
        const auto mc = mc_over_samples(card, world, cls, set.samples, *vp);
        if (mc.n == 0) continue;
        curve.push_back({d, mc.p_recognise_mean});
        ++limits[mc.modal_limit];
        if (mc.p_recognise_mean > best_p)
        {
            best_p = mc.p_recognise_mean;
            best_se = mc.p_recognise_sd / std::sqrt(static_cast<float>(mc.n));   // the search's own draws at the maximum
        }
    }
    if (curve.empty()) return {};
    CanonicalDistance out;
    if (best_p <= 0.0f)   // evaluated, and recognised nowhere: say that, with what most often stops it
    {
        out.kind = CanonicalDistance::Kind::NeverRecognised;
        int most = -1;
        for (const auto& [l, c] : limits) if (c > most) { most = c; out.limit = l; }
        return out;
    }
    // The NEAREST distance statistically as good as the maximum: within one Monte-Carlo standard error of it (exact ties
    // when the SE is 0). On a plateau the maximum is a noise pick; the nearest plateau point is the one worth reporting.
    for (const auto& at : curve)
        if (at.p >= best_p - best_se) { out.m = at.d; out.kind = CanonicalDistance::Kind::Searched; return out; }
    return out;
}

/// "0.62 ± 0.11"
[[nodiscard]] inline std::string pm(float mean, float sd)
{
    return std::format("{} ± {}", see_detail::p2(mean), std::format("{:.2f}", std::max(sd, 0.0f)));
}

/// Everything the imagine_view answer words, as plain data.
struct ImagineFacts
{
    std::string cls, place_words, from_label;
    int shown = 0;                          ///< samples on the canvas
    int accepted = 0, drawn = 0;            ///< of the whole MC draw (the sampler's acceptance)
    int rejected = 0, outside_polygon = 0;
    int inside_count = 0;                   ///< shown samples whose render has the camera inside a part
    std::vector<std::string> excluded;
    std::vector<std::string> assumed;       ///< door assumptions ("assumption: " notes without the prefix), as can_see words them
    bool room_height_known = false;
    float room_height = 2.5f;
    std::uint64_t seed = 0;
    std::string calibration;                ///< the card's, for the "isn't calibrated yet" note
    std::string sensor = "both";
    bool canonical = false;                 ///< the canonical view: a typical instance on the floor in empty space, no room
    float distance_m = 0;                   ///< canonical: how far in front of the robot
    CanonicalDistance::Kind distance_kind = CanonicalDistance::Kind::Searched;
    rc::card::Limit never_limit = rc::card::Limit::None;   ///< NeverRecognised: what most often stops it
};

/// " Assumed: door_1: ...; door_2: ...." from the doors' notes; empty when nothing was assumed.
[[nodiscard]] inline std::string assumed_suffix(const std::vector<std::string>& assumed)
{
    std::string s;
    for (const auto& a : assumed) s += (s.empty() ? "" : "; ") + a;
    return s.empty() ? std::string{} : " Assumed: " + s + ".";
}

/// Each uncertain (or widthless) door's note, without the "assumption: " prefix: world_for put it at its most probable state.
[[nodiscard]] inline std::vector<std::string> door_assumptions(const SeeScene& s)
{
    std::vector<std::string> out;
    for (const auto& d : s.doors)
        if (auto n = door_probable_note(d); not n.empty()) out.push_back(n.starts_with("assumption: ") ? n.substr(12) : n);
    return out;
}

/// The canonical answer's note on the distance: nothing when it was searched; why 2 m otherwise.
[[nodiscard]] inline std::string canonical_note(const ImagineFacts& f)
{
    using K = CanonicalDistance::Kind;
    if (f.distance_kind == K::NeverRecognised)
        return std::format(" (I would not recognise one at any distance from 0.5 to 8 m; what most often stops it: {}; I show it at {:.0f} m)",
                           see_detail::limit_words(f.never_limit, ""), kFreeSpaceDistanceM);
    if (f.distance_kind == K::NoSensor) return " (I have no sensor to evaluate my detector with, so I assumed that distance)";
    return "";
}

[[nodiscard]] inline std::string answer_imagine_view(const ImagineFacts& f, const rc::card::McResult& mc)
{
    using namespace see_detail;
    const std::string pl = plural(f.cls);
    if (f.shown == 0)
        return std::format("I could not place any imagined {} {}: of {} draws, {} fell inside a wall or another object "
                           "(my own body at the viewpoint counts as one: I cannot stand inside what I imagine) and {} outside the "
                           "room. (seed {}){}", f.cls, f.place_words, f.drawn, f.rejected, f.outside_polygon, f.seed, assumed_suffix(f.assumed));
    const std::string at = f.from_label.empty() ? "from here" : "from " + f.from_label;
    std::string out;
    if (f.canonical)
        out = std::format("I imagined {} typical {} standing on the floor {:.1f} m in front of me{}. I would notice that something "
                          "is there with P {}, and recognise it as a {} with P {} ({}).",
                          f.shown, f.shown == 1 ? f.cls : pl, f.distance_m,
                          canonical_note(f),
                          pm(mc.p_notice_mean, mc.p_notice_sd), f.cls, pm(mc.p_recognise_mean, mc.p_recognise_sd), likelihood(mc.p_recognise_mean));
    else
        out = std::format("I imagined {} {} {}, seen {}. I would notice that something is there with P {}, and "
                          "recognise it as a {} with P {} ({}).",
                          f.shown, f.shown == 1 ? f.cls : pl, f.place_words, at, pm(mc.p_notice_mean, mc.p_notice_sd),
                          f.cls, pm(mc.p_recognise_mean, mc.p_recognise_sd), likelihood(mc.p_recognise_mean));
    if (mc.modal_limit != rc::card::Limit::None)
        out += std::format(" What most often limits it: {}.", limit_words(mc.modal_limit, ""));
    if (f.inside_count > 0)
        out += std::format(" In {} of the {} drawn the viewpoint is inside the imagined {}, so the camera sees nothing of it.",
                           f.inside_count, f.shown, f.cls);
    const double rate = f.drawn > 0 ? double(f.accepted) / double(f.drawn) : 1.0;
    if (rate < 0.8)
        out += std::format(" Only {:.0f}% of my draws were physically possible there ({} of {}; {} inside a wall or an object, "
                           "{} outside the room).", 100.0 * rate, f.accepted, f.drawn, f.rejected, f.outside_polygon);
    if (not f.excluded.empty())
    {
        std::string names;
        for (const auto& n : f.excluded) names += (names.empty() ? "" : ", ") + n;
        out += std::format(" I left {} out of the scene, as it is the {} at that place.", names, f.cls);
    }
    out += assumed_suffix(f.assumed);
    if (not f.room_height_known)
        out += std::format(" Room height {:.1f} m assumed (the room does not publish one).", f.room_height);
    out += std::format(" (P over {} draws of the {} prior; seed {}.)", mc.n, f.cls, f.seed);
    if (f.calibration == "prior")
        out += std::format(" (My detector for {} isn't calibrated yet, so treat the numbers as rough.)", pl);
    return out;
}

/// The answer to can_see for a concept with no instance, placed at a named place: Monte-Carlo over the card's size prior.
[[nodiscard]] inline std::string answer_can_see_hypothetical(const rc::card::Card& card, const std::string& cls,
                                                             const std::string& place_words, const std::string& from_label,
                                                             const rc::card::McResult& mc, bool room_height_known, float room_height,
                                                             std::uint64_t seed = 1, const std::vector<std::string>& assumed = {})
{
    using namespace see_detail;
    const std::string at = from_label.empty() ? "from here" : "from " + from_label;
    if (mc.n == 0)   // not one physically possible place: never a P for a chair the robot would be standing in
        return std::format("I cannot judge a {} {} {}: there is no free place for one there (a wall, another object or my own "
                           "body at the viewpoint is in the way).{}{}", cls, place_words, at, assumed_suffix(assumed), calibration_note(card));
    if (mc.modal_limit == rc::card::Limit::Inside)
        return std::format("I cannot judge a {} {} {}: that spot is not somewhere I can look from.{}", cls, place_words, at,
                           calibration_note(card));
    std::string out = std::format("I have not seen a {} there, so this is for an imagined one {}, {}: I would notice that "
                                  "something is there with P {} and recognise it as a {} with P {} ({}).",
                                  cls, place_words, at, pm(mc.p_notice_mean, mc.p_notice_sd), cls,
                                  pm(mc.p_recognise_mean, mc.p_recognise_sd), likelihood(mc.p_recognise_mean));
    if (mc.modal_limit != rc::card::Limit::None)
        out += std::format(" What most often limits it: {}.", limit_words(mc.modal_limit, ""));
    out += assumed_suffix(assumed);
    if (not room_height_known) out += std::format(" Room height {:.1f} m assumed.", room_height);
    out += std::format(" (n={} imagined {}, sizes and place drawn from the prior; seed {}.)", mc.n, plural(cls), seed);
    out += calibration_note(card);
    return out;
}

/// can_see for a hypothetical target (is_hypothetical): the SAME draw as imagine_view's pictures (sizes, heading, physical
/// rejection, the robot's body at the viewpoint an obstacle) at the named place, n = kMcSamples.
[[nodiscard]] inline std::string can_see_hypothetical_reply(const rc::card::Card& card, const SeeScene& s, const std::string& what,
                                                            const std::string& where, const std::string& from)
{
    const std::string cls = canonical_concept(what);
    const auto place = resolve_place(s, cls, where);
    if (not place.has_value()) return place.error();
    const auto vp = resolve_from(s, from);
    if (not vp.has_value()) return vp.error();
    std::vector<std::string> excluded;
    const auto world = imagine_world(s, card, cls, *place, excluded);
    const auto set = rc::card::sample_concept(card, with_robot_body(world, *vp), place->ctx, kMcSamples, kHypotheticalSeed);
    const auto mc = mc_over_samples(card, world, cls, set.samples, *vp);
    return answer_can_see_hypothetical(card, cls, place->words, from, mc, s.room_height_known, s.room_height, kHypotheticalSeed,
                                       door_assumptions(s));
}

/// The refusal for a place that needs the room when there is none. A refusal is a limitation the person must hear as it is
/// (ImagineOutcome::refusal), never a model paraphrase.
inline constexpr const char* kNoRoomRefusal = "I do not know which room I am in, so I cannot imagine it here.";
/// How far in front of the robot a room-less imagined thing stands (the cards carry no typical viewing distance).
struct ImagineOutcome
{
    std::string text;
    bool refusal = false;   ///< `text` is a refusal / limitation: the robot's reply is this text unchanged
    bool drawn = false;     ///< the canvas data were written: the caller draws stock/imagined.py
    std::string caption;
};

/// The whole imagine_view path. Writes <run_dir>/imagined.json + imagined_zed.png when it placed anything.
[[nodiscard]] inline ImagineOutcome imagine_view(const rc::card::Card& card, const SeeScene& scene_in, const ImagineRequest& rq,
                                                 std::uint64_t seed, const std::filesystem::path& run_dir)
{
    ImagineOutcome out;
    // An EMPTY `where` is the canonical view: the robot's idea of the thing, a typical instance on the floor in empty space in
    // front of its real sensors (no room polygon, walls, doors or other objects), whether or not a room is known. Only a
    // named place (a wall, a door, an object, "here") puts it in the room, and then the room is needed.
    const bool canonical = rq.where.empty();
    SeeScene s = scene_in;
    if (not canonical and scene_in.room.empty()) { out.text = kNoRoomRefusal; out.refusal = true; return out; }
    if (canonical)
    {
        s.robot_xy = Eigen::Vector2f(0.0f, 0.0f);
        s.robot_yaw = 0.0f;
        s.polygon.clear(); s.doors.clear(); s.objects.clear();
        s.room_height_known = true;
    }
    const std::string cls = canonical_concept(rq.concept_);
    if (not card.generative)
    {
        out.text = std::format("My model of {} does not say what one looks like, so I cannot imagine it.", see_detail::plural(cls));
        out.refusal = true;
        return out;
    }
    auto place = resolve_place(s, cls, rq.where);
    if (not place.has_value()) { out.text = place.error(); out.refusal = not place.error().ask; return out; }
    const auto vp = resolve_from(s, canonical ? std::string{} : rq.from);   // the canonical view is from the robot itself
    if (not vp.has_value()) { out.text = vp.error(); out.refusal = not vp.error().ask; return out; }
    CanonicalDistance cd;
    if (canonical)
    {
        cd = canonical_distance(card, s, cls);
        place->ctx.at = vp->xy + cd.m * Eigen::Vector2f(std::cos(*vp->yaw), std::sin(*vp->yaw));
        place->words = std::format("standing on the floor {:.1f} m in front of me", cd.m);
    }
    const std::string sensor = lower_ascii(rq.sensor);
    const bool want_zed = sensor != "lidar", want_lidar = sensor != "zed";
    const int n = std::clamp(rq.n, 1, kMaxDisplaySamples);

    ImagineFacts f;
    f.cls = cls; f.place_words = place->words; f.from_label = rq.from;
    f.room_height_known = s.room_height_known; f.room_height = s.room_height;
    f.canonical = canonical; f.distance_m = cd.m; f.distance_kind = cd.kind; f.never_limit = cd.limit;
    f.seed = seed; f.calibration = card.calibration; f.sensor = sensor; f.assumed = door_assumptions(s);
    const auto world = imagine_world(s, card, cls, *place, f.excluded);

    // one draw behind both the canvas (its first n) and the P(detect)
    auto set = rc::card::sample_concept(card, with_robot_body(world, *vp), place->ctx, std::max(n, kMcSamples), seed);
    f.drawn = set.drawn; f.accepted = static_cast<int>(set.samples.size());
    f.rejected = set.rejected; f.outside_polygon = set.outside_polygon;
    const rc::card::SampleSet whole = set;
    if (static_cast<int>(set.samples.size()) > n) set.samples.resize(static_cast<std::size_t>(n));
    f.shown = static_cast<int>(set.samples.size());

    const rc::card::McResult mc = mc_over_samples(card, world, cls, whole.samples, *vp);   // fixed place or not: the pictures' distribution

    std::vector<rc::card::ImaginedImage> images;
    std::array<std::vector<rc::card::ImaginedScan>, 2> scans;
    for (std::size_t i = 0; i < set.samples.size(); ++i)
    {
        const auto& sm = set.samples[i];
        if (viewpoint_in_footprint(*vp, sm.parts)) ++f.inside_count;
        if (want_zed) images.push_back(rc::card::render_zed(world, *vp, sm.parts, card));
        if (want_lidar)
        {
            scans[0].push_back(rc::card::render_lidar(world, *vp, world.helios ? &*world.helios : nullptr, sm.parts, seed + 2 * i, card, cls));
            scans[1].push_back(rc::card::render_lidar(world, *vp, world.bpearl ? &*world.bpearl : nullptr, sm.parts, seed + 2 * i + 1, card, cls));
        }
    }
    if (not s.room_height_known) set.notes.push_back(std::format("room height {:.1f} m assumed", s.room_height));
    out.text = answer_imagine_view(f, mc);
    for (const auto& n : s.notes) out.text += " (" + n + ")";
    if (f.shown == 0) return out;

    out.caption = std::format("{} imagined {} {}: P(notice) {}, P(recognise) {}", f.shown, f.shown == 1 ? cls : see_detail::plural(cls),
                              place->words, pm(mc.p_notice_mean, mc.p_notice_sd), pm(mc.p_recognise_mean, mc.p_recognise_sd));
    out.drawn = write_imagined(run_dir, set, images, scans, mc, out.caption, {"helios", "bpearl"});
    if (not out.drawn) out.text += " (I could not write the picture's data.)";
    return out;
}

}   // namespace rc::chat
