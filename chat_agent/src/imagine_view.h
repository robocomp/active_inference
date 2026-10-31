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
 *   · The viewpoint inside the imagined object is reported here, from render_zed's own Inside test.
 *   · n samples are drawn for display; the P(detect) is a Monte-Carlo over kMcSamples draws (evaluate_mc when the place
 *     is fixed; the evaluator on each drawn place otherwise, same aggregate).
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
[[nodiscard]] inline std::expected<ImaginePlace, std::string> resolve_place(const SeeScene& s, const std::string& cls,
                                                                              const std::string& where)
{
    ImaginePlace pl;
    const std::string w = lower_ascii(where);
    if (w.empty()) { pl.words = "wherever such things usually stand"; return pl; }
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
    return std::unexpected("ASK: which place do you mean? Name a door or an object, or say against the wall.");
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

/// render_zed's own Inside test: the camera (at its mount height) lies within one of the sample's parts.
[[nodiscard]] inline bool camera_inside_part(const rc::card::World& w, const rc::card::Viewpoint& vp, const rc::card::Placed& pl)
{
    if (not w.zed) return false;
    const float mount = w.zed_mount_z > 1e-3f ? w.zed_mount_z : w.zed->height_m;
    for (const auto& b : pl.boxes)
    {
        bool in = false;
        rc::card::detail::slab_hit(rc::card::detail::ray_box(b), vp.xy.x(), vp.xy.y(), mount, 1.0f, 0.0f, 0.0f, &in);
        if (in) return true;
    }
    return false;
}

/// The evaluator over the drawn places AND sizes (a fixed place goes through evaluate_mc instead). Same aggregate as
/// evaluate_mc: mean, population sd, modal limit (ties to the lower enum value).
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
    bool room_height_known = false;
    float room_height = 2.5f;
    std::uint64_t seed = 0;
    std::string calibration;                ///< the card's, for the "isn't calibrated yet" note
    std::string sensor = "both";
};

[[nodiscard]] inline std::string answer_imagine_view(const ImagineFacts& f, const rc::card::McResult& mc)
{
    using namespace see_detail;
    const std::string pl = plural(f.cls);
    if (f.shown == 0)
        return std::format("I could not place any imagined {} {}: of {} draws, {} fell inside a wall or another object and "
                           "{} outside the room. (seed {})", f.cls, f.place_words, f.drawn, f.rejected, f.outside_polygon, f.seed);
    const std::string at = f.from_label.empty() ? "from here" : "from " + f.from_label;
    std::string out = std::format("I imagined {} {} {}, seen {}. I would notice that something is there with P {}, and "
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
                                                             std::uint64_t seed = 1)
{
    using namespace see_detail;
    const std::string at = from_label.empty() ? "from here" : "from " + from_label;
    if (mc.modal_limit == rc::card::Limit::Inside)
        return std::format("I cannot judge a {} {} {}: that spot is not somewhere I can look from.{}", cls, place_words, at,
                           calibration_note(card));
    std::string out = std::format("I have not seen a {} there, so this is for an imagined one {}, {}: I would notice that "
                                  "something is there with P {} and recognise it as a {} with P {} ({}).",
                                  cls, place_words, at, pm(mc.p_notice_mean, mc.p_notice_sd), cls,
                                  pm(mc.p_recognise_mean, mc.p_recognise_sd), likelihood(mc.p_recognise_mean));
    if (mc.modal_limit != rc::card::Limit::None)
        out += std::format(" What most often limits it: {}.", limit_words(mc.modal_limit, ""));
    if (not room_height_known) out += std::format(" Room height {:.1f} m assumed.", room_height);
    out += std::format(" (n={} size samples from the {} prior; seed {}.)", mc.n, cls, seed);
    out += calibration_note(card);
    return out;
}

/// can_see for a hypothetical target (is_hypothetical): the card's size prior at the named place, n = kMcSamples.
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
    rc::card::Hypothesis base;
    base.concept_ = cls;
    base.x = place->ctx.at->x(); base.y = place->ctx.at->y(); base.yaw = 0.0f;
    const auto mc = rc::card::evaluate_mc(card, world, base, *vp, kMcSamples, kHypotheticalSeed);
    return answer_can_see_hypothetical(card, cls, place->words, from, mc, s.room_height_known, s.room_height, kHypotheticalSeed);
}

struct ImagineOutcome
{
    std::string text;
    bool drawn = false;     ///< the canvas data were written: the caller draws stock/imagined.py
    std::string caption;
};

/// The whole imagine_view path. Writes <run_dir>/imagined.json + imagined_zed.png when it placed anything.
[[nodiscard]] inline ImagineOutcome imagine_view(const rc::card::Card& card, const SeeScene& s, const ImagineRequest& rq,
                                                 std::uint64_t seed, const std::filesystem::path& run_dir)
{
    ImagineOutcome out;
    const std::string cls = canonical_concept(rq.concept_);
    if (not card.generative)
    {
        out.text = std::format("My model of {} does not say what one looks like, so I cannot imagine it.", see_detail::plural(cls));
        return out;
    }
    const auto place = resolve_place(s, cls, rq.where);
    if (not place.has_value()) { out.text = place.error(); return out; }
    const auto vp = resolve_from(s, rq.from);
    if (not vp.has_value()) { out.text = vp.error(); return out; }
    const std::string sensor = lower_ascii(rq.sensor);
    const bool want_zed = sensor != "lidar", want_lidar = sensor != "zed";
    const int n = std::clamp(rq.n, 1, kMaxDisplaySamples);

    ImagineFacts f;
    f.cls = cls; f.place_words = place->words; f.from_label = rq.from;
    f.room_height_known = s.room_height_known; f.room_height = s.room_height;
    f.seed = seed; f.calibration = card.calibration; f.sensor = sensor;
    const auto world = imagine_world(s, card, cls, *place, f.excluded);

    // one draw behind both the canvas (its first n) and the P(detect)
    auto set = rc::card::sample_concept(card, world, place->ctx, std::max(n, kMcSamples), seed);
    f.drawn = set.drawn; f.accepted = static_cast<int>(set.samples.size());
    f.rejected = set.rejected; f.outside_polygon = set.outside_polygon;
    const rc::card::SampleSet whole = set;
    if (static_cast<int>(set.samples.size()) > n) set.samples.resize(static_cast<std::size_t>(n));
    f.shown = static_cast<int>(set.samples.size());

    rc::card::McResult mc;
    if (place->ctx.at)
    {
        rc::card::Hypothesis base;
        base.concept_ = cls; base.x = place->ctx.at->x(); base.y = place->ctx.at->y();
        mc = rc::card::evaluate_mc(card, world, base, *vp, kMcSamples, seed);
    }
    else mc = mc_over_samples(card, world, cls, whole.samples, *vp);

    std::vector<rc::card::ImaginedImage> images;
    std::array<std::vector<rc::card::ImaginedScan>, 2> scans;
    for (std::size_t i = 0; i < set.samples.size(); ++i)
    {
        const auto& sm = set.samples[i];
        if (camera_inside_part(world, *vp, sm.parts)) ++f.inside_count;
        if (want_zed) images.push_back(rc::card::render_zed(world, *vp, sm.parts, card));
        if (want_lidar)
        {
            scans[0].push_back(rc::card::render_lidar(world, *vp, world.helios ? &*world.helios : nullptr, sm.parts, seed + 2 * i, card, cls));
            scans[1].push_back(rc::card::render_lidar(world, *vp, world.bpearl ? &*world.bpearl : nullptr, sm.parts, seed + 2 * i + 1, card, cls));
        }
    }
    if (not s.room_height_known) set.notes.push_back(std::format("room height {:.1f} m assumed", s.room_height));
    out.text = answer_imagine_view(f, mc);
    if (f.shown == 0) return out;

    out.caption = std::format("{} imagined {} {}: P(notice) {}, P(recognise) {}", f.shown, f.shown == 1 ? cls : see_detail::plural(cls),
                              place->words, pm(mc.p_notice_mean, mc.p_notice_sd), pm(mc.p_recognise_mean, mc.p_recognise_sd));
    out.drawn = write_imagined(run_dir, set, images, scans, mc, out.caption, {"helios", "bpearl"});
    if (not out.drawn) out.text += " (I could not write the picture's data.)";
    return out;
}

}   // namespace rc::chat
