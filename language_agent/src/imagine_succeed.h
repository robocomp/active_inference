#pragma once
/*
 * imagine_succeed.h — "would the mission succeed if you went through door_2 instead?". PURE.
 * Per leg (a passage) the odds are a Beta over today's closed crossing steps (EM level Step, kind "cross", target =
 * the leg's door; both directions pooled). A route multiplies its legs. LTSM folds the same crossings into its Beta at
 * eviction, so the LTSM Beta is the prior ONLY when today has no record for that door (never both: double counting).
 * Outcomes (t0/t1 are ms, em_schema.h):
 *   success   "satisfied"; "preempted" with room_after non-empty and != room
 *   failure   "door_wont_open", "cross_unreachable", "approach_unreachable", "unreachable"
 *   excluded  "stopped_by_user", "superseded", "lost", "preempted" without a room change, and ANY other outcome
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "imagined_world.h"

namespace rc::chat::imagine
{

struct LegStats
{
    std::string passage, door;
    int n_success = 0, n_fail = 0;
    double mean = 0, sd = 0;
    bool used_ltsm_prior = false;
    std::optional<double> mean_duration_s;
    int n_duration = 0;
    int ok_ab = 0, n_ab = 0, ok_ba = 0, n_ba = 0;   ///< per direction (room -> room_after of the crossing / its reverse)
    std::map<std::string, int> excluded;            ///< crossings seen but not counted, by the plain words for why
};
struct RouteOdds { std::vector<LegStats> legs; double p = 0, sd = 0; double e2 = 1; };

namespace succeed_detail
{
enum class Verdict { Success, Failure, Excluded };
inline Verdict verdict(const rc::em::Record& r)
{
    if (r.outcome == "satisfied") return Verdict::Success;
    if (r.outcome == "preempted") return (not r.room_after.empty() and r.room_after != r.room) ? Verdict::Success : Verdict::Excluded;
    for (const std::string_view f : {"door_wont_open", "cross_unreachable", "approach_unreachable", "unreachable"})
        if (r.outcome == f) return Verdict::Failure;
    return Verdict::Excluded;
}
/// Why a record is not counted, in the person's words.
inline std::string excluded_why(const rc::em::Record& r)
{
    if (r.outcome == "stopped_by_user") return "stopped by you";
    if (r.outcome == "superseded") return "superseded";
    if (r.outcome == "lost") return "lost";
    if (r.outcome == "preempted") return "interrupted without leaving the room";
    return "with another outcome";
}
inline double e2_of(double a, double b) { return a * (a + 1) / ((a + b) * (a + b + 1)); }
}   // namespace succeed_detail

[[nodiscard]] inline RouteOdds route_odds(const ImaginedWorld& w, const std::vector<std::string>& passage_ids)
{
    using namespace succeed_detail;
    RouteOdds out;
    out.p = 1;
    for (const auto& id : passage_ids)
    {
        LegStats L;
        L.passage = id;
        const auto pit = std::ranges::find(w.topology.passages, id, &TopoPassageInfo::id);
        const TopoPassageInfo* P = pit == w.topology.passages.end() ? nullptr : &*pit;
        L.door = (P and P->live_door) ? *P->live_door : id;
        const std::string kd = imagine_detail::norm_name(L.door), ki = imagine_detail::norm_name(id);
        double dur_sum = 0;
        for (const auto& r : w.records)
        {
            if (r.level != rc::em::Level::Step or r.kind != "cross" or not r.t1) continue;
            const std::string kt = imagine_detail::norm_name(r.target);
            if (kt != kd and kt != ki) continue;
            const auto v = verdict(r);
            if (v == Verdict::Excluded) { ++L.excluded[excluded_why(r)]; continue; }
            const bool ab = P and r.room == P->room_a;
            (ab ? L.n_ab : L.n_ba) += 1;
            if (v == Verdict::Success)
            {
                ++L.n_success;
                ++(ab ? L.ok_ab : L.ok_ba);
                dur_sum += (static_cast<double>(*r.t1) - static_cast<double>(r.t0)) / 1000.0;
                ++L.n_duration;
            }
            else ++L.n_fail;
        }
        if (L.n_duration > 0) L.mean_duration_s = dur_sum / L.n_duration;
        double a = 1.0 + L.n_success, b = 1.0 + L.n_fail;
        if (L.n_success + L.n_fail == 0 and P) { a = P->pass_alpha; b = P->pass_beta; L.used_ltsm_prior = true; }
        L.mean = a / (a + b);
        L.sd = std::sqrt(std::max(0.0, e2_of(a, b) - L.mean * L.mean));
        out.p *= L.mean;
        out.e2 *= e2_of(a, b);
        out.legs.push_back(std::move(L));
    }
    out.sd = std::sqrt(std::max(0.0, out.e2 - out.p * out.p));
    return out;
}

namespace succeed_detail
{
inline std::string leg_text(const LegStats& L, const TopoPassageInfo* P, bool em_available)
{
    std::string s;
    const int n = L.n_success + L.n_fail;
    if (n == 0)
    {
        const bool flat = P == nullptr or (P->pass_alpha == 1 and P->pass_beta == 1);
        if (not em_available)
        {
            const std::string_view em = "I am not connected to my episodic memory";
            if (flat) s = std::format("I have no data on {} (0.50 by default): {}", L.door, em);
            else s = std::format("{}: {}; its remembered record gives {:.2f}", L.door, em, L.mean);
        }
        else if (flat) s = std::format("I have no data on {} (0.50 by default)", L.door);
        else s = std::format("{} has no crossing today; its remembered record gives {:.2f}", L.door, L.mean);
    }
    else
    {
        s = std::format("{}: {} of {} crossings today succeeded (mean {:.2f}", L.door, L.n_success, n, L.mean);
        if (P and L.n_ab + L.n_ba > 0)
            s += std::format("; {}->{} {} of {}, {}->{} {} of {}", P->room_a, P->room_b, L.ok_ab, L.n_ab, P->room_b, P->room_a, L.ok_ba, L.n_ba);
        s += ")";
    }
    if (L.mean_duration_s) s += std::format(", about {:.0f} s per crossing over {} successes", *L.mean_duration_s, L.n_duration);
    if (not L.excluded.empty())
    {
        std::string ex;
        for (const auto& [why, k] : L.excluded) ex += std::format("{}{} {}", ex.empty() ? "" : ", ", k, why);
        s += std::format(" ({}, not counted)", ex);
    }
    return s;
}
inline std::string route_text(const ImaginedWorld& w, const RouteOdds& o)
{
    std::string s;
    for (const auto& L : o.legs)
    {
        const auto pit = std::ranges::find(w.topology.passages, L.passage, &TopoPassageInfo::id);
        s += (s.empty() ? "" : "; ") + leg_text(L, pit == w.topology.passages.end() ? nullptr : &*pit, w.em_available);
    }
    return s;
}
inline std::string names_of(const RouteOdds& o)
{
    std::string s;
    for (const auto& L : o.legs) s += (s.empty() ? "" : " + ") + L.door;
    return s;
}
}   // namespace succeed_detail

[[nodiscard]] inline std::string answer_succeed(const ImaginedWorld& w, std::string_view from_room, std::string_view to_room,
                                                std::string_view via)
{
    using namespace succeed_detail;
    if (from_room.empty()) return "I do not know which room I am in, so I cannot judge a trip from here.";
    if (to_room.empty()) return "ASK: where should the mission end?";
    if (norm_name(from_room) == norm_name(to_room)) return std::format("I am already in {}; there is no route to succeed at.", to_room);
    if (w.topology.rooms.empty() and w.topology.passages.empty())
        return "I have no remembered topology (my long-term memory of rooms is not joined, or still empty), so I cannot judge a route.";
    const auto normal = route(w.topology, from_room, to_room);
    if (normal.empty()) return std::format("I know no remembered route from {} to {}.", from_room, to_room);
    const RouteOdds base = route_odds(w, normal);
    if (via.empty())
        return std::format("Going {} -> {} by {}: P(success) ≈ {:.2f} ± {:.2f}. {}.", from_room, to_room, names_of(base), base.p, base.sd,
                           route_text(w, base));
    const auto pid = resolve_passage(w.topology, via);
    if (not pid) return std::format("I don't know a door called {}", via);
    const auto alt = route_via(w.topology, from_room, to_room, *pid);
    if (not alt) return std::format("no remembered route through {} reaches {}", via, to_room);
    const RouteOdds ao = route_odds(w, *alt);
    return std::format("≈{:.2f} via {} vs {:.2f} via {} (± {:.2f} and ± {:.2f}), going {} -> {}. Via {}: {}. Normal route: {}.",
                       ao.p, names_of(ao), base.p, names_of(base), ao.sd, base.sd, from_room, to_room, via, route_text(w, ao),
                       route_text(w, base));
}

}   // namespace rc::chat::imagine
