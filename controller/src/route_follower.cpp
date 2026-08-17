/*
 * route_follower.cpp — see route_follower.h
 */

#include "route_follower.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace rc
{

bool RouteFollower::build(const Eigen::Vector2f &start,
                          const std::vector<Eigen::Vector2f> &waypoints,
                          int laps,
                          const PlanFn &plan,
                          const FreeFn &is_free,
                          float spacing,
                          float smoothing_m)
{
    spline_ = RouteSpline{};
    wp_s_.clear();
    poly_.clear();
    wp_pos_.clear();
    wp_per_lap_ = 0;
    progress_ = 0.f;
    if (waypoints.size() < 2 or not plan) return false;

    laps = std::max(1, laps);
    laps_ = laps;
    wp_per_lap_ = 0;          // set from what lap 1 actually records, not from what was asked for
    spacing_ = std::max(0.02f, spacing);
    smoothing_ = smoothing_m;

    // 1. One polyline for the WHOLE RUN. Each hop is planned, never interpolated: a straight line
    //    between two waypoints is not known to be drivable, and a route containing a segment the planner
    //    never approved is exactly the lossy shortcut the grid planner exists to remove.
    poly_.push_back(start);
    Eigen::Vector2f from = start;
    deferred_.clear();
    int unreachable = 0;
    for (int lap = 0; lap < laps; ++lap)
        for (const auto &wp : waypoints)
        {
            const auto hop = plan(from, wp);
            if (not hop.has_value() or hop->size() < 2)
            {
                // UNREACHABLE, not merely blocked. The caller has already repaired this waypoint to the
                // nearest footprint-FEASIBLE pose — but feasible is not the same as reachable, and
                // nearest_free has no connectivity requirement, so it can land the waypoint in an enclosed
                // pocket (between an obstacle and a wall) that the planner can never enter. Observed live:
                // waypoint 14 repaired 0.70 m into such a pocket, A* exhausted 67k states over 16k free
                // cells, and ONE waypoint killed the whole 30-waypoint tour.
                //
                // Dropping it keeps the tour drivable, which is better than refusing to move at all — the
                // same policy the caller already applies to a boxed-in waypoint. It CHANGES THE ROUTE, so
                // it is said out loud rather than swallowed.
                // ★DEFERRED, NOT DISCARDED (2026-08-17). Dropping it permanently was self-fulfilling:
                // the waypoint leaves the route, so the robot never drives toward it, so the close-range
                // evidence that would free it is never gathered. Remembered with the place it holds in
                // the tour and re-offered by reinstate_deferred() when the robot comes near.
                ++unreachable;
                deferred_.push_back(Deferred{.pos = wp, .after_index = wp_pos_.size()});
                std::printf("[route] waypoint (%.2f,%.2f) is UNREACHABLE from (%.2f,%.2f) — DEFERRED "
                            "(will be re-tried when the robot gets near it). The driven route does not "
                            "match the recorded one until it comes back.\n",
                            wp.x(), wp.y(), from.x(), from.y());
                std::fflush(stdout);
                continue;                       // `from` stays put: the next hop starts where we still are
            }
            for (std::size_t i = 1; i < hop->size(); ++i) poly_.push_back((*hop)[i]);
            wp_pos_.push_back(wp);
            if (lap == 0) ++wp_per_lap_;      // the reachable count, which is what lap_at must divide by
            from = wp;
        }

    if (wp_per_lap_ < 2 or wp_pos_.size() < 2)
    {
        std::printf("[route] build FAILED: only %zu of %d waypoints are reachable — that is not a route.\n",
                    wp_pos_.size(), wp_per_lap_ * laps);
        std::fflush(stdout);
        return false;
    }

    // 2-3. One curve through all of it, then every waypoint's arc length. Both live in
    //      fit_from_polyline so a repair reproduces them by the same code.
    if (not fit_from_polyline(is_free)) return false;

    std::printf("[route] built: %d lap(s) x %d reachable waypoints (%d UNREACHABLE, deferred) -> %.2f m, "
                "%zu samples at %.0f cm, %d feasibility corrections\n",
                laps, wp_per_lap_, unreachable, spline_.length(), spline_.samples().size(),
                spline_.spacing() * 100, spline_.corrections());
    std::fflush(stdout);
    return true;
}

// ── SECOND CHANCE FOR A WAYPOINT THE WORLD REFUSED AT BUILD TIME ────────────────────────────────
// See the header for why a drop is a deferral. The two gates are what make this cheap AND what make it
// mean something: the waypoint must still be AHEAD (splicing in one the robot has driven past means
// going backwards) and the robot must be NEAR it (only then can the map about it have changed — this is
// the "as the robot approaches" the whole mechanism is for, expressed as a condition rather than a
// timer). A waypoint nobody approaches is never re-planned to, so a tour that avoids it costs nothing.
RouteFollower::ReinstateResult RouteFollower::reinstate_deferred(const Eigen::Vector2f &robot_pos,
                                                                 float ahead_m,
                                                                 const PlanFn &plan,
                                                                 const FreeFn &is_free)
{
    ReinstateResult r;
    if (deferred_.empty() or not spline_.valid() or not plan) return r;

    // erase_if rather than an index cursor: five branches, two of which remove and three of which do
    // not, is exactly the shape where a hand-rolled `++d` gets forgotten and the loop spins forever.
    // The predicate runs once per element in order and repair() never touches deferred_, so the side
    // effects below are safe inside it.
    std::erase_if(deferred_, [&](Deferred &dw)
    {
        // Retire the hopeless. At some point "blocked" IS the answer, and re-planning to it on every
        // approach is an A* per cycle that buys nothing — the same livelock the affordance side solved
        // with known_useless_spot().
        if (dw.attempts >= kMaxReinstateAttempts)
        {
            std::printf("[route] waypoint (%.2f,%.2f) stays OUT of the tour: %d approaches and still "
                        "unreachable. Not asking again this run.\n", dw.pos.x(), dw.pos.y(), dw.attempts);
            std::fflush(stdout);
            ++r.retired;
            return true;
        }

        // Where it belongs in arc length: just before the survivor that follows it. That survivor is
        // also what says whether the waypoint is still ahead of the robot at all.
        const float s_next = dw.after_index < wp_s_.size() ? wp_s_[dw.after_index] : spline_.length();
        if (s_next <= progress_ + 0.10f) return false;                       // already driven past it
        if ((robot_pos - dw.pos).norm() > ahead_m) return false;             // not near enough yet

        ++r.tested;
        ++dw.attempts;
        if (const auto hop = plan(robot_pos, dw.pos); not hop.has_value() or hop->size() < 2)
            return false;                       // still blocked; it keeps its place and its counter

        // Reachable again. Re-author the window through it — an unblocked window is exactly the
        // situation a recovery happens in, so the ordinary blockage guard has to be lifted for it.
        constexpr float kBackM = 1.0f;
        const RouteRepairOptions opts{.require_blockage = false,
                                 .extra_via = std::make_pair(std::max(progress_ + 0.20f, s_next - 0.01f),
                                                             dw.pos)};
        if (repair(robot_pos, kBackM, ahead_m, plan, is_free, opts) != RepairResult::Repaired)
        {
            std::printf("[route] waypoint (%.2f,%.2f) is reachable again but the window would not "
                        "re-author through it — leaving it deferred.\n", dw.pos.x(), dw.pos.y());
            std::fflush(stdout);
            return false;
        }
        std::printf("[route] ★waypoint (%.2f,%.2f) is REACHABLE AGAIN and back in the tour after %d "
                    "attempt(s) — the route now matches the recorded one more closely.\n",
                    dw.pos.x(), dw.pos.y(), dw.attempts);
        std::fflush(stdout);
        ++r.recovered;
        return true;
    });
    return r;
}

std::vector<float> RouteFollower::anchor_polyline_arclengths() const
{
    std::vector<float> out;
    if (poly_.size() < 2) return out;
    out.reserve(wp_pos_.size());

    std::vector<float> cum(poly_.size(), 0.f);
    for (std::size_t i = 1; i < poly_.size(); ++i) cum[i] = cum[i - 1] + (poly_[i] - poly_[i - 1]).norm();

    // FORWARD-ONLY, in route order, within a window. The waypoints are vertices of this polyline, so the
    // projection lands on them exactly; the window and the monotone hint are what keep lap 2's copy of a
    // waypoint from binding to lap 1's identical position. A global search cannot tell them apart at all —
    // they are the same point in space, and only the order they are driven in distinguishes them.
    constexpr float kWindow = 10.0f;      // a lap is ~36 m and waypoints ~1.2 m apart: no ambiguity here
    float s_hint = 0.f;
    for (const auto &w : wp_pos_)
    {
        float best_s = s_hint, best_d2 = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i + 1 < poly_.size(); ++i)
        {
            if (cum[i + 1] < s_hint) continue;              // behind the hint: already consumed
            if (cum[i] > s_hint + kWindow) break;
            const Eigen::Vector2f a = poly_[i], ab = poly_[i + 1] - a;
            const float len2 = ab.squaredNorm();
            const float t = len2 > 1e-9f ? std::clamp((w - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
            if (const float d2 = (a + t * ab - w).squaredNorm(); d2 < best_d2)
            {
                best_d2 = d2;
                best_s = cum[i] + t * std::sqrt(len2);
            }
        }
        out.push_back(best_s);
        s_hint = best_s;
    }
    return out;
}

// Arc length of every waypoint. Projected in ORDER with a forward-only hint so a self-crossing route
// cannot bind a waypoint to the wrong passage — this tour crosses itself, so a global nearest-point
// search would do exactly that. ONE place computes these, shared by the fit and the band, so the arc
// lengths can never drift apart from the geometry they describe.
void RouteFollower::rederive_waypoint_arclengths()
{
    wp_s_.clear();
    wp_s_.reserve(wp_pos_.size());
    float s_hint = 0.f;
    for (const auto &w : wp_pos_)
    {
        s_hint = spline_.project(w, s_hint, 8.0f);
        wp_s_.push_back(s_hint);
    }
}

RouteOptimizerReport RouteFollower::deform_window(
    std::function<float(const Eigen::Vector2f &)> distance,
    std::function<Eigen::Vector2f(const Eigen::Vector2f &)> distance_gradient,
    std::size_t freeze_before, std::size_t freeze_after, int iterations)
{
    // The band holds the SAME beliefs the route was built with — bending prior, clearance preference,
    // anchor likelihood, gauge — and differs only in the field it measures them against and in being
    // restricted to a window. Re-deriving the weights here would let the driven geometry and the authored
    // geometry answer to two different objectives, which is how a route silently stops being the tour.
    if (not spline_.valid() or not distance) return RouteOptimizerReport{};
    RouteOptimizerConfig opt = opt_;
    opt.enabled = true;
    opt.distance = std::move(distance);
    opt.distance_gradient = std::move(distance_gradient);
    opt.anchors = wp_pos_;
    opt.anchor_s = anchor_polyline_arclengths();   // polyline metric — see fit_from_polyline
    opt.freeze_before = freeze_before;
    opt.freeze_after = freeze_after;
    opt.iterations = std::max(1, iterations);
    opt.verbose = false;                           // this runs at control rate; per-cycle logging is noise

    const RouteOptimizerReport rep = spline_.deform(opt);
    if (rep.ran) rederive_waypoint_arclengths();
    return rep;
}

bool RouteFollower::fit_from_polyline(const FreeFn &is_free, std::size_t freeze_before,
                                     std::size_t freeze_after)
{
    if (poly_.size() < 2) return false;

    // The optimiser's LIKELIHOOD term is the route's own authored waypoints, so they are filled in here
    // rather than asked of the caller — nothing outside this class knows what the route was authored from.
    RouteOptimizerConfig opt = opt_;
    opt.anchors = wp_pos_;
    // ★POLYLINE arc length, not wp_s_. Two reasons, and both were live defects on a 3-lap run:
    //   • wp_s_ is measured along the SMOOTHED CURVE, while the optimiser's control points are resampled
    //     every h along the POLYLINE and it indexes them as s/h. The curve is shorter than the polyline
    //     (smoothing cuts corners), so the two metrics drift apart with distance — invisible at 36 m,
    //     several control points at 110 m.
    //   • On a fresh build wp_s_ is EMPTY (it is derived from the curve that does not exist yet), so the
    //     optimiser fell back to its nearest-control-point search. Control points are 0.40 m apart, so on
    //     a route that repeats, lap 3's control point can sit 0.02 m from a waypoint where lap 1's sits
    //     0.19 m: the anchor binds to the wrong lap, the forward-only hint follows it to the end of the
    //     route, and every remaining anchor binds there too. Measured on 3 laps: anchor cost 11.87 of a
    //     total 12.09, control points dragged 2.15 m, and 15 m of tour deleted.
    // The polyline exists before the fit and is the metric the control points are actually spaced in, so
    // this is the one quantity that is well defined at the moment the binding is made.
    opt.anchor_s = anchor_polyline_arclengths();
    opt.freeze_before = freeze_before;
    opt.freeze_after = freeze_after;
    const RouteOptimizerConfig *popt = (opt.enabled and opt.distance) ? &opt : nullptr;

    if (not spline_.build(poly_, spacing_, is_free, smoothing_, popt)) return false;

    rederive_waypoint_arclengths();
    return true;
}

RouteFollower::RepairResult RouteFollower::repair(const Eigen::Vector2f &robot_pos,
                                                  float back_m,
                                                  float ahead_m,
                                                  const PlanFn &plan,
                                                  const FreeFn &is_free,
                                                  const RouteRepairOptions &opts)
{
    if (not spline_.valid() or poly_.size() < 2 or not plan) return RepairResult::Failed;

    const float L = spline_.length();
    const float s0 = std::clamp(progress_ - std::max(0.f, back_m), 0.f, L);
    const float s1 = std::clamp(progress_ + std::max(0.f, ahead_m), 0.f, L);
    // Too near the end for a detour to rejoin anything. Not a failure of planning — there is simply no
    // route left to re-author, so say no rather than splice a degenerate window.
    if (s1 - s0 < 0.5f) return RepairResult::NotNeeded;

    // ── IS THE ROUTE ACTUALLY BLOCKED? ────────────────────────────────────────────────────────────
    // Ask the question the caller could not: walk the curve across the window and test it with the SAME
    // footprint predicate the planner and the feasibility pass use. If every pose still fits, the thing
    // that fired the reflex is not on our path — it is beside it, behind it, or already gone — and the
    // route needs no repair. This is the difference between "an obstacle appeared" and "I cannot get
    // through", and conflating them is what produced a detour storm in places that were plainly drivable.
    // How much of the window is impassable, in metres — not merely whether any of it is. The amount is
    // what makes the post-repair check below possible: a repair has to IMPROVE it, and "is there a
    // blockage" cannot express improvement.
    float blocked_before = 0.f;
    if (is_free)
    {
        for (float s = s0; s <= s1; s += spline_.spacing())
            if (not is_free(spline_.position_at(s), spline_.heading_at(s)))
                blocked_before += spline_.spacing();
        // See RepairOptions::require_blockage: recovery must proceed on an UNBLOCKED window, which is
        // the one situation this guard would otherwise refuse. Every other caller wants it — it is what
        // keeps a transient obstacle from becoming a permanent detour.
        if (blocked_before <= 0.f and opts.require_blockage) return RepairResult::NotNeeded;
    }

    // Map the arc-length window onto the POLYLINE, which is what gets spliced. Do it by ARC LENGTH, never
    // by nearest-vertex search: this route crosses itself (the tour has two passes 0.6 m apart), so a
    // spatial search for "the vertex nearest the window start" can bind to the branch driven ten metres
    // ago and splice a detour into the wrong pass. Walking cumulative length is monotone in s by
    // construction, so that failure is not merely unlikely — it is unrepresentable.
    // The polyline and the curve differ in length (smoothing cuts corners), so s is rescaled by their
    // ratio rather than compared raw.
    std::vector<float> cum(poly_.size(), 0.f);
    for (std::size_t i = 1; i < poly_.size(); ++i)
        cum[i] = cum[i - 1] + (poly_[i] - poly_[i - 1]).norm();
    if (cum.back() < 1e-3f) return RepairResult::Failed;
    const float scale = cum.back() / std::max(L, 1e-6f);
    // The bracket must CONTAIN the window, so the two ends round outwards: i0 is the last vertex at or
    // before s0 and i1 the first at or after s1. Rounding both the same way (the obvious lower_bound for
    // each) walks the whole window forward by up to one vertex, which would start the detour AHEAD of the
    // robot — exactly what back_m exists to prevent.
    const auto index_before = [&cum, scale](float s)
    {
        const float target = std::clamp(s * scale, 0.f, cum.back());
        const auto it = std::upper_bound(cum.begin(), cum.end(), target);
        return it == cum.begin() ? std::size_t{0} : static_cast<std::size_t>(it - cum.begin() - 1);
    };
    const auto index_after = [&cum, scale](float s)
    {
        const float target = std::clamp(s * scale, 0.f, cum.back());
        const auto it = std::lower_bound(cum.begin(), cum.end(), target);
        return it == cum.end() ? cum.size() - 1 : static_cast<std::size_t>(it - cum.begin());
    };
    const std::size_t i0 = index_before(s0);
    const std::size_t i1 = index_after(s1);
    if (i1 <= i0 or i1 >= poly_.size()) return RepairResult::Failed;

    // ── PLAN THROUGH THE AUTHORED WAYPOINTS INSIDE THE WINDOW ─────────────────────────────────────
    // A single hop from i0 to i1 asks A* for the SHORTEST path across the window, which cuts whatever
    // excursion the tour was authored to make through it. Measured: three repairs took the route from
    // 35.64 m to 30.99 m, and again 35.96 -> 31.66 m on the next run — about 13% of the tour deleted,
    // one window at a time. A detour is LONGER than what it replaces; anything shorter is a shortcut,
    // and a shortcut silently rewrites the stimulus the run is supposed to measure.
    // So the window is re-planned as a CHAIN through the waypoints it contains. A repair can then route
    // around a blocker but can never straighten the tour.
    std::vector<std::pair<float, Eigen::Vector2f>> via_s;
    for (std::size_t j = 0; j < wp_s_.size() and j < wp_pos_.size(); ++j)
        if (wp_s_[j] > s0 + 0.10f and wp_s_[j] < s1 - 0.10f) via_s.emplace_back(wp_s_[j], wp_pos_[j]);
    // A waypoint being REINSTATED joins the chain at the arc length it holds in the tour, so the detour
    // visits it in the authored order rather than wherever a shortest path would put it.
    if (opts.extra_via.has_value())
    {
        via_s.emplace_back(*opts.extra_via);
        std::ranges::sort(via_s, {}, &std::pair<float, Eigen::Vector2f>::first);
    }
    std::vector<Eigen::Vector2f> via;
    via.reserve(via_s.size());
    for (const auto &[s_at, p] : via_s) via.push_back(p);

    std::vector<Eigen::Vector2f> hop{poly_[i0]};
    Eigen::Vector2f from = poly_[i0];
    int dropped = 0;
    for (const auto &v : via)
    {
        const auto seg = plan(from, v);
        // A waypoint we cannot reach IS the thing we are routing around — drop that one and carry on,
        // rather than failing the whole repair. Dropping it is a real change to the route, so it is
        // counted and reported rather than swallowed.
        if (not seg.has_value() or seg->size() < 2) { ++dropped; continue; }
        for (std::size_t k = 1; k < seg->size(); ++k) hop.push_back((*seg)[k]);
        from = v;
    }
    const auto tail = plan(from, poly_[i1]);
    if (not tail.has_value() or tail->size() < 2) return RepairResult::Failed;
    for (std::size_t k = 1; k < tail->size(); ++k) hop.push_back((*tail)[k]);
    if (hop.size() < 2) return RepairResult::Failed;

    std::vector<Eigen::Vector2f> spliced;
    spliced.reserve(poly_.size() + hop.size());
    spliced.insert(spliced.end(), poly_.begin(), poly_.begin() + static_cast<std::ptrdiff_t>(i0) + 1);
    for (std::size_t i = 1; i < hop.size(); ++i) spliced.push_back(hop[i]);   // hop[0] == poly_[i0]
    spliced.insert(spliced.end(), poly_.begin() + static_cast<std::ptrdiff_t>(i1) + 1, poly_.end());

    auto previous = poly_;
    poly_ = std::move(spliced);
    // FREEZE WHAT THE ROBOT IS ALREADY DRIVING. The refit re-optimises the control polygon, and moving the
    // stretch under the robot would put a step in the cross-track error at the worst possible moment. The
    // window start s0 is behind the robot by construction, so everything before it is safe to pin. Control
    // points sit every `ctrl_step` along the POLYLINE, hence the same rescaling used for the bracket.
    const float ctrl_step = std::max(spacing_, smoothing_);
    const std::size_t freeze = static_cast<std::size_t>(std::max(0.f, s0 * scale / ctrl_step));
    // ...AND FREEZE EVERYTHING PAST THE SPLICE. A repair is a LOCAL edit, so its variable set must be
    // local: without this the solve re-optimises the entire route ahead, against whatever the ESDF holds
    // at the instant of this one blockage. Measured over 31 repairs in a 3-lap run, that compounds — the
    // anchor cost climbed 0.077 -> 0.70 and the route eroded 109.9 -> 106.3 m, so the tour the robot drove
    // on lap 3 was not the tour it was given, and the drift came from transients metres away.
    // The window in POLYLINE arc length is [cum[i0], cum[i0] + |hop|]; kBlendCtrl control points of slack
    // either side let the splice blend into what it joins instead of kinking against a pinned neighbour.
    constexpr std::size_t kBlendCtrl = 3;
    float hop_len = 0.f;
    for (std::size_t i = 1; i < hop.size(); ++i) hop_len += (hop[i] - hop[i - 1]).norm();
    const std::size_t window_end_ctrl =
        static_cast<std::size_t>(std::max(0.f, (cum[i0] + hop_len) / ctrl_step)) + kBlendCtrl;
    // A failed refit must leave the route EXACTLY as it was — not merely re-derived from the same
    // polyline. Re-fitting would re-run the optimiser against the CURRENT distance field, so a repair
    // that failed would still move the route; live, ten consecutive failures did exactly that, each one
    // paying the full re-optimisation it could not use. Keeping the curve costs one copy of a few
    // thousand points and makes a failed repair a genuine no-op.
    auto prev_spline = spline_;
    auto prev_wp_s = wp_s_;
    if (not fit_from_polyline(is_free, freeze, window_end_ctrl))
    {
        poly_ = std::move(previous);
        spline_ = std::move(prev_spline);
        wp_s_ = std::move(prev_wp_s);
        return RepairResult::Failed;
    }

    // ── DID THE REPAIR ACTUALLY IMPROVE ANYTHING? ─────────────────────────────────────────────────
    // A detour is planned on the POLYLINE, but what gets driven is the SMOOTHED curve, and smoothing at
    // 0.40 m can round a narrow detour straight back into the obstacle it was authored to avoid. The
    // route is then still blocked, the reflex fires again next cycle, and each firing splices another
    // detour: the length grows, the tour drifts away from what was authored, and nothing improves.
    // ★The test is COMPARATIVE, not absolute. "Is the window clear now?" is the wrong question, because
    // the window deliberately starts BEHIND the robot — which is usually where the blocker is, the robot
    // having just backed out of it — so an absolute test reverts every repair, including the ones that
    // work. Requiring strictly LESS blocked arc length asks what a repair is actually for, and the
    // stretch the robot is standing in cancels from both sides of the comparison.
    if (is_free and blocked_before > 0.f)
    {
        const float poly_len_new = cum[i0] + hop_len + (cum.back() - cum[i1]);
        const float L_new = spline_.length();
        const float scale_new = poly_len_new / std::max(L_new, 1e-6f);
        const float w0 = std::clamp(cum[i0] / std::max(scale_new, 1e-6f), 0.f, L_new);
        const float w1 = std::clamp((cum[i0] + hop_len) / std::max(scale_new, 1e-6f), 0.f, L_new);
        float blocked_after = 0.f;
        for (float s = w0; s <= w1; s += spline_.spacing())
            if (not is_free(spline_.position_at(s), spline_.heading_at(s)))
                blocked_after += spline_.spacing();
        if (blocked_after >= blocked_before - 0.5f * spline_.spacing())
        {
            std::printf("[route] repair REVERTED: %.2f m of the window was impassable and %.2f m still is, "
                        "so this detour achieved nothing. Holding instead of splicing another one.\n",
                        blocked_before, blocked_after);
            std::fflush(stdout);
            poly_ = std::move(previous);
            spline_ = std::move(prev_spline);
            wp_s_ = std::move(prev_wp_s);
            return RepairResult::Failed;
        }
    }

    // Re-anchor progress on the NEW curve. The arc-length scale changed under us, so the monotonicity
    // invariant of advance() does not carry across a repair — this is the one place progress_ may move
    // backwards, and it does so by construction rather than by a projection accident. Search forward from
    // the window start, which is behind the robot, with a window wide enough to contain the detour.
    const float hint = std::max(0.f, s0 - 0.5f);
    progress_ = spline_.project(robot_pos, hint, std::max(4.f, back_m + ahead_m));

    std::printf("[route] repaired %.2f m window at s=%.2f through %zu authored waypoint(s)"
                "%s -> new length %.2f m, %zu samples, %d feasibility corrections\n",
                s1 - s0, progress_, via.size() - static_cast<std::size_t>(dropped),
                dropped > 0 ? " (some unreachable and DROPPED)" : "",
                spline_.length(), spline_.samples().size(), spline_.corrections());
    std::fflush(stdout);
    return RepairResult::Repaired;
}

float RouteFollower::advance(const Eigen::Vector2f &robot_pos)
{
    if (not spline_.valid()) return progress_;
    // Forward-only, and never past the end.
    progress_ = std::clamp(spline_.project(robot_pos, progress_), progress_, spline_.length());
    return progress_;
}

int RouteFollower::lap_at(float s) const
{
    if (wp_s_.empty() or wp_per_lap_ <= 0) return 1;
    // The lap containing s = how many whole laps of waypoints are already behind it.
    int passed = 0;
    for (const float ws : wp_s_)
        if (s >= ws) ++passed;
    return std::min(laps_, 1 + passed / wp_per_lap_);
}

int RouteFollower::laps_completed_at(float s) const
{
    if (wp_s_.empty() or wp_per_lap_ <= 0) return 0;
    int passed = 0;
    for (const float ws : wp_s_)
        if (s >= ws) ++passed;
    return std::min(laps_, passed / wp_per_lap_);   // NO +1: a lap counts once its last waypoint is behind us
}

bool RouteFollower::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char *m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // A trivial "planner" that returns the straight segment — enough to exercise the concatenation,
    // the arc-length bookkeeping and the lap mapping without a grid.
    const PlanFn straight = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
        -> std::optional<std::vector<Eigen::Vector2f>> { return std::vector<Eigen::Vector2f>{a, b}; };

    // A 2 m square, driven twice.
    const std::vector<Eigen::Vector2f> square{{2.f, 0.f}, {2.f, 2.f}, {0.f, 2.f}, {0.f, 0.f}};
    {
        RouteFollower r;
        check(r.build({0.f, 0.f}, square, 2, straight, {}), "a square route must build");
        check(r.waypoint_arclengths().size() == 8, "2 laps x 4 waypoints = 8 waypoint arc lengths");
        check(r.laps_completed_at(0.f) == 0, "nothing driven yet is zero laps completed");
        check(r.laps_completed_at(r.length()) == 2, "driving the whole route completes every lap");
        // The real path: DRIVE the route (advance is forward-only within a window, so it cannot be
        // teleported to the end), stopping just short exactly as finished() permits. The lap count must
        // still be complete — a strict arc-length test loses the final lap here, because the last
        // waypoint's arc length IS the route length.
        const float stop_at = r.length() - 0.5f * RouteFollower::finish_tol_m;
        for (float s = 0.f; s <= stop_at; s += 0.5f) r.advance(r.spline().position_at(s));
        r.advance(r.spline().position_at(stop_at));
        std::printf("  finish tolerance: stopped at s=%.2f of %.2f m -> %d laps done\n",
                    r.progress(), r.length(), r.laps_done());
        check(r.laps_done() == 2, "a run that ends within the finish tolerance has completed every lap");
        check(r.waypoints_per_lap() == 4, "waypoints per lap must be remembered");
        // 8 m per lap, 2 laps; smoothing cuts the corners so it is a little shorter.
        std::printf("  square x2: length %.2f m (raw 16.00), samples %zu\n", r.length(), r.path().size());
        check(r.length() > 14.f and r.length() < 16.5f, "length must be about two laps of the square");

        // Arc lengths must be strictly increasing — a waypoint bound to the wrong passage on lap 2
        // would show up as a step backwards here, and would silently mis-attribute every metric.
        const auto &ws = r.waypoint_arclengths();
        for (std::size_t i = 1; i < ws.size(); ++i)
            check(ws[i] > ws[i - 1], "waypoint arc lengths must increase monotonically across laps");

        // Lap mapping.
        check(r.lap_at(0.1f) == 1, "the start is lap 1");
        check(r.lap_at(r.length() - 0.1f) == 2, "the end is on lap 2");
        std::printf("  mapping: s=0.1 -> lap %d | s=%.1f -> lap %d\n",
                    r.lap_at(0.1f), r.length() - 0.1f, r.lap_at(r.length() - 0.1f));
    }

    // PROGRESS IS MONOTONE. Feeding poses that step backwards must not move progress backwards: on a
    // route that revisits the same place each lap, a backward jump would re-run a lap of metrics.
    {
        RouteFollower r;
        r.build({0.f, 0.f}, square, 2, straight, {});
        const float a = r.advance({2.f, 1.0f});
        const float b = r.advance({0.5f, 0.f});      // back near the start of lap 1
        check(b >= a, "progress must never run backwards, even if the pose does");
        std::printf("  monotonicity: %.2f m then a backward pose -> %.2f m\n", a, b);
    }

    // LOCAL REPAIR. A blocker appears mid-route; the detour must splice in, keep the waypoint bookkeeping
    // consistent, and leave progress pointing at the robot — not at the old arc length, which the splice
    // has just invalidated.
    {
        RouteFollower r;
        check(r.build({0.f, 0.f}, square, 1, straight, {}), "a square route must build for the repair test");
        const float len_before = r.length();
        const std::size_t wps_before = r.waypoint_arclengths().size();

        r.advance({2.f, 0.8f});                      // partway up the first side
        const float s_before = r.progress();
        check(s_before > 1.f, "the robot must have made progress before the repair");

        // A "planner" that bypasses any hop via a point 1.5 m PERPENDICULAR to it. The offset must be
        // perpendicular, not along a fixed axis: a fixed axis can land the bypass point back on the hop's
        // own endpoint (it did, on the leg running along −x), producing a degenerate "detour" that
        // silently tests nothing.
        Eigen::Vector2f bypass_point{0.f, 0.f};
        const PlanFn detour = [&bypass_point](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        {
            const Eigen::Vector2f d = (b - a).normalized();
            const Eigen::Vector2f n{-d.y(), d.x()};
            bypass_point = 0.5f * (a + b) + 1.5f * n;
            return std::vector<Eigen::Vector2f>{a, bypass_point, b};
        };

        check(r.repair({2.f, 0.8f}, 0.5f, 1.5f, detour, {}) == RepairResult::Repaired,
              "a repair with a reachable detour must succeed");
        check(r.valid(), "the route must still be valid after a repair");
        // Assert the detour is actually IN the curve rather than asserting it got longer — whether a
        // bypass lengthens or shortens a route depends on the planner's geometry, so a length inequality
        // would be testing the stub, not the splice.
        float to_bypass = std::numeric_limits<float>::max();
        for (const auto &p : r.path()) to_bypass = std::min(to_bypass, (p - bypass_point).norm());
        check(to_bypass < 0.40f, "the repaired curve must actually pass through the planned detour");
        check(std::abs(r.length() - len_before) > 1e-3f, "a repair must change the route's length");
        check(r.waypoint_arclengths().size() == wps_before,
              "a repair must not add or drop waypoints — only re-derive their arc lengths");
        check(r.waypoints_per_lap() == 4, "waypoints per lap must survive a repair");
        // Progress must name the robot's own place on the NEW curve. Testing this against a fixed
        // distance would be meaningless — after a detour the robot is legitimately off the route, so any
        // loose bound passes for the wrong reason. The real invariant is that the re-anchored projection
        // is the CLOSEST point on the repaired curve, so compare it against a brute-force scan.
        const Eigen::Vector2f robot{2.f, 0.8f};
        const float err = (r.spline().position_at(r.progress()) - robot).norm();
        float best = std::numeric_limits<float>::max();
        for (const auto &p : r.path()) best = std::min(best, (p - robot).norm());
        check(err < best + 0.05f, "re-anchored progress must be the closest point on the repaired curve");
        std::printf("  repair: length %.2f -> %.2f m, progress %.2f -> %.2f m, "
                    "re-anchor %.3f m (global closest %.3f m)\n",
                    len_before, r.length(), s_before, r.progress(), err, best);

        // A repair with no reachable detour must leave the route untouched and usable.
        const PlanFn none = [](const Eigen::Vector2f &, const Eigen::Vector2f &)
            -> std::optional<std::vector<Eigen::Vector2f>> { return std::nullopt; };
        const float len_kept = r.length();
        check(r.repair({2.f, 0.8f}, 0.5f, 1.5f, none, {}) == RepairResult::Failed,
              "an unreachable detour must report failure");
        check(r.valid() and std::abs(r.length() - len_kept) < 1e-3f,
              "a failed repair must leave the route exactly as it was");
    }

    // A REPAIR MUST NOT EAT THE ROUTE. The window is re-planned through the waypoints it contains, so a
    // shortest-path planner cannot straighten the tour. Measured before this: three repairs took a 35.6 m
    // route to 31.0 m, one window at a time, silently changing the stimulus the run was measuring.
    {
        RouteFollower r;
        // A big square: waypoints far enough apart that a 3 m window contains an interior one.
        const std::vector<Eigen::Vector2f> big{{4.f, 0.f}, {4.f, 4.f}, {0.f, 4.f}, {0.f, 0.f}};
        // A REAL patch on the outbound leg, not an everywhere-blocked world. "Nothing is passable" used
        // to be a convenient way to force the repair to run, but no detour can improve an impassable
        // world, so the improvement check below (correctly) reverts every repair in it — a world with no
        // solution cannot test whether we found one.
        const FreeFn blocked = [](const Eigen::Vector2f &p, float)
        { return not (p.x() > 3.7f and p.x() < 4.3f and p.y() > 2.0f and p.y() < 2.6f); };
        // Detours to the RIGHT around that patch, straight everywhere else. Still the worst case for
        // shortcutting: it cuts every corner it is permitted to.
        const PlanFn side_step = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        {
            std::vector<Eigen::Vector2f> out{a};
            if (std::min(a.y(), b.y()) < 2.8f and std::max(a.y(), b.y()) > 1.8f
                and a.x() > 3.f and b.x() > 3.f)
                for (float y = 1.6f; y <= 3.0f; y += 0.35f) out.push_back({5.2f, y});
            out.push_back(b);
            return out;
        };
        check(r.build({0.f, 0.f}, big, 1, straight, {}), "the big square must build");
        const float len_before = r.length();
        r.advance({4.f, 1.5f});
        const auto res = r.repair({4.f, 1.5f}, 1.0f, 3.0f, side_step, blocked);
        check(res == RepairResult::Repaired, "a blocked route must actually be repaired");
        // The corner at (4,4) lies inside the window. If it were bypassed the route would collapse
        // diagonally and lose metres; planning through it keeps the length.
        std::printf("  no-shortcut: %.2f m -> %.2f m across a repair spanning a corner\n",
                    len_before, r.length());
        check(r.length() > len_before - 0.60f,
              "a repair must not shorten the route — that is a shortcut, not a detour");
    }

    // A REFLEX THAT FIRED ON NOTHING must not re-author the route. This is the common case in the field:
    // a blockage is seen beside the path, or has already gone by the time the repair runs.
    {
        RouteFollower r;
        const FreeFn all_free = [](const Eigen::Vector2f &, float) { return true; };
        check(r.build({0.f, 0.f}, square, 1, straight, all_free), "a square route must build");
        r.advance({2.f, 0.8f});
        const float len_before = r.length();
        const float s_before = r.progress();
        // A "planner" that would produce a wild detour if it were ever consulted — it must NOT be.
        bool planner_called = false;
        const PlanFn spy = [&planner_called](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        { planner_called = true; return std::vector<Eigen::Vector2f>{a, {9.f, 9.f}, b}; };
        check(r.repair({2.f, 0.8f}, 0.5f, 1.5f, spy, all_free) == RepairResult::NotNeeded,
              "a still-feasible route must report NotNeeded");
        check(not planner_called, "NotNeeded must not even consult the planner");
        check(std::abs(r.length() - len_before) < 1e-4f and std::abs(r.progress() - s_before) < 1e-4f,
              "NotNeeded must leave the route and the progress untouched");
        std::printf("  not-needed: feasible route left alone (%.2f m, s=%.2f), planner not consulted\n",
                    r.length(), r.progress());
    }

    // AN UNREACHABLE WAYPOINT MUST NOT KILL THE WHOLE TOUR. `nearest_free` guarantees a waypoint is
    // FEASIBLE, never that it is REACHABLE — it has no connectivity requirement, so it can repair one into
    // an enclosed pocket. Observed live: one such waypoint made a 30-waypoint tour refuse to build at all,
    // repeatedly. Skipping it keeps the route connected (the next hop starts from where we still are) and
    // drivable; it changes the route, which is why it is counted and announced.
    {
        RouteFollower r;
        const PlanFn broken = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        { if (b.y() > 1.5f) return std::nullopt; return std::vector<Eigen::Vector2f>{a, b}; };
        check(r.build({0.f, 0.f}, square, 1, broken, {}),
              "two unreachable waypoints out of four must still yield a drivable route");
        check(r.valid(), "the surviving route must be valid");
        check(r.waypoint_arclengths().size() == 2,
              "only the REACHABLE waypoints may be recorded, or the arc lengths describe a route we skipped");
        std::printf("  unreachable: 2 of 4 waypoints skipped -> %.2f m route still built\n", r.length());
    }

    // ...but a route that has lost almost everything is not a route.
    {
        RouteFollower r;
        const PlanFn nearly_all_broken = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        { if (not (b.x() > 1.5f and b.y() < 0.5f)) return std::nullopt; return std::vector<Eigen::Vector2f>{a, b}; };
        check(not r.build({0.f, 0.f}, square, 1, nearly_all_broken, {}),
              "fewer than two reachable waypoints must fail the build");
        check(not r.valid(), "a failed build must leave nothing valid behind");
    }

    // ── MULTI-LAP ANCHOR BINDING, WITH THE OPTIMISER ON ──────────────────────────────────────────────
    // Every other test here runs with the optimiser OFF (no distance field), which is why a 2-lap route
    // passed them all while the anchor binding was broken. On a repeating route the same waypoint is the
    // same POINT once per lap, so only arc length can say which lap an anchor belongs to — and the
    // failure is silent: the route comes back smooth, well cleared, and missing whole laps of the tour.
    // Live, on 3 laps: 94.73 m where 109.7 m was asked for, i.e. 15 m of tour deleted, and the robot
    // drove into a narrow passage the route had never been authored to enter.
    // The invariant is exact and needs no tuning: N laps of a route is N times one lap of it.
    {
        RouteOptimizerConfig opt;
        // A field with no deficit anywhere, so the clearance term is inert and this test is ABOUT the
        // anchors: if it fails, the binding is wrong, not the trade-off between terms.
        opt.distance = [](const Eigen::Vector2f &) { return 5.f; };
        opt.distance_gradient = [](const Eigen::Vector2f &) { return Eigen::Vector2f::Zero(); };
        opt.h = 0.40f; opt.d_target = 0.60f; opt.rho = 0.49f; opt.sigma_a = 0.30f;
        opt.iterations = 30; opt.verbose = false;

        RouteFollower r1, r3;
        r1.set_optimizer(opt);
        r3.set_optimizer(opt);
        check(r1.build({0.f, 0.f}, square, 1, straight, {}), "1-lap optimised route must build");
        check(r3.build({0.f, 0.f}, square, 3, straight, {}), "3-lap optimised route must build");
        const float ratio = r3.length() / std::max(1e-3f, r1.length());
        const auto &rep = r3.spline().last_optimizer_report();
        std::printf("  multi-lap: 1 lap %.2f m, 3 laps %.2f m (ratio %.3f), anchor cost %.3f, "
                    "max move %.3f m\n", r1.length(), r3.length(), ratio, rep.e_anchor, rep.max_move_m);
        check(ratio > 2.85f and ratio < 3.15f,
              "a 3-lap optimised route must be three times the 1-lap route — anchors bound to the wrong lap");
        check(rep.e_anchor < 1.0f, "the anchor term must not explode on a repeating route");
        check(rep.max_move_m < 3.f * opt.h, "control points must not be dragged across the route");

        // The arc lengths handed to the optimiser must be monotone and cover the whole polyline — the
        // property that makes "which lap" answerable at all.
        const auto as = r3.anchor_polyline_arclengths();
        check(as.size() == 12, "3 laps x 4 waypoints = 12 anchor arc lengths");
        bool monotone = true;
        for (std::size_t i = 1; i < as.size(); ++i) if (as[i] <= as[i - 1]) monotone = false;
        check(monotone, "anchor arc lengths must increase strictly along the route");
    }

    // ── A REPAIR MUST BE LOCAL, AND REPAIRS MUST NOT COMPOUND ────────────────────────────────────────
    // The route far ahead of a blockage is not evidence about the blockage. If a repair re-solves it, the
    // geometry the robot eventually drives there depends on transient occupancy metres away and a lap
    // earlier — and the trades accumulate: 31 repairs took the anchor cost 0.077 -> 0.70 and eroded the
    // route 109.9 -> 106.3 m. Both assertions below are about CONFINEMENT, not quality.
    {
        // A corridor with a distance field, so the optimiser is genuinely active (with no field it is
        // disabled and this test would pass vacuously — the trap the multi-lap test fell into).
        RouteOptimizerConfig opt;
        opt.distance = [](const Eigen::Vector2f &p) { return std::max(0.05f, 2.0f - std::abs(p.y())); };
        opt.distance_gradient = [](const Eigen::Vector2f &p)
        { return Eigen::Vector2f{0.f, p.y() > 0.f ? -1.f : 1.f}; };
        opt.h = 0.40f; opt.d_target = 0.60f; opt.rho = 0.49f; opt.sigma_a = 0.30f;
        opt.iterations = 30; opt.verbose = false;

        const std::vector<Eigen::Vector2f> line{{4.f, 0.f}, {8.f, 0.f}, {12.f, 0.f}, {16.f, 0.f}, {20.f, 0.f}};
        RouteFollower r;
        r.set_optimizer(opt);
        check(r.build({0.f, 0.f}, line, 1, straight, {}), "a long straight route must build");

        // Drive a little, then repair around a blocker near the robot.
        for (float s = 0.f; s <= 3.0f; s += 0.25f) r.advance(r.spline().position_at(s));
        const auto before = r.path();
        const float len_before = r.length();
        // A planner that bulges around x in [3,6] — a detour local to the window.
        // The detour must be WIDE, not just tall: the curve is smoothed at 0.40 m, so a single-vertex
        // spike is rounded straight back into the obstacle it was meant to avoid — which is a real
        // failure mode, not a test artefact, and is how a repair can leave the route still blocked and
        // the reflex still firing.
        const PlanFn detour = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        {
            std::vector<Eigen::Vector2f> out{a};
            if (a.x() < 6.8f and b.x() > 5.0f)
                for (float x = std::max(a.x(), 5.0f); x <= std::min(b.x(), 6.8f); x += 0.4f)
                    out.push_back({x, 1.2f});
            out.push_back(b);
            return out;
        };
        // BETWEEN the waypoints at x=4 and x=8. A blocker sitting ON an authored waypoint is a different
        // problem entirely — the repair plans THROUGH the waypoints in its window, so it cannot route
        // around one, and the post-check below correctly reverts every such attempt.
        const FreeFn blocked_patch = [](const Eigen::Vector2f &p, float)
        { return not (p.x() > 5.5f and p.x() < 6.1f and std::abs(p.y()) < 0.60f); };
        const auto res = r.repair(r.spline().position_at(r.progress()), 1.0f, 4.0f, detour, blocked_patch);
        check(res == RepairResult::Repaired, "a blocked window must repair");

        // FAR from the window, the curve must be untouched — as GEOMETRY. A repair changes the route's
        // LENGTH (a detour is longer than what it replaces), so every sample past the window shifts in
        // arc length even when it has not moved a millimetre in space: comparing before[i] to after[i]
        // measures the splice, not the shape, and reported 0.24 m of "movement" on a curve that was
        // identical. The honest measure is curve-to-curve distance, the same one lap_repeat_* uses.
        const auto &after = r.path();
        float far_move = 0.f;
        const float far_from = 12.0f;      // window was [~2, ~7]; this is well past it
        for (std::size_t i = 0; i < after.size(); ++i)
        {
            if (static_cast<float>(i) * r.spline().spacing() < far_from) continue;
            float nearest = std::numeric_limits<float>::max();
            for (std::size_t j = 0; j < before.size(); ++j)
                nearest = std::min(nearest, (after[i] - before[j]).norm());
            far_move = std::max(far_move, nearest);
        }
        std::printf("  local repair: length %.2f -> %.2f m, curve beyond the window moved %.4f m\n",
                    len_before, r.length(), far_move);
        check(far_move < 0.02f, "a repair must not move the route far ahead of its own window");

        // REPEATED REPAIRS MUST NOT COMPOUND. The live failure was not one repair, it was 31 of them: the
        // reflex fires whenever something is near, and each firing used to re-author the whole route
        // ahead. Once the detour exists the window is drivable again, so every further attempt must come
        // back NotNeeded and change nothing at all — bit-identical, not merely similar.
        {
            const auto settled = r.path();
            const float settled_len = r.length();
            int not_needed = 0;
            for (int k = 0; k < 5; ++k)
                if (r.repair(r.spline().position_at(r.progress()), 1.0f, 4.0f, detour, blocked_patch)
                    == RepairResult::NotNeeded) ++not_needed;
            float drift = 0.f;
            for (std::size_t i = 0; i < std::min(settled.size(), r.path().size()); ++i)
                drift = std::max(drift, (settled[i] - r.path()[i]).norm());
            std::printf("  repeated repairs: %d/5 NotNeeded, length %.2f -> %.2f m, drift %.6f m\n",
                        not_needed, settled_len, r.length(), drift);
            check(not_needed == 5, "a route that is already drivable must not be repaired again");
            check(drift < 1e-6f, "repeated repair attempts must not move the route at all");
        }

        // ── A BLOCKER SITTING ON AN AUTHORED WAYPOINT ───────────────────────────────────────────────
        // This is the case that loops forever, and it is not the one I first guessed. Smoothing alone
        // cannot undo a detour — RouteSpline re-tests every sample and pulls failures back toward the
        // polyline. But when the obstacle covers a WAYPOINT, pulling back lands inside it: the repair
        // re-plans THROUGH the waypoints in its window (by design, so a shortest-path planner cannot
        // straighten the tour), so every attempt reproduces a blocked route, the reflex fires again next
        // cycle, and each firing splices more length. Live, a 3-lap run logged 31 repairs and the
        // optimiser reported route clearance of 0.000 m — a route through an occupied cell.
        // There is no detour to find here. The honest outcome is to fail and hold until the obstacle
        // ages out on its TTL, not to re-author the tour once per cycle.
        {
            RouteFollower rw;
            rw.set_optimizer(opt);
            check(rw.build({0.f, 0.f}, line, 1, straight, {}), "route for the blocked-waypoint case must build");
            for (float s = 0.f; s <= 3.0f; s += 0.25f) rw.advance(rw.spline().position_at(s));
            // Covers the authored waypoint at (4,0).
            const FreeFn on_waypoint = [](const Eigen::Vector2f &p, float)
            { return not ((p - Eigen::Vector2f{4.f, 0.f}).norm() < 0.5f); };
            const auto before_w = rw.path();
            const float len_w = rw.length();
            int failed = 0;
            for (int k = 0; k < 5; ++k)
                if (rw.repair(rw.spline().position_at(rw.progress()), 1.0f, 4.0f, detour, on_waypoint)
                    == RepairResult::Failed) ++failed;
            float moved = 0.f;
            for (std::size_t i = 0; i < std::min(before_w.size(), rw.path().size()); ++i)
                moved = std::max(moved, (before_w[i] - rw.path()[i]).norm());
            std::printf("  blocked waypoint: %d/5 Failed, length %.2f -> %.2f m, curve moved %.6f m\n",
                        failed, len_w, rw.length(), moved);
            check(failed == 5, "a blocker on an authored waypoint has no detour — every attempt must fail");
            check(std::abs(rw.length() - len_w) < 1e-4f,
                  "five impossible repairs must not add a metre of route between them");
            check(moved < 1e-6f, "a reverted repair must leave the route bit-identical");
        }

        // A FAILED repair must be a no-op, not a re-solve. Ten consecutive failures were observed live.
        const PlanFn no_route = [](const Eigen::Vector2f &, const Eigen::Vector2f &)
            -> std::optional<std::vector<Eigen::Vector2f>> { return std::nullopt; };
        // A wall across the whole corridor INSIDE the window, so the route really is blocked and the
        // repair reaches the planner — against `blocked_patch` the detour already exists and the answer
        // would (correctly) be NotNeeded, which tests nothing about the failure path.
        const FreeFn walled = [](const Eigen::Vector2f &p, float)
        { return not (p.x() > 5.5f and p.x() < 6.1f); };
        const auto path_before_fail = r.path();
        const float len_pre_fail = r.length();
        const auto fail = r.repair(r.spline().position_at(r.progress()), 1.0f, 4.0f, no_route, walled);
        check(fail == RepairResult::Failed, "a repair with no plannable detour must fail");
        float fail_move = 0.f;
        for (std::size_t i = 0; i < std::min(path_before_fail.size(), r.path().size()); ++i)
            fail_move = std::max(fail_move, (path_before_fail[i] - r.path()[i]).norm());
        std::printf("  failed repair: length %.2f -> %.2f m, curve moved %.6f m\n",
                    len_pre_fail, r.length(), fail_move);
        check(fail_move < 1e-6f, "a FAILED repair must leave the route bit-identical");
    }

    // ── A BLOCKED WAYPOINT IS DEFERRED, NOT LOST, AND COMES BACK WHEN THE WORLD DOES ─────────────
    // The failure this exists to prevent is silent and permanent: a waypoint unreachable for the few
    // seconds the route was built is dropped for the whole mission, and because it leaves the route the
    // robot never goes near it, so nothing ever revises the verdict.
    {
        RouteFollower r;
        const FreeFn all_free = [](const Eigen::Vector2f &, float) { return true; };
        const std::vector<Eigen::Vector2f> wps{{4.f, 0.f}, {4.f, 4.f}, {0.f, 4.f}};
        // A "door" at (4,4) that is shut while the route is built and open afterwards.
        bool door_shut = true;
        const PlanFn gated = [&door_shut](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        {
            if (door_shut and (b - Eigen::Vector2f{4.f, 4.f}).norm() < 0.1f) return std::nullopt;
            return std::vector<Eigen::Vector2f>{a, b};
        };
        check(r.build({0.f, 0.f}, wps, 1, gated, all_free), "the tour must still build with one shut door");
        std::printf("  deferral: built with %d deferred waypoint(s)\n", r.deferred_count());
        check(r.deferred_count() == 1, "★the unreachable waypoint must be REMEMBERED, not discarded");

        // Far away: nothing is re-planned, because nothing can have been learned about it.
        r.advance({4.f, 0.2f});
        const auto far_res = r.reinstate_deferred({4.f, 0.2f}, 1.0f, gated, all_free);
        check(far_res.tested == 0 and r.deferred_count() == 1,
              "★a waypoint the robot is nowhere near must not cost a search — that is what makes this cheap");

        // The door opens, and the robot comes within reach of it.
        door_shut = false;
        const float len_before = r.length();
        const auto res = r.reinstate_deferred({4.f, 2.6f}, 4.0f, gated, all_free);
        std::printf("  reinstate: tested %d, recovered %d, route %.2f m -> %.2f m, %d still deferred\n",
                    res.tested, res.recovered, len_before, r.length(), r.deferred_count());
        check(res.recovered == 1, "★a waypoint that became reachable must be put BACK into the tour");
        check(r.deferred_count() == 0, "and it must stop being deferred once recovered");
        // It is back because the route now goes there: the corner at (4,4) is on the driven curve.
        float nearest = 1e9f;
        for (const auto &p : r.path()) nearest = std::min(nearest, (p - Eigen::Vector2f{4.f, 4.f}).norm());
        std::printf("  the recovered waypoint is now %.3f m from the driven curve\n", nearest);
        check(nearest < 0.5f, "★the route must actually PASS THROUGH it — a counter is not a recovery");
    }

    // ...AND A WAYPOINT THAT IS GENUINELY WALLED OFF MUST STOP COSTING SEARCHES. Without this, every
    // approach re-runs an A* that has already failed — the livelock the affordance side solved with
    // known_useless_spot().
    {
        RouteFollower r;
        const FreeFn all_free = [](const Eigen::Vector2f &, float) { return true; };
        const std::vector<Eigen::Vector2f> wps{{4.f, 0.f}, {4.f, 4.f}, {0.f, 4.f}};
        int calls = 0;
        const PlanFn walled = [&calls](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        {
            if ((b - Eigen::Vector2f{4.f, 4.f}).norm() < 0.1f) { ++calls; return std::nullopt; }
            return std::vector<Eigen::Vector2f>{a, b};
        };
        check(r.build({0.f, 0.f}, wps, 1, walled, all_free), "the tour must build");
        const int after_build = calls;
        r.advance({4.f, 2.0f});
        int rounds = 0;
        while (r.deferred_count() > 0 and rounds < 20)
        { r.reinstate_deferred({4.f, 2.6f}, 4.0f, walled, all_free); ++rounds; }
        std::printf("  retirement: gave up after %d rounds, %d extra plan calls (cap %d)\n",
                    rounds, calls - after_build, kMaxReinstateAttempts);
        check(r.deferred_count() == 0, "a permanently blocked waypoint must be RETIRED, not retried forever");
        check(calls - after_build <= kMaxReinstateAttempts,
              "★and it must cost at most kMaxReinstateAttempts searches, however long the tour is");
    }

    std::printf("RouteFollower::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
