/*
 * route_spline.cpp — see route_spline.h
 */

#include "route_spline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rc
{

namespace
{
// Resample a polyline at a uniform arc-length step. Returns at least the endpoints.
std::vector<Eigen::Vector2f> resample(const std::vector<Eigen::Vector2f> &in, float step)
{
    std::vector<Eigen::Vector2f> out;
    if (in.size() < 2) return in;
    out.push_back(in.front());
    float carry = 0.f;
    for (std::size_t i = 0; i + 1 < in.size(); ++i)
    {
        const Eigen::Vector2f a = in[i], b = in[i + 1];
        const float seg = (b - a).norm();
        if (seg < 1e-9f) continue;
        const Eigen::Vector2f dir = (b - a) / seg;
        float t = step - carry;
        while (t <= seg)
        {
            out.push_back(a + dir * t);
            t += step;
        }
        carry = seg - (t - step);
    }
    if ((out.back() - in.back()).norm() > 0.25f * step) out.push_back(in.back());
    return out;
}

// Uniform cubic B-spline: C² by construction, and APPROXIMATING, so the curve stays inside the convex
// hull of its control points and cannot overshoot a corner into a wall.
Eigen::Vector2f bspline(const Eigen::Vector2f &p0, const Eigen::Vector2f &p1,
                        const Eigen::Vector2f &p2, const Eigen::Vector2f &p3, float t)
{
    const float t2 = t * t, t3 = t2 * t;
    return ((1.f - 3.f * t + 3.f * t2 - t3) * p0
          + (4.f - 6.f * t2 + 3.f * t3) * p1
          + (1.f + 3.f * t + 3.f * t2 - 3.f * t3) * p2
          + t3 * p3) / 6.f;
}
}  // namespace

bool RouteSpline::build(const std::vector<Eigen::Vector2f> &polyline, float spacing,
                        const std::function<bool(const Eigen::Vector2f &, float)> &is_free,
                        float smoothing_m,
                        const RouteOptimizerConfig *opt)
{
    samples_.clear();
    length_ = 0.f;
    corrections_ = 0;
    last_opt_ = RouteOptimizerReport{};
    max_dev_ = mean_dev_ = 0.f;
    spacing_ = std::max(0.02f, spacing);
    if (polyline.size() < 2) return false;

    // 1. Control points at the SMOOTHING scale, not the output scale. This is what sets how hard a
    //    corner is rounded; the output is resampled finely afterwards.
    const float ctrl_step = std::max(spacing_, smoothing_m);
    auto ctrl = resample(polyline, ctrl_step);
    if (ctrl.size() < 2) return false;
    if (ctrl.size() < 4)   // too short to smooth; the polyline IS the answer
    {
        samples_ = ctrl;
        for (std::size_t i = 0; i + 1 < samples_.size(); ++i) length_ += (samples_[i + 1] - samples_[i]).norm();
        return true;
    }

    // 1b. VARIATIONAL OPTIMISATION of the control polygon, if the caller supplied a distance field.
    //     Here — after the control points exist and BEFORE the curve is evaluated — because the control
    //     polygon is the decision variable. Optimising the samples instead would break the uniform
    //     arc-length spacing that position_at() and every downstream metric assume.
    //     `h` is taken from ctrl_step rather than the caller's guess: the prior's length scale IS the
    //     control spacing, and letting the two disagree would silently mis-weight the bending term.
    if (opt != nullptr and opt->enabled and opt->distance)
    {
        RouteOptimizerConfig cfg = *opt;
        cfg.h = ctrl_step;
        last_opt_ = optimize_route(ctrl, cfg);
    }

    // Retain what a later LOCAL DEFORMATION needs. build() used to keep only the samples, so once a route
    // was installed there was no decision variable left to re-optimise: the control polygon would have to
    // be re-derived from the polyline, discarding every deformation made since. The polygon IS the state.
    ctrl_      = std::move(ctrl);
    ctrl_step_ = ctrl_step;
    polyline_  = polyline;
    is_free_   = is_free;
    return evaluate_from_ctrl();
}

// Steps 2-4 of build(), factored out so a deformation can re-run them on a MODIFIED control polygon
// without refitting from the polyline. Everything below is a deterministic function of ctrl_, which is
// precisely why the control polygon is the thing worth keeping.
bool RouteSpline::evaluate_from_ctrl()
{
    const auto &ctrl      = ctrl_;
    const auto &polyline  = polyline_;
    const auto &is_free   = is_free_;
    const float ctrl_step = ctrl_step_;
    samples_.clear();
    length_ = 0.f;
    corrections_ = 0;
    max_dev_ = mean_dev_ = 0.f;
    if (ctrl.size() < 4)
    {
        samples_ = ctrl;
        for (std::size_t i = 0; i + 1 < samples_.size(); ++i) length_ += (samples_[i + 1] - samples_[i]).norm();
        return valid();
    }

    // 2. Evaluate the C² B-spline. Endpoints are duplicated so the curve starts and ends where the route
    //    does — a route whose first metre is a smoothing artefact would put the robot somewhere it was
    //    never asked to be.
    std::vector<Eigen::Vector2f> pad;
    pad.reserve(ctrl.size() + 4);
    pad.push_back(ctrl.front()); pad.push_back(ctrl.front());
    pad.insert(pad.end(), ctrl.begin(), ctrl.end());
    pad.push_back(ctrl.back()); pad.push_back(ctrl.back());

    std::vector<Eigen::Vector2f> curve;
    // Enough sub-samples that the polyline approximation of the B-spline is finer than the output step,
    // or the curvature estimate below would measure the sub-sampling rather than the curve.
    const int kSub = std::max(8, static_cast<int>(std::ceil(4.f * ctrl_step / spacing_)));
    for (std::size_t i = 0; i + 3 < pad.size(); ++i)
        for (int k = 0; k < kSub; ++k)
            curve.push_back(bspline(pad[i], pad[i + 1], pad[i + 2], pad[i + 3],
                                    static_cast<float>(k) / kSub));
    curve.push_back(pad.back());

    // 3. Back to uniform arc length, so `s` really is distance travelled along the curve.
    samples_ = resample(curve, spacing_);
    if (samples_.size() < 2) { samples_ = ctrl; }

    // 4. FEASIBILITY. Smoothing moved the curve off the only path known to be drivable, so every sample
    //    is re-tested and pulled back toward its nearest polyline point until it passes. Failing to find
    //    a feasible blend leaves the polyline point itself, which is feasible by construction.
    if (is_free)
    {
        const auto nearest_on_polyline = [&polyline](const Eigen::Vector2f &p)
        {
            Eigen::Vector2f best = polyline.front();
            float best_d2 = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i + 1 < polyline.size(); ++i)
            {
                const Eigen::Vector2f a = polyline[i], b = polyline[i + 1];
                const Eigen::Vector2f ab = b - a;
                const float len2 = ab.squaredNorm();
                const float t = len2 > 1e-9f ? std::clamp((p - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
                const Eigen::Vector2f q = a + t * ab;
                if (const float d2 = (p - q).squaredNorm(); d2 < best_d2) { best_d2 = d2; best = q; }
            }
            return best;
        };

        for (std::size_t i = 0; i < samples_.size(); ++i)
        {
            const Eigen::Vector2f fwd = samples_[std::min(i + 1, samples_.size() - 1)]
                                      - samples_[i > 0 ? i - 1 : 0];
            const float h = fwd.squaredNorm() > 1e-12f ? std::atan2(fwd.y(), fwd.x()) : 0.f;
            if (is_free(samples_[i], h)) continue;
            ++corrections_;
            const Eigen::Vector2f anchor = nearest_on_polyline(samples_[i]);
            bool fixed = false;
            for (const float blend : {0.25f, 0.5f, 0.75f, 1.0f})
            {
                const Eigen::Vector2f cand = samples_[i] + blend * (anchor - samples_[i]);
                if (is_free(cand, h)) { samples_[i] = cand; fixed = true; break; }
            }
            if (not fixed) samples_[i] = anchor;
        }
    }

    for (std::size_t i = 0; i + 1 < samples_.size(); ++i) length_ += (samples_[i + 1] - samples_[i]).norm();

    // Fidelity of the fit to the polyline it was fitted to. Measured AFTER the feasibility pass, so it
    // describes the curve that will actually be driven, not the one the spline first proposed.
    {
        const auto nearest_dist = [&polyline](const Eigen::Vector2f &p)
        {
            float best_d2 = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i + 1 < polyline.size(); ++i)
            {
                const Eigen::Vector2f a = polyline[i], ab = polyline[i + 1] - a;
                const float len2 = ab.squaredNorm();
                const float t = len2 > 1e-9f ? std::clamp((p - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
                best_d2 = std::min(best_d2, (p - (a + t * ab)).squaredNorm());
            }
            return std::sqrt(best_d2);
        };
        double acc = 0.0;
        std::size_t n = 0;
        for (std::size_t i = 0; i < samples_.size(); i += 4)   // every 4th: this is O(samples x polyline)
        {
            const float d = nearest_dist(samples_[i]);
            max_dev_ = std::max(max_dev_, d);
            acc += d;
            ++n;
        }
        if (n) mean_dev_ = static_cast<float>(acc / static_cast<double>(n));
    }
    return valid();
}

RouteOptimizerReport RouteSpline::deform(const RouteOptimizerConfig &opt)
{
    // A LOCAL, INCREMENTAL solve on the retained control polygon: same objective and same solver as the
    // build-time optimisation, but warm-started from the curve currently being driven and restricted to a
    // window by the caller's freeze_before/freeze_after. This is the elastic band — the geometry keeps
    // absorbing what the live field says, instead of being fixed once when the route was installed.
    if (ctrl_.size() < 4 or not opt.enabled or not opt.distance) return RouteOptimizerReport{};
    RouteOptimizerConfig cfg = opt;
    // h is the prior's length scale and it IS the control spacing — the same coupling build() enforces.
    // Letting the caller's guess stand here would silently mis-weight the bending term against a polygon
    // whose spacing it does not describe.
    cfg.h = ctrl_step_;
    const RouteOptimizerReport rep = optimize_route(ctrl_, cfg);
    // Re-evaluate unconditionally: optimize_route reverts ctrl_ itself when its acceptance test rejects
    // the solve, so on that path this rebuilds the identical curve rather than a stale one.
    evaluate_from_ctrl();
    last_opt_ = rep;
    return rep;
}

Eigen::Vector2f RouteSpline::position_at(float s) const
{
    if (samples_.empty()) return Eigen::Vector2f::Zero();
    if (samples_.size() == 1) return samples_.front();
    const float u = std::clamp(s, 0.f, length_) / spacing_;
    const auto i = static_cast<std::size_t>(u);
    if (i + 1 >= samples_.size()) return samples_.back();
    return samples_[i] + (u - static_cast<float>(i)) * (samples_[i + 1] - samples_[i]);
}

float RouteSpline::heading_at(float s) const
{
    if (samples_.size() < 2) return 0.f;
    const Eigen::Vector2f a = position_at(std::max(0.f, s - spacing_));
    const Eigen::Vector2f b = position_at(std::min(length_, s + spacing_));
    const Eigen::Vector2f d = b - a;
    return d.squaredNorm() > 1e-12f ? std::atan2(d.y(), d.x()) : 0.f;
}

float RouteSpline::curvature_at(float s) const
{
    if (samples_.size() < 3 or spacing_ < 1e-6f) return 0.f;
    const Eigen::Vector2f a = position_at(std::max(0.f, s - spacing_));
    const Eigen::Vector2f b = position_at(std::clamp(s, 0.f, length_));
    const Eigen::Vector2f c = position_at(std::min(length_, s + spacing_));
    const Eigen::Vector2f d1 = (c - a) / (2.f * spacing_);
    const Eigen::Vector2f d2 = (c - 2.f * b + a) / (spacing_ * spacing_);
    const float den = std::pow(d1.squaredNorm(), 1.5f);
    return den > 1e-9f ? (d1.x() * d2.y() - d1.y() * d2.x()) / den : 0.f;
}

float RouteSpline::project(const Eigen::Vector2f &p, float s_hint, float window_m) const
{
    if (samples_.size() < 2) return 0.f;
    // FORWARD-ONLY within a window. This tour crosses itself; a global nearest-point search would
    // snap progress onto a branch the robot drove ten metres ago, or one it has not reached.
    const float s0 = std::clamp(s_hint - 0.5f * spacing_, 0.f, length_);
    const float s1 = std::clamp(s_hint + window_m, 0.f, length_);
    float best_s = s0, best_d2 = std::numeric_limits<float>::max();
    for (float s = s0; s <= s1; s += spacing_)
    {
        if (const float d2 = (position_at(s) - p).squaredNorm(); d2 < best_d2) { best_d2 = d2; best_s = s; }
    }
    return best_s;
}

bool RouteSpline::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char *m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // (1) A straight line must survive smoothing exactly, and arc length must be the real distance.
    {
        RouteSpline r;
        std::vector<Eigen::Vector2f> line;
        for (int i = 0; i <= 10; ++i) line.push_back({static_cast<float>(i) * 0.5f, 0.f});
        check(r.build(line, 0.1f, {}), "a straight polyline must build");
        check(std::abs(r.length() - 5.0f) < 0.05f, "arc length must match the real length");
        check(std::abs(r.position_at(2.5f).x() - 2.5f) < 0.02f, "s must BE distance along the curve");
        check(std::abs(r.curvature_at(2.5f)) < 1e-3f, "a straight line has zero curvature");
        check(std::abs(r.heading_at(2.5f)) < 1e-3f, "heading along +x must be 0");
    }

    // (2) CURVATURE CONTINUITY is the whole point. Across a sharp corner the curvature of a C1 curve
    //     jumps; a C2 curve's must not. Measured as the largest step between adjacent samples.
    {
        RouteSpline r;
        std::vector<Eigen::Vector2f> corner;
        for (int i = 0; i <= 20; ++i) corner.push_back({static_cast<float>(i) * 0.1f, 0.f});
        for (int i = 1; i <= 20; ++i) corner.push_back({2.0f, static_cast<float>(i) * 0.1f});
        check(r.build(corner, 0.05f, {}), "a right-angle route must build");
        float max_jump = 0.f;
        for (float s = r.spacing(); s < r.length() - r.spacing(); s += r.spacing())
            max_jump = std::max(max_jump, std::abs(r.curvature_at(s) - r.curvature_at(s - r.spacing())));
        std::printf("  right-angle corner: length %.2f m, max curvature step %.2f 1/m per %.0f cm\n",
                    r.length(), max_jump, r.spacing() * 100);
        check(max_jump < 12.f, "curvature must not step discontinuously across a corner");
    }

    // (3) FEASIBILITY WINS OVER SMOOTHNESS. With a predicate that forbids the smoothed shortcut, every
    //     sample must end up somewhere the predicate accepts — never a pretty curve through a wall.
    {
        RouteSpline r;
        std::vector<Eigen::Vector2f> corner;
        for (int i = 0; i <= 20; ++i) corner.push_back({static_cast<float>(i) * 0.1f, 0.f});
        for (int i = 1; i <= 20; ++i) corner.push_back({2.0f, static_cast<float>(i) * 0.1f});
        // Forbid the inside of the corner — exactly where an approximating spline wants to go.
        auto is_free = [](const Eigen::Vector2f &p, float) { return not (p.x() > 1.85f and p.y() > 0.15f
                                                                          and p.x() < 1.99f); };
        check(r.build(corner, 0.05f, is_free), "must still build under a restrictive predicate");
        int violations = 0;
        for (const auto &p : r.samples()) if (not is_free(p, 0.f)) ++violations;
        std::printf("  restrictive corner: %d samples pulled back, %d violations remain\n",
                    r.corrections(), violations);
        check(violations == 0, "NO sample may remain infeasible — the polyline anchor is the fallback");
        check(r.corrections() > 0, "the test predicate should actually have bitten");
    }

    // (4) PROJECTION MUST BE MONOTONE on a self-crossing route. A global nearest-point search would
    //     teleport progress onto the other branch; this tour genuinely crosses itself.
    {
        RouteSpline r;
        std::vector<Eigen::Vector2f> fig;             // out along +x, back along a parallel 10 cm away
        for (int i = 0; i <= 30; ++i) fig.push_back({static_cast<float>(i) * 0.1f, 0.f});
        for (int i = 30; i >= 0; --i) fig.push_back({static_cast<float>(i) * 0.1f, 0.10f});
        check(r.build(fig, 0.05f, {}), "a doubling-back route must build");
        // Standing near the start of the OUTBOUND leg, but already far along the route, progress must
        // not jump back to the outbound branch.
        const float s_late = r.length() - 0.5f;
        const float s = r.project({0.2f, 0.05f}, s_late);
        std::printf("  self-crossing: hint %.2f m -> projected %.2f m (route %.2f m)\n",
                    s_late, s, r.length());
        check(s >= s_late - r.spacing(), "projection must not run backwards past the hint");
    }

    // (N) LOCAL ELASTIC BAND. The invariant that matters is not "does it move the curve" — it is that
    // it moves ONLY the window. Everything behind the robot must be bit-identical, because arc length is
    // measured from s=0: if the curve shifts back there, progress(), every waypoint arc length and the
    // carrot all silently change meaning while the robot is driving.
    {
        RouteSpline r;
        std::vector<Eigen::Vector2f> line;
        for (int i = 0; i <= 40; ++i) line.push_back({static_cast<float>(i) * 0.25f, 0.f});   // 10 m straight
        check(r.build(line, 0.05f, {}), "band: base route must build");
        const auto ctrl_before = r.control_points();
        const std::vector<Eigen::Vector2f> samples_before = r.samples();
        check(ctrl_before.size() >= 12, "band: need enough control points to carve a window");

        // An obstacle NEAR the route at x = 5 m, offset laterally. Distance field = distance to that
        // point, so the one-sided clearance term has a real deficit, and only near x = 5.
        // ★The offset is not cosmetic. With the obstacle EXACTLY on the curve the gradient is purely
        // axial — the band is pushed along the route rather than off it, and nothing moves. That is
        // Zhou 2020 Fig. 2 (ESDF gradients flip sign across the medial axis; a curve straddling an
        // obstacle is pushed apart in opposite directions and deadlocks), and it is a REAL property of
        // this mechanism, not a test artefact: a gradient band cannot escape a dead-centre obstacle on
        // its own. That case belongs to the planner via the blockage trigger, not to the band.
        const Eigen::Vector2f obs{5.f, 0.25f};
        RouteOptimizerConfig cfg;
        cfg.distance = [obs](const Eigen::Vector2f &p) { return (p - obs).norm(); };
        cfg.distance_gradient = [obs](const Eigen::Vector2f &p)
        {
            const Eigen::Vector2f d = p - obs;
            const float n = d.norm();
            return n > 1e-6f ? Eigen::Vector2f(d / n) : Eigen::Vector2f(1.f, 0.f);
        };
        cfg.d_target = 0.60f;
        cfg.iterations = 8;
        cfg.verbose = false;
        // Robot notionally at s = 3 m; freeze everything up to 4 m, deform 4 m -> 7 m.
        const float h = r.control_spacing();
        cfg.freeze_before = static_cast<std::size_t>(4.0f / h);
        cfg.freeze_after  = static_cast<std::size_t>(7.0f / h);
        const auto rep = r.deform(cfg);
        check(rep.ran, "band: deform must run on a retained polygon");

        const auto ctrl_after = r.control_points();
        check(ctrl_after.size() == ctrl_before.size(), "band: deform must not change the polygon size");
        float moved_frozen = 0.f, moved_window = 0.f, moved_tail = 0.f;
        for (std::size_t i = 0; i < ctrl_after.size(); ++i)
        {
            const float d = (ctrl_after[i] - ctrl_before[i]).norm();
            if (i < cfg.freeze_before)      moved_frozen = std::max(moved_frozen, d);
            else if (i < cfg.freeze_after)  moved_window = std::max(moved_window, d);
            else                            moved_tail = std::max(moved_tail, d);
        }
        std::printf("  band: frozen prefix moved %.9f m, window moved %.4f m, frozen tail moved %.9f m\n",
                    moved_frozen, moved_window, moved_tail);
        check(moved_frozen == 0.f, "band: NOTHING before the robot may move");
        check(moved_tail == 0.f, "band: nothing beyond the window may move");
        check(moved_window > 0.01f, "band: the window must actually respond to an obstacle on the route");

        // The curve behind the frozen boundary must be unchanged SAMPLE FOR SAMPLE, not merely close:
        // that is what makes progress() still mean the same thing after a deformation. Checked well
        // inside the frozen region, since a B-spline control point influences four spans either side.
        const auto &samples_after = r.samples();
        const auto n_check = static_cast<std::size_t>(3.0f / r.spacing());
        float worst = 0.f;
        for (std::size_t i = 0; i < std::min({n_check, samples_before.size(), samples_after.size()}); ++i)
            worst = std::max(worst, (samples_after[i] - samples_before[i]).norm());
        std::printf("  band: curve behind the robot moved at most %.9f m over the first 3 m\n", worst);
        check(worst < 1e-6f, "band: the driven curve behind the robot must be unchanged");
    }

    std::printf("RouteSpline::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
