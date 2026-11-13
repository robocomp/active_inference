// view_ledger.h — Task 4 of the door-existence model: the VIEW LEDGER. One record per viewpoint per door, shared by every
// evidence channel. All frames of a view are ONE trial: the view's log-likelihood ratio is recomputed from its aggregate and
// REPLACES the view's previous contribution (never added to). Header-only, DSR-free.
//
// Absorption rule: a frame is scored against every existing view of the SAME visit with the kernel
//   k = exp(-1/2 [ (db/alpha)^2 + (dr/R_eff)^2 ]),  R_eff = hypot(R, spread_m),  alpha = atan(R_eff / d).
// k is P(same view | the two standpoints). Assigning the frame to the more probable hypothesis (k >= 1/2: same view; else a new
// view) is a MAP decision on a posterior, the same kind as door creation at L > 0. The discontinuity it causes is bounded by one
// view's llr; a soft split would create a view record per frame. Same standpoint: k = 1 (a parked stare is one trial).
// Views of different `visit` never merge.
// Domain: R > 0 and s2 > 0 are required inputs. On violation the frame is ignored: observe() returns npos and every add_* /
// recompute on an out-of-range index is a no-op, so it contributes exactly 0 (never NaN).
// add_* leave a view's llr STALE until recompute(v). Actuation accumulates (+=) by design: each satisfied command is one
// observation. All other channels are aggregates replaced on recompute.
// The kernel widths are the door's own scale R (half-diagonal), widened by the pose spread: a poorly known pose cannot tell
// two standpoints apart, so they are one view.
// Keys are door-relative: range = |cam - door|, bearing = angle of (cam - door) measured from door_normal. Both survive any
// change of room frame (rotation + translation) unchanged, so persisted ledgers stay valid after re-registration.
//
// Serialisation (one line, classic locale, std::from_chars, no commas/newlines — fits a CSV cell): whitespace-separated tokens
//   "VL2 <nviews>" then per view, 30 numbers: bearing range visit actuation_llr llr
//   then per Bernoulli channel (Zed, Ricoh): n sum_q sum_hit_w sum_f hit_c ; then geo: n sum_x sum_w mu0 mu1 s2;
//   then the fit-support channel: n sum_x sum_w present_a present_b present_outlier absent_a absent_b absent_outlier.
//   "VL3" appends the support channel's sum_q (the summed overlap q of its frames, see add_support) after that block.
//   "VL1" (the same without the support block) still parses, with an empty support channel; VL1/VL2 read as q = 1.
#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include "detector_llr.h"

namespace rc::door::core
{
using Vec2f = Eigen::Vector2f;
enum class Channel : std::uint8_t { Zed, Ricoh, Geometry, Actuation, Support };

struct ViewKey { float bearing = 0.f; float range = 0.f; int visit = 0; };   // door-relative

/// Per-view Bernoulli aggregate over its frames: sums of q, hit weight (association weight of a hit, 0 for a miss) and the
/// per-frame effective false-alarm rate f_eff (false_alarm_eff). hit_c = ln(hit_rate/false_alarm) of the camera's model.
struct BernoulliAgg { double n = 0, sum_q = 0, sum_hit_w = 0, sum_f = 0; float hit_c = 0.f; };
struct GaussAgg { double n = 0, sum_x = 0, sum_w = 0; float mu0 = 0, mu1 = 0, s2 = 1; };
/// The fit-support channel's per-view aggregate: the weighted mean of x over the view's frames, scored once (support_llr); the
/// model it was scored with travels with it (as geo's mu0/mu1/s2 do), so a restored ledger recomputes without the table.
struct SupportAgg { double n = 0, sum_x = 0, sum_w = 0, sum_q = 0; SupportModel m; };
struct View
{
    ViewKey key; 
    std::array<BernoulliAgg, 2> bern;   // Zed, Ricoh
    GaussAgg geo; SupportAgg sup; float actuation_llr = 0; float llr = 0;
    float llr_percept = 0;   // the perceptual part of llr (recomputed with it; llr - llr_percept = the physical part)
};

class ViewLedger
{
public:
    /// Locate (or open) the view for a frame taken from cam_xy of the door at door_xy; returns its index.
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    std::size_t observe(const Vec2f& cam_xy, const Vec2f& door_xy, const Vec2f& door_normal, float R, float spread_m, int visit)
    {
        if (not (R > 0.f) or not std::isfinite(R) or not std::isfinite(spread_m)) return npos;
        const auto [best, bearing, range] = locate(cam_xy, door_xy, door_normal, R, spread_m, visit);
        if (best != npos) return best;
        View v; v.key = {bearing, range, visit};
        views_.push_back(v);
        return views_.size() - 1;
    }
    /// The view a frame WOULD join (npos: none yet), without opening one: the ricoh's missed frames are logged against it, not observed.
    [[nodiscard]] std::size_t find_view(const Vec2f& cam_xy, const Vec2f& door_xy, const Vec2f& door_normal, float R, float spread_m, int visit) const
    {
        if (not (R > 0.f) or not std::isfinite(R) or not std::isfinite(spread_m)) return npos;
        return std::get<0>(locate(cam_xy, door_xy, door_normal, R, spread_m, visit));
    }
    /// One frame of a camera channel (Zed or Ricoh): q = detect_prob, m = that camera's model (f_eff and the hit constant are
    /// derived here), hit_w = association weight of a hit (0 for a miss frame).
    void add_bernoulli(std::size_t v, Channel c, float q, const DetectorModel& m, float hit_w)
    {
        if (v >= views_.size()) return;
        if (c == Channel::Ricoh and not (hit_w > 0.f)) return;   // ★RICOH IS HITS ONLY (Pablo 2026-11-08): a frame without a detection is no trial
        auto& a = views_[v].bern[c == Channel::Ricoh ? 1 : 0];
        a.n += 1; a.sum_q += q; a.sum_hit_w += hit_w; a.sum_f += false_alarm_eff(m, q);
        if (q > 0.f) a.hit_c = hit_llr(q, m, 1.f);
    }
    /// One geometry frame: statistic x with weight w; hypotheses N(mu0,s2) (absent) vs N(mu1,s2) (present).
    void add_geometry(std::size_t v, float x, float w, float mu0, float mu1, float s2)
    {
        if (v >= views_.size() or not (s2 > 0.f)) return;
        auto& g = views_[v].geo;
        g.n += 1; g.sum_x += double(w) * x; g.sum_w += w; g.mu0 = mu0; g.mu1 = mu1; g.s2 = s2;
    }
    /// One frame of the fit's own support: x = the clutter share of this door's fit this cycle, weight w (the pose overlap κ).
    /// q = P(the window the fit scored lies on the opening it stands for): the expected overlap of the fit's aperture with the
    /// true one given the aperture's along-wall uncertainty (door_opening_edges.h aperture_overlap). With probability 1 - q the
    /// frame looked at wall beside the opening, which a door and no door predict alike: the view's support term is the mixture
    /// log((1 - q) + q e^l) — l itself at q = 1, nothing at q = 0, continuous between (clean35, 2026-11-10: a one-viewpoint hinge
    /// flip moved a real door's fit 0.6-0.8 m onto plain wall, sigma_s 0.42 m, and ONE support row there deleted it).
    void add_support(std::size_t v, float x, float w, const SupportModel& m, float q = 1.f)
    {
        if (v >= views_.size()) return;
        auto& s = views_[v].sup;
        s.n += 1; s.sum_x += double(w) * x; s.sum_w += w; s.sum_q += std::clamp(std::isfinite(q) ? q : 0.f, 0.f, 1.f); s.m = m;   // clamp: domain of a probability
    }
    void add_actuation(std::size_t v, float llr) { if (v < views_.size()) views_[v].actuation_llr += llr; }

    /// Recompute view v's llr from its aggregate (replacing the old one). Returns the CHANGE in sum().
    float recompute(std::size_t v, bool geometry_may_refute)
    {
        if (v >= views_.size()) return 0.f;
        View& w = views_[v];
        float l = 0.f;                          // the perceptual channels; actuation joins it at the end, kept apart (place_belief.h)
        for (std::size_t i = 0; i < w.bern.size(); ++i)
        {
            const auto& a = w.bern[i];
            if (a.n <= 0) continue;
            const float qb = float(a.sum_q / a.n), fb = float(a.sum_f / a.n), db = float(a.sum_hit_w / a.n);
            const float hit = qb > 0.f ? a.hit_c : 0.f;
            l += db * hit;
            if (i == 0) l += (1.f - db) * miss_llr(qb, fb);   // the ZED alone counts misses: the ricoh contributes hits only (a persisted VL1/VL2 ricoh aggregate keeps only its hit part too)
        }
        const auto& g = w.geo;
        if (g.sum_w > 0 and g.n > 0 and g.s2 > 0.f)
        {
            const float xb = float(g.sum_x / g.sum_w), wb = float(g.sum_w / g.n);
            float lg = wb * ((xb - g.mu0) * (xb - g.mu0) - (xb - g.mu1) * (xb - g.mu1)) / (2.f * g.s2);
            if (not geometry_may_refute) lg = std::max(0.f, lg);
            l += lg;
        }
        if (const auto& s = w.sup; s.n > 0 and s.sum_w > 0)
        {   // ONE ℓ per view: the mean x, weighted, scored once — then marginalised over whether the window was on the opening
            const double ls = double(s.sum_w / s.n) * double(support_llr(float(s.sum_x / s.sum_w), s.m));
            const double q = std::clamp(s.sum_q / s.n, 0.0, 1.0);
            if (q >= 1.0) l += float(ls);
            else if (q > 0.0)
            {
                const double a = std::log1p(-q), b = std::log(q) + ls, mx = std::max(a, b);
                l += float(mx + std::log(std::exp(a - mx) + std::exp(b - mx)));
            }
        }
        w.llr_percept = l;
        l += w.actuation_llr;
        const float change = l - w.llr;
        w.llr = l;
        return change;
    }

    /// Every channel's sum: what the per-channel dl's of the trace are differences of. The belief does NOT add this (place_belief.h).
    [[nodiscard]] float sum() const { float s = 0.f; for (const auto& v : views_) s += v.llr; return s; }
    /// Sc: zed + ricoh + geometry + support. Evidence that a door-SHAPED thing is here: it does not separate a door from a look-alike.
    [[nodiscard]] float perceptual_sum() const { float s = 0.f; for (const auto& v : views_) s += v.llr_percept; return s; }
    /// The physical part of the ledger: actuation (opened on command / failed attempts). Only a physical test separates a door from a look-alike.
    [[nodiscard]] float physical_sum() const { float s = 0.f; for (const auto& v : views_) s += v.llr - v.llr_percept; return s; }
    [[nodiscard]] std::span<const View> views() const { return views_; }

    [[nodiscard]] std::string serialise() const
    {
        std::string s = std::format("VL3 {}", views_.size());
        for (const auto& v : views_)
        {
            s += std::format(" {} {} {} {} {}", v.key.bearing, v.key.range, v.key.visit, v.actuation_llr, v.llr);
            for (const auto& a : v.bern) s += std::format(" {} {} {} {} {}", a.n, a.sum_q, a.sum_hit_w, a.sum_f, a.hit_c);
            s += std::format(" {} {} {} {} {} {}", v.geo.n, v.geo.sum_x, v.geo.sum_w, v.geo.mu0, v.geo.mu1, v.geo.s2);
            s += std::format(" {} {} {} {} {} {} {} {} {}", v.sup.n, v.sup.sum_x, v.sup.sum_w, v.sup.m.present_a, v.sup.m.present_b,
                             v.sup.m.present_outlier, v.sup.m.absent_a, v.sup.m.absent_b, v.sup.m.absent_outlier);
            s += std::format(" {}", v.sup.sum_q);
        }
        return s;
    }

    /// Rejects any non-finite number. The stored llr is NOT trusted: every view is recomputed from its aggregate
    /// (geometry_may_refute must match the one used by the caller's recompute calls).
    static std::optional<ViewLedger> parse(std::string_view str, bool geometry_may_refute)
    {
        std::size_t pos = 0;
        const auto tok = [&]() -> std::string_view {
            while (pos < str.size() and str[pos] == ' ') ++pos;
            const std::size_t b = pos;
            while (pos < str.size() and str[pos] != ' ') ++pos;
            return str.substr(b, pos - b);
        };
        const auto num = [&](auto& out) {
            const auto t = tok();
            if (t.empty()) return false;
            const auto r = std::from_chars(t.data(), t.data() + t.size(), out);
            return r.ec == std::errc{} and r.ptr == t.data() + t.size();
        };
        const auto version = tok();
        const bool vl1 = version == "VL1", vl3 = version == "VL3";
        if (not vl1 and not vl3 and version != "VL2") return std::nullopt;
        std::size_t n = 0;
        if (not num(n)) return std::nullopt;
        ViewLedger L;
        for (std::size_t i = 0; i < n; ++i)
        {
            View v; bool ok = num(v.key.bearing) and num(v.key.range) and num(v.key.visit)
                              and num(v.actuation_llr) and num(v.llr);
            for (auto& a : v.bern)
                ok = ok and num(a.n) and num(a.sum_q) and num(a.sum_hit_w) and num(a.sum_f) and num(a.hit_c);
            ok = ok and num(v.geo.n) and num(v.geo.sum_x) and num(v.geo.sum_w) and num(v.geo.mu0) and num(v.geo.mu1) and num(v.geo.s2);
            if (not vl1)   // VL1 predates the support channel: it stays empty
                ok = ok and num(v.sup.n) and num(v.sup.sum_x) and num(v.sup.sum_w) and num(v.sup.m.present_a) and num(v.sup.m.present_b)
                     and num(v.sup.m.present_outlier) and num(v.sup.m.absent_a) and num(v.sup.m.absent_b) and num(v.sup.m.absent_outlier);
            if (vl3) ok = ok and num(v.sup.sum_q);
            else     v.sup.sum_q = v.sup.n;    // before the overlap existed every frame counted in full: q = 1
            if (not ok) return std::nullopt;
            const auto fin = [](float x) { return std::isfinite(x); };
            ok = ok and fin(v.key.bearing) and fin(v.key.range) and fin(v.actuation_llr) and fin(v.llr)
                 and fin(v.geo.mu0) and fin(v.geo.mu1) and fin(v.geo.s2) and std::isfinite(v.geo.n)
                 and std::isfinite(v.geo.sum_x) and std::isfinite(v.geo.sum_w) and std::isfinite(v.sup.n) and std::isfinite(v.sup.sum_x)
                 and std::isfinite(v.sup.sum_w) and std::isfinite(v.sup.sum_q) and support_in_domain(v.sup.m);
            for (const auto& a : v.bern)
                ok = ok and fin(a.hit_c) and std::isfinite(a.n) and std::isfinite(a.sum_q) and std::isfinite(a.sum_hit_w) and std::isfinite(a.sum_f);
            if (not ok) return std::nullopt;
            v.llr = 0.f;
            L.views_.push_back(v);
            L.recompute(L.views_.size() - 1, geometry_may_refute);
        }
        if (not tok().empty()) return std::nullopt;
        return L;
    }

private:
    /// The existing view a frame at this standpoint belongs to (the kernel in the header), and the frame's door-relative key.
    [[nodiscard]] std::tuple<std::size_t, float, float> locate(const Vec2f& cam_xy, const Vec2f& door_xy, const Vec2f& door_normal, float R, float spread_m, int visit) const
    {
        const Vec2f d = cam_xy - door_xy;
        const float range = d.norm();
        const float bearing = std::atan2(door_normal.x() * d.y() - door_normal.y() * d.x(), door_normal.dot(d));
        const float R_eff = std::hypot(R, spread_m);
        const float alpha = std::atan2(R_eff, range);
        std::size_t best = views_.size();
        float kbest = -1.f;
        for (std::size_t i = 0; i < views_.size(); ++i)
        {
            if (views_[i].key.visit != visit) continue;
            float db = std::fmod(bearing - views_[i].key.bearing, 2.f * std::numbers::pi_v<float>);
            if (db > std::numbers::pi_v<float>) db -= 2.f * std::numbers::pi_v<float>;
            if (db < -std::numbers::pi_v<float>) db += 2.f * std::numbers::pi_v<float>;
            const float dr = range - views_[i].key.range;
            const float k = std::exp(-0.5f * ((db / alpha) * (db / alpha) + (dr / R_eff) * (dr / R_eff)));
            if (k > kbest) { kbest = k; best = i; }
        }
        return {best < views_.size() and kbest >= 1.f - kbest ? best : npos, bearing, range};
    }

    std::vector<View> views_;
};
}  // namespace rc::door::core
