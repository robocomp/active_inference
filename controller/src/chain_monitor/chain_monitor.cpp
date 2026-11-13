#include "chain_monitor.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include "../local_policy/contact.h"

namespace rc::chain
{
namespace
{
inline float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }
}

float two_sided_quantile(float risk_eps)
{
    float lo = 0.f, hi = 10.f;
    for (int i = 0; i < 60; ++i) { const float m = 0.5f * (lo + hi); (std::erfc(m / std::sqrt(2.f)) > risk_eps ? lo : hi) = m; }
    return 0.5f * (lo + hi);
}

bool UnmappedTest::operator()(const V2& q, float sigma_loc) const
{
    const float se = std::sqrt(sigma_r * sigma_r + sigma_loc * sigma_loc + sigma_map * sigma_map);
    return gp->distance_at(q) > gp->params.cell_size_m + k_eps * se;
}

WindowResult window_test(const rc::RobotFootprint& body, const RouteSpline& spline, float s_prog, const Pose2& bel,
                         std::span<const Return> cells, float sigma_track, float sigma_track_yaw, float risk_eps, float window_m)
{
    WindowResult r;
    // the chain's NEXT motion starts with whatever turn in place aligns the body with the route (PLAIN pivots at a cusp or a
    // large heading error): that sweep is part of the window, at the robot's own position
    const float h_route = spline.heading_at(s_prog), dh = wrap(h_route - bel.yaw);
    const int n = std::max(1, int(std::ceil(std::abs(dh) / 0.10f)));
    for (int k = 0; k <= n; ++k) r.poses.push_back(Pose2{bel.p, bel.yaw + dh * float(k) / float(n)});
    r.n_pivot = r.poses.size();
    for (float s = s_prog; s <= std::min(spline.length(), s_prog + window_m) + 1e-4f; s += 0.10f)
        r.poses.push_back(Pose2{spline.position_at(s), spline.heading_at(s)});
    // Each SEGMENT in its own frame: the chain's error is lateral to the route at each point (a shared offset), so the window
    // is as risky as its worst segment; a one-pose sweep reads 0 whatever is there, hence segments
    const float lp_ok = std::log1p(-risk_eps);
    for (std::size_t i = 0; i + 1 < r.poses.size(); ++i)
    {
        const Pose2 seg[2] = {r.poses[i], r.poses[i + 1]};
        const float lp = rc::policy::log_p_no_contact_cells(body, seg, cells, sigma_track, sigma_track_yaw, 0.01f);
        r.lp_worst = std::min(r.lp_worst, lp);
        if (r.first_bad < 0 and lp < lp_ok) r.first_bad = int(i);
    }
    return r;
}

float window_s_bad(const WindowResult& w, float s_prog)
{
    if (w.first_bad < 0) return 0.f;
    return s_prog + 0.10f * float(std::max(0, w.first_bad - int(w.n_pivot)));
}

PassResult stretch_blocked_at(const rc::RobotFootprint& body, const RouteSpline& spline, const GridPlanner& gp, float s0,
                              float s1, std::span<const Return> pc, const Pose2& bel, float risk_eps, float k_eps,
                              const std::function<float(const V2&)>& sigma_loc_at, float sigma_map)
{
    // CONNECTIVITY over (route position s, lateral offset d): cell (k, j) is free when the body moving from section k to k+1
    // at offset d (heading = route's, or +-0.3 rad) keeps P(no contact) >= 1 - risk_eps. Passable iff free cells connect the
    // robot's own cell to the last section (8-neighbour). Per-section "some offset passes" is NOT enough: an oblique crossing
    // slides the test pose off the obstacle (mock: apt_gap_closed read passable, the robot wandered 60 s). The lateral range
    // spans the corridor AND the localiser's uncertainty about where the route lies in the LiDAR's frame.
    // ★FLAGGED: ds 0.10 m, dd 0.03 m, +-0.3 rad (resolution choices, not levels).
    const float sig = 0.01f;   // the policy's own relative pose sigma (in.sigma_rel_xy)
    const float lp_ok = std::log1p(-risk_eps);
    const float ds = 0.10f, dd = 0.03f;
    PassResult out;
    const int K = std::max(0, int(std::floor((s1 - s0) / ds)));
    std::vector<float> dmax(std::size_t(K), 0.f);
    int J = 0;
    for (int k = 0; k < K; ++k)
    {
        const V2 q0 = spline.position_at(s0 + float(k) * ds);
        dmax[std::size_t(k)] = std::max(0.f, gp.distance_at(q0) - body.inscribed_radius()) + k_eps * std::hypot(sigma_loc_at(q0), sigma_map);
        J = std::max(J, int(std::floor(dmax[std::size_t(k)] / dd)));
    }
    const int W = 2 * J + 1;
    out.sections = K; out.offsets = W;
    std::vector<signed char> fr(std::size_t(K * W), -1);   // -1 unknown, 0 blocked, 1 free
    const auto free_at = [&](int k, int jj) -> bool {
        auto& f = fr[std::size_t(k * W + jj + J)];
        if (f >= 0) return f == 1;
        const float d = float(jj) * dd;
        if (std::abs(d) > dmax[std::size_t(k)] + 1e-4f) { f = 0; return false; }
        const float sa = s0 + float(k) * ds, sb = sa + ds;
        const V2 q0 = spline.position_at(sa), q1 = spline.position_at(sb);
        const float h0 = spline.heading_at(sa), h1 = spline.heading_at(sb);
        const V2 n0(-std::sin(h0), std::cos(h0)), n1(-std::sin(h1), std::cos(h1));
        f = 0;
        for (const float dh : {0.f, 0.3f, -0.3f})
        {
            const Pose2 seg[2] = {Pose2{q0 + d * n0, h0 + dh}, Pose2{q1 + d * n1, h1 + dh}};
            if (rc::policy::log_p_no_contact_cells(body, seg, pc, sig, sig, 0.01f, lp_ok - 1.f) >= lp_ok) { f = 1; break; }
        }
        return f == 1; };
    bool passable = K == 0;
    int reach_k = -1;
    if (K > 0)
    {
        // the robot's own cell: its lateral offset at section 0 (nearest free cell of that column contiguous with it)
        const V2 q0 = spline.position_at(s0), n0(-std::sin(spline.heading_at(s0)), std::cos(spline.heading_at(s0)));
        const int j0 = std::clamp(int(std::lround((bel.p - q0).dot(n0) / dd)), -J, J);
        std::deque<std::pair<int, int>> qq;
        std::vector<char> seen(std::size_t(K * W), 0);
        for (int r = 0; r <= J and qq.empty(); ++r)
            for (const int jj : {j0 - r, j0 + r})
                if (jj >= -J and jj <= J and free_at(0, jj)) { qq.push_back({0, jj}); seen[std::size_t(jj + J)] = 1; }
        while (not qq.empty() and not passable)
        {
            const auto [k, jj] = qq.front(); qq.pop_front();
            reach_k = std::max(reach_k, k);
            if (k == K - 1) { passable = true; break; }
            for (int dk = -1; dk <= 1; ++dk)
                for (int dj = -1; dj <= 1; ++dj)
                {
                    const int k2 = k + dk, j2 = jj + dj;
                    if ((dk == 0 and dj == 0) or k2 < 0 or k2 >= K or j2 < -J or j2 > J) continue;
                    auto& sn = seen[std::size_t(k2 * W + j2 + J)];
                    if (sn) continue;
                    sn = 1;
                    if (free_at(k2, j2)) qq.push_back({k2, j2});
                }
        }
    }
    out.reach_k = reach_k;
    out.blocked_at = passable ? -1.f : s0 + float(reach_k + 1) * ds;
    return out;
}

// ---------------------------------------------------------------------------------------------------------------- marks
std::vector<std::vector<V2>> MarkSet::polygons(float cell) const
{
    const float h = 0.5f * 1.4143f * cell;
    std::vector<std::vector<V2>> out;
    out.reserve(marks.size() + fail.size());
    const auto sq = [&](const V2& c) { return std::vector<V2>{c + V2(-h, -h), c + V2(h, -h), c + V2(h, h), c + V2(-h, h)}; };
    for (const auto& m : marks) out.push_back(sq(m.c));
    for (const auto& f : fail) out.push_back(sq(f));
    return out;
}

float polyline_length(std::span<const V2> poly)
{
    float L = 0.f;
    for (std::size_t i = 1; i < poly.size(); ++i) L += (poly[i] - poly[i - 1]).norm();
    return L;
}

bool shorter_route(std::span<const V2> candidate, float remaining_m, float cell)
{
    return candidate.size() >= 2 and polyline_length(candidate) + cell < remaining_m;
}

bool MarkSet::add(const V2& c, float cell)
{
    const float half = cell * 1.4143f * 0.5f;   // the square polygons() draws
    for (auto& m : marks)
        if (std::abs(c.x() - m.c.x()) <= half and std::abs(c.y() - m.c.y()) <= half)
            return false;   // no new count either: observe() already scores this sweep's hit on m (no double evidence)
    marks.push_back(Mark{c});
    return true;
}

MarkSet::Look MarkSet::observe(const V2& origin, std::span<const std::pair<float, float>> br, float cell, float near_r,
                               float risk_eps, float lapse_r, bool belief)
{
    Look out;
    if (marks.empty() and fail.empty()) return out;
    // the sweep sorted by bearing ONCE: each mark then reads only the beams of its own cone (binary search), not the whole
    // sweep — live (2026-11-11 chain1) the per-mark full scan cost 92 ms/cycle at 300 marks. The search interval is widened
    // by a hair and the original cone test is applied to every candidate, so the result is identical to the full scan.
    std::vector<std::pair<float, float>> sorted(br.begin(), br.end());
    std::ranges::sort(sorted, {}, &std::pair<float, float>::first);
    // one sweep's look at a mark's cell: +1 a return AT it (hit), -1 every return in its cone beyond it (through), 0 no look
    // (too close, occluded by something in front, no beam in the cone). The cone is the cell's own angular half-width: a
    // beam grazing its edge is not a look through it (mock: real marks cleared and re-made 90x with a wider bin)
    const auto look = [&](const V2& c) -> int {
        const V2 r = c - origin;
        const float rr = r.norm(), b = std::atan2(r.y(), r.x()), half = 0.5f * cell / std::max(rr, 1e-3f);
        if (rr <= near_r) return 0;
        int n = 0; bool hit = false, front = false;
        const auto scan = [&](float lo, float hi) {
            auto it = std::ranges::lower_bound(sorted, lo, {}, &std::pair<float, float>::first);
            for (; it != sorted.end() and it->first <= hi; ++it)
                if (std::abs(wrap(it->first - b)) <= half)
                { ++n; if (it->second < rr - cell) front = true; else if (it->second <= rr + cell) hit = true; } };
        const float pi = std::numbers::pi_v<float>, w = half + 1e-3f, lo = b - w, hi = b + w;
        scan(std::max(lo, -pi - 1e-3f), std::min(hi, pi + 1e-3f));
        if (lo < -pi) scan(lo + 2.f * pi, pi + 1e-3f);           // the cone wraps past -pi
        if (hi > pi) scan(-pi - 1e-3f, hi - 2.f * pi);           // the cone wraps past +pi
        if (hit) return +1;
        if (front or n == 0) return 0;
        return -1; };
    const double p_fa = risk_eps, clear_lo = std::log((1.0 - risk_eps) / risk_eps);
    std::vector<Mark> keep;
    keep.reserve(marks.size());
    for (auto mb : marks)
    {
        bool gone = false;
        const int o = look(mb.c);
        if (not belief) gone = o < 0;   // the old flag: one see-through sweep
        else
        {
            if (o > 0) { mb.throughs += mb.run; mb.run = 0; ++mb.hits; }
            else if (o < 0) ++mb.run;
            if (mb.run > 0)
            {   // log P(run of m throughs | PRESENT, history) = sum_i log (throughs+1+i)/(hits+throughs+2+i)  (Beta-binomial)
                double lp = 0.0;
                for (int i = 0; i < mb.run; ++i) lp += std::log((mb.throughs + 1.0 + i) / (mb.hits + mb.throughs + 2.0 + i));
                gone = double(mb.run) * std::log1p(-p_fa) - lp >= clear_lo;
            }
        }
        if (gone) out.cleared_at.push_back(mb.c); else keep.push_back(mb);
    }
    out.cleared = out.cleared_at.size();
    marks.swap(keep);
    // a failure mark is evidence about the world AS IT WAS when the specialist failed: when an observed obstacle within
    // lapse_r of it is seen gone (the person left the doorway), that evidence lapses with it
    if (not out.cleared_at.empty() and not fail.empty())
    {
        std::vector<V2> kf;
        for (const V2& c : fail)
        {
            const bool lapse = std::ranges::any_of(out.cleared_at, [&](const V2& q) { return (q - c).norm() <= lapse_r; });
            if (not lapse) kf.push_back(c);
        }
        out.lapsed = fail.size() - kf.size();
        fail.swap(kf);
    }
    return out;
}

float MarkSet::carry_if_relocalised(const std::function<V2(const V2&)>& transport, const std::function<float(const V2&)>& sigma_slip,
                                    float k_eps, float cell)
{
    bool jumped = false; float worst = 0.f;
    const auto check = [&](const V2& c) {
        const float disp = (transport(c) - c).norm();
        worst = std::max(worst, disp);
        if (disp > k_eps * sigma_slip(c) and disp > 0.5f * cell) jumped = true; };
    for (const auto& m : marks) check(m.c);
    for (const auto& f : fail) check(f);
    if (not jumped) return -1.f;
    for (auto& m : marks) m.c = transport(m.c);
    for (auto& f : fail) f = transport(f);
    return worst;
}

// ------------------------------------------------------------------------------------------------------- progress
void ProgressMonitor::record(float t, float adv, float s, float rot)
{
    hist_.push_back({t, adv, s, rot});
    while (not hist_.empty() and hist_.front().t < t - 3.f) hist_.pop_front();
}

ProgressMonitor::Verdict ProgressMonitor::judge(const In& in) const
{
    Verdict v;
    const float H = in.T_lag + 0.5f * in.W / in.v_max;
    v.H = H;
    float pred = 0.f, pred_rot = 0.f; int n_in = 0;
    for (const auto& c : hist_)
        if (c.t >= in.t - H - in.T_lag and c.t < in.t - in.T_lag) { pred += std::max(0.f, c.adv) * in.dt; pred_rot += std::abs(c.rot) * in.dt; ++n_in; }
    float s_then = in.s_prog;
    for (const auto& c : hist_) if (c.t <= in.t - H + 1e-6f) { s_then = c.s; v.have = true; }
    const float hsp = 0.10f, va = std::abs(in.v) * in.age;
    const float resn = std::sqrt(in.sigma_pose * in.sigma_pose + hsp * hsp / 12.f + va * va);
    const float wa = std::abs(in.w) * in.age;
    const float yaw_res = std::sqrt(0.01f * 0.01f + wa * wa) + resn / std::max(0.1f, in.circumscribed_r);
    v.pred = pred; v.realised = in.s_prog - s_then; v.resn = resn;
    v.idle = v.have and n_in > 0 and pred <= resn and pred_rot <= yaw_res and in.s_prog - s_then <= resn
             and in.goal_applies and in.goal_dist > in.goal_tol + resn;
    v.censored = s_then <= resn and in.s_prog <= resn;
    v.fail = v.have and not v.censored and ((pred > resn and in.s_prog - s_then <= resn) or v.idle);
    return v;
}
}  // namespace rc::chain
