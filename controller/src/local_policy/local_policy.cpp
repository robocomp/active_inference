#include "local_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace rc::policy
{
namespace
{
constexpr float kPi = 3.14159265f;
constexpr float kInf = std::numeric_limits<float>::infinity();

float wrap(float a) { while (a > kPi) a -= 2.f * kPi; while (a <= -kPi) a += 2.f * kPi; return a; }

// n values evenly over [lo, hi] (one value -> the midpoint).
std::vector<float> linspace(float lo, float hi, int n)
{
    std::vector<float> out;
    if (n <= 1 or hi <= lo) { out.push_back(0.5f * (lo + hi)); return out; }
    for (int i = 0; i < n; ++i) out.push_back(lo + (hi - lo) * float(i) / float(n - 1));
    return out;
}

// ★ The base's deadband: a command it would not execute is not a candidate at all (|x| in (0, min_eff) dropped).
void keep_executable(std::vector<float>& xs, float min_eff)
{
    std::erase_if(xs, [&](float x) { return x != 0.f and std::abs(x) < min_eff; });
    std::ranges::sort(xs);
    const auto [b, e] = std::ranges::unique(xs, [](float a, float c) { return std::abs(a - c) < 1e-5f; });
    xs.erase(b, e);
}

// Heading the goal asks for at `at`: its yaw if set, else the bearing to p (Heading kind only).
std::optional<float> target_yaw(const Goal& g, const Pose2& at)
{
    if (g.yaw) return *g.yaw;
    if (g.kind != Goal::Kind::Heading) return std::nullopt;
    const Eigen::Vector2f d = g.p - at.p;
    if (d.squaredNorm() < 1e-12f) return std::nullopt;
    return std::atan2(d.y(), d.x());
}

// Progress (m) of moving from a to b.
// ★ Place/Doorway: decrease of the geodesic field. If the field is +inf at the START (goal walled off, nothing can
//   be measured) every motion makes zero progress — no direction is preferred and only release can act. If it is
//   finite at the start but +inf at the end, the motion leaves the reachable set: -inf.
// ★ Heading: reduction of |angle error| (rad) times the body's circumscribed radius — a radian of heading and a
//   metre of travel trade at the body's own lever arm (a radian of turn moves its outermost point that far) —
//   minus the distance travelled (controller ruling: a heading goal holds position).
//   Place with `yaw`: the same heading term weighted by w = tol_xy / max(tol_xy, d), d = distance of the START to
//   the goal — full weight inside the tolerance, fading as 1/d beyond it (no new constant: the goal's own
//   tolerance is the scale), so the heading emerges from the path far away and is held near the goal.
float progress(const PolicyInput& in, const Pose2& a, const Pose2& b)
{
    const Goal& g = *in.goal;
    const float lever = in.body->circumscribed_radius();
    float pr = 0.f, wt = 1.f;
    if (g.kind != Goal::Kind::Heading)
    {
        const float fa = in.field->at(a.p);
        if (not std::isfinite(fa)) return 0.f;
        const float fb = in.field->at(b.p);
        if (not std::isfinite(fb)) return -kInf;
        pr = fa - fb;
        // ★ ROUND 5 (field resolution): a decrease the field cannot resolve (CostToGo::resolution_m — octile metric +
        //   interpolation) is not progress; it counts as zero (release_side_on's 9 mm "two-step progress" was this).
        if (pr > 0.f and pr < in.field->resolution_m((b.p - a.p).norm())) pr = 0.f;
        if (not g.yaw) return pr;
        wt = g.tol_xy / std::max(g.tol_xy, (a.p - g.p).norm());
    }
    const auto ty = target_yaw(g, a);
    const float turn = ty ? (std::abs(wrap(*ty - a.yaw)) - std::abs(wrap(*ty - b.yaw))) * lever : 0.f;
    // ★ A heading goal HOLDS POSITION: travel is pure cost (metres), so a turn in place beats an arc that lands the
    //   same heading 0.6 m away (measured before this term: open space chose "back up arcing right v=-0.50 w=-1.20").
    if (g.kind == Goal::Kind::Heading) return turn - (b.p - a.p).norm();
    return pr + wt * turn;
}

// Inside the goal's own tolerances at `at` (they are the goal's, not the policy's).
bool arrived(const PolicyInput& in, const Pose2& at)
{
    const Goal& g = *in.goal;
    if (g.kind == Goal::Kind::Heading)
    {
        const auto ty = target_yaw(g, at);
        return not ty or std::abs(wrap(*ty - at.yaw)) <= g.tol_yaw;
    }
    const bool xy = g.kind == Goal::Kind::Doorway ? in.field->at(at.p) <= g.tol_xy : (at.p - g.p).norm() <= g.tol_xy;
    return xy and (not g.yaw or std::abs(wrap(*g.yaw - at.yaw)) <= g.tol_yaw);
}

// prog = ONE-step progress (start -> end); prog2 = TWO-step progress (round-4 ruling): the best of the end itself and every
// ELIGIBLE coarse continuation from it, measured from the decision's start.
// rate = best progress RATE (m/s, round 5): max(prog / dur, max over eligible continuations of
// progress(start, cont end) / (dur + cont duration)); ub_rate: the same over ALL continuations (no contact sweep).
struct Cand { float v, w; float lps = 0.f, prog = 0.f, prog2 = 0.f, ub2 = 0.f, dur = 0.f, rate = 0.f, tau = 0.f, ub_rate = 0.f, lpc = 0.f; Pose2 end{}, plan_end{};
              bool elig = false, lpc_done = false; int conts = -1; };

// Contact log-probability of (v, w) started at `from` with velocity (v0, w0); `end` receives the rest pose.
// lps_floor: below it only a bound is needed (contact.h) — candidates need exact values only while releasable
// (P < release_risk_max), continuations only while eligible (lps >= log(1 - eps)).
// Duration (s) of a rollout: its poses are dt apart except the last step, which may be shorter — (n-1) dt is an upper
// bound by less than dt, the same for every candidate (it only rescales rates by at most dt / horizon).
float duration(const std::vector<Pose2>& sw, const PolicyParams& p) { return std::max(p.dt, float(sw.size() - 1) * p.dt); }

float lp_safe(const PolicyInput& in, const PolicyParams& p, const Pose2& from, float v0, float w0, float v, float w, Pose2* end, float lps_floor,
              float* dur = nullptr)
{
    const auto sw = rollout(from, v0, w0, v, w, p.horizon_s, in.base, p.dt);
    if (end) *end = sw.back();
    if (dur) *dur = duration(sw, p);
    return log_p_no_contact_cells(*in.body, sw, in.returns, in.sigma_rel_xy, in.sigma_rel_yaw, in.body_geometry_sigma_m, lps_floor);
}

bool eligible(float lps, const PolicyParams& p) { return 1.f - std::exp(lps) <= p.risk_eps; }

std::string words(float v, float w)
{
    const char* lin = v > 0.f ? "forward" : v < 0.f ? "back up" : "turn in place";
    const char* ang = w > 0.f ? "left" : w < 0.f ? "right" : (v == 0.f ? "" : "straight");
    char buf[128];
    if (v == 0.f) std::snprintf(buf, sizeof buf, "%s %s (w=%+.2f)", lin, ang, w);
    else if (w == 0.f) std::snprintf(buf, sizeof buf, "%s %s (v=%+.2f)", lin, ang, v);
    else std::snprintf(buf, sizeof buf, "%s arcing %s (v=%+.2f w=%+.2f)", lin, ang, v, w);
    return buf;
}

bool finite_input(const PolicyInput& in)
{
    auto ok = [](float x) { return std::isfinite(x); };
    if (not (ok(in.pose.p.x()) and ok(in.pose.p.y()) and ok(in.pose.yaw) and ok(in.v_now) and ok(in.w_now) and ok(in.sigma_rel_xy)
             and ok(in.sigma_rel_yaw) and ok(in.pose_drift_sigma_m) and ok(in.pose_drift_time_s) and ok(in.returns_age_s) and ok(in.latency_s) and ok(in.body_geometry_sigma_m))) return false;
    for (const Return& r : in.returns)
        if (not (ok(r.p.x()) and ok(r.p.y()) and ok(r.sigma) and ok(r.p_exist) and ok(r.sxx) and ok(r.sxy) and ok(r.syy))) return false;
    return true;
}
}  // namespace

const char* mode_name(Decision::Mode m)
{
    switch (m)
    {
        case Decision::Mode::Drive: return "Drive";
        case Decision::Mode::Release: return "Release";
        case Decision::Mode::Arrived: return "Arrived";
        case Decision::Mode::Stuck: return "Stuck";
        case Decision::Mode::Blind: return "Blind";
    }
    return "?";
}

Pose2 policy_start(const PolicyInput& in) { return advance(in.pose, in.v_now, in.w_now, std::max(0.f, in.latency_s)); }

// ★ Continuability: from a pose AT REST (every candidate ends braked), the coarse set spans the dynamic window
//   from rest — v in n_cont_v values over [-min(v_max, a_lin*T), +...], w likewise with a_rot — minus the deadband
//   and minus (0,0) (standing still is always "safe" and is not a way to continue). k = members that are eligible
//   AND make progress >= 0; log P = log((k+1)/(N+2)) (Laplace: never 0 or 1 from a finite sample).
// ★ A pose inside the goal's tolerances has nothing left to continue: k = N. Without this, a pose 0.1 m from the
//   goal scores as a TRAP (every continuation overshoots, progress < 0) and the look-ahead steers away from arrival
//   (measured: side-on test, the best end pose 0.11 m from the goal read k = 4/14).
namespace
{
std::vector<std::pair<float, float>> coarse_set(const PolicyInput& in, const PolicyParams& p)
{
    const float vc = std::min(in.base.v_max, in.base.a_lin * p.horizon_s);
    const float wc = std::min(in.base.w_max, in.base.a_rot * p.horizon_s);
    auto vs = linspace(-vc, vc, p.n_cont_v), ws = linspace(-wc, wc, p.n_cont_w);
    keep_executable(vs, in.base.v_min_eff);
    keep_executable(ws, in.base.w_min_eff);
    std::vector<std::pair<float, float>> out;
    for (float v : vs) for (float w : ws) if (v != 0.f or w != 0.f) out.emplace_back(v, w);
    return out;
}

// The coarse continuation ROLLOUTS from a pose at rest, computed once per end pose and cached (round 4: they feed both
// the continuability count and the second step of the two-step progress, and the Drive bound below).
struct Conts { std::vector<std::vector<Pose2>> sw; bool arrived = false, done = false; int n = 0; };
Conts make_conts(const PolicyInput& in, const PolicyParams& p, const Pose2& from)
{
    Conts c; c.done = true;
    const auto set = coarse_set(in, p);
    c.n = int(set.size());
    if (in.body == nullptr or in.goal == nullptr or (in.goal->kind != Goal::Kind::Heading and in.field == nullptr)) return c;
    if (arrived(in, from)) { c.arrived = true; return c; }
    for (const auto& [v, w] : set) c.sw.push_back(rollout(from, 0.f, 0.f, v, w, p.horizon_s, in.base, p.dt));
    return c;
}
// `in.returns` are surface cells already (decide() merges once per cycle).
// ref (optional): also return, over the ELIGIBLE continuations, the largest progress(ref, continuation end) (best_prog)
// and the largest RATE progress(ref, end) / (base_dur + continuation duration) (best_rate); -inf if none.
// ★ EXACT pruning: a continuation's contact sweep (its rollout is cached) is evaluated only if its answer can change
//   something — count it for continuability (progress from `from` >= 0), or raise best_prog / best_rate.
float eval_conts(const PolicyInput& in, const PolicyParams& p, const Pose2& from, const Conts& c,
                 const Pose2* ref = nullptr, float base_dur = 0.f, float* best_prog = nullptr, float* best_rate = nullptr,
                 float* best_rate_dur = nullptr, Pose2* best_rate_end = nullptr)
{
    if (best_rate_dur) *best_rate_dur = 0.f;
    if (best_prog) *best_prog = -kInf;
    if (best_rate) *best_rate = -kInf;
    if (c.arrived) return std::log((float(c.n) + 1.f) / (float(c.n) + 2.f));   // nothing left to continue: k = N
    if (c.sw.empty()) return std::log(1.f / (float(c.n) + 2.f));
    struct Item { std::size_t i; float pf, pr, rt, dur; };
    std::vector<Item> items;
    for (std::size_t i = 0; i < c.sw.size(); ++i)
    {
        const float pr = ref ? progress(in, *ref, c.sw[i].back()) : -kInf;
        const float dj = duration(c.sw[i], p);
        items.push_back({i, progress(in, from, c.sw[i].back()), pr, ref ? pr / (base_dur + dj) : -kInf, dj});
    }
    std::ranges::sort(items, std::greater{}, &Item::rt);
    int k = 0; float bp = -kInf, br = -kInf;
    for (const Item& it : items)
    {
        const bool need_k = it.pf >= 0.f, need_b = ref != nullptr and (it.pr > bp or it.rt > br);
        if (not (need_k or need_b)) continue;
        const float lps = log_p_no_contact_cells(*in.body, c.sw[it.i], in.returns, in.sigma_rel_xy, in.sigma_rel_yaw, in.body_geometry_sigma_m,
                                                 std::log1p(-p.risk_eps));
        if (not eligible(lps, p)) continue;
        if (need_k) ++k;
        if (ref) { bp = std::max(bp, it.pr); if (it.rt > br) { br = it.rt; if (best_rate_dur) *best_rate_dur = it.dur; if (best_rate_end) *best_rate_end = c.sw[it.i].back(); } }
    }
    if (best_prog) *best_prog = bp;
    if (best_rate) *best_rate = br;
    return std::log((float(k) + 1.f) / (float(c.n) + 2.f));
}
float log_p_continuable_cells(const PolicyInput& in, const PolicyParams& p, const Pose2& from)
{ return eval_conts(in, p, from, make_conts(in, p, from)); }
}  // namespace

float log_p_continuable(const PolicyInput& in0, const PolicyParams& p, const Pose2& from)
{
    if (in0.body == nullptr) return log_p_continuable_cells(in0, p, from);   // (it answers the no-body case itself)
    const auto cells = surface_cells(in0.returns, in0.pose.p, bearing_bin_rad(*in0.body));
    PolicyInput in = in0; in.returns = cells;
    return log_p_continuable_cells(in, p, from);
}

Decision decide(const PolicyInput& in0, const PolicyParams& p, ProgressMonitor* mon)
{
    Decision d{.mode = Decision::Mode::Stuck, .why = {}};
    if (in0.body == nullptr or in0.goal == nullptr) { d.why = "no body or no goal"; return d; }
    // ---- a NaN/inf anywhere would silently poison every probability: no prediction, stop
    if (not finite_input(in0)) { d.mode = Decision::Mode::Blind; d.why = "non-finite input"; return d; }
    // ★ returns -> surface cells ONCE per cycle (contact.h): every candidate and continuation sweeps the same cells.
    //   Sensor origin for the bearing bins = the robot centre at the DECISION pose `pose` (where the returns were
    //   re-expressed; the live buffer's returns come from nearby poses and two off-centre LiDARs — an approximation
    //   whose error is the bin's arc, not a bias in range).
    const auto cells = surface_cells(in0.returns, in0.pose.p, bearing_bin_rad(*in0.body));
    PolicyInput in = in0; in.returns = cells;
    const Goal& g = *in.goal;
    if (g.kind != Goal::Kind::Heading and in.field == nullptr) { d.why = "place/doorway goal without a field"; return d; }

    if (arrived(in, in.pose)) { d.mode = Decision::Mode::Arrived; d.why = "inside the goal tolerances"; return d; }
    // ---- Blind: returns older than the window they describe — no contact prediction can be trusted
    if (in.returns_age_s > p.returns_max_age_s)
    {
        char buf[96]; std::snprintf(buf, sizeof buf, "returns %.2f s old (> %.2f s): stop", in.returns_age_s, p.returns_max_age_s);
        d.mode = Decision::Mode::Blind; d.why = buf; return d;
    }

    // ★ LATENCY (review ruling): the command decided now takes effect latency_s later (pose age + command latency);
    //   until then the base keeps (v_now, w_now). Every candidate is rolled out from THAT pose, starting at the
    //   current velocity — the coast of a moving base is part of every motion, including "stop and turn".
    const Pose2 start = policy_start(in);

    // ---- ROUND 6: judge the pending prediction (local_policy.h, ProgressMonitor) on what actually happened
    const bool monitored = mon != nullptr and g.kind != Goal::Kind::Heading;
    bool prediction_failed = false;
    char failed_why[160] = "";
    if (monitored and mon->pending and in.now_s - mon->t0 >= double(mon->tau))
    {
        const CostToGo& pf = in.progress_field ? *in.progress_field : *in.field;   // round 8: trap-free
        const float f_now = pf.at(in.pose.p);
        const float realized = mon->f0 - f_now;
        // ★ ROUND 7: the instrument's full resolution — field interpolation and the pose noise seen through the slope
        const float slope = [&] {
            const float h = pf.cell();
            const float fx1 = pf.at(in.pose.p + Eigen::Vector2f(h, 0.f)), fx0 = pf.at(in.pose.p - Eigen::Vector2f(h, 0.f));
            const float fy1 = pf.at(in.pose.p + Eigen::Vector2f(0.f, h)), fy0 = pf.at(in.pose.p - Eigen::Vector2f(0.f, h));
            if (not (std::isfinite(fx1) and std::isfinite(fx0) and std::isfinite(fy1) and std::isfinite(fy0))) return 1.f;
            return 0.5f * std::hypot(fx1 - fx0, fy1 - fy0) / h; }();
        const float res_field = pf.resolution_m((in.pose.p - mon->p0.p).norm());
        // round 8: the CHANGE of the absolute pose error over the elapsed time (OU: sigma sqrt(2 (1 - e^{-tau/T})))
        const double el = in.now_s - mon->t0;
        const float sd = in.pose_drift_sigma_m > 0.f and in.pose_drift_time_s > 0.f
                       ? in.pose_drift_sigma_m * std::sqrt(2.f * (1.f - std::exp(-float(el) / in.pose_drift_time_s))) : 0.f;
        const float res = std::hypot(res_field, slope * sd);
        if (std::isfinite(realized) and realized > res) mon->reset();              // the plan delivered: forget it
        else if (mon->dF_pred > res)
        {
            prediction_failed = true;
            std::snprintf(failed_why, sizeof failed_why, "predicted %.3f m in %.2f s, realized %.3f m (resolution %.3f)",
                          mon->dF_pred, mon->tau, std::isfinite(realized) ? realized : -1.f, res);
            mon->reset();
        }
        else mon->reset();                                                         // too small to judge
    }

    // ---- Candidates: n_v x n_w over the dynamic window, plus pure turns of both signs and a straight reverse
    const float T = p.horizon_s;
    const float vlo = std::max(-in.base.v_max, in.v_now - in.base.a_lin * T), vhi = std::min(in.base.v_max, in.v_now + in.base.a_lin * T);
    const float wlo = std::max(-in.base.w_max, in.w_now - in.base.a_rot * T), whi = std::min(in.base.w_max, in.w_now + in.base.a_rot * T);
    auto vs = linspace(vlo, vhi, p.n_v), ws = linspace(wlo, whi, p.n_w);
    // ★ always present: v = 0 (pure turns), w = 0 (straight lines), a reverse speed and turns of BOTH signs —
    //   whatever the current velocity, "stop and turn" and "back straight out" are always on the table (and are
    //   rolled out from the current velocity like every other candidate).
    const float vr = std::min(in.base.v_max, in.base.a_lin * T), wr = std::min(in.base.w_max, in.base.a_rot * T);
    vs.push_back(0.f); ws.push_back(0.f);
    if (vlo >= 0.f) vs.push_back(-vr);
    if (wlo >= 0.f) ws.push_back(-wr);
    if (whi <= 0.f) ws.push_back(wr);
    keep_executable(vs, in.base.v_min_eff);
    keep_executable(ws, in.base.w_min_eff);

    std::vector<Cand> cands;
    for (float v : vs)
        for (float w : ws)
        {
            if (v == 0.f and w == 0.f) continue;     // standing still is the absence of a decision, not a candidate
            Cand c{.v = v, .w = w};
            c.lps = lp_safe(in, p, start, in.v_now, in.w_now, v, w, &c.end, std::log1p(-p.release_risk_max), &c.dur);
            c.elig = eligible(c.lps, p);
            if (c.elig) { c.prog = progress(in, start, c.end); ++d.eligible; }
            cands.push_back(c);
        }
    // ★ one continuation rollout set per candidate end pose, cached (make_conts); its sweeps yield BOTH the continuability
    //   and the second step of the two-step progress (ruling round 4) — no rollout is ever repeated.
    std::vector<Conts> conts;
    conts.reserve(cands.size());
    auto ensure_conts = [&](Cand& c) -> const Conts& { if (c.conts < 0) { c.conts = int(conts.size()); conts.push_back(make_conts(in, p, c.end)); } return conts[c.conts]; };
    auto ensure_lpc = [&](Cand& c)
    {
        if (c.lpc_done) return;
        float second = -kInf, second_rate = -kInf, second_dur = 0.f;
        Pose2 second_end = c.end;
        c.lpc = eval_conts(in, p, c.end, ensure_conts(c), &start, c.dur, &second, &second_rate, &second_dur, &second_end);
        c.prog2 = std::max(c.prog, second);
        c.rate = std::max(c.prog / c.dur, second_rate);
        const bool one_step = c.prog / c.dur >= second_rate;
        c.tau = one_step ? c.dur : c.dur + second_dur;          // the plan the rate belongs to
        c.plan_end = one_step ? c.end : second_end;
        c.lpc_done = true;
    };
    auto finish = [&](Decision::Mode m, const Cand& c, std::string why)
    {
        d.mode = m; d.v = c.v; d.w = c.w; d.log_p_safe = c.lps; d.log_p_cont = c.lpc; d.progress_m = c.prog2; d.why = std::move(why);
        d.prediction_failed = m == Decision::Mode::Release and prediction_failed;
        return d;
    };

    // ---- Drive: best score among ELIGIBLE candidates whose TWO-STEP progress is > 0 (risk_eps gates normal motion).
    // ★ ROUND-4 RULING (the user's "evaluate its movements ahead"): a candidate's progress is
    //     prog2 = progress(start, end) max over its eligible coarse continuations c of progress(start, c.end)
    //   i.e. field(start) - min(field(end), min field(continuation ends)) for Place/Doorway, the same heading-agreement
    //   term for Heading goals. "Turn to line up, then go" is a Drive: the turn makes no field progress by itself, the
    //   continuation it enables does. Release (and the trap the driver lays on it) is left for the genuine wedge: NO
    //   eligible candidate gets anywhere even one step later. (Round 5: scored by RATE, below.)
    // ★ ROUND 5: PROGRESS IS A RATE. A candidate is scored by its best descent RATE (m/s): its own progress over its own
    //   duration (hold + brake), or its progress together with an eligible continuation's over both durations. Turning
    //   in place first and driving later takes the time of both, so it halves the rate of driving now: the borrowed
    //   progress of round 4 no longer wins the turn-forever dither, and no rule was added for it.
    //   Rate scale = progress_scale_m / horizon_s (0.10 m/s per nat with the defaults): the descent rate worth one nat,
    //   the same meaning as the old 0.10 m over one horizon. Drive iff some eligible candidate has prog2 > 0
    //   (two-step, at the field's resolution); among those, score = lps + rate / rate_scale + lpc.
    // ★ EXACT branch-and-bound (budget): rate <= ub_rate over ALL cached continuation ends (no contact sweep) and
    //   lpc <= lpc_top < 0; candidates are swept in decreasing bound until the bound cannot beat the best exact score.
    const float rate_scale = p.progress_scale_m / std::max(p.horizon_s, 1e-3f);
    const float lpc_top = [&] { const float n = float(coarse_set(in, p).size()); return std::log((n + 1.f) / (n + 2.f)); }();
    std::vector<Cand*> drive;
    for (auto& c : cands)
    {
        if (not c.elig) continue;
        const Conts& cc = ensure_conts(c);
        c.ub2 = c.prog; c.ub_rate = c.prog / c.dur;
        for (const auto& sw : cc.sw)
        {
            const float pr = progress(in, start, sw.back());
            c.ub2 = std::max(c.ub2, pr);
            c.ub_rate = std::max(c.ub_rate, pr / (c.dur + duration(sw, p)));
        }
        if (c.ub2 > 0.f) drive.push_back(&c);
    }
    auto bound = [&](const Cand* c) { return c->lps + c->ub_rate / rate_scale + lpc_top; };
    std::ranges::sort(drive, std::greater{}, bound);
    Cand* best_drive = nullptr; float best_s = -kInf;
    for (Cand* c : drive)
    {
        if (bound(c) <= best_s) break;
        ensure_lpc(*c);
        if (not (c->prog2 > 0.f)) continue;
        const float s = c->lps + c->rate / rate_scale + c->lpc;
        if (s > best_s) { best_s = s; best_drive = c; }
    }
    if (best_drive != nullptr and not prediction_failed)
    {
        if (monitored and not mon->pending)
        {
            // the prediction this Drive commits to: its plan's descent over its plan's duration
            mon->pending = true; mon->t0 = in.now_s; mon->tau = best_drive->tau;
            // round 8: the plan's descent on the TRAP-FREE field, from the pose the realized descent will be measured from
            const CostToGo& pf = in.progress_field ? *in.progress_field : *in.field;
            mon->f0 = pf.at(in.pose.p); mon->dF_pred = mon->f0 - pf.at(best_drive->plan_end.p); mon->p0 = in.pose;
            if (not std::isfinite(mon->dF_pred)) mon->reset();
        }
        return finish(Decision::Mode::Drive, *best_drive, "drive: " + words(best_drive->v, best_drive->w));
    }

    // ---- No eligible motion makes progress: the robot is in a wedge, and the only option is to move back and turn
    //      a bit (user's rule). First the impossible case: a return inside the BARE hull contradicts "not touching
    //      now" (the robot cannot overlap a wall) — a sensing/pose error, not a place a motion can be chosen from.
    //      Returns count by their existence (a ghost inside the body is not a contradiction).
    // ★ D1 (Task 6): judged on the SURFACE CELLS — each one the estimate of the nearest surface on its bearing — not on
    //   the raw hits: one noisy hit (sigma_r 0.02) beside a wall 2.5 cm away lands inside the hull ~10 % of the time
    //   (22 of them at tour3_1_leg3's start held the robot Stuck at t = 0 with nothing touching it).
    float inside = 0.f;
    // ★ ROUND 6 (B1): LiDAR-derived cells only (aggregate == true). An unseen-contact hypothesis inside the body is not
    //   a sensing contradiction — it is a hypothesis the body's presence refutes (UnseenContact does that).
    for (const Return& r : in.returns)
        if (r.aggregate and r.p_exist > 0.f and hull_distance(*in.body, start, r.p) < 0.f) inside += r.p_exist;
    if (inside > 0.f)
    {
        char buf[96]; std::snprintf(buf, sizeof buf, "%.1f returns inside the body (sensing/pose error): hold", inside);
        d.why = buf; return d;
    }

    // ---- Release (controller ruling, fix round 5): NOT held to risk_eps. Among ALL candidates with v <= 0 (reverse
    //      arcs and in-place turns), lexicographically: (a) those that INCREASE continuability
    //      (log_p_cont(end) > log_p_cont(now)) if any exist, else all of them; (b) the smallest P(contact); ties go
    //      to the larger continuability gain, then to the larger reverse travel.
    // ★ STUCK: no release candidate is more likely contact-free than not (P < release_risk_max, the policy's second
    //   stated risk level — local_policy.h). A release that will probably bump is not a release; the task layer is told.
    // ★ A candidate at P >= release_risk_max can never be released, whichever pass picks it (the pick would fall
    //   through to the next pass and finally to Stuck), so its continuations are never rolled out: the scans run
    //   over the RELEASABLE candidates only — the same decision as scanning all and falling through, at a fraction
    //   of the cost (the reviewer's wedge: 47 x 14 continuation sweeps -> 0).
    // ★ Exact scan: visiting in decreasing log_p_safe, the first candidate with a gain is the (a)-member with the
    //   smallest P; only its equal-lps ties can beat it (larger gain), and a tie reaching lpc_max ends the scan.
    //   Pass (b) reuses every continuation already rolled out.
    const float lps_stuck = std::log1p(-p.release_risk_max);     // releasable <=> P(contact) < release_risk_max
    std::vector<Cand*> rel;
    bool any_reverse = false;
    for (auto& c : cands)
        if (c.v <= 0.f)
        {
            any_reverse = true;
            if (c.lps > lps_stuck) rel.push_back(&c);
        }
    if (not any_reverse) { d.why = "no progress, and no reverse or in-place motion to release with: hold"; return d; }
    if (rel.empty())
    {
        // ★ the candidates' lps are BOUNDS below the release bar (their sweeps stopped once provably above it), so the
        //   least likely contact is re-swept exactly here to name it — a terminal branch, worth the cost.
        const Cand* least = nullptr; float least_lps = -kInf;
        for (auto& c : cands)
            if (c.v <= 0.f)
            {
                const float lps = lp_safe(in, p, start, in.v_now, in.w_now, c.v, c.w, nullptr, -kInf);
                if (lps > least_lps) { least_lps = lps; least = &c; }
            }
        char buf[160];
        std::snprintf(buf, sizeof buf, "stuck: best release %s has P(contact)=%.3f >= release_risk_max %.2f: hold", words(least->v, least->w).c_str(), 1.f - std::exp(least_lps), p.release_risk_max);
        d.why = buf; return d;
    }
    const float lpc_now = eval_conts(in, p, start, make_conts(in, p, start));
    const float n_cont = float(coarse_set(in, p).size());
    const float lpc_max = std::log((n_cont + 1.f) / (n_cont + 2.f));
    std::ranges::sort(rel, std::greater{}, [](const Cand* x) { return std::pair{x->lps, -x->v}; });
    constexpr float kTie = 1e-6f;
    Cand* best = nullptr;
    if (lpc_now < lpc_max)                        // (a): a gain is possible at all
        for (Cand* c : rel)
        {
            if (best != nullptr and c->lps < best->lps - kTie) break;
            ensure_lpc(*c);
            if (c->lpc > lpc_now and (best == nullptr or c->lpc > best->lpc)) best = c;
            if (best != nullptr and best->lpc >= lpc_max) break;
        }
    if (best == nullptr)                          // (b): nobody gains — the least likely contact, ties to continuability
        for (Cand* c : rel)
        {
            if (best != nullptr and c->lps < best->lps - kTie) break;
            ensure_lpc(*c);
            if (best == nullptr or c->lpc > best->lpc) best = c;
            if (best->lpc >= lpc_max) break;
        }
    char buf[400];
    std::snprintf(buf, sizeof buf, "release%s%s: %s; P(contact)=%.3f; continuability %.2f -> %.2f nats", prediction_failed ? " (prediction failed: " : "",
                  prediction_failed ? (std::string(failed_why) + ")").c_str() : "", words(best->v, best->w).c_str(),
                  1.f - std::exp(best->lps), lpc_now, best->lpc);
    return finish(Decision::Mode::Release, *best, buf);
}
}  // namespace rc::policy
