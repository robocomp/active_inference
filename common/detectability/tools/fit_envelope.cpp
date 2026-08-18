/*
 * fit_envelope.cpp — OFFLINE maximum-likelihood fit of rc::detect::DetectorEnvelope from a live agent log.
 *
 * WHAT THIS IS FOR. Every stand-off the fleet now computes is the argmax of P(detect | fill), and that curve
 * is currently an admitted PRIOR (min_fill 0.10 / max_fill 0.60 / soft 0.06) — a guess about YOLO, not a
 * measurement of it. Since rc::nbv derives every viewpoint from it, those three numbers are the dominant
 * source of error in the whole next-best-view stack. This tool replaces them with a fit, AND with the
 * covariance of that fit — which is what seeds the online (phase-2) recursive filter.
 *
 * THE MODEL. Each logged cycle is one Bernoulli trial: the object projected to `fill`, and the detector
 * either fired or it did not. P(detect) is unimodal in fill (too few pixels below, frame overflow above):
 *
 *     p(fill; a, b, s) = σ((fill − a)/s) · σ((b − fill)/s)          a = min_fill, b = max_fill, s = soft
 *
 * fitted by WEIGHTED maximum likelihood, then Laplace-approximated at the optimum for the covariance.
 *
 * ★THE WEIGHTS ARE THE POINT, not a refinement. A miss has three possible causes: bad framing, occlusion, or
 * the object was never there. Attribute all of them to the envelope and it absorbs occlusion and phantoms and
 * you learn a curve that is too pessimistic — and the error is CIRCULAR, because the removal channel uses
 * p_detect to decide absence, so a corrupted envelope corrupts the very existence estimates that label this
 * data. The fix is not to exclude doubtful cycles (that is a threshold) but to weight each trial by
 * P(exists) — the `ex_p` column — so an existence-doubtful cycle contributes little information rather than
 * wrong information. `--no-weights` turns this off to show you how much it mattered.
 *
 * ★THE ASPECT CHECK is not optional either. `fill` is max(Δcol/W, Δrow/H), so a grazing or edge-on view is a
 * tall thin sliver whose MAX is large. Those views miss for reasons that have nothing to do with framing, and
 * their misses land in high-fill bins where they drag the fitted max_fill shoulder down to explain a
 * non-framing effect. This tool therefore ALWAYS refits on the compact subset and prints both, so the bias is
 * visible rather than baked in. (This is why table_instance.h logs the two axes separately: the max alone
 * cannot be un-mixed after the fact.)
 *
 * ★LOCALE. These machines run LANG=es_ES.UTF-8 and the agent that WROTE this file is a Qt program. Parsing is
 * std::from_chars ONLY — strtof/atof/stod/>> read through LC_NUMERIC and would stop at the '.' of "0.626452"
 * and silently return 0. main() calls setlocale(LC_ALL, "") deliberately so this harness runs under the SAME
 * locale as the agent and cannot answer a different question than the one you asked. See CLAUDE.md.
 *
 * BUILD (self-contained; Eigen only):
 *     g++ -std=c++23 -O2 -I/usr/include/eigen3 -o fit_envelope fit_envelope.cpp
 *
 * RUN:
 *     ./fit_envelope ../../../table_concept/etc/ai2_log.csv
 *     ./fit_envelope <csv> [--label det_alive|fresh] [--no-weights] [--bins N]
 *                          [--hfov DEG --vfov DEG --camh M]      # for the stand-off consequence table
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <locale>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "../detectability.h"
#include "../../nbv/viewpoint_score.h"

namespace
{

// ─── locale-independent parsing (CLAUDE.md: from_chars ONLY) ──────────────────────────────────────────────
std::optional<float> parse_float(std::string_view s)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '\t')) s.remove_prefix(1);
    while (not s.empty() and (s.back()  == ' ' or s.back()  == '\t' or s.back() == '\r')) s.remove_suffix(1);
    if (s.empty()) return std::nullopt;
    float v{};
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} or not std::isfinite(v)) return std::nullopt;
    return v;
}

std::vector<std::string_view> split(std::string_view line, char sep = ',')
{
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i)
        if (i == line.size() or line[i] == sep) { out.emplace_back(line.substr(start, i - start)); start = i + 1; }
    return out;
}

// ─── one logged trial ─────────────────────────────────────────────────────────────────────────────────────
struct Trial
{
    float fill = 0.0f, fill_h = 0.0f, fill_v = 0.0f;
    float weight = 1.0f;      // P(exists) — see the header note on credit assignment
    bool  fired = false;
    // Aspect ratio of the projected bbox, min/max ∈ (0,1]. 1 = square, → 0 = a sliver (grazing / edge-on).
    float fill_lo() const { return std::min(fill_h, fill_v); }
    float aspect() const
    {
        const float hi = std::max(fill_h, fill_v), lo = std::min(fill_h, fill_v);
        return (hi > 1e-6f) ? lo / hi : 0.0f;
    }
};

// ─── the envelope, in an UNCONSTRAINED parameterisation ───────────────────────────────────────────────────
// θ = (a, log(b−a), log s). Ordering (b > a) and positivity (s > 0) then hold BY CONSTRUCTION, so the
// optimiser never needs a clamp and the Laplace covariance lives in a space where a Gaussian is sensible.
// This is the same trick the online filter will use, which is why the covariance below transfers directly.
struct Params { float a = 0.0f, b = 0.0f, s = 0.0f; };

Params to_params(const Eigen::Vector3d& th)
{
    Params p;
    p.a = static_cast<float>(th(0));
    p.b = static_cast<float>(th(0) + std::exp(th(1)));
    p.s = static_cast<float>(std::exp(th(2)));
    return p;
}
Eigen::Vector3d from_params(const Params& p)
{
    return { p.a, std::log(std::max(1e-6f, p.b - p.a)), std::log(std::max(1e-6f, p.s)) };
}

// ★TWO-AXIS, matching rc::detect::p_detect: the size shoulder tests the SHORT axis (enough pixels across to
// segment) and the fit shoulder the LONG one (what runs off the frame edge first). Fitting the max-only form
// against a model that uses both would calibrate the wrong thing — and it is what let an edge-on door read as
// perfectly framed. Same three parameters; only the axis each governs changes.
double p_fire(double fill_max, double fill_min, const Eigen::Vector3d& th)
{
    const double a = th(0), b = a + std::exp(th(1)), s = std::exp(th(2));
    const auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    return sig((fill_min - a) / s) * sig((b - fill_max) / s);
}

// Weighted negative log-likelihood. The 1e-9 floors keep log() finite for a trial the model calls impossible;
// they bound the penalty rather than changing the optimum.
double nll(const std::vector<Trial>& d, const Eigen::Vector3d& th)
{
    double acc = 0.0;
    for (const auto& t : d)
    {
        const double p = std::clamp(p_fire(t.fill, t.fill_lo(), th), 1e-9, 1.0 - 1e-9);
        acc -= t.weight * (t.fired ? std::log(p) : std::log(1.0 - p));
    }
    return acc;
}

// Central-difference gradient + Hessian. 3 parameters, smooth objective: analytic derivatives would buy
// nothing here and cost a class of transcription bugs.
void derivatives(const std::vector<Trial>& d, const Eigen::Vector3d& th,
                 Eigen::Vector3d& g, Eigen::Matrix3d& H)
{
    constexpr double h = 1e-4;
    for (int i = 0; i < 3; ++i)
    {
        Eigen::Vector3d tp = th, tm = th;
        tp(i) += h; tm(i) -= h;
        g(i) = (nll(d, tp) - nll(d, tm)) / (2.0 * h);
    }
    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
        {
            Eigen::Vector3d pp = th, pm = th, mp = th, mm = th;
            pp(i) += h; pp(j) += h;   pm(i) += h; pm(j) -= h;
            mp(i) -= h; mp(j) += h;   mm(i) -= h; mm(j) -= h;
            H(i, j) = H(j, i) = (nll(d, pp) - nll(d, pm) - nll(d, mp) + nll(d, mm)) / (4.0 * h * h);
        }
}

struct Fit
{
    Params      p;
    Eigen::Vector3d theta   = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cov     = Eigen::Matrix3d::Zero();   // Laplace covariance IN θ-SPACE (seeds phase 2)
    std::array<float, 3> sigma{};                        // σ(min_fill), σ(max_fill), σ(soft) — reported units
    double      nll_final   = 0.0;
    double      n_eff       = 0.0;
    bool        ok          = false;
};

// Levenberg-damped Newton. Damping is what keeps it alive when the Hessian is indefinite — which it WILL be
// whenever a shoulder is unidentified (e.g. no data at high fill, so max_fill is only bounded from below).
Fit fit_envelope(const std::vector<Trial>& d, const Params& seed)
{
    Fit f;
    if (d.size() < 10) return f;
    Eigen::Vector3d th = from_params(seed);
    double lambda = 1e-3, cur = nll(d, th);

    for (int iter = 0; iter < 200; ++iter)
    {
        Eigen::Vector3d g; Eigen::Matrix3d H;
        derivatives(d, th, g, H);
        bool stepped = false;
        for (int trial = 0; trial < 12; ++trial)
        {
            const Eigen::Matrix3d Hd = H + lambda * Eigen::Matrix3d::Identity();
            const Eigen::Vector3d step = Hd.ldlt().solve(-g);
            if (not step.allFinite()) { lambda *= 10.0; continue; }
            const Eigen::Vector3d cand = th + step;
            const double n = nll(d, cand);
            if (std::isfinite(n) and n < cur)
            {
                if ((cur - n) < 1e-10) { th = cand; cur = n; stepped = true; break; }
                th = cand; cur = n; lambda = std::max(1e-9, lambda * 0.3); stepped = true; break;
            }
            lambda *= 10.0;
        }
        if (not stepped) break;
        if (g.norm() < 1e-8) break;
    }

    Eigen::Vector3d g; Eigen::Matrix3d H;
    derivatives(d, th, g, H);
    f.theta = th;
    f.p     = to_params(th);
    f.nll_final = cur;
    for (const auto& t : d) f.n_eff += t.weight;

    // Laplace: Σ = H⁻¹ at the optimum. A non-PSD H means the optimum is not a well-formed peak in some
    // direction — reported honestly as an infinite σ rather than papered over, because that direction is
    // exactly the one the online filter must not pretend to know.
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H);
    if (es.info() == Eigen::Success and es.eigenvalues().minCoeff() > 1e-9)
    {
        f.cov = H.inverse();
        // θ → reported units. a is already min_fill. b = a + e^θ1 and s = e^θ2, so propagate first order.
        const double eb = std::exp(th(1)), es_ = std::exp(th(2));
        Eigen::Matrix3d J = Eigen::Matrix3d::Identity();     // ∂(a, b, s)/∂θ
        J(1, 0) = 1.0; J(1, 1) = eb; J(1, 2) = 0.0;
        J(2, 0) = 0.0; J(2, 1) = 0.0; J(2, 2) = es_;
        const Eigen::Matrix3d C = J * f.cov * J.transpose();
        for (int i = 0; i < 3; ++i)
            f.sigma[i] = static_cast<float>(std::sqrt(std::max(0.0, C(i, i))));
        f.ok = true;
    }
    else
        for (auto& s : f.sigma) s = std::numeric_limits<float>::infinity();
    return f;
}

void print_fit(const char* title, const Fit& f, std::size_t n_rows)
{
    std::printf("\n── %s ──\n", title);
    if (not f.ok and f.n_eff <= 0.0)
    { std::printf("  too few rows to fit (%zu)\n", n_rows); return; }
    std::printf("  rows %zu   effective N (Σ weights) %.1f   final NLL %.2f\n", n_rows, f.n_eff, f.nll_final);
    std::printf("  min_fill %.4f ± %.4f\n", f.p.a, f.sigma[0]);
    std::printf("  max_fill %.4f ± %.4f\n", f.p.b, f.sigma[1]);
    std::printf("  soft     %.4f ± %.4f\n", f.p.s, f.sigma[2]);
    rc::detect::DetectorEnvelope e{f.p.a, f.p.b, f.p.s};
    // best_fill is the SQUARE-object optimum (fill_min == fill_max). An elongated object cannot reach it on
    // both axes at once, which is a property of the object, not of the fit.
    std::printf("  ⇒ best_fill %.4f (square object)   peak p_detect %.3f\n",
                rc::detect::best_fill(e), rc::detect::p_detect(rc::detect::best_fill(e), 1.0f, e));
    if (not f.ok)
        std::printf("  ⚠ Hessian not positive-definite: at least one parameter is NOT identified by this data.\n"
                    "    Usually max_fill — the planner deliberately avoids the too-close regime, so the upper\n"
                    "    shoulder is only bounded from below. Treat it as a prior, not a measurement.\n");
}

// Empirical hit rate per fill bin against the fitted curve. The eyeball test: if the curve does not track the
// bins, the two-logistic form is wrong for this detector and no amount of refitting will save it.
void print_reliability(const std::vector<Trial>& d, const Fit& f, int bins)
{
    if (not (bins > 0)) return;
    float hi = 0.0f;
    for (const auto& t : d) hi = std::max(hi, t.fill);
    hi = std::max(0.2f, std::min(hi, 2.0f));
    std::printf("\n  (fitted column assumes a SQUARE bbox at the bin centre; elongated views sit lower)\n");
    std::printf("\n  fill bin      n    Σw    fired   emp.rate   fitted   \n");
    std::printf("  ----------------------------------------------------\n");
    for (int i = 0; i < bins; ++i)
    {
        const float lo_f = hi * static_cast<float>(i) / bins, hi_f = hi * static_cast<float>(i + 1) / bins;
        double w = 0.0, wf = 0.0; int n = 0;
        for (const auto& t : d)
            if (t.fill >= lo_f and t.fill < hi_f) { ++n; w += t.weight; if (t.fired) wf += t.weight; }
        if (n == 0) continue;
        const double mid = 0.5 * (lo_f + hi_f);
        std::printf("  %.2f-%.2f  %5d  %6.1f  %6.1f    %6.3f    %6.3f\n",
                    lo_f, hi_f, n, w, wf, (w > 0 ? wf / w : 0.0), p_fire(mid, mid, f.theta));
    }
}

}  // namespace

int main(int argc, char** argv)
{
    // ★Deliberate: run under the SAME locale as the agent that wrote the file, so this harness cannot
    // silently answer a different question than the agent would. Parsing is from_chars, so it is immune.
    std::setlocale(LC_ALL, "");

    if (argc < 2)
    { std::printf("usage: %s <ai2_log.csv> [--label det_alive|fresh] [--no-weights] [--bins N]\n"
                  "          [--hfov DEG --vfov DEG --camh M]\n", argv[0]); return 1; }

    std::string path = argv[1], label = "det_alive";   // overridden to "fired" below when the probe column exists
    bool use_weights = true; int bins = 16;
    float hfov_deg = 110.0f, vfov_deg = 77.6f, cam_h = 0.945f;   // the live sim rig, overridable
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if      (a == "--no-weights") use_weights = false;
        else if (a == "--label" and i + 1 < argc) label = argv[++i];
        else if (a == "--bins"  and i + 1 < argc) bins  = std::atoi(argv[++i]);
        else if (a == "--hfov"  and i + 1 < argc) hfov_deg = parse_float(argv[++i]).value_or(hfov_deg);
        else if (a == "--vfov"  and i + 1 < argc) vfov_deg = parse_float(argv[++i]).value_or(vfov_deg);
        else if (a == "--camh"  and i + 1 < argc) cam_h    = parse_float(argv[++i]).value_or(cam_h);
    }

    std::ifstream in(path);
    if (not in.is_open()) { std::printf("cannot open %s\n", path.c_str()); return 1; }

    std::string header;
    if (not std::getline(in, header)) { std::printf("empty file\n"); return 1; }
    std::unordered_map<std::string, int> col;
    { int i = 0; for (auto f : split(header)) col[std::string(f)] = i++; }

    // Columns are looked up BY NAME, never by position: this schema is ~90 columns wide and grows, and a
    // positional reader would silently read the wrong field the next time someone inserts one.
    const auto need = [&](const char* n) -> int
    {
        const auto it = col.find(n);
        if (it == col.end()) { std::printf("column '%s' missing — is this an ai2_log.csv from a build\n"
                                           "  that includes the calibration columns?\n", n); std::exit(1); }
        return it->second;
    };
    // Optional lookup: -1 when absent, so a file that simply does not carry a column is handled without
    // pretending it does.
    const auto opt = [&](const char* n) -> int
    { const auto it = col.find(n); return it == col.end() ? -1 : it->second; };
    const int c_fill = need("roi_fill"), c_h = need("roi_fill_h"), c_v = need("roi_fill_v");
    // ★ALSO ACCEPTS detect_probe.csv (rc::probe), which carries a REAL per-cycle outcome column.
    // `det_alive` is a LATCHED liveness flag — measured 1 in 9432/9432 hood rows, still 1 at
    // frames_since_det = 6172 — so on an ai2 log the only usable label is `fresh`. The probe file
    // settles it by logging `fired` directly; when that column exists it wins, and the two ai2-only
    // columns are then optional rather than required.
    const int c_valid = need("roi_valid");
    const int c_fired = opt("fired");
    if (c_fired >= 0) label = "fired";   // the probe file settles the label; say so in the report
    const int c_alive = (c_fired >= 0) ? opt("det_alive") : need("det_alive");
    const int c_fsd   = (c_fired >= 0) ? opt("frames_since_det") : need("frames_since_det");
    const int c_exp = (opt("ex_p_prior") >= 0) ? opt("ex_p_prior") : need("ex_p");   // probe files name it for the PRIOR it actually is

    std::vector<Trial> all;
    std::size_t lines = 0, dropped_invalid = 0, dropped_parse = 0;
    for (std::string line; std::getline(in, line); )
    {
        if (line.empty()) continue;
        ++lines;
        const auto f = split(line);
        const int wide = std::max({c_fill, c_h, c_v, c_valid, c_alive, c_fsd, c_exp, c_fired});
        if (static_cast<int>(f.size()) <= wide) { ++dropped_parse; continue; }

        const auto fill = parse_float(f[c_fill]); const auto fh = parse_float(f[c_h]);
        const auto fv = parse_float(f[c_v]);      const auto valid = parse_float(f[c_valid]);
        const auto alive = (c_alive >= 0) ? parse_float(f[c_alive]) : std::optional<float>{0.0f};
        const auto fsd   = (c_fsd   >= 0) ? parse_float(f[c_fsd])   : std::optional<float>{0.0f};
        const auto fired_v = (c_fired >= 0) ? parse_float(f[c_fired]) : std::optional<float>{0.0f};
        const auto pex = parse_float(f[c_exp]);
        if (not (fill and fh and fv and valid and alive and fsd and fired_v and pex)) { ++dropped_parse; continue; }
        if (*valid < 0.5f) { ++dropped_invalid; continue; }   // degenerate projection: fill is meaningless

        Trial t;
        t.fill = *fill; t.fill_h = *fh; t.fill_v = *fv;
        // Two label conventions, because they answer different questions: `det_alive` is the agent's own
        // liveness flag (may latch across frames), `fresh` is "a mask arrived THIS cycle". If the two fits
        // disagree materially, the latch is smearing detections across cycles the detector did not fire in.
        t.fired  = (c_fired >= 0) ? (*fired_v > 0.5f)
                                  : ((label == "fresh") ? (*fsd <= 0.5f) : (*alive > 0.5f));
        t.weight = use_weights ? std::clamp(*pex, 0.0f, 1.0f) : 1.0f;
        all.push_back(t);
    }

    std::printf("read %s\n", path.c_str());
    std::printf("  %zu data lines · %zu usable · %zu dropped (roi_valid=0) · %zu dropped (parse/short)\n",
                lines, all.size(), dropped_invalid, dropped_parse);
    if (all.empty()) { std::printf("nothing to fit\n"); return 1; }

    std::size_t pos = 0; double wsum = 0.0, wpos = 0.0;
    for (const auto& t : all) { if (t.fired) ++pos; wsum += t.weight; wpos += t.weight * (t.fired ? 1 : 0); }
    std::printf("  label='%s'  positives %zu/%zu (%.1f%%)  weighted %.1f/%.1f (%.1f%%)  weights=%s\n",
                label.c_str(), pos, all.size(), 100.0 * pos / all.size(), wpos, wsum,
                (wsum > 0 ? 100.0 * wpos / wsum : 0.0), use_weights ? "P(exists)" : "OFF");
    if (pos == 0 or pos == all.size())
    { std::printf("\n⚠ all trials have the SAME outcome — a detection RATE cannot be fitted from this.\n"
                  "  You need a tour where the object is sometimes framed well and sometimes not.\n"); return 1; }

    const Params seed{0.10f, 0.60f, 0.06f};   // the current prior, used only as a starting point
    const Fit full = fit_envelope(all, seed);
    print_fit("FIT (all usable rows)", full, all.size());
    print_reliability(all, full, bins);

    // ── the aspect-bias check (always run; see the header) ────────────────────────────────────────────────
    // "Compact" = the projected bbox is not a sliver. Slivers are grazing/edge-on views whose max-fill is
    // large for a reason unrelated to framing, and whose misses would otherwise pull max_fill down.
    std::vector<Trial> compact;
    for (const auto& t : all) if (t.aspect() > 0.5f) compact.push_back(t);
    std::printf("\n  aspect split: %zu compact (min/max > 0.5) · %zu slivers\n",
                compact.size(), all.size() - compact.size());
    const Fit comp = fit_envelope(compact, seed);
    print_fit("FIT (compact views only — the aspect-bias control)", comp, compact.size());

    if (full.ok and comp.ok)
    {
        const float d_max = std::abs(comp.p.b - full.p.b), d_min = std::abs(comp.p.a - full.p.a);
        std::printf("\n  ⇒ aspect bias: min_fill moves %.4f, max_fill moves %.4f between the two fits.\n", d_min, d_max);
        std::printf("     %s\n", (d_max > 2.0f * std::max(1e-4f, full.sigma[1]))
            ? "MATERIAL — sliver views ARE dragging the upper shoulder. Prefer the COMPACT fit, and treat\n"
              "     fill=max(axis) as too lossy for this detector; the envelope may need an aspect term."
            : "not material — the max-over-axes simplification is holding up on this data.");
    }

    // ── the consequence: what these numbers do to the stand-offs ──────────────────────────────────────────
    // A fit is only meaningful if you can see what it CHANGES. Same objects as the headless model check.
    const Fit& use = comp.ok ? comp : full;
    if (use.ok)
    {
        rc::nbv::Sensor s;
        s.hfov_rad = hfov_deg * std::numbers::pi_v<float> / 180.0f;
        s.vfov_rad = vfov_deg * std::numbers::pi_v<float> / 180.0f;
        s.height_m = cam_h;
        rc::nbv::Sensor before = s; before.env = rc::detect::DetectorEnvelope{};
        rc::nbv::Sensor after  = s; after.env  = rc::detect::DetectorEnvelope{use.p.a, use.p.b, use.p.s};

        struct Obj { const char* name; rc::nbv::Target t; };
        const std::array<Obj, 3> objs{{
            {"dining table 2.4x0.9", {0, 0, 0.0f, 2.40f, 0.90f, 0.0f, 0.75f}},
            {"chair 0.6x0.52",       {0, 0, 0.0f, 0.60f, 0.52f, 0.0f, 0.90f}},
            {"fridge 0.6x0.6x1.7",   {0, 0, 0.0f, 0.60f, 0.60f, 0.0f, 1.70f}},
        }};
        std::printf("\n── stand-off consequence (hfov %.1f° vfov %.1f° cam_h %.2f m) ──\n", hfov_deg, vfov_deg, cam_h);
        std::printf("  %-24s %10s %10s\n", "object", "prior", "fitted");
        for (const auto& o : objs)
        {
            // ★Eigen::Vector2f, NOT auto: centre()/axis_x() return by VALUE, so `auto` would deduce a
            // CwiseBinaryOp holding references to those temporaries — dangling the moment the statement ends,
            // and read later inside standoff_band. The symptom was every stand-off collapsing to the floor.
            const Eigen::Vector2f face = o.t.centre() + o.t.axis_x() * (0.5f * o.t.w);
            const auto b0 = rc::nbv::standoff_band(o.t, before, face, o.t.axis_x(), 0.30f);
            const auto b1 = rc::nbv::standoff_band(o.t, after,  face, o.t.axis_x(), 0.30f);
            std::printf("  %-24s %8.2f m %8.2f m\n", o.name, b0.best, b1.best);
        }
    }

    // ── the circularity caveat, printed UNCONDITIONALLY ───────────────────────────────────────────────────
    // ex_p weighting DAMPS the credit-assignment problem; it does not break the loop. The removal channel
    // uses p_detect to decide existence, so `ex_p` — the weight — is itself downstream of the very envelope
    // being fitted. And the data was gathered under the OLD envelope, whose stand-offs chose which fills were
    // ever sampled. A fit that moves a lot has therefore changed the sampling distribution that produced it,
    // and is only trustworthy once it reproduces itself on data logged AFTER it went live.
    if (use.ok)
    {
        const rc::detect::DetectorEnvelope prior{};   // what was live while this log was being written
        const float d_a = std::abs(use.p.a - prior.min_fill);
        const float d_b = std::abs(use.p.b - prior.max_fill);
        const bool  moved = d_a > 0.02f or d_b > 0.05f;
        std::printf("\n── caveat: this fit is not yet a fixed point ──\n");
        if (use_weights)
            std::printf("  · trials were weighted by ex_p = P(exists), which is itself computed USING p_detect.\n"
                        "    That damps the circularity (a doubtful cycle contributes little) but does not\n"
                        "    remove it. Re-run with --no-weights: if the two fits disagree, existence doubt is\n"
                        "    carrying real weight and the labels deserve scrutiny before you trust either.\n");
        else
            std::printf("  · weights are OFF, so every miss is charged to the envelope — including misses caused\n"
                        "    by occlusion or by the object not being there. This fit is a PESSIMISTIC bound.\n");
        std::printf("  · the log was written under the PRIOR envelope (%.3f / %.3f / %.3f), whose stand-offs\n"
                    "    decided which fills were ever sampled. Fitted moves: min_fill %+.3f, max_fill %+.3f.\n",
                    prior.min_fill, prior.max_fill, prior.soft, use.p.a - prior.min_fill, use.p.b - prior.max_fill);
        std::printf("    %s\n", moved
            ? "MOVED MATERIALLY ⇒ deploy it, log a fresh tour, and re-run this fit. Only if the second fit\n"
              "    lands on the first is it a fixed point rather than an artefact of the old sampling."
            : "close to the prior ⇒ the sampling distribution barely changed; one pass is likely enough.");
    }

    // ── seed for the ONLINE filter (phase 2) ──────────────────────────────────────────────────────────────
    // Printed in the SAME unconstrained θ-space the online filter uses — θ = (a, log(b−a), log s) — so it
    // transfers without a reparameterisation step, which is where this sort of handoff usually goes wrong.
    if (use.ok)
    {
        std::printf("\n── phase-2 seed (θ = [a, log(b−a), log s], the online filter's own space) ──\n");
        std::printf("  mu    = [% .6f, % .6f, % .6f]\n", use.theta(0), use.theta(1), use.theta(2));
        std::printf("  Sigma = [% .3e, % .3e, % .3e\n", use.cov(0,0), use.cov(0,1), use.cov(0,2));
        std::printf("           % .3e, % .3e, % .3e\n", use.cov(1,0), use.cov(1,1), use.cov(1,2));
        std::printf("           % .3e, % .3e, % .3e]\n", use.cov(2,0), use.cov(2,1), use.cov(2,2));
        std::printf("\n  config (DetectMinFill / DetectMaxFill / DetectSoft):  %.4f  %.4f  %.4f\n",
                    use.p.a, use.p.b, use.p.s);
    }
    return 0;
}
