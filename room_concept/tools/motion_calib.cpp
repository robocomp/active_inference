/*
 *  motion_calib.cpp — calibrate the motion model's noise densities and scale errors from a logged run.
 *
 *  WHAT IT ESTIMATES, AND WHY IN THIS FORM
 *  ---------------------------------------
 *  se2_preintegration.h splits the odometry's error into two classes that scale DIFFERENTLY with the
 *  length of the interval:
 *
 *      random  (density sigma)   accumulates as a Wiener process   ->  std grows like sqrt(T)
 *      scale   (fraction s)      is one constant, fully correlated ->  error grows like T
 *
 *  That difference IS the estimator. Over a window of duration T the odometry's error against a
 *  reference is
 *
 *      e(T)  =  s * Delta(T)  +  eps,        eps ~ N(0, sigma^2 * T)
 *
 *  so a weighted least squares of e against Delta separates them in one pass: the SLOPE is the scale,
 *  the RESIDUAL is the density. Nothing has to be assumed about which one dominates — the data says.
 *
 *  ★ AND THE WLS SOLUTION IS THE EVIDENCE OPTIMUM, not a separate heuristic. The negative log evidence
 *  of the motion factors under (s, sigma) is
 *
 *      F(s,sigma) = sum_k [ (e_k - s*Delta_k)^2 / (2*sigma^2*T_k) + (1/2) log(sigma^2 T_k) ] + const
 *
 *  dF/ds = 0 gives the weighted slope; dF/dsigma = 0 gives sigma^2 = (1/n) sum (e_k - s Delta_k)^2 / T_k.
 *  Both in closed form. So "calibrate the model" and "maximise the evidence for the model" are the same
 *  computation here, which is the point worth making: the covariance is INFERRED under the same
 *  functional the localiser already minimises, not tuned beside it.
 *
 *  THE REFERENCE, AND ITS ONE HONEST WEAKNESS
 *  ------------------------------------------
 *  The reference is the localiser's own posterior track, which is anchored to a FIXED room polygon and
 *  is therefore an independent witness of true motion in a way dead reckoning can never be. But it is
 *  not fully independent: the posterior was computed USING the motion prior. Attributing the whole
 *  discrepancy to odometry is exactly the error that made the online learner fail — it used a
 *  post-optimisation residual, so it charged the optimiser's own correction to the encoder and
 *  concluded the encoder was uninformative.
 *
 *  Two things keep that under control here, and the tool reports both rather than hiding them:
 *    1. The estimate is computed at SEVERAL window lengths. A longer window is more SDF-dominated and
 *       less contaminated, so a systematic drift of sigma-hat with T is the contamination made visible.
 *       Read the long-T end, and read the trend, not one number.
 *    2. The scale is a SLOPE, not a mean residual. The motion prior pulls the posterior toward the
 *       odometry roughly uniformly; it does not manufacture a linear dependence on Delta.
 *
 *  ⚠ WHAT THIS CANNOT DO IN SIMULATION. Measured 2026-08-13 over 24434 parked frames, this simulator's
 *  odometry has sigma_v = 3.9e-7 m/sqrt(s) and sigma_omega = 2.2e-8 rad/sqrt(s) — five to six orders
 *  below the configured values — because webots-bridge publishes the supervisor's GROUND-TRUTH velocity
 *  and there is no encoder in the loop. Calibrating there would drive the motion prior to zero variance
 *  and make the SDF inert. In simulation this tool has exactly one legitimate use: score it against a
 *  KNOWN injected error (RoomConcept.OdomInject*, printed as "[OdomInject] GROUND TRUTH" at startup).
 *  The real calibration is a HARDWARE measurement.
 *
 *  BUILD / RUN
 *      make -C build motion_calib && ../bin/motion_calib <debug_log.csv>
 *      ../bin/motion_calib --selftest        # synthetic truth, no log needed
 */
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <clocale>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    // ── Locale-safe CSV reading ──────────────────────────────────────────────────────────────────
    // These machines run LANG=es_ES.UTF-8 and the agent is a Qt program, so Qt's setlocale(LC_ALL,"")
    // makes the C library's decimal separator a COMMA. strtof/atof/stod would then stop dead at the '.'
    // in a file written with decimal POINTS and return the integer part, SILENTLY: "0.260417" -> 0.
    // std::from_chars is locale-independent by definition and reports failure instead of guessing.
    // ★ main() calls setlocale(LC_ALL,"") on purpose: a standalone harness otherwise stays in the "C"
    // locale, the bug vanishes, and the harness answers a different question than the agent would.
    inline bool parse_float(std::string_view s, float& out)
    {
        while (!s.empty() and (s.front() == ' ' or s.front() == '\t')) s.remove_prefix(1);
        while (!s.empty() and (s.back()  == ' ' or s.back()  == '\r')) s.remove_suffix(1);
        if (s.empty()) return false;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
        return r.ec == std::errc{};
    }

    /// ⚠ SEPARATE DOUBLE OVERLOAD, AND IT IS NOT A STYLE CHOICE. `ts_ms` is an epoch millisecond
    /// stamp, ~1.755e12. float32 carries ~7 significant decimal digits, so that value quantises to
    /// steps of 2^17 ms ≈ 131 s: every consecutive pair of frames parses to the SAME instant, every
    /// window interval comes out as exactly zero, and the whole calibration silently returns n = 0
    /// windows with no error anywhere. Measured — that is precisely what the first run of this tool
    /// did on an 8700-frame log. Timestamps are doubles; pose and velocity values are small and float
    /// is fine for them.
    inline bool parse_double(std::string_view s, double& out)
    {
        while (!s.empty() and (s.front() == ' ' or s.front() == '\t')) s.remove_prefix(1);
        while (!s.empty() and (s.back()  == ' ' or s.back()  == '\r')) s.remove_suffix(1);
        if (s.empty()) return false;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
        return r.ec == std::errc{};
    }

    std::vector<std::string_view> split(std::string_view line, std::vector<std::string_view>& buf)
    {
        buf.clear();
        size_t start = 0;
        for (size_t i = 0; i <= line.size(); ++i)
            if (i == line.size() or line[i] == ',')
            {
                buf.push_back(line.substr(start, i - start));
                start = i + 1;
            }
        return buf;
    }

    /// Locale-independent number OUTPUT. The mirror of parse_float, and just as load-bearing:
    /// printf formats through LC_NUMERIC, so under this machine's es_ES locale it emits "0,0298".
    /// The suggested-config block at the end is meant to be PASTED INTO A TOML FILE, where a comma
    /// decimal separator is either a parse error or a different value — so a tool that reads
    /// locale-safely and then writes locale-unsafely has simply moved the bug to its output.
    /// ★ Note what this does NOT do: it does not touch LC_NUMERIC. The setlocale(LC_ALL,"") in main()
    /// stays, so that any locale-dependent parsing added to this file later still misbehaves here
    /// exactly as it would inside the agent. Pinning the facet globally would remove that guard.
    inline std::string num(double v, int prec = 6)
    {
        char buf[64];
        const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::general, prec);
        return (r.ec == std::errc{}) ? std::string(buf, r.ptr) : std::string("nan");
    }

    inline float wrap_pi(float a)
    {
        while (a >  static_cast<float>(M_PI)) a -= 2.f * static_cast<float>(M_PI);
        while (a < -static_cast<float>(M_PI)) a += 2.f * static_cast<float>(M_PI);
        return a;
    }

    /// One localiser frame, reduced to what calibration needs.
    struct Frame
    {
        double ts_s   = 0;      // lidar stamp, seconds
        float  px = 0, py = 0, pth = 0;   // POSTERIOR pose (pred + innovation), room frame
        float  ox = 0, oy = 0, oth = 0;   // measured-odometry increment for THIS frame, room frame
        bool   ok = false;                // odometry valid and fresh, pose usable
    };

    /// The joint ML / evidence-maximising fit of  e = s*Delta + N(0, sigma^2 T).
    struct Fit
    {
        double s = 0;           // scale (dimensionless)
        double sigma = 0;       // density (m/sqrt(s) or rad/sqrt(s))
        double s_stderr = 0;
        int    n = 0;
        double span = 0;        // |Delta| range covered, so a degenerate slope is visible
    };

    Fit fit_scale_and_density(const std::vector<double>& delta,   // regressor: the odometry increment
                              const std::vector<double>& err,     // response: odom - reference
                              const std::vector<double>& T)       // window duration, seconds
    {
        Fit f;
        f.n = static_cast<int>(delta.size());
        if (f.n < 3) return f;

        // Weights 1/T: the noise variance is proportional to T, so this is exactly the ML weighting.
        double sxx = 0, sxy = 0;
        for (int i = 0; i < f.n; ++i)
        {
            const double w = 1.0 / std::max(T[i], 1e-9);
            sxx += w * delta[i] * delta[i];
            sxy += w * delta[i] * err[i];
        }
        f.s = (sxx > 1e-18) ? sxy / sxx : 0.0;

        double ss = 0;
        for (int i = 0; i < f.n; ++i)
        {
            const double r = err[i] - f.s * delta[i];
            ss += r * r / std::max(T[i], 1e-9);
        }
        // dF/dsigma = 0.  n-1 rather than n because one parameter (the slope) was consumed.
        f.sigma = std::sqrt(ss / std::max(1, f.n - 1));
        f.s_stderr = (sxx > 1e-18) ? f.sigma / std::sqrt(sxx) : 0.0;

        auto [lo, hi] = std::minmax_element(delta.begin(), delta.end());
        f.span = *hi - *lo;
        return f;
    }

    /// Accumulate over disjoint windows of `stride` frames and fit. `stride` in frames.
    struct BandResult { int stride; double T_med; Fit rot, trans; int rejected_jump = 0; };

    BandResult calibrate_at(const std::vector<Frame>& fr, int stride)
    {
        BandResult br; br.stride = stride;
        int rejected_jump = 0;
        std::vector<double> d_rot, e_rot, T_rot, d_tr, e_tr, T_tr, Ts;

        for (size_t a = 0; a + static_cast<size_t>(stride) < fr.size(); a += static_cast<size_t>(stride))
        {
            const size_t b = a + static_cast<size_t>(stride);
            // Reject a window containing anything unusable; a relocalisation inside it would show up
            // as an enormous "odometry error" that is nothing of the kind.
            bool good = true;
            for (size_t i = a; i <= b; ++i) if (!fr[i].ok) { good = false; break; }
            // ★ AND REJECT WINDOWS CONTAINING A POSE DISCONTINUITY. The reference is the localiser's
            // posterior, and a relocalisation moves it by more than the robot could have travelled —
            // measured on this log, 5 steps above 15 cm between consecutive frames, two of them 0.61 m
            // and 0.39 m around a recovery episode. That is not a report about motion, so charging it
            // to the odometry would put a 0.6 m outlier into a weighted least squares whose typical
            // increment is ~1 m. Validity flags do not catch it: those frames are perfectly valid, the
            // localiser simply changed its mind. The bound is kinematic (~3x the base's top speed), so
            // it can only ever reject a physical impossibility, never real motion.
            if (good)
                for (size_t i = a; i < b; ++i)
                {
                    const double step = std::hypot(fr[i+1].px - fr[i].px, fr[i+1].py - fr[i].py);
                    const double gap  = std::max(1e-3, fr[i+1].ts_s - fr[i].ts_s);
                    if (step / gap > 2.0) { good = false; ++rejected_jump; break; }
                }
            if (!good) continue;

            const double T = fr[b].ts_s - fr[a].ts_s;
            if (T <= 1e-3 or T > 30.0) continue;     // stalls and gaps are not intervals; 192-frame
                                                     // windows are ~10 s, so the cap cannot sit below that

            // Odometry increment over the window: the same additive accumulation of global-frame
            // increments the agent performs (each was integrated at its own running heading).
            double ox = 0, oy = 0, oth = 0;
            for (size_t i = a + 1; i <= b; ++i) { ox += fr[i].ox; oy += fr[i].oy; oth += fr[i].oth; }

            // Reference increment from the posterior track.
            const double rx = fr[b].px - fr[a].px;
            const double ry = fr[b].py - fr[a].py;
            const double rth = wrap_pi(fr[b].pth - fr[a].pth);

            // ★ THE REGRESSOR IS THE REFERENCE INCREMENT, NOT THE ODOMETRY ONE. This is not a
            // presentational choice, it is the difference between a biased and an unbiased estimator.
            // Writing odom = ref*(1+s) + eps, the response is e = odom - ref = s*ref + eps. Regressing
            // e on ODOM puts eps on both sides — errors in variables — and the slope converges to
            //     [ s(1+s) Var(ref) + Var(eps) ] / [ (1+s)^2 Var(ref) + Var(eps) ],
            // which at s = 0 is Var(eps)/(Var(ref)+Var(eps)) > 0: pure noise is reported as a positive
            // scale. Measured on synthetic truth with s = 0 exactly, that bias was +0.0029 rad/rad and
            // +0.0018 m/m, and it grows as the noise does — precisely the regime where the estimate
            // matters most. Regressing on the reference makes Cov(ref, e) = s*Var(ref) and the slope
            // unbiased. (The reference carries its own error, so this is a reduction rather than an
            // elimination; that error is the posterior's, which is the smaller of the two by the same
            // argument that makes the posterior usable as a reference at all.)
            //
            // ---- rotation channel -------------------------------------------------------------
            d_rot.push_back(rth);
            e_rot.push_back(oth - rth);
            T_rot.push_back(T);

            // ---- translation channel ----------------------------------------------------------
            // Project the error on the DIRECTION OF TRAVEL. A scale error acts along the path; the
            // cross-path component is a heading error and belongs to the rotation channel, so mixing
            // them would let a yaw error masquerade as a speed error. The direction, like the
            // regressor, is taken from the reference.
            const double mag = std::hypot(rx, ry);
            if (mag > 1e-4)
            {
                const double ux = rx / mag, uy = ry / mag;
                d_tr.push_back(mag);
                e_tr.push_back((ox - rx) * ux + (oy - ry) * uy);
                T_tr.push_back(T);
            }
            Ts.push_back(T);
        }

        std::sort(Ts.begin(), Ts.end());
        br.T_med = Ts.empty() ? 0.0 : Ts[Ts.size() / 2];
        br.rejected_jump = rejected_jump;
        br.rot   = fit_scale_and_density(d_rot, e_rot, T_rot);
        br.trans = fit_scale_and_density(d_tr,  e_tr,  T_tr);
        return br;
    }

    void report(const std::vector<BandResult>& bands)
    {
        std::printf("\n%-8s %8s %7s | %-34s | %-34s\n", "window", "T_med", "n",
                    "ROTATION   s_omega / sigma_omega", "TRANSLATION  s_v / sigma_v");
        std::printf("%-8s %8s %7s | %34s | %34s\n", "frames", "s", "", "", "");
        for (const auto& b : bands)
        {
            std::printf("%-8d %8s %7d | %9s +-%-9s %11s | %9s +-%-9s %11s\n",
                        b.stride, num(b.T_med, 3).c_str(), b.rot.n,
                        num(b.rot.s, 4).c_str(), num(b.rot.s_stderr, 3).c_str(), num(b.rot.sigma, 4).c_str(),
                        num(b.trans.s, 4).c_str(), num(b.trans.s_stderr, 3).c_str(), num(b.trans.sigma, 4).c_str());
        }
        std::printf("\n  sigma_omega is rad/sqrt(s), sigma_v is m/sqrt(s) — these are DENSITIES, the\n"
                    "  PreintOdomSigma* config keys. s_* are the PreintOdomScale* keys.\n");
        std::printf("  ★ READ THE TREND, NOT ONE ROW. sigma-hat rising with the window length means the\n"
                    "    reference is contaminated by the motion prior (a longer window is more\n"
                    "    SDF-dominated); a flat sigma-hat is the Wiener model holding. Take the long-T end.\n");
    }

    // ── Synthetic recovery test ──────────────────────────────────────────────────────────────────
    // Generates a track with KNOWN sigma and s and checks the estimator recovers them. This is the
    // step that can be done without a robot and without the simulator, and it is the only way to know
    // the estimator itself is unbiased before pointing it at data whose truth is unknown.
    int selftest()
    {
        std::printf("motion_calib selftest — synthetic truth, exact\n\n");
        const double dt = 0.05;
        const int    N  = 20000;
        int failures = 0;

        struct Case { const char* name; double s_w, sig_w, s_v, sig_v; };
        const Case cases[] = {
            {"scale only, no noise",   0.070, 0.0000, 0.050, 0.0000},
            {"noise only, no scale",   0.000, 0.0300, 0.000, 0.0100},
            {"both, realistic",        0.070, 0.0300, 0.050, 0.0100},
            {"both, hardware-ish",     0.030, 0.0020, 0.020, 0.0010},
        };

        for (const auto& c : cases)
        {
            std::mt19937 rng(20260813);
            std::normal_distribution<double> g(0.0, 1.0);
            std::vector<Frame> fr(N);
            double px = 0, py = 0, pth = 0.3;      // TRUE track (the "posterior" reference)
            fr[0] = {0.0, (float)px, (float)py, (float)pth, 0, 0, 0, true};
            for (int k = 1; k < N; ++k)
            {
                // A varied manoeuvre so the regressor spans a real range: without variation in Delta
                // the slope is unidentifiable and the fit would silently return noise.
                const double phase = 2.0 * M_PI * k / 900.0;
                const double v = 0.35 + 0.25 * std::sin(phase);
                const double w = 0.60 * std::sin(0.7 * phase);

                const double dth_true = w * dt;
                const double thm = pth + 0.5 * dth_true;
                const double dpx_true = v * dt * -std::sin(thm);   // body +y forward
                const double dpy_true = v * dt *  std::cos(thm);
                px += dpx_true; py += dpy_true; pth += dth_true;

                // Odometry: true motion, scaled, plus a Wiener increment of density sigma.
                const double dth_o = dth_true * (1.0 + c.s_w) + c.sig_w * std::sqrt(dt) * g(rng);
                const double along = v * dt * (1.0 + c.s_v)   + c.sig_v * std::sqrt(dt) * g(rng);
                const double dpx_o = along * -std::sin(thm);
                const double dpy_o = along *  std::cos(thm);

                fr[k] = {k * dt, (float)px, (float)py, (float)pth,
                         (float)dpx_o, (float)dpy_o, (float)dth_o, true};
            }

            std::vector<BandResult> bands;
            for (int s : {1, 2, 4, 8, 16}) bands.push_back(calibrate_at(fr, s));
            const auto& B = bands.back();

            // A slope is scored against ITS OWN standard error, not a fixed tolerance: with truth 0 a
            // relative tolerance is meaningless, and a fixed absolute one just encodes how noisy this
            // particular synthetic case happens to be. 3 sigma is the honest bar.
            auto slope_ok = [](const Fit& f, double want) {
                return std::abs(f.s - want) <= std::max(3.0 * f.s_stderr, 1e-3);
            };
            auto close = [](double got, double want, double atol, double rtol) {
                return std::abs(got - want) <= atol + rtol * std::abs(want);
            };
            const bool ok =
                slope_ok(B.rot,   c.s_w) and
                slope_ok(B.trans, c.s_v) and
                close(B.rot.sigma, c.sig_w, 2e-3, 0.10) and
                close(B.trans.sigma, c.sig_v, 1e-3, 0.10);
            if (!ok) ++failures;

            std::printf("  %-22s %s\n", c.name, ok ? "PASS" : "FAIL");
            std::printf("      s_omega %9s (truth %9s)   sigma_omega %11s (truth %11s)\n",
                        num(B.rot.s,4).c_str(), num(c.s_w,4).c_str(),
                        num(B.rot.sigma,4).c_str(), num(c.sig_w,4).c_str());
            std::printf("      s_v     %9s (truth %9s)   sigma_v     %11s (truth %11s)\n",
                        num(B.trans.s,4).c_str(), num(c.s_v,4).c_str(),
                        num(B.trans.sigma,4).c_str(), num(c.sig_v,4).c_str());
        }

        // Identifiability guard. If the manoeuvre never varies, the slope has nothing to lean on. The
        // tool must SAY so rather than return a confident number, so the span is reported and this
        // case is here to keep that honest.
        {
            std::mt19937 rng(7);
            std::normal_distribution<double> g(0.0, 1.0);
            std::vector<Frame> fr(4000);
            double px=0, py=0, pth=0;
            fr[0] = {0.0, 0, 0, 0, 0, 0, 0, true};
            for (size_t k = 1; k < fr.size(); ++k)      // pure constant-speed straight line
            {
                const double dpy_true = 0.4 * dt;
                py += dpy_true;
                fr[k] = {k * dt, (float)px, (float)py, (float)pth,
                         0.f, (float)(dpy_true * 1.05 + 0.01 * std::sqrt(dt) * g(rng)), 0.f, true};
            }
            const auto B = calibrate_at(fr, 8);
            // The guard is on the span RELATIVE to the mean increment: an absolute threshold just
            // encodes the speed of this particular case. With the regressor taken from the reference,
            // a constant-speed leg gives a genuinely degenerate regressor and the ratio collapses.
            const double mean_delta = 0.4 * 8 * dt;
            const double ratio = B.trans.span / mean_delta;
            const bool ok = ratio < 0.02;
            std::printf("  %-22s %s\n", "constant speed (degenerate)", ok ? "PASS" : "FAIL");
            std::printf("      |Delta| span/mean = %s over %d windows — a slope fitted here is not identified\n",
                        num(ratio,3).c_str(), B.trans.n);
            if (!ok) ++failures;
        }

        std::printf("\n%s (%d failure%s)\n\n", failures == 0 ? "ALL PASS" : "FAILURES",
                    failures, failures == 1 ? "" : "s");
        return failures == 0 ? 0 : 1;
    }
} // namespace

int main(int argc, char** argv)
{
    // Reproduce the AGENT's locale — see the note on parse_float. Without this the harness would run
    // in "C", any locale-dependent parsing bug would vanish here and survive in the agent, and the two
    // would disagree about the same file.
    std::setlocale(LC_ALL, "");

    if (argc >= 2 and std::string_view(argv[1]) == "--selftest")
        return selftest();

    if (argc < 2)
    {
        std::fprintf(stderr,
            "usage: motion_calib <debug_log.csv>\n"
            "       motion_calib --selftest\n");
        return 2;
    }

    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    std::string line;
    if (!std::getline(in, line)) { std::fprintf(stderr, "empty file\n"); return 2; }

    std::vector<std::string_view> buf;
    std::map<std::string, int> col;
    {
        std::string_view hv(line);
        split(hv, buf);
        for (size_t i = 0; i < buf.size(); ++i) col[std::string(buf[i])] = static_cast<int>(i);
    }
    const char* need[] = {"ts_ms","pred_x","pred_y","pred_theta","innov_x","innov_y","innov_theta",
                          "meas_dx","meas_dy","meas_dth","meas_valid","meas_fresh"};
    for (const char* n : need)
        if (!col.count(n)) { std::fprintf(stderr, "column '%s' missing — wrong log format\n", n); return 2; }

    std::vector<Frame> fr;
    fr.reserve(1 << 16);
    int ragged = 0;
    const size_t ncols = col.size();
    while (std::getline(in, line))
    {
        std::string_view lv(line);
        split(lv, buf);
        // The debug log has a documented history of ragged rows; a short row silently shifts every
        // field after the break, so drop it rather than read a neighbour's value.
        if (buf.size() < ncols) { ++ragged; continue; }

        auto get = [&](const char* k, float& out) { return parse_float(buf[col[k]], out); };
        Frame f;
        double ts = 0;
        float ix, iy, ith, mv, mf;
        bool ok = true;
        ok &= parse_double(buf[col["ts_ms"]], ts);   // DOUBLE — see parse_double's note
        ok &= get("pred_x", f.px); ok &= get("pred_y", f.py); ok &= get("pred_theta", f.pth);
        ok &= get("innov_x", ix);  ok &= get("innov_y", iy);  ok &= get("innov_theta", ith);
        ok &= get("meas_dx", f.ox); ok &= get("meas_dy", f.oy); ok &= get("meas_dth", f.oth);
        ok &= get("meas_valid", mv); ok &= get("meas_fresh", mf);
        if (!ok) continue;

        f.ts_s = ts / 1000.0;
        f.px += ix; f.py += iy; f.pth = wrap_pi(f.pth + ith);   // pred + innovation = POSTERIOR
        f.ok = (mv > 0.5f) and (mf > 0.5f);
        fr.push_back(f);
    }

    std::printf("motion_calib — %s\n", argv[1]);
    std::printf("  %zu frames read", fr.size());
    if (ragged) std::printf(", %d ragged rows dropped", ragged);
    const size_t usable = static_cast<size_t>(std::count_if(fr.begin(), fr.end(),
                                                            [](const Frame& f){ return f.ok; }));
    std::printf(", %zu with valid+fresh odometry\n", usable);
    if (fr.size() < 100) { std::fprintf(stderr, "  too few frames to calibrate\n"); return 1; }

    // How much MOTION is in here? A parked log calibrates the density floor and NOTHING about scale,
    // and the tool has to say which of its own numbers to disbelieve.
    double tot_turn = 0, tot_travel = 0;
    for (const auto& f : fr) if (f.ok) { tot_turn += std::abs(f.oth); tot_travel += std::hypot(f.ox, f.oy); }
    std::printf("  odometry content: %s rad of turning, %s m of travel\n",
                num(tot_turn,4).c_str(), num(tot_travel,4).c_str());
    if (tot_turn < 2.0)
        std::printf("  ⚠ under ~2 rad of total turning the rotation SCALE is not identified — "
                    "read sigma_omega only.\n");
    if (tot_travel < 2.0)
        std::printf("  ⚠ under ~2 m of total travel the translation SCALE is not identified — "
                    "read sigma_v only.\n");

    std::vector<BandResult> bands;
    // ★ THE LADDER MUST REACH LONG WINDOWS OR THE SCALE READS LOW. The posterior reference is
    // contaminated by the motion prior at short T, which drags the slope toward zero; the
    // contamination falls off as the window lengthens. Measured against an injected truth of +0.10:
    // 0.038 at 8 frames, 0.065 at 16, 0.071 at 32, 0.087 at 64, 0.091 at 128, 0.0945 +-0.0074 at 192
    // (9.7 s) — monotone, and only the last is within one sigma of the truth. A ladder stopping at 32
    // would have reported 71% recovery and looked like estimator bias when it was window length.
    for (int s : {1, 2, 4, 8, 16, 32, 64, 128, 192}) bands.push_back(calibrate_at(fr, s));
    report(bands);

    // ── ★ IS THE ESTIMATE REFERENCE-LIMITED? ─────────────────────────────────────────────────────
    // The regression above measures the discrepancy between the odometry and the POSTERIOR, and
    // charges all of it to the odometry. That is only legitimate while the odometry is the noisier of
    // the two. When it is not, the tool is measuring its own reference and will report the localiser's
    // jitter as though it were an encoder property — which is a confident, plausible, completely wrong
    // number, and exactly the failure mode that has to be caught by an independent measurement rather
    // than by inspection.
    //
    // The independent measurement is free: while the robot is PARKED the odometry's own stream is
    // pure noise about zero, so its sample variance gives the density directly, with the localiser
    // nowhere in the loop.  sigma = std(v_reported) * sqrt(sample interval).
    double direct_v = -1, direct_w = -1;
    bool reference_limited = false;
    if (col.count("odom_adv_norm") and col.count("odom_rot_norm") and col.count("odom_ingress_ts"))
    {
        const bool has_raw = col.count("odom_adv_raw") and col.count("odom_rot_raw");
        std::ifstream in2(argv[1]);
        std::string l2; std::getline(in2, l2);
        std::vector<double> av, rv, its;
        std::vector<std::string_view> b2;
        while (std::getline(in2, l2))
        {
            std::string_view lv(l2);
            split(lv, b2);
            if (b2.size() < ncols) continue;
            float a, r; double t;
            if (!parse_float(b2[col["odom_adv_norm"]], a)) continue;
            if (!parse_float(b2[col["odom_rot_norm"]], r)) continue;
            if (!parse_double(b2[col["odom_ingress_ts"]], t)) continue;
            // ★ STILLNESS IS DECIDED ON THE *RAW* CHANNEL, THE NOISE MEASURED ON THE CONSUMED ONE.
            // The two differ exactly when synthetic error is being injected, and that is precisely
            // when this estimator matters. Testing |norm| < 1e-3 asks "did the pipeline BELIEVE the
            // robot was still", and with noise injected the answer is never — so the parked sample
            // count fell below the minimum, the whole direct check silently skipped, and the scorecard
            // fell back to the reference-limited regression and reported sigma_v 6.6x its true value.
            // The raw columns are the pre-injection ground truth; on a real robot raw == norm and this
            // is a no-op.
            float ar = a, rr = r;
            if (has_raw)
            {
                if (!parse_float(b2[col["odom_adv_raw"]], ar)) continue;
                if (!parse_float(b2[col["odom_rot_raw"]], rr)) continue;
            }
            if (std::abs(ar) < 1e-3f and std::abs(rr) < 1e-3f) { av.push_back(a); rv.push_back(r); its.push_back(t); }
        }
        if (av.size() > 200)
        {
            auto sd = [](const std::vector<double>& v) {
                double m = 0; for (double x : v) m += x; m /= static_cast<double>(v.size());
                double s = 0; for (double x : v) s += (x - m) * (x - m);
                return std::sqrt(s / static_cast<double>(v.size() - 1));
            };
            std::vector<double> d;
            for (size_t i = 1; i < its.size(); ++i) if (its[i] > its[i-1]) d.push_back((its[i]-its[i-1]) / 1000.0);
            std::sort(d.begin(), d.end());
            const double dt_s = d.empty() ? 0.05 : d[d.size()/2];
            const double dir_v = sd(av) * std::sqrt(dt_s);
            const double dir_w = sd(rv) * std::sqrt(dt_s);
            direct_v = dir_v; direct_w = dir_w;

            std::printf("\n  ── INDEPENDENT CHECK: the odometry stream's OWN noise, parked ──\n");
            std::printf("    %zu parked samples at %s s spacing\n", av.size(), num(dt_s,3).c_str());
            std::printf("    sigma_v     = %s m/sqrt(s)   (direct, localiser not involved)\n", num(dir_v,4).c_str());
            std::printf("    sigma_omega = %s rad/sqrt(s)\n", num(dir_w,4).c_str());

            const auto& B = bands.back();
            const double ratio_v = (dir_v > 0) ? B.trans.sigma / dir_v : 0;
            const double ratio_w = (dir_w > 0) ? B.rot.sigma   / dir_w : 0;
            std::printf("    the regression above is %sx (translation) and %sx (rotation) LARGER\n",
                        num(ratio_v,3).c_str(), num(ratio_w,3).c_str());
            reference_limited = (ratio_v > 3.0 or ratio_w > 3.0);
            if (reference_limited)
                std::printf(
                    "    ⚠⚠ REFERENCE-LIMITED — DO NOT USE THE REGRESSION'S sigma AS A CALIBRATION.\n"
                    "       The odometry is far quieter than the posterior it is being compared against,\n"
                    "       so the regression is reporting the LOCALISER's jitter (SDF residual noise,\n"
                    "       odometry/lidar window misalignment), not an encoder property. Only the\n"
                    "       SCALE columns survive this, and only at the long-window end where the\n"
                    "       jitter has averaged down. Use the direct numbers above for sigma.\n");
            else
                std::printf("    (within 3x — the odometry dominates the comparison, the regression's "
                            "sigma is usable)\n");
        }
    }

    // Pre-registered selection, used by BOTH the scorecard and the suggestion so they cannot
    // disagree: the LONGEST window that still has 30 windows behind it. The longest row overall is
    // not safe to read — measured, the 192-frame row had n = 25 and its rotation fit blew up to
    // -0.724 +-0.206 with sigma 0.336 against ~0.008 on every other row, because over ~10 s windows
    // the robot returns to similar headings and the regressor collapses. A rule chosen after seeing
    // the answer is not a rule, so this one is fixed and the degenerate row is simply not read.
    // ★ SELECTED PER CHANNEL, and guarded on the fitted sigma rather than on n alone. The two
    // channels degenerate at DIFFERENT window lengths: over ~10 s windows this robot's net heading
    // change collapses (it turns back and forth), so the rotation regressor loses its span while the
    // translation regressor still has plenty. Measured on a 17k-frame run, the 192-frame row gave
    // rotation s = -0.555 +-0.110 with sigma 0.2135 against -0.056 +-0.003 and sigma 0.0095 one row
    // up — a 22x jump in a quantity that is a PHYSICAL CONSTANT and cannot legitimately move like
    // that. n was 62 there, so an n threshold does not catch it; the sigma consistency does. Same
    // rule, fixed in advance, applied independently to each channel.
    const auto pick = [&bands](bool rotation) -> const BandResult* {
        double sig_min = std::numeric_limits<double>::max();
        for (const auto& b : bands)
        {
            const Fit& f = rotation ? b.rot : b.trans;
            if (f.n >= 30 and f.sigma > 0) sig_min = std::min(sig_min, f.sigma);
        }
        const BandResult* best = nullptr;
        for (const auto& b : bands)
        {
            const Fit& f = rotation ? b.rot : b.trans;
            if (f.n >= 30 and f.sigma <= 3.0 * sig_min) best = &b;
        }
        return best;
    };
    const BandResult* best_rot   = pick(true);
    const BandResult* best_trans = pick(false);
    const BandResult* best = (best_trans != nullptr) ? best_trans : best_rot;

    // ── SCORECARD: if this log carries injected ground truth, grade the estimate against it ──────
    // The point of the inj_* columns is that an archived log is self-describing, so this needs no
    // side channel and no memory of which run was which.
    if (col.count("inj_active"))
    {
        std::ifstream in3(argv[1]);
        std::string l3; std::getline(in3, l3);
        std::vector<std::string_view> b3;
        float act = 0, tsv = 0, tsw = 0, tcv = 0, tcw = 0;
        bool got = false;
        while (std::getline(in3, l3))
        {
            std::string_view lv(l3);
            split(lv, b3);
            if (b3.size() < ncols) continue;
            if (!parse_float(b3[col["inj_active"]], act)) continue;
            parse_float(b3[col["inj_sigma_v"]], tsv);
            parse_float(b3[col["inj_sigma_w"]], tsw);
            parse_float(b3[col["inj_scale_v"]], tcv);
            parse_float(b3[col["inj_scale_w"]], tcw);
            got = true;
            break;
        }
        if (got and act > 0.5f and best_rot != nullptr and best_trans != nullptr)
        {
            const auto& BR = *best_rot;
            const auto& BT = *best_trans;
            std::printf("\n  ══ RECOVERY SCORECARD — this log carries injected ground truth ══\n");
            std::printf("    %-14s %14s %14s %10s\n", "parameter", "TRUTH", "RECOVERED", "ratio");
            auto row = [&](const char* n, double truth, double got_v) {
                std::printf("    %-14s %14s %14s %10s\n", n, num(truth,5).c_str(), num(got_v,5).c_str(),
                            std::abs(truth) > 1e-12 ? num(got_v/truth,4).c_str() : "-");
            };
            std::printf("    (s_v read at %d frames / %s s, n=%d;  s_omega at %d frames / %s s, n=%d)\n",
                        BT.stride, num(BT.T_med,3).c_str(), BT.trans.n,
                        BR.stride, num(BR.T_med,3).c_str(), BR.rot.n);
            row("sigma_v",     tsv, direct_v > 0 ? direct_v : BT.trans.sigma);
            row("sigma_omega", tsw, direct_w > 0 ? direct_w : BR.rot.sigma);
            row("s_v",         tcv, BT.trans.s);
            row("s_omega",     tcw, BR.rot.s);
            std::printf("    ★ sigma is scored against the DIRECT parked measurement (the regression is\n"
                        "      reference-limited); the scales against the long-window regression.\n"
                        "    ⚠ s_* truth is the REALISED draw for this run, which is what the estimator\n"
                        "      can possibly recover — not the configured prior width.\n");
        }
        else if (got)
            std::printf("\n  (inj_active = 0 — no injected truth in this log; a normal run.)\n");
    }

    std::printf("\n  Suggested config (from the longest window with n >= 30):\n");

    if (best)
    {
        // When the regression is reference-limited its sigma describes the LOCALISER, not the
        // odometry, so emitting it here would hand the user a number the warning above just told
        // them not to use. Fall back to the direct parked measurement, and say which was used.
        const bool use_direct = reference_limited and direct_v > 0 and direct_w > 0;
        const double out_w = use_direct ? direct_w : (best_rot   ? best_rot->rot.sigma   : 0.0);
        const double out_v = use_direct ? direct_v : (best_trans ? best_trans->trans.sigma : 0.0);
        std::printf("    # sigma from: %s\n", use_direct
                    ? "the DIRECT parked measurement (the regression is reference-limited)"
                    : "the regression");
        std::printf("    PreintOdomSigmaOmega = %s\n", num(out_w, 5).c_str());
        std::printf("    PreintOdomSigmaVLat  = %s\n", num(out_v, 5).c_str());
        std::printf("    PreintOdomSigmaVLong = %s\n", num(out_v, 5).c_str());
        std::printf("    PreintOdomScaleOmega = %s   # |s_omega| measured\n",
                    num(best_rot   ? std::abs(best_rot->rot.s)     : 0.0, 5).c_str());
        std::printf("    PreintOdomScaleV     = %s   # |s_v| measured\n",
                    num(best_trans ? std::abs(best_trans->trans.s) : 0.0, 5).c_str());
        std::printf("\n  ⚠ The scale keys are PRIOR WIDTHS over an unknown constant, and what is measured\n"
                    "    here is ONE realisation of it. A single run cannot separate 'the scale is 0.07'\n"
                    "    from 'the scale is drawn from N(0, 0.07^2)'. Use |s| as a lower bound on the\n"
                    "    prior width and widen it across sessions, or make s a state and let the SDF\n"
                    "    infer it per run.\n");
    }
    else
        std::printf("    (not enough windows — need a longer or more varied run)\n");
    return 0;
}
