/*  calibration_estimator.h — joint batch estimation of the robot's motion-model parameters.
 *
 *  WHAT IT IS
 *  ----------
 *  A sliding window of EPISODES (one open-loop prediction ramp plus the correction that ended it) is
 *  solved jointly for the parameters of the motion model. The localiser's own correction is the
 *  measurement, so this needs no ground truth and no calibration rig, and behaves identically on
 *  hardware and in simulation.
 *
 *  WHY JOINT, AND NOT ONE FILTER PER PARAMETER
 *  -------------------------------------------
 *  The regressors are not orthogonal in real driving. Estimating them independently is what made the
 *  forward scale oscillate around 1.0 for an entire session: every turn cancelled what the straights
 *  taught, because a turn's pose correction carries a component the translation parameters cannot
 *  explain and an independent Jacobian hands it to them anyway. That was patched by inflating R with
 *  |dtheta| — a symptom fix. A joint solve handles it by construction.
 *
 *  It is also the ONLY way some parameters can be separated at all. eps_yaw and a sensor lever arm
 *  produce the SAME component of the correction and differ only in which covariate they scale with
 *  (distance vs rotation); so do the gyro's scale and its bias (rotation vs elapsed time). No
 *  independent scalar filter can tell those apart.
 *
 *  WHAT "GOOD" MEANS — AND WHY IT IS NOT A THRESHOLD
 *  ------------------------------------------------
 *  Two criteria fall out of the solve itself: the posterior sigma has SHRUNK against the prior
 *  (information was gained), and the information matrix is well-conditioned for that parameter (it is
 *  observable in THIS data, not merely in principle). Reported per parameter, because a run supplies
 *  some covariates and not others: measured live, a turn-heavy run leaves eps_yaw untaught with a
 *  WIDENING sigma while a straight-heavy run does the same to the gyro scale. ★A parameter's VALUE
 *  cannot distinguish "converged" from "never asked"; only its precision can.
 *
 *  THE COLD-START TRAP
 *  -------------------
 *  A window with no motion has near-zero residual, which reads as "my measurements are perfect" and
 *  makes every future manoeuvre look pointless. afford_calib already paid for this: before its fix a
 *  brand-new robot advertised the SMALLEST information gain of its life (0.03 nats) exactly when it
 *  had most to learn, rising to 3.46 once a conjugate prior at a MEASURED density was added.
 *  ★Zero motion is not evidence of a quiet sensor. The prior here plays the same role and is not
 *  optional.
 */
#pragma once
#include <string>
#include <locale>
#include <charconv>
#include <fstream>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <string_view>

namespace rc::calib
{
    /// The parameters, in solve order. Extend at the END so stored covariances stay comparable.
    enum Param : int
    {
        P_K_V = 0,      ///< translation odometry scale (fractional, 0 = correct)
        P_EPS_YAW,      ///< body/mount yaw offset (rad)
        P_K_OMEGA,      ///< gyro scale (fractional)
        P_B_OMEGA,      ///< gyro bias (rad/s) — separated from scale ONLY by time-vs-rotation
        P_K_LAT,        ///< wheel LATERAL scale (fractional). Excitable only on a base that can
                        ///< strafe; on a differential base its covariate is identically zero and it
                        ///< correctly stays at its prior for ever.
        P_DK_WHEEL,     ///< per-wheel mismatch (rad per metre travelled). Unequal effective wheel
                        ///< radii make a commanded straight line CURVE, so it lands on the heading
                        ///< component but is driven by DISTANCE — which is the only thing separating
                        ///< it from the gyro scale (rotation) and the gyro bias (time). Three
                        ///< parameters, one component, three covariates.
        P_COUNT
    };

    [[nodiscard]] constexpr std::string_view param_name(int p)
    {
        switch (p)
        {
            case P_K_V:     return "k_v";
            case P_EPS_YAW: return "eps_yaw";
            case P_K_OMEGA: return "k_omega";
            case P_B_OMEGA: return "b_omega";
            case P_K_LAT:    return "k_lat";
            case P_DK_WHEEL: return "dk_wheel";
            default:        return "?";
        }
    }

    /// One ramp-plus-correction. All quantities are accumulated over the whole episode.
    struct Episode
    {
        // Covariates: the motion that would reveal each parameter.
        float d_forward = 0.f;   ///< m, body +Y, as the model believed it
        float d_lateral = 0.f;   ///< m, body +X
        float d_theta   = 0.f;   ///< rad, as the model believed it
        float duration  = 0.f;   ///< s — the covariate that separates a gyro BIAS from a scale

        // The measurement: what the optimizer had to add, in the ROBOT frame.
        float r_forward = 0.f;   ///< m
        float r_lateral = 0.f;   ///< m
        float r_theta   = 0.f;   ///< rad

        // How much this episode should be believed.
        float pos_var   = 1e-4f; ///< m^2, localiser posterior + model-error terms
        float theta_var = 1e-4f; ///< rad^2
    };

    struct Result
    {
        Eigen::Matrix<float, P_COUNT, 1> value = Eigen::Matrix<float, P_COUNT, 1>::Zero();
        Eigen::Matrix<float, P_COUNT, 1> sigma = Eigen::Matrix<float, P_COUNT, 1>::Zero();
        /// True where the posterior actually shrank against the prior — i.e. this window TAUGHT
        /// something about that parameter, as opposed to leaving it where the prior put it.
        Eigen::Matrix<bool, P_COUNT, 1> informed = Eigen::Matrix<bool, P_COUNT, 1>::Zero();
        int   episodes = 0;
        /// Condition number of the CORRELATION-NORMALISED information matrix, D^-1/2 H D^-1/2.
        /// Normalising is not cosmetic: on the raw H the parameters have wildly different units
        /// (a dimensionless scale ~1e-2 against a bias in rad/s ~1e-4), so its eigenvalues span
        /// orders of magnitude whatever the geometry, and the number says nothing about
        /// collinearity. Measured on synthetic data: a SEPARABLE window scored 157145 and a
        /// deliberately COLLINEAR one 118002 -- i.e. the raw condition number ranked them backwards.
        /// Normalised, it measures what it is meant to: how nearly two columns point the same way.
        float condition = 0.f;
        bool  ok = false;
    };

    /// Prior: ZERO mean, with a stated sigma per parameter.
    ///
    /// ★★★THE MEAN STAYS AT ZERO, AND RE-CENTRING IT ON THE RUNNING ESTIMATE IS A TRAP. It was tried
    /// (2026-08-23) to stop a sliding window forgetting what an earlier window taught. It does that,
    /// and it also gives a weakly-excited parameter a RATCHET: the solve nudges the mean, the prior
    /// follows it there, the next solve nudges again, and with no restoring force the value random-
    /// walks. Measured live: the gyro bias walked from +0.0012 to -0.0140 deg/s over nine minutes
    /// without a single window ever reporting it as informed, against a true injected value of
    /// +0.0029 -- wrong sign, five times the magnitude, and quietly feeding into the prediction.
    ///
    /// With a fixed zero mean, an unexcited parameter simply returns to its prior: "I don't know",
    /// which is the truth. Pair it with Result::informed so a consumer can tell that from a
    /// measurement. The cost is real -- a turn-heavy window will read eps_yaw ~ 0 even though an
    /// earlier window measured it -- and is accepted deliberately: a value that is honestly absent is
    /// better than one that is silently wrong. Making this a proper recursive estimator means
    /// carrying the posterior COVARIANCE forward too, which cannot be combined with re-solving a
    /// sliding window because the same episodes would be counted repeatedly.
    ///
    /// ★NOT optional — see THE COLD-START TRAP. It is also what keeps the solve well-posed when a
    /// window happens to contain no motion of the kind a given parameter needs.
    struct Prior
    {
        float sigma_k_v     = 0.02f;    ///< 2% — a wheel radius is not wrong by more than this
        float sigma_eps_yaw = 0.0175f;  ///< 1 deg of mount/axis misalignment
        float sigma_k_omega = 0.02f;    ///< 2%
        float sigma_b_omega = 5.0e-4f;  ///< rad/s ~ 0.03 deg/s, a plausible post-calibration residual
        float sigma_k_lat   = 0.05f;    ///< 5% — roller slip makes a mecanum's lateral channel much
                                        ///< worse than its forward one, so the prior is looser
        float sigma_dk_wheel = 0.02f;   ///< rad/m — 2 cm of lateral drift per metre driven straight
    };

    class BatchEstimator
    {
    public:
        void configure(const Prior &p, std::size_t window) { prior_ = p; window_ = std::max<std::size_t>(window, 8); }

        void add(const Episode &e)
        {
            eps_.push_back(e);
            while (eps_.size() > window_) eps_.pop_front();
        }

        [[nodiscard]] std::size_t size() const noexcept { return eps_.size(); }
        void clear() noexcept { eps_.clear(); }

        /// ── A CLOSED PIVOT, as one more residual row ─────────────────────────────────────────────────
    /// A closure is a direct, map-free measurement of the rotation model: turn through N complete
    /// turns, come back to the heading you started from, and the robot turned exactly 2*pi*N radians.
    /// No map, no survey, no localiser anywhere in that number — which makes it a STRONGER instrument
    /// than the episode rows, whose reference is the optimizer's own correction.
    ///
    /// ★ IT IS THE SAME TWO COVARIATES AS THE HEADING ROW, so it drops into the existing solve with
    ///   no new machinery. The heading row is r_theta ~ d_theta*k_omega + duration*b_omega; a closure
    ///   supplies exactly that pair, with the rotation being the truth and the duration being
    ///   truth/rate. The residual is the heading the model still owes: truth - (what the CORRECTED
    ///   odometry accumulated), so a perfectly calibrated robot contributes r = 0 and teaches nothing,
    ///   which is the correct behaviour.
    ///
    /// ★ AND IT IS WHY TWO RATES ARE RUN. At one rate the two unknowns enter through a single number,
    ///   k + b/w, and no solve can separate them. Two blocks at different rates give two rows whose
    ///   b-covariate differs while the k-covariate does not — the same separation-by-covariate that
    ///   the episode rows get from rotation versus elapsed time.
    ///
    /// `sigma_s` is the closure's own resolution (|heading miss| / |truth|), so the weight comes out
    /// as one over the heading miss squared — the run is believed exactly as precisely as it closed.
    void add_closure(double truth_rad, double turned_rad, double rate_rad_s, double sigma_s)
    {
        if (not (truth_rad > 0.0) or not (rate_rad_s > 0.0) or not (sigma_s > 0.0)) return;
        if (not std::isfinite(turned_rad)) return;
        ClosureRow c;
        c.r_theta  = static_cast<float>(truth_rad - turned_rad);
        c.d_theta  = static_cast<float>(truth_rad);
        c.duration = static_cast<float>(truth_rad / rate_rad_s);
        const double sigma_r = sigma_s * truth_rad;                 // radians
        c.weight   = static_cast<float>(1.0 / (sigma_r * sigma_r));
        cls_.push_back(c);
        // Closures are rare and expensive -- each is minutes of the robot's time -- so the window is
        // generous and bounded rather than tuned. Evicting one throws away a measurement no amount of
        // ordinary driving reproduces.
        while (cls_.size() > 16) cls_.pop_front();
    }
    [[nodiscard]] std::size_t closures() const noexcept { return cls_.size(); }

    /// Forget everything measured and return to the priors. The window is emptied; nothing else
    /// changes, so the very next solve reports each parameter at its prior sigma and `informed`
    /// false — which is the honest description of a robot that has just been told to un-learn.
    void reset() noexcept { eps_.clear(); cls_.clear(); }

    /// ── PERSIST THE EVIDENCE, NOT THE CONCLUSION ─────────────────────────────────────────────────
    /// What is written is the WINDOW: the episodes and the closed pivots. Not the parameters.
    ///
    /// ★ THAT DISTINCTION IS THE WHOLE DESIGN. Saving the fitted values and restoring them as a prior
    ///   mean is re-centring, and re-centring was tried on 2026-08-23 and measured to fail: a weakly
    ///   excited parameter gets no restoring force, so the solve nudges the mean, the prior follows
    ///   it there, and the value random-walks. The gyro bias walked to -0.0140 deg/s over nine
    ///   minutes, wrong sign and five times the magnitude, without one window ever reporting it
    ///   informed. Restoring the DATA has none of that: the prior stays at zero, the solve is the
    ///   same solve it would have been had the run never stopped, and a parameter nothing excited
    ///   still comes back saying "I don't know".
    ///
    /// ★ A closure is worth minutes of the robot's time and cannot be reproduced by ordinary driving,
    ///   which is the strongest reason for this file to exist at all.
    ///
    /// Locale: written through the classic locale and read with std::from_chars, because these
    /// machines run es_ES where strtof would stop at the decimal point and return the integer part,
    /// silently. See CLAUDE.md.
    bool save(const std::string& path) const
    {
        std::ofstream f(path, std::ios::out | std::ios::trunc);
        if (not f.is_open()) return false;
        f.imbue(std::locale::classic());
        f << "# motion calibration window — evidence, not parameters. Delete to return to the priors.\n";
        f << "# E,d_forward,d_lateral,d_theta,duration,r_forward,r_lateral,r_theta,pos_var,theta_var\n";
        f << "# C,r_theta,d_theta,duration,weight\n";
        for (const auto& e : eps_)
            f << "E," << e.d_forward << ',' << e.d_lateral << ',' << e.d_theta << ',' << e.duration
              << ',' << e.r_forward << ',' << e.r_lateral << ',' << e.r_theta
              << ',' << e.pos_var << ',' << e.theta_var << '\n';
        for (const auto& c : cls_)
            f << "C," << c.r_theta << ',' << c.d_theta << ',' << c.duration << ',' << c.weight << '\n';
        return true;
    }

    /// Returns how many rows were restored; 0 for "no file", which is the ordinary first-run state
    /// and not an error.
    std::size_t load(const std::string& path)
    {
        std::ifstream f(path);
        if (not f.is_open()) return 0;
        eps_.clear(); cls_.clear();
        std::string line; std::size_t n = 0;
        while (std::getline(f, line))
        {
            if (line.empty() or line[0] == '#') continue;
            std::vector<float> v; v.reserve(9);
            const char* p = line.data() + 2;              // past the tag and its comma
            const char* end = line.data() + line.size();
            while (p < end)
            {
                float x = 0.f;
                const auto [next, ec] = std::from_chars(p, end, x);
                if (ec != std::errc{}) break;             // a malformed row is dropped, not guessed
                v.push_back(x);
                p = (next < end and *next == ',') ? next + 1 : end;
            }
            if (line[0] == 'E' and v.size() == 9)
            {
                Episode e;
                e.d_forward = v[0]; e.d_lateral = v[1]; e.d_theta   = v[2]; e.duration = v[3];
                e.r_forward = v[4]; e.r_lateral = v[5]; e.r_theta   = v[6];
                e.pos_var   = v[7]; e.theta_var = v[8];
                eps_.push_back(e); ++n;
            }
            else if (line[0] == 'C' and v.size() == 4)
            {
                ClosureRow c;
                c.r_theta = v[0]; c.d_theta = v[1]; c.duration = v[2]; c.weight = v[3];
                cls_.push_back(c); ++n;
            }
        }
        while (eps_.size() > window_) eps_.pop_front();
        while (cls_.size() > 16)      cls_.pop_front();
        return n;
    }

    /// Solve the window. p = (J'WJ + P0^-1)^-1 J'W r, and that same inverse IS the covariance.
        [[nodiscard]] Result solve() const
        {
            Result out;
            out.episodes = static_cast<int>(eps_.size());
            if (eps_.size() < 4) return out;

            Eigen::Matrix<float, P_COUNT, P_COUNT> H = Eigen::Matrix<float, P_COUNT, P_COUNT>::Zero();
            Eigen::Matrix<float, P_COUNT, 1>       b = Eigen::Matrix<float, P_COUNT, 1>::Zero();

            // Prior information. Zero mean, so it contributes to H only -- see Prior for why
            // re-centring it on the running estimate is a ratchet rather than a memory.
            Eigen::Matrix<float, P_COUNT, 1> p0;
            p0 << prior_.sigma_k_v, prior_.sigma_eps_yaw, prior_.sigma_k_omega, prior_.sigma_b_omega,
                  prior_.sigma_k_lat, prior_.sigma_dk_wheel;
            for (int i = 0; i < P_COUNT; ++i)
                H(i, i) += 1.f / std::max(p0[i] * p0[i], 1e-18f);
            const Eigen::Matrix<float, P_COUNT, P_COUNT> H_prior = H;

            for (const auto &e : eps_)
            {
                const float wp = 1.f / std::max(e.pos_var, 1e-12f);
                const float wt = 1.f / std::max(e.theta_var, 1e-12f);

                // ALONG-track row: only the translation scale moves the robot along its own heading.
                Eigen::Matrix<float, P_COUNT, 1> j_along = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_along[P_K_V] = e.d_forward;
                H += wp * j_along * j_along.transpose();
                b += wp * j_along * e.r_forward;

                // CROSS-track row: a yaw offset rotates FORWARD travel off-axis, while a lateral
                // scale error mis-measures LATERAL travel. Same component, different covariates —
                // which is exactly what lets them be told apart, and only while the robot does both.
                // The sign on eps follows the integrator's body->world mapping, where +eps moves the
                // step by -d_forward laterally.
                Eigen::Matrix<float, P_COUNT, 1> j_cross = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_cross[P_EPS_YAW] = -e.d_forward;
                j_cross[P_K_LAT]   =  e.d_lateral;
                H += wp * j_cross * j_cross.transpose();
                b += wp * j_cross * e.r_lateral;

                // THREE parameters share this component and are separated only by covariate:
                // a SCALE by rotation, a BIAS by elapsed time, and a per-wheel mismatch by DISTANCE
                // (unequal wheel radii make a commanded straight curve). No pair of scalar filters
                // could do this; it is the clearest case for solving jointly.
                Eigen::Matrix<float, P_COUNT, 1> j_th = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_th[P_K_OMEGA]  = e.d_theta;
                j_th[P_B_OMEGA]  = e.duration;
                j_th[P_DK_WHEEL] = e.d_forward;
                H += wt * j_th * j_th.transpose();
                b += wt * j_th * e.r_theta;
            }

            // ── Closed pivots, on the same two covariates as the heading row above ────────────────
            for (const auto &c : cls_)
            {
                Eigen::Matrix<float, P_COUNT, 1> j_cl = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_cl[P_K_OMEGA] = c.d_theta;
                j_cl[P_B_OMEGA] = c.duration;
                H += c.weight * j_cl * j_cl.transpose();
                b += c.weight * j_cl * c.r_theta;
            }

            const Eigen::LDLT<Eigen::Matrix<float, P_COUNT, P_COUNT>> ldlt(H);
            if (ldlt.info() != Eigen::Success) return out;
            out.value = ldlt.solve(b);
            const Eigen::Matrix<float, P_COUNT, P_COUNT> cov = H.inverse();
            if (not cov.allFinite()) return out;

            for (int i = 0; i < P_COUNT; ++i)
            {
                out.sigma[i] = std::sqrt(std::max(cov(i, i), 0.f));
                // "Informed" = this window actually shrank the posterior for THIS parameter. A
                // parameter the driving never excited comes back at its prior sigma and is reported
                // as uninformed rather than as a confident zero.
                out.informed[i] = out.sigma[i] < 0.9f * (1.f / std::sqrt(H_prior(i, i)));
            }
            // Normalise to correlation form before asking about conditioning -- see Result::condition.
            Eigen::Matrix<float, P_COUNT, 1> d;
            for (int i = 0; i < P_COUNT; ++i) d[i] = 1.f / std::sqrt(std::max(H(i, i), 1e-30f));
            const Eigen::Matrix<float, P_COUNT, P_COUNT> Hn = d.asDiagonal() * H * d.asDiagonal();
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, P_COUNT, P_COUNT>> es(Hn);
            const auto ev = es.eigenvalues();
            out.condition = ev.minCoeff() > 0.f ? ev.maxCoeff() / ev.minCoeff()
                                                : std::numeric_limits<float>::infinity();
            out.ok = out.value.allFinite();
            return out;
        }

    private:
        Prior prior_{};
        std::size_t window_ = 64;
        struct ClosureRow { float r_theta = 0.f, d_theta = 0.f, duration = 0.f, weight = 0.f; };
        std::deque<ClosureRow> cls_;
        std::deque<Episode> eps_;
    };
}
