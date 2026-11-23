#pragma once
// ── THE ROOM'S CHANGE BELIEF (2026-11-23, AuRo journal item B) ──────────────────────────────────────────────────
//
// Replaces the known-room offer's NEGLECT DRIVE (0.5·log(1 + age/120 s), 4.49 nats for a never-visited cell: a
// computed quantity, but not information from any belief the room holds) with the expected information of
// re-observing parts of the room that may have CHANGED since they were last seen.
//
// THE MODEL
//   · Changes (furniture or objects appearing, leaving, moving) arrive as a Poisson process in space and time with
//     hazard λ per m² per second. Occupancy is correlated over a length ℓ (coverage_info.h), so a cell of area A holds
//     n = A/ℓ² independent regions, each of area ℓ².
//   · λ is unknown. Its belief is the exact 1-D posterior on a log-spaced grid, starting from a Gamma(α₀, β₀) prior.
//     The predictive probability that a region last observed `a` seconds ago has changed is the posterior average
//     p(a) = 1 − E[e^{−λ ℓ² a}].
//   · Re-observing a region is a noisy test of "changed?": a change shows with probability d, a false alarm with f.
//     Its expected information is the binary mutual information I(p; d, f) ≤ ln 2 (the same formula as the objects'
//     existence information). A view's change information = Σ over the cells it would observe of n · I(p(a_k)).
//   · Cells the robot observes NOW have a ≈ 0, hence I ≈ 0: the term is marginal over staying put by construction.
//     A cell never observed in this run has an unknown age, taken from a stated prior (the time since the last tour).
//
// LEARNING λ (per re-observation): a cell seen before is observed again after age a; whether its furniture state
// FLIPPED (matter ↔ no matter) is the noisy change test. Exact Bayes: the posterior over λ is multiplied by
//   P(flip | λ) = d·c(λ) + f·(1 − c(λ)),  c(λ) = 1 − e^{−λ·A·a}  (A the cell's area), or its complement for no flip.
// (2026-11-23: an online EM step was tried first; with responsibilities taken at the current λ it stalled at half the
//  prior in a static home and under-estimated a changing one. The grid posterior has no such bias.)
//
// No decay constant and no drive weight remain; λ's prior, d, f and the unseen-age prior are stated priors.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include "coverage_info.h"

namespace rc::change
{
/// Binary mutual information between "changed" (prior p) and the observation, detection d, false alarm f. ≤ ln 2.
[[nodiscard]] inline double mutual_info(double p, double d, double f)
{
    using rc::coverage::binary_entropy;
    p = std::clamp(p, 0.0, 1.0);
    return std::max(0.0, binary_entropy(p * d + (1.0 - p) * f) - p * binary_entropy(d) - (1.0 - p) * binary_entropy(f));
}

struct Params
{
    double hazard_mean      = 3e-6;     ///< prior mean of λ, changes per m² per second (~5 per 20 m² room per day)
    double hazard_shape     = 1.0;      ///< prior α (weak: one change's worth of evidence)
    double detect           = 0.85;     ///< P(the re-observation shows a change | changed)
    double false_alarm      = 0.05;     ///< P(it shows one | unchanged)
    double unseen_age_s     = 86400.0;  ///< prior age of a cell never observed in this run (the time since the last tour)
};

class ChangeBelief
{
public:
    struct Cell
    {
        double t_obs = -1.0;        ///< end of the last observation episode; < 0: never observed in this run
        std::int8_t matter = -1;    ///< furniture state at that episode (majority of its sweeps); -1 unknown
        bool in_view = false;       ///< an episode is open now
        int sweeps = 0, hits = 0;   ///< sweeps of the open episode, and how many saw matter in the cell
    };

    explicit ChangeBelief(Params p = {}) : p_(p)
    {
        // λ grid: 1e-10 .. 1e-2 changes per m² per s, log-spaced; Gamma(α₀, β₀) prior with mean hazard_mean
        const double a0 = std::max(1e-3, p_.hazard_shape), b0 = a0 / std::max(1e-30, p_.hazard_mean);
        for (int i = 0; i < kGrid; ++i)
        {
            const double lam = std::pow(10.0, -10.0 + 8.0 * i / (kGrid - 1));
            lam_[i] = lam;
            // Gamma density × the log-grid Jacobian (λ dlogλ)
            logw_[i] = a0 * std::log(lam) - b0 * lam;
        }
        normalise();
    }

    void init(const Eigen::Vector2f& origin, int cols, int rows, float cell_size)
    {
        origin_ = origin; cols_ = std::max(1, cols); rows_ = std::max(1, rows); cell_ = cell_size;
        cells_.assign(static_cast<std::size_t>(cols_) * rows_, Cell{});
    }
    [[nodiscard]] bool ready() const { return not cells_.empty(); }
    [[nodiscard]] int size() const { return static_cast<int>(cells_.size()); }
    [[nodiscard]] const Cell& cell(int k) const { return cells_[k]; }
    [[nodiscard]] double hazard() const   ///< posterior mean λ
    {
        double m = 0.0;
        for (int i = 0; i < kGrid; ++i) m += w_[i] * lam_[i];
        return m;
    }
    [[nodiscard]] std::uint64_t tests() const { return tests_; }
    [[nodiscard]] const Params& params() const { return p_; }

    /// Predictive P(one ℓ×ℓ region changed) after `age_s`.
    [[nodiscard]] double p_changed(double age_s, double ell) const { return p_area(age_s, ell * ell); }
    /// Predictive P(a change somewhere in `area_m2`) after `age_s`.
    [[nodiscard]] double p_area(double age_s, double area_m2) const
    {
        if (not (age_s > 0.0)) return 0.0;
        double q = 0.0;
        for (int i = 0; i < kGrid; ++i) q += w_[i] * std::exp(-lam_[i] * area_m2 * age_s);
        return std::clamp(1.0 - q, 0.0, 1.0);
    }
    [[nodiscard]] double age(int k, double now_s) const
    {
        const auto& c = cells_[k];
        return c.t_obs < 0.0 ? p_.unseen_age_s : std::max(0.0, now_s - c.t_obs);
    }
    /// Expected information (nats) of observing cell k now. A cell IN VIEW is being observed: nothing left to offer.
    [[nodiscard]] double info(int k, double now_s, double ell) const
    {
        if (cells_[k].in_view) return 0.0;
        const double n = static_cast<double>(cell_) * cell_ / std::max(1e-6, ell * ell);
        return n * mutual_info(p_changed(age(k, now_s), ell), p_.detect, p_.false_alarm);
    }

    /// Cell k was observed at `now_s`; `matter` = furniture returns were seen in it. When the cell had been observed
    /// before, the flip (or its absence) multiplies λ's posterior by its likelihood. One call per OBSERVATION EPISODE
    /// (the caller's job: a cell kept in view is one test, not one per sweep — a per-sweep test would count sensor
    /// flicker as many false alarms).
    void observe(int k, double now_s, bool matter, double /*ell*/ = 0.0)
    {
        auto& c = cells_[k];
        const std::int8_t m = matter ? 1 : 0;
        if (c.t_obs >= 0.0 and c.matter >= 0)
        {
            const double a = now_s - c.t_obs;
            if (a > 0.0)
            {
                const double A = static_cast<double>(cell_) * cell_;
                const double d = p_.detect, f = p_.false_alarm;
                const bool flip = m != c.matter;
                for (int i = 0; i < kGrid; ++i)
                {
                    const double ch = 1.0 - std::exp(-lam_[i] * A * a);
                    const double pf = d * ch + f * (1.0 - ch);
                    logw_[i] += std::log(std::max(1e-300, flip ? pf : 1.0 - pf));
                }
                normalise();
                ++tests_;
            }
        }
        c.t_obs = now_s;
        c.matter = m;
    }

    /// One sweep: `visible` = the cells observed now (sorted or not), `matter(k)` = furniture returns landed in cell k.
    /// A cell entering view opens an episode; a cell leaving view closes it, and the closed episode is ONE test
    /// (observe) with the majority state of its sweeps — so a cell kept in view, flickering, is never counted as change.
    template <class MatterFn>
    void sweep(const std::vector<int>& visible, MatterFn&& matter, double now_s)
    {
        mark_.assign(cells_.size(), 0);
        for (const int k : visible)
        {
            if (k < 0 or k >= size()) continue;
            mark_[k] = 1;
            auto& c = cells_[k];
            if (not c.in_view) { c.in_view = true; c.sweeps = 0; c.hits = 0; }
            ++c.sweeps;
            c.hits += matter(k) ? 1 : 0;
        }
        for (int k = 0; k < size(); ++k)
            if (cells_[k].in_view and not mark_[k]) close_episode(k, now_s);
    }
    void close_episode(int k, double now_s)
    {
        auto& c = cells_[k];
        if (not c.in_view) return;
        const bool m = 2 * c.hits > c.sweeps;
        c.in_view = false;
        observe(k, now_s, m);
    }

    /// Forget this run's observations (a new room); λ's posterior is kept — it is a property of the home.
    void clear_cells() { for (auto& c : cells_) c = Cell{}; }

private:
    static constexpr int kGrid = 161;
    void normalise()
    {
        double mx = -1e300;
        for (int i = 0; i < kGrid; ++i) mx = std::max(mx, logw_[i]);
        double z = 0.0;
        for (int i = 0; i < kGrid; ++i) { logw_[i] -= mx; w_[i] = std::exp(logw_[i]); z += w_[i]; }
        for (int i = 0; i < kGrid; ++i) w_[i] /= z;
    }
    Params p_;
    std::array<double, kGrid> lam_{}, logw_{}, w_{};
    std::uint64_t tests_ = 0;
    std::vector<std::uint8_t> mark_;
    Eigen::Vector2f origin_{0.f, 0.f};
    int cols_ = 0, rows_ = 0;
    float cell_ = 0.5f;
    std::vector<Cell> cells_;
};
}   // namespace rc::change
