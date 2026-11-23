#pragma once
// ── THE ROOM'S CHANGE BELIEF (2026-11-23, AuRo journal item B; revised after auro's referee notes the same day) ────
//
// Replaces the known-room offer's NEGLECT DRIVE (0.5·log(1 + age/120 s), 4.49 nats for a never-visited cell: a
// computed quantity, but not information from any belief the room holds) with the expected information of
// re-observing parts of the room that may have CHANGED since they were last seen.
//
// THE MODEL
//   · Changes (furniture or objects appearing, leaving, moving) arrive as a Poisson process in space and time with
//     hazard λ per m² per second. Occupancy is correlated over a length ℓ (coverage_info.h), so a cell of area A holds
//     n = A/ℓ² independent regions, each of area ℓ².
//   · Re-observing a cell is a noisy test of "changed?": the cell's furniture state FLIPS (matter ↔ no matter) with
//     probability d if it changed and f (false alarm) if it did not.
//   · λ, d and f are all unknown and LEARNT: their belief is the exact joint posterior on a grid (λ log-spaced ×
//     d × f), from Gamma(λ) × Beta(d) × Beta(f) priors, multiplied at every test by P(flip | λ, d, f) =
//     d·c + f·(1 − c), c = 1 − e^{−λ A a}, or its complement for no flip (a = the cell's age).
//   · The predictive probability that one region last observed `a` seconds ago has changed is the posterior average
//     p(a) = 1 − E[e^{−λ ℓ² a}]; the information of re-observing it is the binary mutual information I(p; d̄, f̄) with
//     the posterior means of d and f (a plug-in for the sensor parameters: averaging I over their grid at planning
//     time would cost ~100× per cell). A view's change information = Σ over the cells it would observe of n·I.
//   · A cell observed NOW (in view) offers nothing: the term is marginal over staying put by construction.
//   · A cell with NO observation on record — neither in this run nor in the room's remembered record — has no age:
//     what a look at it resolves is its unknown STATE, so it is valued like unknown floor (coverage_info.h):
//     n · h(P(occupied)), passed in by the caller from the learnt occupancy prior.
//
// EPISODES: a cell kept in view is one observation, not one per sweep (a per-sweep test would count sensor flicker as
// many false alarms). Its state is the majority of the episode's sweeps; closing the episode is the test.
//
// Times are WALL-CLOCK seconds, so a room's record carries across runs (the caller persists it; export/import).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
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
    double hazard_mean    = 3e-6;   ///< prior mean of λ, changes per m² per second (~5 per 20 m² room per day)
    double hazard_shape   = 1.0;    ///< prior Gamma shape (weak: one change's worth of evidence)
    double detect_mean    = 0.85;   ///< prior mean of d = P(flip | changed)
    double detect_n       = 10.0;   ///< prior strength of d, in tests
    double false_mean     = 0.05;   ///< prior mean of f = P(flip | unchanged)
    double false_n        = 20.0;   ///< prior strength of f, in tests
};

/// One change test (an episode closed on a cell observed before) — what was promised and what was delivered.
struct Test
{
    int cell = -1;
    double t = 0.0, age = 0.0;
    double p_cell = 0.0;      ///< predictive P(the cell changed) before the test
    bool flip = false;
    double promised = 0.0;    ///< expected information of the test, I(p_cell; d̄, f̄)
    double delivered = 0.0;   ///< realised: KL(P(changed | outcome) ‖ P(changed)) — ≥ 0, its expectation is `promised`
    bool first = false;       ///< a FIRST observation (no record): its unknown state resolved, promised = delivered = n·h(P(occ))
};

class ChangeBelief
{
public:
    struct Cell
    {
        double t_obs = -1.0;        ///< wall-clock end of the last observation episode; < 0: no observation on record
        std::int8_t matter = -1;    ///< furniture state at that episode (majority of its sweeps); -1 unknown
        bool in_view = false;       ///< an episode is open now
        int sweeps = 0, hits = 0;   ///< sweeps of the open episode, and how many saw matter in the cell
    };

    explicit ChangeBelief(Params p = {}) : p_(p)
    {
        const double a0 = std::max(1e-3, p_.hazard_shape), b0 = a0 / std::max(1e-30, p_.hazard_mean);
        for (int i = 0; i < kL; ++i)
        {
            lam_[i] = std::pow(10.0, -10.0 + 8.0 * i / (kL - 1));
            prior_l_[i] = a0 * std::log(lam_[i]) - b0 * lam_[i];   // Gamma × the log-grid Jacobian
        }
        const auto beta_log = [](double x, double mean, double n)
        { const double a = mean * n, b = (1.0 - mean) * n; return (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x); };
        for (int j = 0; j < kD; ++j) { d_[j] = 0.30 + 0.69 * (j + 0.5) / kD; prior_d_[j] = beta_log(d_[j], p_.detect_mean, p_.detect_n); }
        // f on a log grid 1e-3 .. ~0.4: the Beta density times the log-grid Jacobian f
        for (int j = 0; j < kF; ++j) { f_[j] = std::pow(10.0, -3.0 + 2.6 * (j + 0.5) / kF); prior_f_[j] = beta_log(f_[j], p_.false_mean, p_.false_n) + std::log(f_[j]); }
        for (int i = 0; i < kL; ++i) for (int j = 0; j < kD; ++j) for (int k = 0; k < kF; ++k)
            logw_[at(i, j, k)] = prior_l_[i] + prior_d_[j] + prior_f_[k];
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
    [[nodiscard]] const Params& params() const { return p_; }
    [[nodiscard]] std::uint64_t tests() const { return tests_; }

    [[nodiscard]] double hazard() const { double m = 0.0; for (int i = 0; i < kL; ++i) m += wl_[i] * lam_[i]; return m; }
    [[nodiscard]] double detect() const { double m = 0.0; for (int j = 0; j < kD; ++j) m += wd_[j] * d_[j]; return m; }
    [[nodiscard]] double false_alarm() const { double m = 0.0; for (int k = 0; k < kF; ++k) m += wf_[k] * f_[k]; return m; }

    /// Information (nats) per region of a cell with no observation on record: set by the caller from the occupancy
    /// prior (h(P(occupied)), coverage_info.h).
    void set_unknown_nats(double nats_per_region) { unknown_nats_ = std::max(0.0, nats_per_region); }

    [[nodiscard]] double p_changed(double age_s, double ell) const { return p_area(age_s, ell * ell); }
    /// Predictive P(a change somewhere in `area_m2`) after `age_s`.
    [[nodiscard]] double p_area(double age_s, double area_m2) const
    {
        if (not (age_s > 0.0)) return 0.0;
        double q = 0.0;
        for (int i = 0; i < kL; ++i) q += wl_[i] * std::exp(-lam_[i] * area_m2 * age_s);
        return std::clamp(1.0 - q, 0.0, 1.0);
    }
    /// Expected information (nats) of observing cell k now.
    [[nodiscard]] double info(int k, double now_s, double ell) const
    {
        const auto& c = cells_[k];
        if (c.in_view) return 0.0;
        const double n = static_cast<double>(cell_) * cell_ / std::max(1e-6, ell * ell);
        if (c.t_obs < 0.0) return n * unknown_nats_;
        return n * mutual_info(p_changed(std::max(0.0, now_s - c.t_obs), ell), d_mean_, f_mean_);
    }

    /// One sweep: `visible` = the cells observed now, `matter(k)` = furniture returns landed in cell k.
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
        c.in_view = false;
        observe(k, now_s, 2 * c.hits > c.sweeps);
    }
    /// Close every open episode (leaving the room, or shutting down): the record then holds this run's observations.
    void close_all(double now_s) { for (int k = 0; k < size(); ++k) close_episode(k, now_s); }

    /// Cell k was observed (an episode closed) at `now_s`. When it had been observed before, the flip (or none) is a
    /// test: the joint posterior over (λ, d, f) is multiplied by its likelihood, and the test is reported.
    void observe(int k, double now_s, bool matter)
    {
        auto& c = cells_[k];
        const std::int8_t m = matter ? 1 : 0;
        if (c.t_obs >= 0.0 and c.matter >= 0 and now_s > c.t_obs)
        {
            const double a = now_s - c.t_obs, A = static_cast<double>(cell_) * cell_;
            const bool flip = m != c.matter;
            Test t;
            t.cell = k; t.t = now_s; t.age = a; t.flip = flip;
            t.p_cell = p_area(a, A);
            const double d = d_mean_, f = f_mean_, pc = t.p_cell;
            t.promised = mutual_info(pc, d, f);
            const double py = flip ? d * pc + f * (1.0 - pc) : (1.0 - d) * pc + (1.0 - f) * (1.0 - pc);
            const double post = py > 0.0 ? (flip ? d * pc : (1.0 - d) * pc) / py : pc;
            const auto kl = [](double q, double p0)
            {
                double v = 0.0;
                if (q > 0.0) v += q * std::log(q / std::max(1e-300, p0));
                if (q < 1.0) v += (1.0 - q) * std::log((1.0 - q) / std::max(1e-300, 1.0 - p0));
                return std::max(0.0, v);
            };
            t.delivered = kl(post, pc);
            for (int i = 0; i < kL; ++i)
            {
                const double ch = 1.0 - std::exp(-lam_[i] * A * a);
                for (int j = 0; j < kD; ++j) for (int q = 0; q < kF; ++q)
                {
                    const double pf = d_[j] * ch + f_[q] * (1.0 - ch);
                    logw_[at(i, j, q)] += std::log(std::max(1e-300, flip ? pf : 1.0 - pf));
                }
            }
            normalise();
            ++tests_;
            if (on_test) on_test(t);
        }
        else if (c.t_obs < 0.0 and on_test)
        {
            Test t;
            t.cell = k; t.t = now_s; t.first = true;
            t.promised = t.delivered = static_cast<double>(cell_) * cell_ / std::max(1e-6, ell_for_first_ * ell_for_first_) * unknown_nats_;
            on_test(t);
        }
        c.t_obs = now_s;
        c.matter = m;
    }
    /// ℓ used to count regions when a first observation is reported (the caller keeps it current).
    void set_ell(double ell) { ell_for_first_ = std::max(1e-3, ell); }

    std::function<void(const Test&)> on_test;   ///< every test, for the promised-vs-delivered log

    /// The cells' record (for persistence by the caller): centre, last observation, state. Cells never observed skipped.
    struct Record { Eigen::Vector2f centre; double t_obs; std::int8_t matter; };
    [[nodiscard]] std::vector<Record> export_cells() const
    {
        std::vector<Record> out;
        for (int k = 0; k < size(); ++k)
            if (cells_[k].t_obs >= 0.0) out.push_back({centre(k), cells_[k].t_obs, cells_[k].matter});
        return out;
    }
    /// Restore a record into the current grid (nearest cell; the newer observation wins).
    void import_cells(const std::vector<Record>& rs)
    {
        for (const auto& r : rs)
        {
            const float fx = (r.centre.x() - origin_.x()) / cell_, fy = (r.centre.y() - origin_.y()) / cell_;
            if (fx < 0.f or fy < 0.f or fx >= cols_ or fy >= rows_) continue;
            auto& cl = cells_[static_cast<std::size_t>(fy) * cols_ + static_cast<std::size_t>(fx)];
            if (r.t_obs > cl.t_obs) { cl.t_obs = r.t_obs; cl.matter = r.matter; }
        }
    }
    /// The joint posterior, for persistence (it is a property of the home, not of a room or a run).
    [[nodiscard]] std::vector<double> export_posterior() const { return {logw_.begin(), logw_.end()}; }
    bool import_posterior(const std::vector<double>& lw)
    {
        if (lw.size() != logw_.size()) return false;
        std::ranges::copy(lw, logw_.begin());
        normalise();
        return true;
    }
    [[nodiscard]] Eigen::Vector2f centre(int k) const
    { return {origin_.x() + (static_cast<float>(k % cols_) + 0.5f) * cell_, origin_.y() + (static_cast<float>(k / cols_) + 0.5f) * cell_}; }

    /// Forget the cells (a new room); the posterior is kept.
    void clear_cells() { for (auto& c : cells_) c = Cell{}; }

    /// ROOMS: one cell grid per room, in that room's own (published) frame; the posterior is shared — it is a property
    /// of the home. Entering a room closes the open episodes of the one being left, stashes its grid, and restores the
    /// entered room's grid when it has one (this run or a persisted record), else starts it empty.
    void enter_room(const std::string& name, const Eigen::Vector2f& origin, int cols, int rows, float cell_size, double now_s)
    {
        if (name == room_ and ready()) return;
        if (not room_.empty() and ready())
        {
            close_all(now_s);
            stash_[room_] = export_cells();
        }
        room_ = name;
        init(origin, cols, rows, cell_size);
        if (const auto it = stash_.find(name); it != stash_.end()) import_cells(it->second);
    }
    [[nodiscard]] const std::string& room() const { return room_; }
    /// Every room's record (the current one included, open episodes counted as observed now) — for persistence.
    [[nodiscard]] std::map<std::string, std::vector<Record>> all_records() const
    {
        auto all = stash_;
        if (not room_.empty() and ready()) all[room_] = export_cells();
        return all;
    }
    void restore_records(std::map<std::string, std::vector<Record>> recs) { for (auto& [k, v] : recs) stash_[k] = std::move(v); }

private:
    static constexpr int kL = 121, kD = 8, kF = 10;
    static constexpr std::size_t at(int i, int j, int k) { return (static_cast<std::size_t>(i) * kD + j) * kF + k; }
    void normalise()
    {
        double mx = -1e300;
        for (const double v : logw_) mx = std::max(mx, v);
        wl_.fill(0.0); wd_.fill(0.0); wf_.fill(0.0);
        double z = 0.0;
        for (int i = 0; i < kL; ++i) for (int j = 0; j < kD; ++j) for (int k = 0; k < kF; ++k)
        {
            auto& lw = logw_[at(i, j, k)];
            lw -= mx;
            const double w = std::exp(lw);
            wl_[i] += w; wd_[j] += w; wf_[k] += w; z += w;
        }
        for (auto& w : wl_) w /= z;
        for (auto& w : wd_) w /= z;
        for (auto& w : wf_) w /= z;
        d_mean_ = detect(); f_mean_ = false_alarm();
    }
    Params p_;
    std::array<double, kL> lam_{}, prior_l_{}, wl_{};
    std::array<double, kD> d_{}, prior_d_{}, wd_{};
    std::array<double, kF> f_{}, prior_f_{}, wf_{};
    std::array<double, static_cast<std::size_t>(kL) * kD * kF> logw_{};
    double d_mean_ = 0.85, f_mean_ = 0.05, unknown_nats_ = std::log(2.0), ell_for_first_ = 0.3;
    Eigen::Vector2f origin_{0.f, 0.f};
    int cols_ = 0, rows_ = 0;
    float cell_ = 0.5f;
    std::vector<Cell> cells_;
    std::vector<std::uint8_t> mark_;
    std::string room_;
    std::map<std::string, std::vector<Record>> stash_;
    std::uint64_t tests_ = 0;
};
}   // namespace rc::change
