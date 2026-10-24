/*
 * room_posterior.h — joint posterior over (room k, pose in k) + "new", and the staged commit rule.
 * Pure math (Eigen + STL), no graph. Consumed by ltsm's RoomRecognizer.
 *
 * Model. For each known room k two experts give a density over the SAME pose x in k's frame:
 *   metric    p_m(x | chain)  — the robot pose propagated through the seam chain, with P(inside polygon_k)
 *   appearance p_a(x | views) — the map match's pose mixture, with a cumulative log-likelihood ratio
 *                               loglr_k = log p(views | k) / p(views | none)
 * The joint is  pi_k * p_m(x) * p_a(x) / p(x),  p(x) uniform over the polygon x SO(2) (1 / (area_k * 2 pi)),
 * so the shared prior is not counted twice. Integrating x gives the evidence
 *   E_k = [ sum_ij w_i w_j N(mu_i - mu_j; 0, S_i + S_j) ] * area_k * 2 pi * exp(log_inside) * exp(loglr)
 * (the bracket is the Gaussian-mixture product integral; pose disagreement lowers it, no extra term).
 * The mixture shape carries LOCATION, the llr carries room IDENTITY. An unavailable expert's factor is
 * OMITTED (not zeroed): metric alone gives E_k = exp(log_inside); appearance alone gives exp(loglr).
 * (The brief says "times area_k"; the pose is SE(2), so the uniform pose prior is 1/(area_k*2pi) and the 2pi
 * is needed for E_k -> exp(log_inside) when the appearance pose is uninformative.)
 * "new": pi_new * exp(metric_log_new); appearance is its reference (llr 0 by construction).
 *
 * Decision boundaries (the only ones, both flagged at their site): the commit is the Bayes MAP under symmetric
 * 0-1 loss, and the prior is uniform over {known rooms} U {new}.
 */
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#include "../place_memory/place_map.h"   // rc::place::wrap_angle

namespace rc::recog
{

struct PoseGauss { Eigen::Vector3f mu; Eigen::Matrix3f S; float w; };   // SE(2), angle wrapped
using PoseMix = std::vector<PoseGauss>;
struct MetricEvidence { PoseMix pose_in_k; float log_inside = 0.f; bool available = false; };
struct AppearanceEvidence { PoseMix pose_in_k; float loglr = 0.f; bool available = false; };
struct Hypothesis { int room = -1;  /* -1 = new */ float log_post = 0.f; PoseMix pose_in_k; };

namespace detail
{
inline double logsumexp(std::span<const double> v)
{
    if (v.empty()) return -std::numeric_limits<double>::infinity();
    const double m = *std::ranges::max_element(v);
    if (not std::isfinite(m)) return m;
    double s = 0.0;
    for (const double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

// log N(d; 0, S) for a 3-vector d
inline double log_gauss0(const Eigen::Vector3d& d, const Eigen::Matrix3d& S)
{
    const Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
    const double logdet = ldlt.vectorD().array().log().sum();
    const double q = d.dot(ldlt.solve(d));
    return -0.5 * (3.0 * std::log(2.0 * M_PI) + logdet + q);
}

inline Eigen::Vector3d pose_diff(const Eigen::Vector3f& a, const Eigen::Vector3f& b)   // a - b, angle wrapped
{
    Eigen::Vector3d d = (a - b).cast<double>();
    d.z() = rc::place::wrap_angle(float(d.z()));
    return d;
}

// Normalised weights of a mixture (a mixture is a density: its weights sum to 1).
inline std::vector<double> norm_w(const PoseMix& m)
{
    std::vector<double> w; w.reserve(m.size());
    for (const auto& g : m) w.push_back(double(g.w));
    const double s = std::accumulate(w.begin(), w.end(), 0.0);
    if (s > 0.0) for (auto& x : w) x /= s;
    else         for (auto& x : w) x = 1.0 / double(w.size());
    return w;
}

// log of the product integral  sum_ij w_i w_j N(mu_i - mu_j; 0, S_i + S_j)  and the fused (product) mixture.
inline double log_pair_integral(const PoseMix& a, const PoseMix& b, PoseMix* fused)
{
    const auto wa = norm_w(a), wb = norm_w(b);
    std::vector<double> terms; terms.reserve(a.size() * b.size());
    PoseMix out;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
        {
            const Eigen::Matrix3d Si = a[i].S.cast<double>(), Sj = b[j].S.cast<double>();
            const Eigen::Vector3d d = pose_diff(a[i].mu, b[j].mu);
            const double lt = std::log(wa[i]) + std::log(wb[j]) + log_gauss0(d, Si + Sj);
            terms.push_back(lt);
            if (fused)
            {   // product of N(x; mu_i, S_i) N(x; mu_j, S_j): S = (Si^-1+Sj^-1)^-1, mu = mu_i - S Si^-1 (mu_i - mu_j)
                const Eigen::Matrix3d S = (Si.inverse() + Sj.inverse()).inverse();
                Eigen::Vector3d mu = a[i].mu.cast<double>() - S * Si.inverse() * d;
                mu.z() = rc::place::wrap_angle(float(mu.z()));
                out.push_back({mu.cast<float>(), S.cast<float>(), float(lt)});   // log-weight for now
            }
        }
    const double lse = detail::logsumexp(terms);
    if (fused)
    {
        for (auto& g : out) g.w = std::isfinite(lse) ? float(std::exp(double(g.w) - lse)) : 0.f;
        *fused = std::move(out);
    }
    return lse;
}
}   // namespace detail

// metric/app are indexed by known room; polygon_area_m2 likewise. Result: known rooms in input order, then new
// last; log_post normalised (logsumexp == 0).
inline std::vector<Hypothesis> posterior(std::span<const MetricEvidence> metric,
                                         std::span<const AppearanceEvidence> app,
                                         std::span<const float> polygon_area_m2,
                                         float metric_log_new)
{
    const size_t n = polygon_area_m2.size();
    // FLAGGED: uniform prior over {known rooms} U {new}. No evidence of how often a never-crossed door leads
    // somewhere new vs known; the error costs are symmetric (user), so indifference. prior_sensitivity pins that
    // halving pi_new moves the posterior < 0.05 once |loglr| >= 5.
    const double log_prior = -std::log(double(n + 1));

    std::vector<double> lp(n + 1);
    std::vector<Hypothesis> out(n + 1);
    for (size_t k = 0; k < n; ++k)
    {
        const bool has_m = k < metric.size() and metric[k].available;
        const bool has_a = k < app.size() and app[k].available;
        double ll = log_prior;
        out[k].room = int(k);
        if (has_m) ll += metric[k].log_inside;
        if (has_a) ll += app[k].loglr;
        if (has_m and has_a and not metric[k].pose_in_k.empty() and not app[k].pose_in_k.empty())
        {
            ll += detail::log_pair_integral(metric[k].pose_in_k, app[k].pose_in_k, &out[k].pose_in_k)
                  + std::log(double(polygon_area_m2[k])) + std::log(2.0 * M_PI);
        }
        else if (has_m) out[k].pose_in_k = metric[k].pose_in_k;
        else if (has_a) out[k].pose_in_k = app[k].pose_in_k;
        lp[k] = ll;
    }
    out[n].room = -1;
    lp[n] = log_prior + double(metric_log_new);

    const double z = detail::logsumexp(lp);
    for (size_t i = 0; i <= n; ++i) out[i].log_post = float(lp[i] - z);
    return out;
}

// Sum of per-frame log-likelihood ratios over a window of dt_s seconds, rescaled to the number of INDEPENDENT
// looks (n_eff_per_s * dt_s) instead of the number of frames, so correlated frames cannot manufacture confidence.
inline float discounted_loglr(std::span<const float> frame_llr, float dt_s, float n_eff_per_s)
{
    if (frame_llr.empty()) return 0.f;
    const double s = std::accumulate(frame_llr.begin(), frame_llr.end(), 0.0);
    return float(s * double(n_eff_per_s) * double(dt_s) / double(frame_llr.size()));
}

struct Commit { enum class State { Entering, Committed } state; int room; };

namespace detail
{
// Index of the strict argmax of log_post, or -2 when empty / tied (no unique MAP).
inline int unique_argmax(std::span<const Hypothesis> h)
{
    if (h.empty()) return -2;
    const auto it = std::ranges::max_element(h, {}, &Hypothesis::log_post);
    const auto ties = std::ranges::count_if(h, [&](const Hypothesis& x) { return x.log_post == it->log_post; });
    return ties == 1 ? int(it - h.begin()) : -2;
}
}   // namespace detail

// Staged rule. Entering -> Committed when the MAP over evidence since entry is a known room; Committed ->
// Entering when the MAP over evidence gathered AFTER the commit (prior reset) is `new` or a different room.
inline Commit commit_step(Commit prev, std::span<const Hypothesis> post_since_entry,
                          std::span<const Hypothesis> post_since_commit)
{
    // FLAGGED: Bayes MAP under symmetric 0-1 loss (user: both errors are disastrous) — argmax, no mass
    // threshold. An exact tie has no unique MAP and decides nothing (symmetric rooms stay Entering).
    if (prev.state == Commit::State::Entering)
    {
        const int i = detail::unique_argmax(post_since_entry);
        if (i >= 0 and post_since_entry[i].room >= 0) return {Commit::State::Committed, post_since_entry[i].room};
        return {Commit::State::Entering, -1};
    }
    const int i = detail::unique_argmax(post_since_commit);
    if (i >= 0 and post_since_commit[i].room != prev.room) return {Commit::State::Entering, -1};
    return prev;
}

}   // namespace rc::recog
