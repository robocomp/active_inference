// reloc_rivals.h -- the relocaliser's rival-pose belief (pure: Eigen + std, no DSR, no Qt).
// Spec: docs/superpowers/specs/2026-11-01-relocaliser-redesign.md sec 4.1-4.4.
// The incumbent pose H0 is never stored here; a rival is a competing pose with a log Bayes factor lambda
// against H0. Rivals are only ever ADDED by a proposer and only ever PROMOTED by accumulated evidence.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

namespace rc::rivals
{
struct Rival
{
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();   // x, y, yaw
    Eigen::Matrix3f cov = Eigen::Matrix3f::Identity();
    float lambda = 0.f;          // ln posterior odds against H0
    float lambda_birth = 0.f;    // the birth price: lambda never falls below it (sec 4.3)
    std::uint64_t id = 0;
};

struct Params
{
    float pi_kid = 1e-3f;                 // prior that the robot was kidnapped (sec 4.2)
    float log_band = std::log(19.f);      // SPRT band alpha=beta=0.05 -- the existing flagged constant (sec 4.4)
    int k_max = 4;                        // bookkeeping bound on the rival list, not a model quantity
    float ln_eps = std::log(0.01f);       // ln(RelocOutlierFrac): per-scan LLR lower bound (heavy-tailed mixture)
};

inline float wrap_pi(float a)
{
    return a - 2.f * std::numbers::pi_v<float> * std::floor((a + std::numbers::pi_v<float>) / (2.f * std::numbers::pi_v<float>));
}

// ln det(2 pi S) / 2
inline float half_log_det_2pi(const Eigen::Matrix3f& S)
{
    return 0.5f * std::log(std::max((2.f * std::numbers::pi_v<float> * S).determinant(), 1e-30f));
}

// sec 4.2: lambda_r(0) = ln pi/(1-pi) + [ll_r - ll_0] + ln(sqrt|2 pi S_r| / V) - ln N(x_0; pred, P_pred) - 1/2 ln|2 pi S_0|
inline float birth_lambda(const Params& p, float ll_r, const Eigen::Matrix3f& cov_r, float ll_0, const Eigen::Vector3f& pose_0,
                          const Eigen::Matrix3f& cov_0, const Eigen::Vector3f& pred, const Eigen::Matrix3f& P_pred, float volume)
{
    Eigen::Vector3f d = pose_0 - pred;
    d.z() = wrap_pi(d.z());
    const float ln_N0 = -half_log_det_2pi(P_pred) - 0.5f * d.dot(P_pred.inverse() * d);
    return std::log(p.pi_kid / (1.f - p.pi_kid)) + (ll_r - ll_0)
         + half_log_det_2pi(cov_r) - std::log(volume)
         - ln_N0 - half_log_det_2pi(cov_0);
}

class Belief
{
public:
    explicit Belief(Params p = {}) : p_(p) {}
    const Params& params() const { return p_; }

    void add(Rival r)
    {
        for (auto& e : rivals_)
        {
            Eigen::Vector3f d = e.pose - r.pose;
            d.z() = wrap_pi(d.z());
            // THRESHOLD IN DISGUISE (spec 10.5): "same Laplace mode" = Mahalanobis <= 3 under the summed covariances.
            // No model-level term replaces it: identity of two modes is a clustering decision.
            if (d.dot((e.cov + r.cov).inverse() * d) <= 9.f)
            {
                if (r.lambda > e.lambda) e = r;
                return;
            }
        }
        rivals_.push_back(std::move(r));
        if (static_cast<int>(rivals_.size()) > p_.k_max)
            rivals_.erase(std::ranges::min_element(rivals_, {}, &Rival::lambda));
    }

    // Rigid carry by the robot's motion expressed in the body frame (dx, dy, dyaw): the rival's own yaw rotates the
    // body-frame delta into the map frame. cov is NOT propagated (first order; no constant process noise, by rule).
    void carry(const Eigen::Vector3f& odo)
    {
        for (auto& r : rivals_)
        {
            const float c = std::cos(r.pose.z()), s = std::sin(r.pose.z());
            r.pose.x() += c * odo.x() - s * odo.y();
            r.pose.y() += s * odo.x() + c * odo.y();
            r.pose.z() = wrap_pi(r.pose.z() + odo.z());
        }
    }

    // sec 4.3. dll[i] = ll_i(z) - ll_0(z) for rival i on the same scan.
    void update(const std::vector<float>& dll, float novelty)
    {
        const float w = std::clamp(novelty, 0.f, 1.f);   // novelty is a weight in [0,1] by definition
        for (std::size_t i = 0; i < rivals_.size() and i < dll.size(); ++i)
            rivals_[i].lambda = std::max(rivals_[i].lambda + w * std::max(dll[i], p_.ln_eps), rivals_[i].lambda_birth);
    }

    std::optional<std::size_t> winner() const
    {
        std::optional<std::size_t> best;
        for (std::size_t i = 0; i < rivals_.size(); ++i)
            if (rivals_[i].lambda >= p_.log_band and (not best or rivals_[i].lambda > rivals_[*best].lambda))
                best = i;
        return best;
    }

    // Rival i becomes the incumbent (returned); the old incumbent takes its place as a rival with lambda = -lambda_i.
    // The demoted incumbent STARTS at -lambda_i but its floor is the kidnap birth price floor_birth (the caller passes
    // birth_lambda for that pose), like any rival, so it decays away instead of sticking near -lambda_i.
    Rival swap(std::size_t i, const Eigen::Vector3f& pose_0, const Eigen::Matrix3f& cov_0, float floor_birth)
    {
        Rival won = rivals_.at(i);
        Rival old;
        old.pose = pose_0; old.cov = cov_0;
        old.lambda = -won.lambda; old.lambda_birth = floor_birth;
        old.id = won.id + 1000000;   // distinct from any proposer id in practice; callers may overwrite
        rivals_[i] = old;
        return won;
    }

    float p_rival() const
    {
        if (rivals_.empty()) return 0.f;
        float m = 0.f;
        for (const auto& r : rivals_) m = std::max(m, r.lambda);
        float s = 0.f;                                   // log-sum-exp, H0 has lambda 0
        for (const auto& r : rivals_) s += std::exp(r.lambda - m);
        const float num = std::exp(m) * s, den = 1.f + num;
        return std::isfinite(num) ? num / den : 1.f;
    }

    // sum_k w_k d_k d_k^T about the published pose; H0 has d = 0, weights are posterior masses.
    Eigen::Matrix3f spread(const Eigen::Vector3f& pose_0) const
    {
        float z = 1.f;
        for (const auto& r : rivals_) z += std::exp(std::min(r.lambda, 80.f));
        Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
        for (const auto& r : rivals_)
        {
            Eigen::Vector3f d = r.pose - pose_0;
            d.z() = wrap_pi(d.z());
            S += (std::exp(std::min(r.lambda, 80.f)) / z) * d * d.transpose();
        }
        return S;
    }

    void clear() { rivals_.clear(); }
    const std::vector<Rival>& rivals() const { return rivals_; }

private:
    Params p_;
    std::vector<Rival> rivals_;
};
}  // namespace rc::rivals
