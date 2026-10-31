#include "helios_mount_stage.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <filesystem>

namespace sc
{
namespace
{
Eigen::Matrix2d rot2(double a) { Eigen::Matrix2d R; R << std::cos(a), -std::sin(a), std::sin(a), std::cos(a); return R; }

// harvester pose of b in a, from the odometry: chord on the midpoint heading, rotated into frame a.
Eigen::Vector3d odo_pose(const sc::calib::Pair& p)
{
    const Eigen::Vector2d t = rot2(0.5 * p.dth) * Eigen::Vector2d(p.d_fwd, p.d_lat);
    return {t.x(), t.y(), p.dth};
}
}  // namespace

Eigen::Vector2d pair_innovation(const sc::calib::Pair& p, const sc::calib::ScanRegResult& r)
{
    const Eigen::Vector2d chord = rot2(-0.5 * p.dth) * r.xyt.head<2>();
    return chord - Eigen::Vector2d(p.d_fwd, p.d_lat);
}

HeliosMountStage::HeliosMountStage(HeliosMountConfig cfg, sc::calib::HarvesterConfig hcfg) : cfg_(std::move(cfg)), hcfg_(hcfg) {}

HeliosMountStage::M6 HeliosMountStage::prior() const
{
    V6 d;
    d << 1.0 / std::pow(hcfg_.prior_sigma_xy, 2), 1.0 / std::pow(hcfg_.prior_sigma_xy, 2),
         1.0 / std::pow(hcfg_.prior_sigma_yaw, 2), 1.0 / std::pow(cfg_.gyro_bias_sigma, 2), 1.0 / std::pow(cfg_.gyro_scale_sigma, 2),
         1.0 / std::pow(cfg_.wheel_scale_sigma, 2);
    return d.asDiagonal();
}

void HeliosMountStage::store_keyframe(std::int64_t stamp_ms, const std::vector<Eigen::Vector3f>& pts)
{
    std::vector<Eigen::Vector2f> band;
    band.reserve(pts.size());
    for (const auto& q : pts)   // Shadow (X lateral, Y forward) -> harvester (x forward, y left)
        if (q.z() >= cfg_.band_min_m and q.z() <= cfg_.band_max_m and std::isfinite(q.x()) and std::isfinite(q.y()))
            band.emplace_back(q.y(), -q.x());
    sweeps_[stamp_ms] = std::move(band);   // never pruned here: process_queue() drops what no pair can reference
}

std::optional<sc::calib::ScanRegResult> HeliosMountStage::register_pair(const sc::calib::Pair& p)
{
    last_reg_ms_ = 0.0;
    const auto ia = sweeps_.find(p.stamp_a_ms), ib = sweeps_.find(p.stamp_b_ms);
    if (ia == sweeps_.end() or ib == sweeps_.end() or ia->second.empty() or ib->second.empty()) { ++missing_; return std::nullopt; }
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = sc::calib::register_scans_2d(ia->second, ib->second, odo_pose(p), cfg_.reg);
    last_reg_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (not r.ok or not r.xyt.allFinite() or not r.cov.allFinite()) { ++rejected_; return std::nullopt; }
    return r;
}

bool HeliosMountStage::add_pair(const sc::calib::Pair& p)
{
    const auto r = register_pair(p);
    return r.has_value() and add_registration(p, *r);
}

void HeliosMountStage::enqueue(const std::vector<sc::calib::Pair>& pairs) { queue_.insert(queue_.end(), pairs.begin(), pairs.end()); }

std::size_t HeliosMountStage::process_queue(std::optional<std::int64_t> ring_front_ms)
{
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t done = 0;
    while (not queue_.empty())
    {
        const sc::calib::Pair p = queue_.front();   // copy: add_pair never touches the queue, but keep the pop first-class
        queue_.pop_front();
        add_pair(p);
        ++done;
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() >= cfg_.registration_budget_ms)
            break;   // a compute budget, not a data gate: the rest waits for the next tick
    }
    // A sweep is held while a queued pair, or a pair the harvester may still commit, can reference it. With nothing
    // pending at all keep only the newest sweep.
    std::int64_t bound = std::numeric_limits<std::int64_t>::max();
    for (const auto& q : queue_) bound = std::min(bound, q.stamp_a_ms);
    if (ring_front_ms) bound = std::min(bound, *ring_front_ms);
    else if (not sweeps_.empty()) bound = std::min(bound, sweeps_.rbegin()->first);
    sweeps_.erase(sweeps_.begin(), sweeps_.lower_bound(bound));
    return done;
}

bool HeliosMountStage::add_registration(const sc::calib::Pair& p, const sc::calib::ScanRegResult& r)
{
    // Three rows: the chord innovation (fwd, lat) and the ROTATION innovation. The sensor is rigid on the body, so the
    // registered rotation is the true body rotation whatever the mount; its difference from the fused odometry
    // rotation is -(heading error) = -(b dt + k dth) and nothing else. Without it a gyro bias is indistinguishable
    // from a mount yaw on straight pairs (the two columns d_fwd and d_fwd dt/2 are collinear when dt is constant).
    const Eigen::Vector2d yc = pair_innovation(p, r);
    const Eigen::Vector3d y(yc.x(), yc.y(), r.xyt.z() - p.dth);
    const double s = 2.0 * std::sin(0.5 * p.dth), dt = double(p.stamp_b_ms - p.stamp_a_ms) * 1e-3;
    Rows w;
    // Last column: wheel translation scale eps (odometry chord = (1+eps) x true): y = reg - odo = -eps (d_fwd, d_lat).
    w.J << -s, 0.0, 0.0, 0.5 * p.d_lat * dt, 0.5 * p.d_lat * p.dth, -p.d_fwd,
           0.0, -s, p.d_fwd, -0.5 * p.d_fwd * dt, -0.5 * p.d_fwd * p.dth, -p.d_lat,
           0.0, 0.0, 0.0, -dt, -p.dth, 0.0;
    const Eigen::Matrix2d Rm = rot2(-0.5 * p.dth);
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();   // registration cov (x, y, t) -> (fwd, lat, rot)
    T.topLeftCorner<2, 2>() = Rm;
    Eigen::Matrix3d R = T * r.cov * T.transpose();
    // p.var = reg_sigma_m^2 (the harvester's ASSUMED registration floor, used only to score pairs before any scan is
    // registered) + the odometry terms. Subtract the floor so only the odometry part is added: the measured registration
    // covariance (in R already) replaces it. The max() only guards a user-set p.var below the floor; it is not a model gate.
    // chord_sigma^2 goes on the LATERAL row only: a heading error rotates the chord perpendicular to itself, and the
    // chord lies along the midpoint heading (fwd), so its effect is first order on lat and second order on fwd.
    R(0, 0) += std::max(p.var(0) - hcfg_.reg_sigma_m * hcfg_.reg_sigma_m, 0.0);
    R(1, 1) += std::max(p.var(1) - hcfg_.reg_sigma_m * hcfg_.reg_sigma_m, 0.0) + p.chord_sigma * p.chord_sigma;
    R(2, 2) += p.dth_sigma * p.dth_sigma;
    // The same per-pair random heading error delta (sd dth_sigma) enters BOTH y_lat and y_rot. In the J rows above the
    // heading error dtheta has coefficient -d_fwd/2 on lat and -1 on rot, so Cov(y_lat, y_rot) = (-d_fwd/2)(-1) sigma^2
    // = +d_fwd dth_sigma^2 / 2. (The lat VARIANCE keeps the harvester's conservative chord_sigma^2 = d^2 sigma^2, which
    // is >= the d^2 sigma^2 / 4 of this derivation, so R stays positive definite: |cross|^2 = d^2 sigma^4 / 4 <= d^2 sigma^4.)
    if (cfg_.heading_cross_cov) { R(1, 2) += 0.5 * p.d_fwd * p.dth_sigma * p.dth_sigma; R(2, 1) = R(1, 2); }
    if (not y.allFinite() or not R.allFinite()) { ++rejected_; return false; }
    w.R = R; w.W = R.inverse(); w.y = y;
    rows_.push_back(w);
    dirty_ = true;
    solve();

    open_csv(pairs_csv_, "pairs", "stamp_a_ms,stamp_b_ms,dth,d_fwd,d_lat,reg_x,reg_y,reg_t,y_fwd,y_lat,y_rot,sd_y_fwd,sd_y_lat,sd_y_rot,n_used,inlier_share,rms,reg_ms,gamma,maha");
    if (pairs_csv_.is_open())
    {
        pairs_csv_ << p.stamp_a_ms << ',' << p.stamp_b_ms << ',' << p.dth << ',' << p.d_fwd << ',' << p.d_lat << ',' << r.xyt.x() << ','
                   << r.xyt.y() << ',' << r.xyt.z() << ',' << y.x() << ',' << y.y() << ',' << y.z() << ',' << std::sqrt(R(0, 0)) << ',' << std::sqrt(R(1, 1)) << ',' << std::sqrt(R(2, 2))
                   << ',' << r.n_used << ',' << r.inlier_share << ',' << r.rms << ',' << last_reg_ms_ << ',' << rows_.back().gamma << ',' << rows_.back().maha << '\n';
        pairs_csv_.flush();
    }
    last_reg_ms_ = 0.0;
    return true;
}

// EM over the pair responsibilities. Weighted normal equations A x = bb (A = prior + sum gamma_i J_i' W_i J_i), then
// each gamma_i from the predictive density of pair i given all the OTHERS (see the header), until gamma settles.
void HeliosMountStage::solve() const
{
    if (not dirty_) return;
    const M6 P = prior();
    const double pi_out = std::clamp(cfg_.outlier_prior, 0.0, 1.0 - 1e-12);   // a probability, not a gate
    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    Q.diagonal() << cfg_.outlier_sigma_m * cfg_.outlier_sigma_m, cfg_.outlier_sigma_m * cfg_.outlier_sigma_m,
                    cfg_.outlier_sigma_rad * cfg_.outlier_sigma_rad;
    const auto info = [](const Rows& w) { return M6(w.J.transpose() * w.W * w.J); };
    const auto rhs = [](const Rows& w) { return V6(w.J.transpose() * w.W * w.y); };
    const auto logdet_quad = [](const Eigen::Matrix3d& S, const Eigen::Vector3d& r, double& quad) {
        const Eigen::LDLT<Eigen::Matrix3d> ldlt(S);
        quad = r.dot(ldlt.solve(r));
        return ldlt.vectorD().array().log().sum();
    };
    constexpr int kMaxIter = 12;   // EM sweeps: a compute budget (gamma is warm-started from the previous solve)
    M6 A; V6 bb;
    for (int it = 0; it < kMaxIter; ++it)
    {
        A = P; bb.setZero();
        for (const auto& w : rows_) { A += w.gamma * info(w); bb += w.gamma * rhs(w); }
        double moved = 0.0;
        for (auto& w : rows_)
        {
            const M6 Ai = A - w.gamma * info(w);
            const V6 bi = bb - w.gamma * rhs(w);
            const M6 Ci = Ai.inverse();
            const Eigen::Vector3d r = w.y - w.J * (Ci * bi);
            const Eigen::Matrix3d S = w.R + w.J * Ci * w.J.transpose();
            double q_in = 0.0, q_out = 0.0;
            const double ld_in = logdet_quad(S, r, q_in), ld_out = logdet_quad(S + Q, r, q_out);
            w.maha = std::sqrt(std::max(q_in, 0.0));
            const double g = pi_out > 0.0 ? 1.0 / (1.0 + std::exp(std::clamp(
                (std::log(pi_out) - 0.5 * (ld_out + q_out)) - (std::log(1.0 - pi_out) - 0.5 * (ld_in + q_in)), -700.0, 700.0))) : 1.0;
            moved = std::max(moved, std::abs(g - w.gamma));
            w.gamma = g;
        }
        if (moved < 1e-4) break;
    }
    A = P; bb.setZero();
    for (const auto& w : rows_) { A += w.gamma * info(w); bb += w.gamma * rhs(w); }
    C_ = A.inverse();
    x_ = A.ldlt().solve(bb);
    dirty_ = false;
}

double HeliosMountStage::effective_pairs() const
{
    solve();
    double n = 0.0;
    for (const auto& w : rows_) n += w.gamma;
    return n;
}

std::vector<double> HeliosMountStage::responsibilities() const
{
    solve();
    std::vector<double> g;
    for (const auto& w : rows_) g.push_back(w.gamma);
    return g;
}

Eigen::Vector3d HeliosMountStage::mean() const
{
    solve();
    return x_.head<3>();   // joint solution; the first three are the marginal mean
}
Eigen::Vector3d HeliosMountStage::sigma() const
{
    solve();   // top-left 3x3 of the inverse == inverse of the Schur complement over the nuisances
    return C_.diagonal().head<3>().cwiseMax(0.0).cwiseSqrt();
}
Eigen::Vector3d HeliosMountStage::nuisance() const { solve(); return x_.tail<3>(); }

std::string HeliosMountStage::summary() const
{
    const Eigen::Vector3d m = mean(), s = sigma();
    constexpr double deg = 180.0 / M_PI;
    char buf[260];
    std::snprintf(buf, sizeof buf, "[helios] x %.1f ± %.1f  y %.1f ± %.1f mm  yaw %.3f ± %.3f deg (%ld pairs, n_eff %.1f, %zu queued, %ld missing, %ld rejected)",
                  m.x() * 1e3, s.x() * 1e3, m.y() * 1e3, s.y() * 1e3, m.z() * deg, s.z() * deg, pairs_used(), effective_pairs(), queue_.size(), missing_, rejected_);
    return buf;
}

std::string HeliosMountStage::clock_summary() const
{
    if (clock_ms_.empty()) return "[helios] clock: no samples";
    std::vector<double> v = clock_ms_;
    std::ranges::sort(v);
    const auto q = [&](double f) { return v[std::size_t(f * double(v.size() - 1))]; };
    char buf[160];
    std::snprintf(buf, sizeof buf, "[helios] clock: keyframe stamp - nearest odometry stamp, median %.1f ms, spread (p10..p90) %.1f..%.1f ms (%zu keyframes)",
                  q(0.5), q(0.1), q(0.9), v.size());
    return buf;
}

void HeliosMountStage::open_csv(std::ofstream& f, const char* stem, const char* header)
{
    if (f.is_open() or cfg_.log_dir.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(cfg_.log_dir, ec);
    char stamp[32];
    const std::time_t now_t = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H-%M-%S", std::localtime(&now_t));
    f.imbue(std::locale::classic());   // CLAUDE.md: data files never acquire a comma separator (before open)
    f.open(cfg_.log_dir + "/" + stem + "_" + stamp + ".csv");
    f << header << '\n';
}

void HeliosMountStage::log_state(std::int64_t stamp_ms)
{
    open_csv(state_csv_, "mount", "stamp_ms,dx,dy,dyaw,sd_dx,sd_dy,sd_dyaw,gyro_bias,gyro_scale,wheel_scale,n_pairs");
    if (not state_csv_.is_open()) return;
    const Eigen::Vector3d m = mean(), s = sigma();
    const Eigen::Vector3d nu = nuisance();
    state_csv_ << stamp_ms << ',' << m.x() << ',' << m.y() << ',' << m.z() << ',' << s.x() << ',' << s.y() << ',' << s.z() << ','
               << nu.x() << ',' << nu.y() << ',' << nu.z() << ',' << pairs_used() << '\n';
    state_csv_.flush();
}
}  // namespace sc
