#include "helios_mount_stage.h"

#include <algorithm>
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

HeliosMountStage::M5 HeliosMountStage::prior() const
{
    V5 d;
    d << 1.0 / std::pow(hcfg_.prior_sigma_xy, 2), 1.0 / std::pow(hcfg_.prior_sigma_xy, 2),
         1.0 / std::pow(hcfg_.prior_sigma_yaw, 2), 1.0 / std::pow(cfg_.gyro_bias_sigma, 2), 1.0 / std::pow(cfg_.gyro_scale_sigma, 2);
    return d.asDiagonal();
}

void HeliosMountStage::store_keyframe(std::int64_t stamp_ms, const std::vector<Eigen::Vector3f>& pts)
{
    std::vector<Eigen::Vector2f> band;
    band.reserve(pts.size());
    for (const auto& q : pts)   // Shadow (X lateral, Y forward) -> harvester (x forward, y left)
        if (q.z() >= cfg_.band_min_m and q.z() <= cfg_.band_max_m and std::isfinite(q.x()) and std::isfinite(q.y()))
            band.emplace_back(q.y(), -q.x());
    sweeps_[stamp_ms] = std::move(band);
    const std::int64_t keep = std::int64_t((hcfg_.ring_span_s + 1.0) * 1000.0);
    sweeps_.erase(sweeps_.begin(), sweeps_.lower_bound(stamp_ms - keep));
}

bool HeliosMountStage::add_pair(const sc::calib::Pair& p)
{
    const auto ia = sweeps_.find(p.stamp_a_ms), ib = sweeps_.find(p.stamp_b_ms);
    if (ia == sweeps_.end() or ib == sweeps_.end() or ia->second.empty() or ib->second.empty()) { ++missing_; return false; }
    const auto r = sc::calib::register_scans_2d(ia->second, ib->second, odo_pose(p), cfg_.reg);
    if (not r.ok or not r.xyt.allFinite() or not r.cov.allFinite()) { ++rejected_; return false; }

    // Three rows: the chord innovation (fwd, lat) and the ROTATION innovation. The sensor is rigid on the body, so the
    // registered rotation is the true body rotation whatever the mount; its difference from the fused odometry
    // rotation is -(heading error) = -(b dt + k dth) and nothing else. Without it a gyro bias is indistinguishable
    // from a mount yaw on straight pairs (the two columns d_fwd and d_fwd dt/2 are collinear when dt is constant).
    const Eigen::Vector2d yc = pair_innovation(p, r);
    const Eigen::Vector3d y(yc.x(), yc.y(), r.xyt.z() - p.dth);
    const double s = 2.0 * std::sin(0.5 * p.dth), dt = double(p.stamp_b_ms - p.stamp_a_ms) * 1e-3;
    Eigen::Matrix<double, 3, 5> J;
    J << -s, 0.0, 0.0, 0.5 * p.d_lat * dt, 0.5 * p.d_lat * p.dth,
         0.0, -s, p.d_fwd, -0.5 * p.d_fwd * dt, -0.5 * p.d_fwd * p.dth,
         0.0, 0.0, 0.0, -dt, -p.dth;
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
    if (not y.allFinite() or not R.allFinite()) { ++rejected_; return false; }
    const Eigen::Matrix3d W = R.inverse();
    H_ += J.transpose() * W * J;
    b_ += J.transpose() * W * y;
    ++n_;

    open_csv(pairs_csv_, "pairs", "stamp_a_ms,stamp_b_ms,dth,d_fwd,d_lat,reg_x,reg_y,reg_t,y_fwd,y_lat,y_rot,sd_y_fwd,sd_y_lat,sd_y_rot,n_used,inlier_share,rms");
    if (pairs_csv_.is_open())
    {
        pairs_csv_ << p.stamp_a_ms << ',' << p.stamp_b_ms << ',' << p.dth << ',' << p.d_fwd << ',' << p.d_lat << ',' << r.xyt.x() << ','
                   << r.xyt.y() << ',' << r.xyt.z() << ',' << y.x() << ',' << y.y() << ',' << y.z() << ',' << std::sqrt(R(0, 0)) << ',' << std::sqrt(R(1, 1)) << ',' << std::sqrt(R(2, 2))
                   << ',' << r.n_used << ',' << r.inlier_share << ',' << r.rms << '\n';
        pairs_csv_.flush();
    }
    return true;
}

Eigen::Vector3d HeliosMountStage::mean() const
{
    const V5 x = (H_ + prior()).ldlt().solve(b_);
    return x.head<3>();   // joint solution; the first three are the marginal mean
}
Eigen::Vector3d HeliosMountStage::sigma() const
{
    const M5 C = (H_ + prior()).inverse();   // top-left 3x3 of the inverse == inverse of the Schur complement over (b, k)
    return C.diagonal().head<3>().cwiseMax(0.0).cwiseSqrt();
}
Eigen::Vector2d HeliosMountStage::nuisance() const { return ((H_ + prior()).ldlt().solve(b_)).tail<2>(); }

std::string HeliosMountStage::summary() const
{
    const Eigen::Vector3d m = mean(), s = sigma();
    constexpr double deg = 180.0 / M_PI;
    char buf[200];
    std::snprintf(buf, sizeof buf, "[helios] x %.1f ± %.1f  y %.1f ± %.1f mm  yaw %.3f ± %.3f deg (%ld pairs, %ld missing, %ld rejected)",
                  m.x() * 1e3, s.x() * 1e3, m.y() * 1e3, s.y() * 1e3, m.z() * deg, s.z() * deg, n_, missing_, rejected_);
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
    open_csv(state_csv_, "mount", "stamp_ms,dx,dy,dyaw,sd_dx,sd_dy,sd_dyaw,gyro_bias,gyro_scale,n_pairs");
    if (not state_csv_.is_open()) return;
    const Eigen::Vector3d m = mean(), s = sigma();
    const Eigen::Vector2d nu = nuisance();
    state_csv_ << stamp_ms << ',' << m.x() << ',' << m.y() << ',' << m.z() << ',' << s.x() << ',' << s.y() << ',' << s.z() << ','
               << nu.x() << ',' << nu.y() << ',' << n_ << '\n';
    state_csv_.flush();
}
}  // namespace sc
