/*
 * helios_mount_stage -- the Helios PLANAR mount (dx, dy, yaw) from scan registration between the two keyframes of each
 * harvester pair (plan Task 4.4). Pure: no Qt, no DSR, no DDS; the agent hands it plain data.
 *
 * FLOW. store_keyframe(stamp, points_robot) keeps the 2-D wall band of every keyframe sweep; add_pair(Pair) looks up
 * the pair's two sweeps, registers b onto a (sc::calib::register_scans_2d, initialised from the odometry), and turns
 * the result into a 2-row innovation  y = (registered chord on the pair's fused midpoint heading) - (d_fwd, d_lat)
 * in the harvester's axes (+x forward, +y left; the robot/Shadow frame of the points is +Y forward, +X lateral,
 * converted on the way in).
 *
 * MODEL (derived in helios_mount_stage.cpp and pinned by tests/helios_mount_test.cpp). The state is the CORRECTION
 * that restores truth: for a sensor truly yawed by +psi_t at lever l (Shadow axes) from its nominal mount the
 * estimate is (dx, dy, psi) = (-l_X, -l_Y, -psi_t). With s = 2 sin(dth/2):
 *       y_fwd = -s dx             + 1/2 d_lat  dtheta
 *       y_lat = -s dy + d_fwd psi - 1/2 d_fwd  dtheta
 * where dtheta = b dt + k dth is the HEADING ERROR of the odometry chord (gyro bias b [rad/s], gyro scale k): a
 * COMMON MODE shared by every pair, kept as two extra unknowns (dx, dy, psi, b, k) with priors gyro_bias_sigma and
 * gyro_scale_sigma, and marginalised by the Schur complement (the posterior of the first three is the marginal of
 * the joint). NOTE Pair::J's x row has the opposite sign (it is written for axes x = left); the information is
 * identical, only the sign of dx in the report differs, and this stage reports in Shadow axes.
 *
 * WEIGHTING. Row covariance = diag(p.var - reg_sigma_m^2)   (the odometry part; the registration floor is REPLACED by
 *   the measured covariance) + R(-thm) cov_xy R(thm) of the registration + diag(0, chord_sigma^2).
 * A degenerate axis (corridor) arrives with a huge variance and so contributes nothing: no gate, no "straight" /
 * "turning" classifier. A registration that is not ok (not converged / not finite) adds nothing.
 *
 * Keyframe sweeps are dropped once no pair can use them: older than the harvester's ring_span_s + 1 s.
 */
#pragma once
#include "../../common/sensor_calib/scan_registration.h"
#include "../../common/sensor_calib/trajectory_harvester.h"

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace sc
{
struct HeliosMountConfig {
    double band_min_m = 1.5, band_max_m = 2.0;   // room_concept's Media.lidar_high_{min,max}_height: the 2-D wall band
    double gyro_bias_sigma = 0.002;              // rad/s prior on the common gyro bias (nuisance)
    double gyro_scale_sigma = 0.01;              // fractional prior on the common gyro scale (nuisance)
    std::string log_dir = "tmp/helios_mount";    // pairs_<ts>.csv + mount_<ts>.csv; "" = none
    sc::calib::ScanRegConfig reg{};
};

// The pair innovation in Pair::J's rows from a registration (cur = b, ref = a, both robot frame):
// y = (registered chord on the pair's fused midpoint heading) - (d_fwd, d_lat), harvester axes (+x fwd, +y left).
[[nodiscard]] Eigen::Vector2d pair_innovation(const sc::calib::Pair& p, const sc::calib::ScanRegResult& r);

class HeliosMountStage {
public:
    explicit HeliosMountStage(HeliosMountConfig cfg, sc::calib::HarvesterConfig hcfg);
    void store_keyframe(std::int64_t stamp_ms, const std::vector<Eigen::Vector3f>& points_robot); // band -> 2-D
    bool add_pair(const sc::calib::Pair& p);     // false = a sweep missing or registration not ok (logged, nothing added)
    [[nodiscard]] Eigen::Vector3d mean() const;  // (dx, dy, psi) helios correction (Shadow axes), nuisances marginalised
    [[nodiscard]] Eigen::Vector3d sigma() const;
    [[nodiscard]] Eigen::Vector2d nuisance() const;   // posterior mean of (b [rad/s], k)
    [[nodiscard]] long pairs_used() const { return n_; }
    [[nodiscard]] std::string summary() const;   // "[helios] x ±  y ± mm  yaw ± deg (n pairs)"
    void log_state(std::int64_t stamp_ms);

    // Clock check (plan step 5): offset of a keyframe stamp from the nearest odometry stamp. Reported, never corrected.
    void note_clock_offset_ms(double ms) { clock_ms_.push_back(ms); }
    [[nodiscard]] std::size_t clock_samples() const { return clock_ms_.size(); }
    [[nodiscard]] std::string clock_summary() const;   // "[helios] clock ..."

private:
    using V5 = Eigen::Matrix<double, 5, 1>;
    using M5 = Eigen::Matrix<double, 5, 5>;
    [[nodiscard]] M5 prior() const;
    void open_csv(std::ofstream& f, const char* stem, const char* header);

    HeliosMountConfig cfg_;
    sc::calib::HarvesterConfig hcfg_;
    std::map<std::int64_t, std::vector<Eigen::Vector2f>> sweeps_;   // stamp -> band points, harvester axes
    M5 H_ = M5::Zero();
    V5 b_ = V5::Zero();
    long n_ = 0, missing_ = 0, rejected_ = 0;
    std::vector<double> clock_ms_;
    std::ofstream pairs_csv_, state_csv_;
};
}  // namespace sc
