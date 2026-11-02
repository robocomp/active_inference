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
 * COMMON MODE shared by every pair. A third common mode is the wheel TRANSLATION scale: an odometry chord (1+eps)
 * times the true one gives y = -eps (d_fwd, d_lat), which on one-direction arcs masquerades as a lever (dx ~ eps rho).
 * All three are kept as extra unknowns (dx, dy, psi, b, k, eps) with priors gyro_bias_sigma, gyro_scale_sigma and
 * wheel_scale_sigma, and marginalised by the Schur complement (the posterior of the first three is the marginal of
 * the joint). NOTE Pair::J's x row has the opposite sign (it is written for axes x = left); the information is
 * identical, only the sign of dx in the report differs, and this stage reports in Shadow axes.
 *
 * OUTLIER MODEL (pair level, continuous). A registration can converge into a wrong basin and still report a small
 * covariance. Every pair therefore carries an EM responsibility gamma = P(inlier | rows, rest of the data): inlier =
 * Gaussian on the 3-row innovation with S = R + J Sigma_-i J' (Sigma_-i the posterior over the 6 states WITHOUT this
 * pair, i.e. the predictive density: with the pair included a pair that dominates its own posterior would mask itself);
 * outlier = the same plus a broad slip diag(outlier_sigma_m^2, outlier_sigma_m^2, outlier_sigma_rad^2), prior share
 * outlier_prior. Each pair's information and right-hand side are weighted by gamma; gamma depends on the estimate, so
 * the (J, R, y) rows of every pair are kept and gamma is re-evaluated for all of them on each solve. No hard cut.
 * outlier_prior = 0 gives gamma = 1 exactly (the control of the tests).
 *
 * WEIGHTING. Row covariance = diag(p.var - reg_sigma_m^2)   (the odometry part; the registration floor is REPLACED by
 *   the measured covariance) + R(-thm) cov_xy R(thm) of the registration + diag(0, chord_sigma^2).
 * A degenerate axis (corridor) arrives with a huge variance and so contributes nothing: no gate, no "straight" /
 * "turning" classifier. A registration that is not ok (not converged / not finite) adds nothing.
 *
 * SCHEDULING. add_pair() registers synchronously (tests). The agent instead enqueue()s the harvester's pairs and calls
 * process_queue() once per tick: pairs are registered FIFO until registration_budget_ms of wall time is spent (a
 * COMPUTE budget so the 50 Hz tick that polls the odometry is never starved; at least one pair per tick, so the queue
 * always drains). Keyframe sweeps are dropped by process_queue() only when neither a queued pair nor the harvester's
 * pending pairs (oldest_pending_stamp_ms) can still reference them.
 */
#pragma once
#include "../../common/sensor_calib/scan_registration.h"
#include "../../common/sensor_calib/trajectory_harvester.h"

#include <Eigen/Dense>
#include <cstdint>
#include <deque>
#include <optional>
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
    double wheel_scale_sigma = 0.03;             // fractional prior on the common wheel translation scale (nuisance)
    double outlier_prior = 0.05;                 // model prior: share of pairs whose registration slipped into a wrong basin
    double outlier_sigma_m = 0.3;                // model prior: whole-pair registration slip ~ the scan's feature spacing [m]
    double outlier_sigma_rad = 0.1;              // ... and its rotation counterpart (0.3 m of slip over ~3 m of range) [rad]
    bool heading_cross_cov = true;               // model: the per-pair heading error enters lat AND rot (R(1,2) != 0); false = independent rows (the control of the nuisance tests only: with the cross term the per-pair rotation row already explains most of a gyro bias away)
    double registration_budget_ms = 8.0;         // COMPUTE budget: wall time of scan registration per process_queue() call
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
    // Synchronous: register + accumulate. false = a sweep missing or registration not ok (counted, nothing added).
    bool add_pair(const sc::calib::Pair& p);
    // The two halves, public so a test can substitute the registration. register_pair counts missing_/rejected_.
    [[nodiscard]] std::optional<sc::calib::ScanRegResult> register_pair(const sc::calib::Pair& p);
    bool add_registration(const sc::calib::Pair& p, const sc::calib::ScanRegResult& r);   // false = non-finite rows
    void enqueue(const std::vector<sc::calib::Pair>& pairs);
    // Register queued pairs FIFO until the wall-time budget is spent (>= 1 pair when any is queued), then drop the
    // sweeps nothing can reference any more. ring_front = the harvester's oldest_pending_stamp_ms(). Returns pairs consumed.
    std::size_t process_queue(std::optional<std::int64_t> ring_front_ms);
    [[nodiscard]] std::size_t queued() const { return queue_.size(); }
    [[nodiscard]] std::size_t sweeps_held() const { return sweeps_.size(); }
    [[nodiscard]] long missing() const { return missing_; }
    [[nodiscard]] long rejected() const { return rejected_; }
    [[nodiscard]] double effective_pairs() const;   // sum of the responsibilities
    [[nodiscard]] std::vector<double> responsibilities() const;   // gamma of every accepted pair, in order
    [[nodiscard]] Eigen::Vector3d mean() const;  // (dx, dy, psi) helios correction (Shadow axes), nuisances marginalised
    [[nodiscard]] Eigen::Vector3d sigma() const;
    [[nodiscard]] Eigen::Vector3d nuisance() const;   // posterior mean of (b [rad/s], k, eps)
    [[nodiscard]] long pairs_used() const { return long(rows_.size()); }
    [[nodiscard]] std::string summary() const;   // "[helios] x ±  y ± mm  yaw ± deg (n pairs)"
    void log_state(std::int64_t stamp_ms);

    // Clock check (plan step 5): offset of a keyframe stamp from the nearest odometry stamp. Reported, never corrected.
    void note_clock_offset_ms(double ms) { clock_ms_.push_back(ms); }
    [[nodiscard]] std::size_t clock_samples() const { return clock_ms_.size(); }
    [[nodiscard]] std::string clock_summary() const;   // "[helios] clock ..."

private:
    using V6 = Eigen::Matrix<double, 6, 1>;
    using M6 = Eigen::Matrix<double, 6, 6>;
    struct Rows { Eigen::Matrix<double, 3, 6> J; Eigen::Matrix3d R, W; Eigen::Vector3d y; double gamma = 1.0, maha = 0.0; };
    [[nodiscard]] M6 prior() const;
    void solve() const;   // EM over the pair responsibilities (lazy, cached)
    void open_csv(std::ofstream& f, const char* stem, const char* header);

    HeliosMountConfig cfg_;
    sc::calib::HarvesterConfig hcfg_;
    std::map<std::int64_t, std::vector<Eigen::Vector2f>> sweeps_;   // stamp -> band points, harvester axes
    mutable std::vector<Rows> rows_;   // gamma is re-evaluated by solve()
    mutable bool dirty_ = true;
    mutable V6 x_ = V6::Zero();
    mutable M6 C_ = M6::Zero();
    std::deque<sc::calib::Pair> queue_;
    double last_reg_ms_ = 0.0;
    long missing_ = 0, rejected_ = 0;
    std::vector<double> clock_ms_;
    std::ofstream pairs_csv_, state_csv_;
};
}  // namespace sc
