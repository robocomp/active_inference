/*
 * trajectory_harvester.h -- the opportunistic TRAJECTORY harvester of the sensor_calibration agent (plan Task 2.4).
 *
 * Plain data in / plain data out: no DDS, no DSR, no Qt. The agent feeds it
 *     add_odometry(source_ts_ms, adv, side, rot)   the producer's velocity samples (piecewise-constant hold)
 *     add_keyframe(stamp_ms, stalled)              one per Helios sweep, with the input layer's stall verdict
 * and takes SELECTED PAIRS back. A pair is two keyframes plus everything a later factor (Phase 4.4: scan
 * registration between the two sweeps) needs to turn them into evidence: the predicted relative motion, the row
 * Jacobian of the mount error and the row noise R. THE MEASUREMENT IS NOT HERE: information depends on the motion
 * geometry, not on what the registration returned, so the pair is scored before any scan is registered.
 *
 * MODEL (room_concept/src/mount_factors.h, KinematicMount). Planar Helios mount error m = (dx, dy, psi) in the
 * order of the state vector below. For two keyframes with relative odometry rotation dth and chord (d_fwd, d_lat)
 * expressed on the MIDPOINT heading:
 *       y_fwd = +2 sin(dth/2) dx                      + noise(var)
 *       y_lat = -2 sin(dth/2) dy + d_fwd psi          + noise(var)
 * so the yaw row grows with straights (d_fwd) and the lever rows with rotation. There is NO "straight" / "turning"
 * / "stopped" classifier anywhere: the Jacobian decides, and a stop has J = 0 and so contributes exactly nothing.
 *
 * ROW NOISE (config; defaults are ASSUMPTIONS to be replaced by Phase 0 data). The measurement is
 * (registration - odometry), so its error is the registration floor plus the odometry error accumulated over the
 * pair: var = reg_sigma_m^2 + (odo_sigma_per_m * path)^2 + (odo_sigma_per_rad * rotated)^2, with `path` and
 * `rotated` the odometry's travelled length and |rotation| between the two keyframes. Longer pairs are noisier;
 * that is a covariance that grows with the covariate, not a cut-off.
 *
 * PAIR SELECTION (what is maximised). Over the keyframes held in the ring, choose a set of DISJOINT pairs that
 * maximises  sum_p tr( Sigma  I_p ),  Sigma = (H_accumulated + H_prior)^-1,  I_p = J_p' R_p^-1 J_p. This is the
 * first-order (linearised) gain of log det of the posterior information, i.e. the D-optimal gain, and it is the
 * natural scalar because (a) Sigma carries the PRIOR scale of each DoF (a metre of lever and a radian of yaw are
 * comparable only through it) and (b) a linearisation is additive over pairs, so the best SEGMENTATION of the ring
 * is a dynamic programme, not a greedy guess. Because information is quadratic in the geometry a whole turn is
 * worth more as ONE pair than as pieces (a pi hairpin: 2 sin(pi/2) = 2 as one pair; the thesis lost 9.9 -> 0.64
 * information units by cutting it at 0.20 rad), and a straight followed by a turn is worth more as TWO pairs than
 * merged (the merged chord is perpendicular to the midpoint heading, so the yaw row dies); the DP finds both.
 * Keyframes of a chosen pair are CONSUMED, and so is everything between them: nothing is counted twice. Two pairs
 * never share a keyframe (a shared sweep would carry one registration error into both).
 *
 * WHEN. Pairs are committed when the ring exceeds its budget (ring_span_s of time or max_keyframes), never at the
 * newest keyframe: a pair touching the newest keyframe may be a turn still in progress, so it stays in the ring
 * (unless the ring is over budget with nothing else to commit). finish() commits everything (end of an episode).
 *
 * STALL = HOLD. A keyframe flagged stalled is not ingested; the ring is committed up to there and cleared, so no
 * pair spans a stalled interval. The input layer does not withhold data; this class is where it is refused.
 *
 * COMPUTE BUDGETS (not thresholds): keyframe_spacing_s (a decimator), ring_span_s, max_keyframes. The tests show
 * that the posterior does not depend on the first.
 */
#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace sc::calib
{

struct HarvesterConfig
{
    // compute budgets
    double      keyframe_spacing_s = 0.5;     ///< a sweep becomes a keyframe at most this often
    double      ring_span_s        = 30.0;    ///< time the ring may hold before pairs are committed
    std::size_t max_keyframes      = 120;     ///< hard memory/compute bound on the ring
    // prior on the mount error (dx, dy) [m] and psi [rad]: same defaults as KinematicMount::Params
    double      prior_sigma_xy     = 0.05;
    double      prior_sigma_yaw    = 0.0175;
    // row noise model (see header)
    double      reg_sigma_m        = 0.01;    ///< scan-registration floor [m]
    double      odo_sigma_per_m    = 0.002;   ///< odometry error per metre travelled [m/m]
    double      odo_sigma_per_rad  = 0.001;   ///< odometry TRANSLATION error per rad rotated [m/rad]; small: a pivot barely moves the chord
    double      history_s          = 90.0;    ///< odometry history kept to place keyframes (>= ring_span_s)
};

/// A selected pair, ready for a measurement. State order (dx, dy, psi).
struct Pair
{
    std::uint64_t id_a = 0, id_b = 0;                 ///< keyframe ids (monotone)
    std::int64_t  stamp_a_ms = 0, stamp_b_ms = 0;
    double dth = 0.0;                                  ///< odometry rotation a -> b [rad] (unwrapped)
    double d_fwd = 0.0, d_lat = 0.0;                   ///< chord on the midpoint heading [m]
    double path = 0.0, rotated = 0.0;                  ///< odometry path length [m] and sum |rot| [rad]
    Eigen::Matrix<double, 2, 3> J = Eigen::Matrix<double, 2, 3>::Zero();   ///< rows: (fwd, lat)
    Eigen::Vector2d var = Eigen::Vector2d::Zero();     ///< per-row noise variance [m^2]
    Eigen::Matrix3d info = Eigen::Matrix3d::Zero();    ///< J' R^-1 J
};

/// Pair geometry from two odometry poses is exposed for the tests.
[[nodiscard]] Pair make_pair_from_motion(double dth, double d_fwd, double d_lat, double path, double rotated,
                                         const HarvesterConfig& cfg);

/// Accumulated evidence for (dx, dy, psi): what a measurement fills in once a pair has one.
class MountEvidence
{
public:
    explicit MountEvidence(const HarvesterConfig& cfg);
    /// y_fwd, y_lat = (registration - odometry) relative motion of the pair, on the pair's midpoint axes.
    void add(const Pair& p, double y_fwd, double y_lat);
    [[nodiscard]] Eigen::Vector3d mean() const;
    [[nodiscard]] Eigen::Vector3d sigma() const;
    [[nodiscard]] const Eigen::Matrix3d& information() const { return H_; }   ///< data only (no prior)
    [[nodiscard]] long n_pairs() const { return n_; }
private:
    Eigen::Matrix3d H_ = Eigen::Matrix3d::Zero(), Hp_ = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b_ = Eigen::Vector3d::Zero();
    long n_ = 0;
};

/// Small per-episode summary (Phase 6 change detection reads these).
struct EpisodeSummary
{
    std::int64_t    t_begin_ms = 0, t_end_ms = 0;
    long            keyframes = 0, dropped_stalled = 0, pairs = 0;
    Eigen::Matrix3d info = Eigen::Matrix3d::Zero();   ///< information gained this episode
};

class TrajectoryHarvester
{
public:
    explicit TrajectoryHarvester(HarvesterConfig cfg = {});

    /// Piecewise-constant hold: a sample's velocity applies from its source_ts_ms to the next sample's. Out-of-order
    /// or duplicate stamps are ignored. adv/side/rot as OdometryIn (body +Y forward).
    void add_odometry(std::int64_t source_ts_ms, double adv, double side, double rot);
    /// Offer a sweep. Returns true when it became a keyframe (false: decimated, stalled, or no odometry covers it).
    bool add_keyframe(std::int64_t stamp_ms, bool stalled);
    /// Commit every remaining pair and clear the ring.
    void finish();

    [[nodiscard]] std::vector<Pair> take_pairs();
    [[nodiscard]] const Eigen::Matrix3d& information() const { return H_acc_; }   ///< accumulated, no prior
    [[nodiscard]] std::size_t ring_size() const { return ring_.size(); }
    /// Close the running episode and start a new one.
    [[nodiscard]] EpisodeSummary close_episode();

private:
    struct Node { std::int64_t ts = 0; double x = 0, y = 0, th = 0, path = 0, rot_acc = 0, adv = 0, side = 0, rot = 0; };
    struct Keyframe { std::uint64_t id = 0; std::int64_t stamp_ms = 0; double x = 0, y = 0, th = 0, path = 0, rot = 0; };

    [[nodiscard]] static Node advance(const Node& n, double dt_s);
    [[nodiscard]] bool over_budget() const;
    void commit(bool final_flush);
    [[nodiscard]] Pair pair_of(const Keyframe& a, const Keyframe& b) const;

    HarvesterConfig cfg_;
    std::deque<Node> hist_;
    std::deque<Keyframe> ring_;
    std::vector<Pair> out_;
    Eigen::Matrix3d H_acc_ = Eigen::Matrix3d::Zero(), H_prior_ = Eigen::Matrix3d::Zero();
    std::uint64_t next_id_ = 1;
    std::int64_t last_kf_ms_ = 0;
    bool have_kf_ = false;
    EpisodeSummary ep_;
};

}   // namespace sc::calib
