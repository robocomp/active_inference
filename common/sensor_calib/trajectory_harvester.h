/*
 * trajectory_harvester.h -- the opportunistic TRAJECTORY harvester of the self_calibration agent (plan Task 2.4).
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
 * maximises  sum_p tr( Sigma  I_p ),  Sigma = (H_accumulated + H_prior)^-1,  I_p = J_p' R_p^-1 J_p: the
 * PRIOR-SCALED INFORMATION TRACE. Sigma puts a metre of lever and a radian of yaw on the same footing (through the
 * prior) and the sum is additive over pairs, so the best SEGMENTATION of the ring is an exact dynamic programme, not
 * a greedy guess. The segmentation is OPTIMAL UNDER THE STATED NOISE MODEL; it is not "one pair per turn". Lever
 * information of a piece is ~ 4 sin^2(dth/2)/var, which is superadditive for small angles (cutting a turn into
 * 0.20 rad pieces destroys most of it: the thesis lost 9.9 -> 0.64 information units) but ties at pi (one pi pair =
 * two pi/2 pairs), and when the row noise grows with the rotation covered (odo_sigma_per_rad) splitting a large
 * turn legitimately wins. A straight followed by a turn is worth two pairs (the merged chord is perpendicular to
 * the midpoint heading, so the yaw row dies). Keyframes of a chosen pair are CONSUMED, and so is everything
 * between them: nothing is counted twice. Two pairs never share a keyframe.
 *
 * HEADING (ruling B, "heading as two sensor factors"). dth of a pair is the precision-weighted fusion of the wheel
 * rotation and the gyro (IMU yaw rate integrated piecewise-constant, add_gyro). Each has a variance that grows with
 * the rotation covered plus a floor; Shadow's wheel rotation over-reports ~41 %, so the config gives the wheels a
 * wide relative sigma and the gyro a narrow one. Translation (the chord) is the wheels' speed integrated along the
 * FUSED heading, fused per odometry interval with the same precision weighting (integrating along the wheel
 * heading would rotate the chord by the wheel's 41 % over-report and bias the rows); the chord is decomposed on
 * the fused midpoint heading, and Pair::chord_sigma = |chord| * dth_sigma states its heading-driven uncertainty.
 * Without gyro coverage the wheel heading is used.
 *
 * ASSUMPTIONS TO VERIFY IN PHASE 4 (no code enforces them): (1) the IMU z axis is the body up axis, so
 * ImuIn::gyro.z is the yaw rate with the same sign as OdometryIn::rot; (2) the Helios capture stamp, the IMU
 * stamp_ms and the odometry source_ts_ms are on one clock; (3) odometry side/rot sign convention is left = +side,
 * CCW = +rot, to be pinned against KinematicMount's (fwd, lat) axes.
 *
 * KEYFRAME SPACING BUDGET: 0.1-0.5 s leaves the posterior sigma stable (<10 %, trajectory_harvester_test); 0.5 s is
 * the UPPER BOUND, at 1.0 s a coarse grid loses up to a spacing of hairpin rotation at each end (sigma +18 %).
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
#include <optional>
#include <vector>

namespace sc::calib
{

struct HarvesterConfig
{
    // compute budgets
    double      keyframe_spacing_s = 0.5;     ///< a sweep becomes a keyframe at most this often; 0.5 s is the UPPER BOUND of the budget
    double      ring_span_s        = 30.0;    ///< time the ring may hold before pairs are committed
    std::size_t max_keyframes      = 120;     ///< hard memory/compute bound on the ring
    // prior on the mount error (dx, dy) [m] and psi [rad]: same defaults as KinematicMount::Params
    double      prior_sigma_xy     = 0.05;
    double      prior_sigma_yaw    = 0.0175;
    // row noise model (see header)
    double      reg_sigma_m        = 0.01;    ///< scan-registration floor [m]
    double      odo_sigma_per_m    = 0.002;   ///< odometry error per metre travelled [m/m]
    double      odo_sigma_per_rad  = 0.005;   ///< odometry error per radian rotated [m/rad] (assumed, to be calibrated in Phase 0)
    // heading fusion: sigma^2 = (rel * |rotation covered|)^2 + floor^2 * dt  (wheels over-report ~41 %: wide)
    double      wheel_rot_sigma_rel   = 0.2;
    double      wheel_rot_sigma_floor = 0.01;    ///< rad / sqrt(s)
    double      gyro_sigma_rel        = 0.01;
    double      gyro_sigma_floor      = 0.002;   ///< rad / sqrt(s)
    double      history_s          = 90.0;    ///< odometry history kept to place keyframes (>= ring_span_s)
};

/// A selected pair, ready for a measurement. State order (dx, dy, psi).
struct Pair
{
    std::uint64_t id_a = 0, id_b = 0;                 ///< keyframe ids (monotone)
    std::int64_t  stamp_a_ms = 0, stamp_b_ms = 0;
    double dth = 0.0, dth_sigma = 0.0;                 ///< fused rotation a -> b [rad] (unwrapped) and its sigma
    double dth_wheel = 0.0, dth_gyro = 0.0;            ///< the two inputs (dth_gyro = dth_wheel when no gyro)
    double d_fwd = 0.0, d_lat = 0.0;                   ///< chord on the (fused) midpoint heading [m]
    double chord_sigma = 0.0;                          ///< |chord| * dth_sigma [m]
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
    /// Gyro yaw rate [rad/s] about the body up axis, held piecewise-constant like the odometry.
    void add_gyro(std::int64_t ts_ms, double yaw_rate);
    void add_odometry(std::int64_t source_ts_ms, double adv, double side, double rot);
    /// Offer a sweep. Returns true when it became a keyframe (false: decimated, stalled, or no odometry covers it).
    bool add_keyframe(std::int64_t stamp_ms, bool stalled);
    /// Commit every remaining pair and clear the ring.
    void finish();

    [[nodiscard]] std::vector<Pair> take_pairs();
    [[nodiscard]] const Eigen::Matrix3d& information() const { return H_acc_; }   ///< accumulated, no prior
    [[nodiscard]] std::size_t ring_size() const { return ring_.size(); }
    /// Stamp of the oldest keyframe a FUTURE or not-yet-taken pair can still reference: the front of the ring, or the
    /// oldest stamp_a of a committed pair not yet taken. nullopt = nothing pending. A consumer of the pairs keeps the
    /// sweeps of every stamp >= this (and of its own queued pairs), and may drop the older ones.
    [[nodiscard]] std::optional<std::int64_t> oldest_pending_stamp_ms() const;
    /// Close the running episode and start a new one.
    [[nodiscard]] EpisodeSummary close_episode();

private:
    struct Node { std::int64_t ts = 0; double x = 0, y = 0, th = 0, path = 0, rot_acc = 0, adv = 0, side = 0, rot = 0; };
    struct Keyframe { std::uint64_t id = 0; std::int64_t stamp_ms = 0; double x = 0, y = 0, th = 0, path = 0, rot = 0, gth = 0, gabs = 0; bool has_gyro = false; };
    struct GNode { std::int64_t ts = 0; double angle = 0, abs = 0, rate = 0; };

    [[nodiscard]] static Node advance(const Node& n, double dt_s);
    [[nodiscard]] bool over_budget() const;
    void commit(bool final_flush);
    struct Chord { double cx = 0, cy = 0, dth = 0; bool ok = false; };
    [[nodiscard]] Chord integrate_chord(std::int64_t t0, std::int64_t t1) const;
    [[nodiscard]] Pair pair_of(const Keyframe& a, const Keyframe& b) const;

    HarvesterConfig cfg_;
    std::deque<Node> hist_;
    std::deque<GNode> gyro_;
    std::deque<Keyframe> ring_;
    std::vector<Pair> out_;
    Eigen::Matrix3d H_acc_ = Eigen::Matrix3d::Zero(), H_prior_ = Eigen::Matrix3d::Zero();
    std::uint64_t next_id_ = 1;
    std::int64_t last_kf_ms_ = 0;
    bool have_kf_ = false;
    EpisodeSummary ep_;
};

}   // namespace sc::calib
