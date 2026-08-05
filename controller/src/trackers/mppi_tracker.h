#ifndef RC_MPPI_TRACKER_H
#define RC_MPPI_TRACKER_H

#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "tracker.h"

namespace rc
{

// THE MPPI SAMPLER — warm start + Gaussian perturbations + structured injection seeds, scored against
// the live ESDF and combined by a softmax whose temperature adapts to rollout dominance.
//
// It used to be steps 5..14 of TrajectoryController::compute plus eight private methods on that class,
// which is why the two other trackers could be read in one sitting and this one could not. Nothing about
// the arithmetic changed in the move; the only difference is that the sampler now names exactly what it
// reads — PathWorld for the body and the route, FieldWorld for the obstacle field.
//
// ★It takes BOTH halves, and it must: unlike PlainTracker it does its own avoidance (obstacle cost, the
// lateral terms, the CBF barrier, the collision predicate and a backstop safety gate), so every one of
// those is a query against the live field.
//
// ★It also needs the RAW CLOUD, which is not part of TrackerInput and deliberately must not be: the
// Safety-Guard exploration ramp and the gate's arming test both count frontal LiDAR returns rather than
// reading the ESDF. Putting a cloud into TrackerInput would hand PlainTracker obstacle data through the
// back door and break the guarantee written at the top of tracker.h — so the controller lends it here,
// for the duration of the call, through set_cycle_cloud().
class MppiTracker final : public Tracker
{
public:
    MppiTracker(const PathWorld& path, const FieldWorld& field) : path_(path), world_(field) {}

    const char* name() const override { return "mppi"; }

    ControlOutput& compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p) override;

    // A trajectory sample: sequence of (adv, rot) commands
    struct Seed
    {
        std::vector<float> adv;
        std::vector<float> rot;
    };

    // Result of simulating a seed
    struct SimResult
    {
        std::vector<Eigen::Vector2f> positions;
        float G_total = 0.f;
        float G_goal = 0.f;
        float G_obs = 0.f;
        float G_lat = 0.f;
        float G_cbf = 0.f;
        float G_smooth = 0.f;
        float G_vel = 0.f;
        float min_esdf = 1e9f;
        bool  collides = false;
    };

    // Previous optimal control sequence (T steps of [adv, rot])
    // This is the core warm-start: each cycle shifts it and samples around it
    struct ControlStep { float adv = 0.f; float rot = 0.f; };

    /// The cycle's LiDAR cloud in the robot frame, lent by the controller for the duration of compute().
    /// Non-owning; the controller clears it again afterwards so it can never be read stale.
    void set_cycle_cloud(const std::vector<Eigen::Vector3f>* points) { lidar_points_ = points; }

    /// Seed the sampler — see TrajectoryController::set_seed, which forwards here.
    void set_seed(std::uint32_t seed) { rng_.seed(seed); }

    /// Everything a NEW path resets (was the MPPI half of TrajectoryController::reset_mppi_state).
    void reset_state(const TrackerParams& p);
    /// Everything a stop clears (was the MPPI half of TrajectoryController::stop).
    void clear_state();

    /// The command actually returned last cycle — the reference for the continuity cost. Not
    /// prev_optimal_[0], which is what was planned rather than what was executed. The controller reports
    /// the commands it issues OUTSIDE this tracker (the arrival rotation) through these, because the
    /// continuity term must be anchored to what the base was told, whoever told it.
    void note_executed_command(float adv, float rot)
    { last_cmd_adv_ = adv; last_cmd_rot_ = rot; last_cmd_valid_ = true; }
    void forget_executed_command() { last_cmd_valid_ = false; }

private:
    const PathWorld&  path_;
    const FieldWorld& world_;
    const std::vector<Eigen::Vector3f>* lidar_points_ = nullptr;

    // Output smoothing
    Eigen::Vector3f smoothed_vel_ = Eigen::Vector3f::Zero();
    bool has_prev_vel_ = false;

    // ---- MPPI state ----
    std::vector<ControlStep> prev_optimal_;

    // Adaptive noise sigmas
    float adaptive_sigma_adv_;
    float adaptive_sigma_rot_;

    // Adaptive MPPI state
    int   adaptive_K_;               // current number of samples
    int   adaptive_T_;               // current horizon length
    float adaptive_lambda_;          // current MPPI temperature
    float ess_smooth_ = 0.f;        // EMA-smoothed ESS
    float dominance_smooth_ = 0.5f; // EMA-smoothed dominance in [0,1]
    float explore_ = 0.f;           // continuous exploration signal [0,1] = 1 - dominance
    float sg_explore_gate_smooth_ = 0.f; // EMA-smoothed Safety-Guard gating factor in [0,1]
    float last_mppi_ms_ = 0.f;      // last MPPI wall-clock time
    int   safety_guard_mood_cooldown_ = 0; // cycles until next mood bump allowed

    float last_cmd_adv_ = 0.f, last_cmd_rot_ = 0.f;
    bool  last_cmd_valid_ = false;

    // RNG
    std::mt19937 rng_{std::random_device{}()};
    std::normal_distribution<float> normal_{0.f, 1.f};

    // Compute ESS for diagnostics and adapt from dominance
    float compute_ess(const std::vector<float>& weights, int K, const TrackerParams& p) const;
    void adapt_from_dominance(float dominance, int K, float sg_gate, const TrackerParams& p);

    // Nominal control toward carrot (initial guess for warm-start)
    Seed compute_nominal(const Eigen::Vector2f& carrot_robot, int steps, const TrackerParams& p) const;

    // MPPI sampling: generate K perturbations around the nominal
    std::vector<Seed> sample_trajectories(const Eigen::Vector2f& carrot_robot,
                                          const Seed& nominal,
                                          const TrackerParams& p);

    // Takes no sampling-mean argument: the only thing that ever needed it was the
    // information-theoretic correction, deleted on purpose (see the block comment in
    // simulate_and_score for why it must not come back).
    SimResult simulate_and_score(const Seed& seed,
                                 const Eigen::Vector2f& carrot_robot,
                                 const Eigen::Vector2f& goal_robot,
                                 const TrackerParams& p);
    void optimize_seed(Seed& seed, const Eigen::Vector2f& carrot_robot, const TrackerParams& p);

    // Obstacle scoring helpers (single-weight 2-stage quadratic model)
    float effective_d_safe_for_goal_dist(float goal_dist, const TrackerParams& p) const;
    float obstacle_step_cost(float esdf_val, float d_safe_eff, float body_r,
                             const TrackerParams& p) const;
    float obstacle_repulsion_strength(float esdf_val, float d_safe_eff, float body_r,
                                      const TrackerParams& p) const;
};

}   // namespace rc

#endif
