/*
 * common/ai_belief/persistent_engine.h  —  the shared persistent estimation engine (header-only, pure)
 *
 * Spec docs/superpowers/specs/2026-11-23-shared-engine-design.md ("Engine shape"); the update itself is spec
 * 2026-11-15 lidar-measures §7c.2 / 7c.7 / 7c.8. Lifted out of countable_concept's CountableBelief (its first model) so
 * chair, table, refrigerator and hood run the same machinery:
 *
 *   - the joint state x = (θ, b), θ ∈ ℝᴺ the model's own state, b ∈ ℝᴷ the persistent shared-error latents
 *     (persistent_update.h), K = 3 + 6N in this layout (offsets below):
 *       e      (kChain, 3)  the robot's pose error in the room, measured through chain_jacobian(lever) (7c.7 b);
 *       b_cam  (kCam,   N)  the camera points' sensor-specific shared error;
 *       b_L    (kL,     N)  the LiDAR rays' sensor-specific shared error;
 *       u_cam  (kFc,    N)  the camera's per-frame ego-motion share, a new source every update ("fix A", rev 3.3.13);
 *       u_L    (kFL,    N)  the same for the LiDAR;
 *       s_cam  (kSc,    N)  the camera's shape misfit (ρ = 1, marginal Δ²/12 with Δ = the model's shape_spread());
 *       s_L    (kSL,    N)  the LiDAR's shape misfit (likewise).
 *     The camera measures θ + B_chain e + b_cam + u_cam + s_cam, the rays θ + B_chain e + b_L + u_L + s_L;
 *   - the damped Gauss-Newton / Levenberg–Marquardt loop with Nielsen's λ rule and the posterior-σ stop rule (7c.8 rev
 *     3.3.7, 3.3.8), each channel linearised at what it measures (the consistent linearisation);
 *   - the class prior at the class mean counted once, with the birth cloud's excess position covariance ("fix B",
 *     rev 3.3.14);
 *   - the smooth beam model of the LiDAR rays (beam_channel.h);
 *   - the GnTrace diagnostic (one row per GN step, and a finite-difference check of every channel's gradient).
 *
 * The MODEL (concept PersistentModel) supplies the shape: the camera-point hooks of recursive_laplace.h, its camera-only and
 * θ-only extra terms (each with its energy, the GN acceptance objective), its beam parts, how a robot pose error moves θ,
 * the class prior, the constraints. Conventions the engine relies on: θ(0), θ(1) are the room-frame centre (cx, cy) — the
 * lever is taken there and the birth position covariance lands there — and θ(M::kYaw) is an angle (wrapped; −1 = none).
 *
 * All solves in double; fixed-size Eigen throughout. No thresholds: the solver's tolerances are solver constants
 * (SolverConfig), and the only case split on a latent is σ_prev = 0 (a new error source).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "beam_channel.h"
#include "foreign_box.h"
#include "persistent_update.h"
#include "viewpoint.h"

namespace rc::ai::engine
{

using rc::ai::beam::ForeignBox;   // a neighbour's published belief (foreign_box.h; spec shared-engine G3)

// The per-frame inputs of the latents (spec 7c.7 a, b; rev 3.3.13), in θ's component order. Read by the engine's latent
// predict; the model fills them from its own frame.
template <int N>
struct FrameLatents
{
    std::array<float, N> rho_cam{};      // lag-1 correlation of b_cam since the camera's previous update (1 parked, 0 new view)
    std::array<float, N> rho_L{};        // the same for b_L
    std::array<float, N> sig2_cam{};     // b_cam's marginal variance at this frame (σ_now²)
    std::array<float, N> sig2_L{};       // b_L's
    std::array<float, N> fresh_c{};      // the camera's fresh (ego-motion) variance: u_cam's, re-drawn every update
    std::array<float, N> fresh_L{};      // the LiDAR's
    float           rho_chain  = 0.0f;   // e's correlation with its last update (exp(−½cᵀΣ_pose⁻¹c))
    Eigen::Matrix3d sigma_pose = Eigen::Matrix3d::Zero();   // the robot's pose covariance in the room (x, y, ψ)
    Eigen::Vector2f chain_dr   = Eigen::Vector2f::Zero();   // the robot's room displacement since the last update
    Eigen::Vector2f robot_xy   = Eigen::Vector2f::Zero();   // the robot's room position at the capture stamp
    bool camera_measured = false;        // the camera channel measures this update
    bool lidar_measured  = false;        // the LiDAR channel measures this update
    // G3 (spec shared-engine, "Answers to the builder check" 5): the neighbours' boxes, a FIXED foreign density for this update
    // built from their previous-cycle published beliefs. Each camera point's membership compares this object's mixture with
    // them: an ordinary mixture, (1 − ε)·L_model(p) + ε·Σ_k g_k(p) (foreign_box.h), ε = mask_error_rate, the camera's
    // probability that a mask pixel is not this object's (Model.MaskPixelErrorRate). No usable neighbour, or ε outside
    // (0, 1) ⇒ the model's own mixture, bit for bit. (The LiDAR rays read neighbours through their staged p_f, beam_room.h.)
    std::vector<ForeignBox> foreign_boxes;
    double mask_error_rate = 0.0;
};

// Solver constants (spec 7c.7 d, 7c.8 rev 3.3.7/3.3.8), not modelling thresholds.
struct SolverConfig
{
    int    iters_max = 25;               // GN steps (accepted + rejected) per update
    double tol_m     = 1e-3;             // the restart scheme's tolerance (metres / other components)
    double tol_rad   = 1e-3;             // ... and on the angle component
    double tol_sigma = 0.1;              // the stop rule: undamped step < tol_sigma·σ_k on every θ component
    bool   reject_all_for_test = false;  // TEST HOOK: every step rejected, as if J never fell
    int    gn_trace  = 0;                // GnTrace: 1 = steps + the FD check on unconverged updates, 2 = the check always
    // The 7c.7 (d) restart scheme (x̃_k = (θ_k, b̄), θ-only damping, ×4/÷4 λ, the step-size convergence rule) in place of
    // the consistent linearisation. Kept for one test hook (the "Reduction" comparison); no model sets it in production.
    bool   restart   = false;
};

// GnTrace (diagnostic): one row per GN step of the last update, and the finite-difference check.
template <int N>
struct GnTraceStep
{
    int    step = 0;             // 1-based step count (accepted + rejected, as gn_it)
    double lambda = 0.0;         // λ the step was solved at (0 = the undamped stop-rule proposal)
    double rho = 0.0;            // Nielsen gain ratio (−1 when the predicted decrease is ≤ 0; NaN for the λ = 0 proposal)
    double pred = 0.0;           // the quadratic model's predicted decrease
    double J_before = 0.0, J_after = 0.0;
    int    accepted = 0;
    Eigen::Matrix<double, N, 1> d = Eigen::Matrix<double, N, 1>::Zero();             // θ step after constraints
    Eigen::Matrix<double, N, 1> d0_sigma = Eigen::Matrix<double, N, 1>::Constant(std::numeric_limits<double>::quiet_NaN());   // undamped step / σ_k at this linearisation (NaN: not computed)
};
struct GnTraceFd
{
    int at = 0;                  // 0 = θ₀ (b̄), 1 = the returned θ (b_post)
    int channel = 0;             // 0 camera, 1 LiDAR, 2 θ-only, 3 all channels along θ (b fixed)
    int k = 0;                   // θ component
    double grad_an = 0.0;        // ∂E/∂θ_k the solver uses, from the raw (unmarginalised) information: −bd_k (channel 3: −G_k)
    double grad_marg = 0.0;      // −g_k after the fresh marginalisation (channels 0, 1; = grad_an otherwise)
    double grad_fd3 = 0.0;       // central difference at h = 1e-3
    double grad_fd4 = 0.0;       // central difference at h = 1e-4
    double curv_an = 0.0;        // the information's diagonal I_kk (Gauss-Newton curvature)
    double curv_fd = 0.0;        // second difference at h = 1e-3
};

// What a model provides. Required hooks beyond the camera-point ones of recursive_laplace.h:
//   Frame::points, Frame::R                    the camera points (indexable Eigen::Vector3f) and their per-point variances
//                                              (indexable float; empty or ≤ 0 ⇒ sigma2()), read by the engine's point loop;
//   accumulate_extra(s, f, I, b, E)            the θ-only channel (what the latents must not absorb); with E non-null each
//                                              term ALSO adds its energy ½λr² (the GN acceptance objective), E = nullptr when
//                                              only the information is wanted — the engine calls this one signature;
//   point_energy(p, s, R)                      one camera point's −log mixture likelihood (double);
//   beam_parts(s), beam_rays(f), beam_config() the LiDAR channel's convex parts, staged rays and beam config;
//   chain_jacobian(s, lever)                   how a robot pose error (δx, δy, δψ) moves θ;
//   class_prior_mean(seed, f)                  the first update's prior mean (the class mean; position at the seed);
//   prior_cov_diag(), process_noise_diag()     Σ₀ and Q diagonals (as recursive_laplace.h);
//   state(), set_state(s), covariance(), set_covariance(Σ)   the model's θ and its float N×N mirror of P_θθ (the model's
//                                              own terms may read the mirror: the engine sets it to the predicted P̄_θθ
//                                              before the first linearisation, and to the posterior after);
//   shape_spread()                             Δ, the class's layer spread per θ component (spec shared-engine G4): the
//                                              range over which the layer a sensor sees may lie; the shape misfit is
//                                              uniform over it, so its marginal is Δ²/12 (0 ⇒ no layering on that DOF).
// Optional (detected by requires-expressions; absent ⇒ the stated default):
//   accumulate_camera_extra(s, f, I, b, E)     the camera channel's non-point terms, same energy convention (absent: none);
//   solver()                                   SolverConfig (absent: SolverConfig{});
//   accumulate_lidar_override(s, f, I*, b*, E*) -> bool   route the LiDAR channel elsewhere (a test hook; absent: never).
// The camera channel's flag is FrameLatents::camera_measured alone: it gates both the camera's accumulation (information and
// energy) and its channel in the joint step, so G and J cannot disagree about whether the camera measured.
template <class M>
concept PersistentModel =
    requires {
        { M::N } -> std::convertible_to<int>;
        { M::kYaw } -> std::convertible_to<int>;
        typename M::State;
        typename M::Frame;
    }
    and requires(const M& cm, M& m, typename M::State& s, const typename M::State& cs, const typename M::Frame& f,
                 const Eigen::Vector3f& p, float R, const Eigen::Vector2f& lever, Eigen::Matrix<float, M::N, M::N>& I,
                 Eigen::Matrix<float, M::N, 1>& b, double* E, const Eigen::Matrix<float, M::N, 1>& v,
                 const Eigen::Matrix<float, M::N, M::N>& S) {
        { M::State::from_vec(v) } -> std::same_as<typename M::State>;
        { cs.vec() } -> std::convertible_to<Eigen::Matrix<float, M::N, 1>>;
        { cm.sdf_prim(p, cs, 0) } -> std::convertible_to<float>;
        { cm.sdf_jacobian(p, cs, 0) } -> std::convertible_to<Eigen::Matrix<float, M::N, 1>>;
        { cm.responsibilities(p, cs, R)[0] } -> std::convertible_to<float>;
        { cm.n_prims() } -> std::convertible_to<int>;
        { cm.sigma2() } -> std::convertible_to<float>;
        { cm.point_energy(p, cs, R) } -> std::convertible_to<double>;
        { f.points.size() } -> std::convertible_to<std::size_t>;
        { f.points[0] } -> std::convertible_to<const Eigen::Vector3f&>;
        { f.R.size() } -> std::convertible_to<std::size_t>;
        { f.R[0] } -> std::convertible_to<float>;
        cm.accumulate_extra(cs, f, I, b, E);
        { cm.beam_parts(cs) } -> std::convertible_to<std::vector<rc::ai::beam::BeamPart>>;
        { cm.beam_rays(f) } -> std::convertible_to<const rc::ai::beam::BeamRays&>;
        { cm.beam_config() } -> std::convertible_to<const rc::ai::beam::BeamConfig&>;
        { cm.chain_jacobian(cs, lever) } -> std::convertible_to<Eigen::Matrix<double, M::N, 3>>;
        cm.apply_constraints(s);
        cm.canonicalize(s);
        { cm.shape_spread() } -> std::convertible_to<Eigen::Matrix<float, M::N, 1>>;
        { cm.prior_cov_diag() } -> std::convertible_to<Eigen::Matrix<float, M::N, 1>>;
        { cm.process_noise_diag() } -> std::convertible_to<Eigen::Matrix<float, M::N, 1>>;
        { cm.class_prior_mean(cs, f) } -> std::convertible_to<Eigen::Matrix<double, M::N, 1>>;
        { cm.state() } -> std::convertible_to<const typename M::State&>;
        { cm.covariance() } -> std::convertible_to<Eigen::Matrix<float, M::N, M::N>>;
        m.set_state(cs);
        m.set_covariance(S);
    };

template <PersistentModel M>
class PersistentEngine
{
public:
    static constexpr int N = M::N;
    static constexpr int K = 3 + 6 * N;
    static constexpr int kChain = 0, kCam = 3, kL = 3 + N, kFc = 3 + 2 * N, kFL = 3 + 3 * N, kSc = 3 + 4 * N, kSL = 3 + 5 * N;
    static constexpr int kYaw = M::kYaw;
    using State = typename M::State;
    using Frame = typename M::Frame;
    using MatNf = Eigen::Matrix<float, N, N>;
    using VecNf = Eigen::Matrix<float, N, 1>;
    using VecNd = Eigen::Matrix<double, N, 1>;
    using MatJd = Eigen::Matrix<double, N + K, N + K>;
    using VecKd = Eigen::Matrix<double, K, 1>;
    using VecXd = Eigen::Matrix<double, N + K, 1>;

    // ── State ────────────────────────────────────────────────────────────────────────────────────────────────────────
    const MatJd& P() const { return P_; }
    MatJd&       P_for_test() { return P_; }
    MatNf        Sigma_theta() const { return P_.template topLeftCorner<N, N>().template cast<float>(); }
    const VecKd& latent_mean() const { return b_; }
    void         set_latent_mean(const VecKd& b) { b_ = b; }
    VecKd        latent_std() const { return P_.diagonal().template tail<K>().cwiseMax(0.0).cwiseSqrt(); }
    const rc::ai::view::ChainLever& chain_lever() const { return lever_; }
    const Eigen::Matrix3d&          last_sigma_pose() const { return last_sigma_pose_; }
    bool joint_init() const { return joint_init_; }
    bool first_update_done() const { return first_update_done_; }
    void set_first_update_done(bool v) { first_update_done_ = v; }
    // Fix B (spec 7c.8 rev 3.3.14): the birth cloud's EXCESS position covariance (room frame), added on θ's (cx, cy) block of
    // the first update's class prior, and now on P if the joint is already seeded. Zero ⇒ the diagonal σ_pos² of the prior.
    void set_birth_position_cov(const Eigen::Matrix2f& excess)
    {
        birth_pos_excess_ = excess;
        if (joint_init_) P_.template topLeftCorner<2, 2>() += excess.template cast<double>();
    }
    const Eigen::Matrix2f& birth_position_excess() const { return birth_pos_excess_; }

    // The joint is seeded from the model's θ covariance the first time it is used (a model's parameters may be set after it
    // is built).
    void ensure_joint(const MatNf& Sigma0)
    {
        if (joint_init_) return;
        P_.setZero();
        P_.template topLeftCorner<N, N>() = Sigma0.template cast<double>();
        b_.setZero(); sig_prev_.setZero();
        lever_ = {};
        last_sigma_pose_.setZero();
        joint_init_ = true;
    }

    // A degenerate P is self-sealing (it freezes the instance): re-seed θθ to the class P, b_cam / b_L to their last
    // marginal (σ_prev²), e to the last Σ_pose (its own 3×3), cross terms 0, b = 0 (ℓ kept), and the class prior re-enters.
    // Returns true when it re-seeded (the model then resets what it carries beside the joint).
    bool sanitize(M& m)
    {
        ensure_joint(m.covariance());
        bool bad = not (P_.allFinite() and b_.allFinite());
        for (int i = 0; i < N and not bad; ++i) bad = not (P_(i, i) > 0.0);
        for (int j = 0; j < K and not bad; ++j) bad = not (P_(N + j, N + j) >= 0.0);
        if (not bad) return false;
        P_.setZero();
        P_.diagonal().template head<N>() = m.prior_cov_diag().template cast<double>();
        P_.diagonal().template tail<K>() = sig_prev_.cwiseProduct(sig_prev_);
        P_.template block<3, 3>(N + kChain, N + kChain) = last_sigma_pose_;
        b_.setZero();
        first_update_done_ = false;                                       // the class prior re-enters once, at the class mean
        m.set_covariance(Sigma_theta());
        return true;
    }

    // θ's predict on the θθ block of P: Σ ← Σ + q_scale·Q (F = I, static), then the clamp_to_prior congruence on θ's rows AND
    // columns of P, so the θ–b cross blocks scale with it and P stays PSD. The latents are predicted only on the update that
    // measures them (their ρ spans the time since their channel's last update).
    void predict(M& m, float q_scale)
    {
        ensure_joint(m.covariance());
        const VecNf q  = m.process_noise_diag();
        const VecNf p0 = m.prior_cov_diag();
        for (int i = 0; i < N; ++i) P_(i, i) += double(std::max(0.0f, q_scale)) * double(q(i));
        for (int i = 0; i < N; ++i)
            if (p0(i) > 0.0f and P_(i, i) > double(p0(i)))
            {
                const double sc = std::sqrt(double(p0(i)) / P_(i, i));   // symmetric rescale keeps P PSD
                P_.row(i) *= sc;
                P_.col(i) *= sc;
            }
        m.set_covariance(Sigma_theta());
    }

    // Step 1 on the latents (spec 7c.7 a, b). e is predicted whenever a channel measures: a NEW SOURCE when it has no lever
    // yet or its previous Σ_pose was zero (ρ = 0 for that transition) — the lever is then taken at (θ's centre − robot);
    // otherwise the lever moves with the robot, ℓ ← ρ(ℓ − Δr) + (1 − ρ)(c − r), BEFORE the predict, and e is carried by
    // F = chain_transport(Δr) with correlation ρ_chain and fresh variance (1 − ρ²) Σ_pose. b_cam / b_L: F = I, per-DOF ρ,
    // Σ_now = diag(σ_now²), new source where σ_prev = 0, predicted only when their channel measures. u_cam / u_L: re-drawn
    // every update (rev 3.3.13). s_cam / s_L: ρ = 1, born at diag(Δ²/12) with Δ = shape_spread(), when their channel measures.
    void predict_latents(const M& m, const FrameLatents<N>& fl)
    {
        namespace pu = rc::ai::persistent;
        namespace vw = rc::ai::view;
        const bool camera = fl.camera_measured, lidar = fl.lidar_measured;
        if (not camera and not lidar) return;
        const VecNf th = m.state().vec();
        VecXd x; x << th.template cast<double>(), b_;
        {
            const bool new_source = not lever_.set or last_sigma_pose_.isZero(0.0);
            const Eigen::Vector2f centre(th(0), th(1));
            if (new_source) vw::lever_start(lever_, centre, fl.robot_xy);
            else            vw::lever_step(lever_, fl.chain_dr, fl.rho_chain, centre, fl.robot_xy);
            pu::predict_block<N, K>(P_, x, kChain, 3, vw::chain_transport(fl.chain_dr),
                                    Eigen::VectorXd::Constant(3, double(fl.rho_chain)), fl.sigma_pose, new_source);
            last_sigma_pose_ = fl.sigma_pose;
        }
        const auto block = [&](int off, const float* r, const float* s2)
        {   // F = I; σ_prev = 0 ⇒ a new source (ρ = 0 for that latent); Σ_now = diag(σ_now²)
            Eigen::VectorXd rv(N), sv(N);
            bool all_new = true;
            for (int j = 0; j < N; ++j)
            {
                const bool nw = (sig_prev_(off + j) == 0.0);
                rv(j) = nw ? 0.0 : double(r[j]);
                sv(j) = std::max(0.0, double(s2[j]));
                all_new = all_new and nw;
            }
            pu::predict_block<N, K>(P_, x, off, N, Eigen::MatrixXd::Identity(N, N), rv, sv.asDiagonal().toDenseMatrix(), all_new);
            for (int j = 0; j < N; ++j) sig_prev_(off + j) = std::sqrt(sv(j));
        };
        if (camera) block(kCam, fl.rho_cam.data(), fl.sig2_cam.data());
        if (lidar)  block(kL,   fl.rho_L.data(),   fl.sig2_L.data());
        // u_cam, u_L (rev 3.3.13): re-drawn every update as a new source (ρ = 0 marginalises the last frame's u out exactly):
        // mean 0, covariance diag(fresh_c) when channel c measures this update, else 0 (inert), cross terms 0.
        const auto fresh_block = [&](int off, const std::array<float, N>& fr, bool on)
        {
            Eigen::VectorXd sv(N);
            for (int j = 0; j < N; ++j) sv(j) = on ? std::max(0.0, double(fr[j])) : 0.0;
            pu::predict_block<N, K>(P_, x, off, N, Eigen::MatrixXd::Identity(N, N), Eigen::VectorXd::Zero(N),
                                    sv.asDiagonal().toDenseMatrix(), /*new_source*/ true);
        };
        fresh_block(kFc, fl.fresh_c, camera);
        fresh_block(kFL, fl.fresh_L, lidar);
        // s_cam, s_L (shape misfit, spec shared-engine G4): the layer a sensor sees lies anywhere in the class's layer range
        // Δ = shape_spread(), uniformly, so the latent is born at Δ²/12 (a new source while σ_prev = 0) and carried with ρ = 1:
        // the view never changes which layer the object has. Once born its marginal is the one carried (σ_prev), so a
        // permute_theta relabelling (w ↔ h) keeps each DOF's spread with the DOF it moved to. Δ = 0 keeps the block exactly 0.
        const VecNf spread = m.shape_spread();
        const auto shape_block = [&](int off)
        {
            std::array<float, N> s2{}, ones{};
            for (int j = 0; j < N; ++j)
            {
                const double carried = sig_prev_(off + j);
                s2[j] = carried > 0.0 ? float(carried * carried) : spread(j) * spread(j) / 12.0f;
                ones[j] = 1.0f;
            }
            block(off, ones.data(), s2.data());
        };
        if (camera) shape_block(kSc);
        if (lidar)  shape_block(kSL);
        b_ = x.template tail<K>();
    }

    // A symmetry of the model's θ (e.g. the countable box's 90° yaw mode, which relabels w ↔ h): θ component i becomes old
    // component perm[i], on θ's rows/columns of P and on every N-sized latent block (b_cam, b_L, u_cam, u_L, s_cam, s_L) of
    // P, b and σ_prev, together. e and the lever are room-frame quantities and stay. The model permutes its own State.
    void permute_theta(const std::array<int, N>& perm)
    {
        if (not joint_init_) return;
        std::array<int, N + K> full{};
        for (int i = 0; i < N + K; ++i) full[i] = i;
        for (const int off : {0, N + kCam, N + kL, N + kFc, N + kFL, N + kSc, N + kSL})
            for (int i = 0; i < N; ++i) full[off + i] = off + perm[i];
        const MatJd P0 = P_;
        const VecKd b0 = b_, s0 = sig_prev_;
        for (int r = 0; r < N + K; ++r)
            for (int c = 0; c < N + K; ++c) P_(r, c) = P0(full[r], full[c]);
        for (int j = 0; j < K; ++j) { b_(j) = b0(full[N + j] - N); sig_prev_(j) = s0(full[N + j] - N); }
    }

    // ── The channels ────────────────────────────────────────────────────────────────────────────────────────────────
    // Each channel is linearised at its own state: s_th = θ_k, s_cam = θ_k + B_cam b, s_L = θ_k + B_L b. Camera (when
    // `camera`) = the points' GN information + the model's camera extra terms; LiDAR = the beam set (when `lidar`) or the
    // model's override; θ-only = the model's accumulate_extra.
    void accumulate_channels(const M& m, const State& s_th, const State& s_cam, const State& s_L, const Frame& f, bool camera,
                             bool lidar, MatNf& Id_cam, VecNf& bd_cam, MatNf& Id_L, VecNf& bd_L, MatNf& Id_th, VecNf& bd_th) const
    {
        Id_cam.setZero(); bd_cam.setZero(); Id_L.setZero(); bd_L.setZero(); Id_th.setZero(); bd_th.setZero();
        if (camera)
        {
            accumulate_points(m, s_cam, f, Id_cam, bd_cam);
            camera_extra(m, s_cam, f, Id_cam, bd_cam, nullptr);
        }
        if (not lidar_override(m, s_L, f, &Id_L, &bd_L, nullptr) and lidar)
            beam_.accumulate(key_of(s_L), [&] { return m.beam_parts(s_L); }, m.beam_rays(f), m.beam_config(), &Id_L, &bd_L);
        m.accumulate_extra(s_th, f, Id_th, bd_th, nullptr);
    }

    // The energy whose gradients accumulate_channels builds, term by term, with the same per-frame gates: the camera points'
    // −log mixture likelihood and the camera extra terms at s_cam, the beam energy (or the override's) at s_L, the θ-only
    // terms at s_th. ∇E = −(bd_cam + bd_L + bd_th) up to the state dependence of the terms' weights, which the GN information
    // also holds fixed.
    double data_energy(const M& m, const State& s_cam, const State& s_L, const State& s_th, const Frame& f, bool camera,
                       bool lidar) const
    {
        MatNf Id = MatNf::Zero();   // the terms' information is not wanted here
        VecNf bd = VecNf::Zero();
        double E = 0.0;
        if (camera)
        {
            const float sig2 = m.sigma2();
            for (std::size_t i = 0; i < f.points.size(); ++i)
            {
                const float R = R_of(f, i, sig2);
                E += neighbour_mix(m.point_energy(f.points[i], s_cam, R), f.points[i], R, nullptr);
            }
            camera_extra(m, s_cam, f, Id, bd, &E);
        }
        if (not lidar_override(m, s_L, f, nullptr, nullptr, &E) and lidar)
            E += beam_.accumulate(key_of(s_L), [&] { return m.beam_parts(s_L); }, m.beam_rays(f), m.beam_config(), nullptr, nullptr);
        m.accumulate_extra(s_th, f, Id, bd, &E);
        return E;
    }

    // ── One update (spec 7c.2 steps 1–4) ─────────────────────────────────────────────────────────────────────────────
    // Predict; per GN iterate linearise each channel at what it measures (the ego-motion share is the latent u_c, rev
    // 3.3.13), joint damped step; joint covariance at the returned θ. Returns the camera points' surface energy at what the
    // camera measures (0 when it did not measure), the recursive-Laplace engine's diagnostic on the biased quantity.
    float update(M& m, const Frame& f, const FrameLatents<N>& fl)
    {
        using MatN = MatNf;
        using VecN = VecNf;
        namespace pu = rc::ai::persistent;
        const SolverConfig cfg = solver_of(m);
        const NeighbourScope nb_scope(*this, fl);                          // G3: the neighbours, fixed for this update

        predict(m, 1.0f);                                                  // step 1, θ
        dbg_iterates_.clear();
        dbg_gn_energy_.clear();
        dbg_gn_steps_.clear(); dbg_gn_fd_.clear(); ++gn_update_serial_;   // GnTrace (diagnostic)
        last_gn_iters_ = 0; last_gn_converged_ = false; last_lambda_ = 0.0;   // gn_it = 0 ⇒ no measured update this cycle
        last_gn_step_.setZero();
        last_beam_staged_ = 0; last_gap_rays_ = 0; beam_.reset_ms();
        dbg_Pbar_tt_ = m.covariance();
        const bool camera = fl.camera_measured;
        // The LiDAR channel measures only a ray-set that is a measurement (spec 7c.8 rev 3.3.11): an empty one (or Σq = 0)
        // silently does not; a misaligned one or a BeamConfig not built by BeamConfig::make is refused, loudly, once per
        // engine. A model's LiDAR override (a test hook) reads its own rays and is not judged here.
        const bool lidar = fl.lidar_measured and lidar_ready(m, f);
        if (not camera and not lidar)
            return 0.0f;                                                   // predict-only

        FrameLatents<N> fl_used = fl;
        fl_used.lidar_measured = lidar;                                    // the latents follow the channels that measure
        predict_latents(m, fl_used);                                       // step 1, latents
        const typename rc::ai::beam::BeamChannel<N>::Scope cache_scope(beam_);   // the beam cache lives for this update
        MatJd P_bar = P_;
        VecXd x_bar; x_bar << m.state().vec().template cast<double>(), b_;
        if (not first_update_done_)
        {   // The class prior as a factor AT THE CLASS MEAN, counted once (spec 7c.1 amendment 11-17): Σ₀ = the class P, cross
            // terms 0 as at birth, plus the birth cloud's spread on (cx, cy) (rev 3.3.14). The seed stays the GN start.
            x_bar.template head<N>() = m.class_prior_mean(m.state(), f);
            P_bar.template topRows<N>().setZero();
            P_bar.template leftCols<N>().setZero();
            P_bar.diagonal().template head<N>() = m.prior_cov_diag().template cast<double>();
            P_bar.template topLeftCorner<2, 2>() += birth_pos_excess_.template cast<double>();
        }
        m.set_covariance(P_bar.template topLeftCorner<N, N>().template cast<float>());   // the model's terms read the predicted Σ
        dbg_Pbar_tt_ = m.covariance();

        // Channel selectors: the camera measures θ + B_chain e + b_cam + u_cam + s_cam, the rays θ + B_chain e + b_L + u_L + s_L;
        // B_chain = chain_jacobian(ℓ) is fixed over the GN iterates (the lever is instance state, spec 7c.7 b).
        const Eigen::Matrix<double, N, K> B_cam = selector(m, true), B_L = selector(m, false);
        // The ego-motion share is the latent u_c (rev 3.3.13), so the channels' information enters unmarginalised (Σf = 0:
        // marginalise_fresh is the identity), and G is the gradient of the J the step is accepted against.
        const MatN Sf_cam = MatN::Zero(), Sf_L = MatN::Zero();
        // The consistent linearisation (spec 7c.8 rev 3.3.7): x̃_k = (θ_k, b_k), every channel linearised at θ_k + B b_k, LM
        // damping on the whole joint state. cfg.restart keeps the 7c.7 restart scheme (x̃_k = (θ_k, b̄), θ-only damping).
        const bool consistent = not cfg.restart;
        const auto S = [](const VecNd& v) { return State::from_vec(v.template cast<float>()); };

        std::vector<pu::Channel<N, K>> channels;
        pu::FreshInfo<N> theta_only;
        MatN Id_cam, Id_L, Id_th; VecN bd_cam, bd_L, bd_th;
        const auto build = [&](const VecNd& th, const VecKd& bb)
        {
            accumulate_channels(m, S(th), S(th + B_cam * bb), S(th + B_L * bb), f, camera, lidar, Id_cam, bd_cam, Id_L, bd_L, Id_th, bd_th);
            channels.clear();
            if (camera) { pu::Channel<N, K> c; c.info = pu::marginalise_fresh<N>(Id_cam, bd_cam, Sf_cam); c.B = B_cam; channels.push_back(c); }
            if (lidar)  { pu::Channel<N, K> c; c.info = pu::marginalise_fresh<N>(Id_L,   bd_L,   Sf_L);   c.B = B_L;   channels.push_back(c); }
            theta_only.I = Id_th.template cast<double>();
            theta_only.g = bd_th.template cast<double>();
        };

        // Step 3, damped GN (spec 7c.7 d, rev 3.3.7). Each step: (𝟙 + P̄(W + λM))δ = P̄G − (x̃_k − x̄), M = diag W over θ and b;
        // the candidate's θ after apply_constraints. J = data_energy at the candidate (each channel at θ + B b) + ½ yᵀ(x − x̄).
        // Nielsen's λ rule: gain ratio ϱ = (J_k − J_{k+1}) / (J_k − J_model(δ)); ϱ > 0 ⇒ accept, λ ← λ·max(⅓, 1 − (2ϱ − 1)³),
        // ν ← 2, rebuild; else λ ← λν, ν ← 2ν and re-solve on the same linearisation (first λ = 10⁻³, ν = 2). The restart
        // scheme keeps ×4 / ÷4. Stop rule: once per linearisation, the UNDAMPED proposal; below tol_sigma·σ_k on every θ
        // component ⇒ applied if J does not rise, and the update stops (gn_conv = 1); else after iters_max steps.
        const auto data_J = [&](const VecXd& x)
        {
            const VecNd th = x.template head<N>();
            const VecKd bb = x.template tail<K>();
            return data_energy(m, S(th + B_cam * bb), S(th + B_L * bb), S(th), f, camera, lidar);
        };
        const auto wrapped = [](VecXd d) { if constexpr (kYaw >= 0) d(kYaw) = std::remainder(d(kYaw), 2.0 * M_PI); return d; };
        const auto below_tol = [&](const VecNd& d)
        {
            bool ok = true;
            for (int k = 0; k < N; ++k) ok = ok and std::abs(d(k)) < (k == kYaw ? cfg.tol_rad : cfg.tol_m);
            return ok;
        };
        // The θ step of a solve from x̃ after apply_constraints (NaN if the solve is not finite). The iterate's angle is put
        // back on x̄'s branch, so the next solve's prior pull and the acceptance's prior energy see the same unwrapped
        // difference even when an iterate crosses ±π. θ is wrapped once, after the loop.
        const auto theta_step = [&](const VecXd& x_tilde, VecXd& xn, const pu::DampedStep<N, K>& st) -> VecNd
        {
            xn = x_tilde + st.delta;
            if (not (xn.allFinite() and st.y.allFinite())) return VecNd::Constant(std::numeric_limits<double>::quiet_NaN());
            State sn = State::from_vec(xn.template head<N>().template cast<float>());
            m.apply_constraints(sn);
            xn.template head<N>() = sn.vec().template cast<double>();
            if constexpr (kYaw >= 0) xn(kYaw) -= 2.0 * M_PI * std::round((xn(kYaw) - x_bar(kYaw)) / (2.0 * M_PI));
            return wrapped(xn - x_tilde).template head<N>();
        };
        const auto chans = [&] { return std::span<const pu::Channel<N, K>>(channels); };
        VecNd theta = m.state().vec().template cast<double>();            // θ₀ = the current mean (at birth: the data seed)
        VecKd b_post = b_;
        const VecNd theta0 = theta;                                        // GnTrace: the FD check at θ₀, b̄
        const VecKd b_bar0 = b_;
        VecNd trace_d0 = VecNd::Constant(std::numeric_limits<double>::quiet_NaN());   // undamped step / σ at this linearisation
        double J_cur, Ep_cur = 0.0;                                        // J and its prior part at the accepted iterate
        {   // J at x₀ = (θ₀, b̄): x₀ − x̄ is zero on b, and on θ unless this is the first update (class mean ≠ seed), where P̄'s θ
            // block is diagonal with zero cross terms; the pseudo-inverse is then exact (and 0 when x₀ = x̄).
            VecXd x0; x0 << theta, b_;
            const VecXd dx = wrapped(x0 - x_bar);
            const double Ep = dx.isZero(0.0) ? 0.0 : 0.5 * dx.dot(P_bar.completeOrthogonalDecomposition().solve(dx));
            J_cur = data_J(x0) + Ep;
            Ep_cur = Ep;
        }
        dbg_gn_energy_.push_back(J_cur);
        double lambda = 1e-3, nu = 2.0;
        VecNd d_last = VecNd::Zero();                                      // the last ACCEPTED θ step (zero if none)
        bool converged = false;
        int steps = 0;
        bool undamped_checked = false;                                     // the λ = 0 proposal at this linearisation was tested
        build(theta, b_);                                                  // step 2: x̃_0 = (θ_0, b̄)
        while (steps < cfg.iters_max)
        {
            VecXd x_tilde; x_tilde << theta, (consistent ? b_post : b_);   // consistent: (θ_k, b_k); restart: (θ_k, b̄)
            if (consistent and not undamped_checked)
            {   // The stop rule (spec 7c.8 rev 3.3.8): once per linearisation, the UNDAMPED proposal from the current θ; below
                // tol_sigma·σ_k on every θ component (σ_k = √P_kk, P = (𝟙 + P̄W)⁻¹P̄ here) ⇒ APPLY it when J does not rise, stop.
                undamped_checked = true;
                VecXd x0n;
                const auto st0 = pu::joint_step_damped<N, K>(x_tilde, x_bar, P_bar, chans(), theta_only, 0.0);
                const VecNd d0 = theta_step(x_tilde, x0n, st0);
                const MatJd P_lin = pu::joint_cov<N, K>(P_bar, chans(), theta_only);
                bool small = d0.allFinite();
                for (int k = 0; k < N and small; ++k)
                    small = std::abs(d0(k)) < cfg.tol_sigma * std::sqrt(std::max(0.0, P_lin(k, k)));
                if (cfg.gn_trace > 0)
                    for (int k = 0; k < N; ++k) trace_d0(k) = d0(k) / std::sqrt(std::max(1e-300, P_lin(k, k)));
                if (small and not cfg.reject_all_for_test)
                {
                    const double Ep0 = pu::prior_energy<N, K>(st0, x_bar + wrapped(x0n - x_bar), x_bar);
                    const double J0 = data_J(x0n) + Ep0;
                    if (cfg.gn_trace > 0)
                    {
                        GnTraceStep<N> r; r.step = steps + 1; r.lambda = 0.0; r.rho = std::numeric_limits<double>::quiet_NaN();
                        r.J_before = J_cur; r.J_after = J0; r.accepted = (J0 <= J_cur) ? 1 : 0; r.d = d0; r.d0_sigma = trace_d0;
                        dbg_gn_steps_.push_back(r);
                    }
                    if (J0 <= J_cur)
                    {
                        ++steps;
                        J_cur = J0; Ep_cur = Ep0; d_last = d0;
                        theta = x0n.template head<N>(); b_post = x0n.template tail<K>();
                        dbg_iterates_.push_back(theta.template cast<float>());
                        dbg_gn_energy_.push_back(J_cur);
                    }
                    converged = true;
                    break;
                }
            }
            ++steps;
            const auto st = pu::joint_step_damped<N, K>(x_tilde, x_bar, P_bar, chans(), theta_only, lambda, /*damp_latents*/ consistent);
            VecXd xn;
            const VecNd d = theta_step(x_tilde, xn, st);
            bool accept = false;
            double rho_gain = 0.0;
            GnTraceStep<N> trace_row; trace_row.step = steps; trace_row.lambda = lambda; trace_row.J_before = J_cur; trace_row.d = d;
            trace_row.d0_sigma = trace_d0;
            trace_d0.setConstant(std::numeric_limits<double>::quiet_NaN());   // the d0 belongs to the first step of a linearisation
            if (d.allFinite())
            {
                const double Ep_new = pu::prior_energy<N, K>(st, x_bar + wrapped(xn - x_bar), x_bar);
                const double J_new = data_J(xn) + Ep_new;
                trace_row.J_after = J_new;
                if (consistent)
                {   // the quadratic model's decrease: Ep_k − Ep_new + Gᵀδ − ½δᵀWδ (G, W of this linearisation, undamped)
                    MatJd Wj; VecXd Gj;
                    pu::detail::information<N, K>(chans(), theta_only, Wj, Gj);
                    const double pred = Ep_cur - Ep_new + Gj.dot(st.delta) - 0.5 * st.delta.dot(Wj * st.delta);
                    rho_gain = pred > 0.0 ? (J_cur - J_new) / pred : -1.0;
                    accept = rho_gain > 0.0;
                    trace_row.pred = pred;
                }
                else accept = J_new < J_cur;
                accept = accept and not cfg.reject_all_for_test;
                if (accept) { J_cur = J_new; Ep_cur = Ep_new; d_last = d; }
            }
            if (cfg.gn_trace > 0)
            {
                trace_row.rho = rho_gain; trace_row.accepted = accept ? 1 : 0;
                dbg_gn_steps_.push_back(trace_row);
            }
            if (not accept)
            {
                if (not consistent and d.allFinite() and below_tol(d) and not undamped_checked)
                {   // (the restart scheme) a rejected sub-tolerance step: converged only if the UNDAMPED proposal from this θ is
                    // below the tolerance (one extra solve per linearisation)
                    undamped_checked = true;
                    VecXd x0n;
                    const auto st0 = pu::joint_step_damped<N, K>(x_tilde, x_bar, P_bar, chans(), theta_only, 0.0);
                    const VecNd d0 = theta_step(x_tilde, x0n, st0);
                    converged = d0.allFinite() and below_tol(d0);
                    if (converged) break;
                }
                if (consistent) { lambda *= nu; nu *= 2.0; }                // Nielsen
                else lambda *= 4.0;                                        // restart scheme; re-solve on the same linearisation
                continue;
            }
            converged = not consistent and below_tol(d);                  // the restart scheme only; see the stop rule above
            if (consistent)
            {
                const double c = 2.0 * rho_gain - 1.0;
                lambda *= std::max(1.0 / 3.0, 1.0 - c * c * c);
                nu = 2.0;
            }
            else lambda /= 4.0;
            theta  = xn.template head<N>();
            b_post = xn.template tail<K>();                                // the accepted iterate's b is the posterior b
            dbg_iterates_.push_back(theta.template cast<float>());
            dbg_gn_energy_.push_back(J_cur);
            if (converged) break;
            build(theta, consistent ? b_post : b_);                        // step 2 at θ_{k+1}
            undamped_checked = false;
        }
        last_gn_iters_ = steps; last_gn_converged_ = converged; last_lambda_ = lambda; last_gn_step_ = d_last;
        State s = State::from_vec(theta.template cast<float>());
        m.apply_constraints(s);
        m.canonicalize(s);
        m.set_state(s);
        const VecNd theta_post = m.state().vec().template cast<double>();

        // Step 4 at the returned θ (consistent: at θ + B b_post; restart: b̄), plus the last ACCEPTED step's outer product on θθ
        // (zero if none): a frame that stopped while still moving carries its last move as uncertainty.
        build(theta_post, consistent ? b_post : b_);
        P_ = pu::joint_cov<N, K>(P_bar, chans(), theta_only);
        P_.template topLeftCorner<N, N>() += d_last * d_last.transpose();
        if (cfg.gn_trace >= 2 or (cfg.gn_trace == 1 and not converged))
        {   // GnTrace: does the solver's G / W match the energy J it accepts against? Per channel, the raw −bd, the
            // marginalised −g, central differences of data_energy with only that channel's state perturbed, and the
            // information diagonal against the second difference. Channel 3: the joint θ gradient against J along θ.
            const auto fd_check = [&](const VecNd& th, const VecKd& bb, int at)
            {
                MatN Ic, IL, It; VecN gc, gL, gt;
                const VecNd sc0 = th + B_cam * bb, sl0 = th + B_L * bb;
                accumulate_channels(m, S(th), S(sc0), S(sl0), f, camera, lidar, Ic, gc, IL, gL, It, gt);
                const auto mc = pu::marginalise_fresh<N>(Ic, gc, Sf_cam);
                const auto mL = pu::marginalise_fresh<N>(IL, gL, Sf_L);
                const double E0 = data_energy(m, S(sc0), S(sl0), S(th), f, camera, lidar);
                for (int ch = 0; ch < 3; ++ch)
                {
                    if (ch == 0 and not camera) continue;
                    if (ch == 1 and not lidar) continue;
                    for (int k = 0; k < N; ++k)
                    {
                        const auto E = [&](double h)
                        {
                            VecNd a = sc0, l = sl0, t = th;
                            (ch == 0 ? a : ch == 1 ? l : t)(k) += h;
                            return data_energy(m, S(a), S(l), S(t), f, camera, lidar);
                        };
                        GnTraceFd r; r.at = at; r.channel = ch; r.k = k;
                        const VecN& g = ch == 0 ? gc : ch == 1 ? gL : gt;
                        const MatN& I = ch == 0 ? Ic : ch == 1 ? IL : It;
                        r.grad_an   = -double(g(k));
                        r.grad_marg = ch == 0 ? -mc.g(k) : ch == 1 ? -mL.g(k) : r.grad_an;
                        const double ep3 = E(1e-3), em3 = E(-1e-3);
                        r.grad_fd3 = (ep3 - em3) / 2e-3;
                        r.grad_fd4 = (E(1e-4) - E(-1e-4)) / 2e-4;
                        r.curv_an  = double(I(k, k));
                        r.curv_fd  = (ep3 - 2.0 * E0 + em3) / 1e-6;
                        dbg_gn_fd_.push_back(r);
                    }
                }
                build(th, bb);
                MatJd Wj; VecXd Gj;
                pu::detail::information<N, K>(chans(), theta_only, Wj, Gj);
                for (int k = 0; k < N; ++k)
                {
                    const auto E = [&](double h) { VecXd x; x << th, bb; x(k) += h; return data_J(x); };
                    GnTraceFd r; r.at = at; r.channel = 3; r.k = k;
                    r.grad_an = r.grad_marg = -Gj(k);
                    const double ep3 = E(1e-3), em3 = E(-1e-3);
                    r.grad_fd3 = (ep3 - em3) / 2e-3;
                    r.grad_fd4 = (E(1e-4) - E(-1e-4)) / 2e-4;
                    r.curv_an  = Wj(k, k);
                    VecXd x0; x0 << th, bb;
                    r.curv_fd  = (ep3 - 2.0 * data_J(x0) + em3) / 1e-6;
                    dbg_gn_fd_.push_back(r);
                }
            };
            fd_check(theta0, b_bar0, 0);
            fd_check(theta_post, consistent ? b_post : b_, 1);
            build(theta_post, consistent ? b_post : b_);                   // restore the final linearisation for dbg_Id_* below
        }
        b_ = b_post;
        m.set_covariance(Sigma_theta());
        dbg_Id_th_ = Id_th; dbg_Id_cam_ = Id_cam; dbg_Id_L_ = Id_L;
        if (lidar)
        {   // the beam log (spec 7c.8 a0 + stated limit 1), at the returned θ and what the rays measure there (θ + B_L b)
            last_beam_staged_ = static_cast<int>(m.beam_rays(f).endpoints.size());
            const State s_L = State::from_vec((theta_post + B_L * b_post).template cast<float>());
            last_gap_rays_ = rc::ai::beam::count_gap_rays(m.beam_parts(s_L), m.beam_rays(f), m.beam_config());
        }
        first_update_done_ = true;                                         // the class prior at the class mean has entered, once
        birth_pos_excess_.setZero();                                       // the birth spread entered with it, once (rev 3.3.14)

        if (not camera or f.points.empty()) return 0.0f;
        // Per-point energy at what the camera MEASURES, θ + B_cam b (posterior b), not at the state θ — the engine's surface
        // energy on the biased quantity. Diagnostic.
        const State s_cam = State::from_vec((theta_post + B_cam * b_).template cast<float>());
        const float sig2 = m.sigma2();
        const int   n_prims = m.n_prims();
        double energy = 0.0;
        for (std::size_t i = 0; i < f.points.size(); ++i)
        {
            const float R = R_of(f, i, sig2);
            const auto  r = m.responsibilities(f.points[i], s_cam, R);
            const float omega = model_share(m, f.points[i], s_cam, R);
            for (int prim = 0; prim < n_prims; ++prim)
            {
                const float d = m.sdf_prim(f.points[i], s_cam, prim);
                energy += 0.5 * (omega * r[prim]) * d * d / R;
            }
        }
        return static_cast<float>(energy / static_cast<double>(f.points.size()));
    }

    // ── Diagnostics of the last update ───────────────────────────────────────────────────────────────────────────────
    const std::vector<VecNf>&           dbg_iterates()  const { return dbg_iterates_; }   // ACCEPTED θ iterates
    const std::vector<double>&          dbg_gn_energy() const { return dbg_gn_energy_; }  // J at θ₀ and after every accepted step
    const std::vector<GnTraceStep<N>>&  dbg_gn_steps()  const { return dbg_gn_steps_; }
    const std::vector<GnTraceFd>&       dbg_gn_fd()     const { return dbg_gn_fd_; }
    long   gn_update_serial()  const { return gn_update_serial_; }
    int    last_gn_iters()     const { return last_gn_iters_; }
    bool   last_gn_converged() const { return last_gn_converged_; }
    double last_lambda()       const { return last_lambda_; }
    const VecNd& last_gn_step() const { return last_gn_step_; }
    const MatNf& dbg_Pbar_tt() const { return dbg_Pbar_tt_; }
    const MatNf& dbg_Id_th()   const { return dbg_Id_th_; }
    const MatNf& dbg_Id_cam()  const { return dbg_Id_cam_; }
    const MatNf& dbg_Id_L()    const { return dbg_Id_L_; }
    int    last_beam_staged() const { return last_beam_staged_; }
    int    last_gap_rays()    const { return last_gap_rays_; }
    double last_beam_ms()     const { return beam_.ms(); }

private:
    static float R_of(const Frame& f, std::size_t i, float sig2) { return (i < f.R.size() and f.R[i] > 0.0f) ? f.R[i] : sig2; }
    static typename rc::ai::beam::BeamChannel<N>::Key key_of(const State& s)
    {
        const VecNf v = s.vec();
        typename rc::ai::beam::BeamChannel<N>::Key k{};
        for (int i = 0; i < N; ++i) k[i] = v(i);
        return k;
    }

    // The camera points' GN information (the recursive-Laplace accumulate loop, unsaturated). With neighbours each point's
    // responsibilities are scaled by the model's share ω of the neighbour mixture: the EM identity of neighbour_mix's energy.
    void accumulate_points(const M& m, const State& s, const Frame& f, MatNf& Id, VecNf& bd) const
    {
        const float sig2 = m.sigma2();
        const int   n_prims = m.n_prims();
        for (std::size_t i = 0; i < f.points.size(); ++i)
        {
            const float R = R_of(f, i, sig2);
            const auto  r = m.responsibilities(f.points[i], s, R);
            const float omega = model_share(m, f.points[i], s, R);      // 1 exactly without neighbours
            for (int prim = 0; prim < n_prims; ++prim)
            {
                const float w = omega * r[prim] / R;
                if (w < 1e-6f) continue;
                const float d = m.sdf_prim(f.points[i], s, prim);
                const VecNf J = m.sdf_jacobian(f.points[i], s, prim);
                Id.noalias() += w * (J * J.transpose());
                bd.noalias() += -w * J * d;
            }
        }
    }

    // ── G3: the neighbours' foreign density on the camera points (foreign_box.h) ──
    // One point's energy under the neighbour mixture, E = −log((1 − ε)·e^{−E_m} + ε·Σ_k g_k(p)), from the model's own
    // E_m = point_energy; *omega (when given) = the model's share (1 − ε)e^{−E_m} / ((1 − ε)e^{−E_m} + ε Σ g_k), by which
    // every model responsibility is scaled. No neighbour this update ⇒ (E_m, 1): the model's mixture, bit for bit.
    double neighbour_mix(double E_m, const Eigen::Vector3f& p, float R, double* omega) const
    {
        if (omega) *omega = 1.0;
        if (nb_.empty()) return E_m;
        const Eigen::Vector3d pd = p.template cast<double>();
        double G = 0.0;
        for (const auto& b : nb_) G += rc::ai::beam::foreign_point_density(b, pd, double(R));
        const double a = std::log1p(-nb_eps_) - E_m;                  // log of the model's term
        if (not (G > 0.0)) return -a;                                  // every neighbour's density underflowed: ω = 1
        const double c = std::log(nb_eps_ * G);                        // log of the neighbours' term
        const double mx = std::max(a, c);
        const double lse = mx + std::log(std::exp(a - mx) + std::exp(c - mx));
        if (omega) *omega = std::exp(a - lse);
        return -lse;
    }
    float model_share(const M& m, const Eigen::Vector3f& p, const State& s, float R) const
    {
        if (nb_.empty()) return 1.0f;
        double omega = 1.0;
        neighbour_mix(m.point_energy(p, s, R), p, R, &omega);
        return float(omega);
    }
    // The neighbours of the running update: set from FrameLatents at its start (the usable ones, Review Focus 5), cleared
    // on every exit, so the public channel/energy calls outside an update see none.
    struct NeighbourScope
    {
        const PersistentEngine& e;
        NeighbourScope(const PersistentEngine& eng, const FrameLatents<N>& fl) : e(eng)
        {
            e.nb_.clear();
            e.nb_eps_ = fl.mask_error_rate;
            if (not (fl.mask_error_rate > 0.0 and fl.mask_error_rate < 1.0)) return;   // ε ∉ (0, 1): no foreign mass to give
            for (const auto& b : fl.foreign_boxes)
                if (rc::ai::beam::usable(b)) e.nb_.push_back(b);
        }
        ~NeighbourScope() { e.nb_.clear(); e.nb_eps_ = 0.0; }
        NeighbourScope(const NeighbourScope&) = delete;
        NeighbourScope& operator=(const NeighbourScope&) = delete;
    };

    // The model's optional camera extra terms: none when absent.
    static void camera_extra(const M& m, const State& s, const Frame& f, MatNf& Id, VecNf& bd, double* E)
    {
        if constexpr (requires { m.accumulate_camera_extra(s, f, Id, bd, E); })
            m.accumulate_camera_extra(s, f, Id, bd, E);
    }

    // The model's optional solver constants: SolverConfig{} when absent.
    static SolverConfig solver_of(const M& m)
    {
        if constexpr (requires { { m.solver() } -> std::convertible_to<SolverConfig>; })
            return m.solver();
        else
            return SolverConfig{};
    }

    // Whether this frame's staged ray-set measures (rc::ai::beam::beam_readiness), logging a refusal once per engine. A model
    // whose LiDAR override handles the channel is ready by its own account.
    bool lidar_ready(const M& m, const Frame& f) const
    {
        if (lidar_override(m, m.state(), f, nullptr, nullptr, nullptr)) return true;
        namespace bm = rc::ai::beam;
        const bm::BeamRays& rays = m.beam_rays(f);
        switch (bm::beam_readiness(rays, m.beam_config()))
        {
            case bm::Readiness::kMeasured: return true;
            case bm::Readiness::kEmpty:    return false;
            case bm::Readiness::kInconsistent:
                if (not beam_inconsistent_logged_)
                    std::fprintf(stderr, "PersistentEngine: beam ray-set refused — m (%zu) / p_f (%zu) do not match the %zu staged "
                                 "rays (the LiDAR channel does not measure)\n", rays.m.size(), rays.p_f.size(), rays.endpoints.size());
                beam_inconsistent_logged_ = true;
                return false;
            case bm::Readiness::kUnready:
                if (not beam_unready_logged_)
                    std::fprintf(stderr, "PersistentEngine: beam not ready — the model's BeamConfig was not built by "
                                 "BeamConfig::make; the staged rays are NOT used (the LiDAR channel does not measure)\n");
                beam_unready_logged_ = true;
                return false;
        }
        return false;
    }

    // The model's optional LiDAR override (a test hook): true when it handled the channel.
    static bool lidar_override(const M& m, const State& s, const Frame& f, MatNf* Id, VecNf* bd, double* E)
    {
        if constexpr (requires { { m.accumulate_lidar_override(s, f, Id, bd, E) } -> std::convertible_to<bool>; })
            return m.accumulate_lidar_override(s, f, Id, bd, E);
        else
            return false;
    }

    // [B_chain(ℓ) | 𝟙 on the channel's own latents | 0]: the camera measures θ + B_chain e + b_cam + u_cam + s_cam, the rays
    // θ + B_chain e + b_L + u_L + s_L.
    Eigen::Matrix<double, N, K> selector(const M& m, bool camera) const
    {
        Eigen::Matrix<double, N, K> B = Eigen::Matrix<double, N, K>::Zero();
        B.template block<N, 3>(0, kChain) = m.chain_jacobian(m.state(), lever_.l);
        B.template block<N, N>(0, camera ? kCam : kL) = Eigen::Matrix<double, N, N>::Identity();
        B.template block<N, N>(0, camera ? kFc : kFL) = Eigen::Matrix<double, N, N>::Identity();
        B.template block<N, N>(0, camera ? kSc : kSL) = Eigen::Matrix<double, N, N>::Identity();
        return B;
    }

    MatJd P_ = MatJd::Zero();
    VecKd b_ = VecKd::Zero();
    VecKd sig_prev_ = VecKd::Zero();                 // σ each latent was last predicted at (0 ⇒ the next predict is a new source)
    rc::ai::view::ChainLever lever_;
    Eigen::Matrix3d last_sigma_pose_ = Eigen::Matrix3d::Zero();
    bool joint_init_ = false;
    bool first_update_done_ = false;                 // the class prior at the class mean has entered
    Eigen::Matrix2f birth_pos_excess_ = Eigen::Matrix2f::Zero();
    rc::ai::beam::BeamChannel<N> beam_;
    mutable bool beam_unready_logged_ = false;       // "beam not ready" printed (once per engine)
    mutable bool beam_inconsistent_logged_ = false;  // "beam ray-set refused" printed (once per engine)
    mutable std::vector<ForeignBox> nb_;              // G3: this update's usable neighbours (NeighbourScope)
    mutable double nb_eps_ = 0.0;                      // and ε, the prior weight of their foreign mass
    std::vector<VecNf> dbg_iterates_;
    std::vector<double> dbg_gn_energy_;
    std::vector<GnTraceStep<N>> dbg_gn_steps_;
    std::vector<GnTraceFd> dbg_gn_fd_;
    long   gn_update_serial_ = 0;
    int    last_gn_iters_ = 0;
    bool   last_gn_converged_ = false;
    double last_lambda_ = 0.0;
    VecNd  last_gn_step_ = VecNd::Zero();
    MatNf  dbg_Pbar_tt_ = MatNf::Zero();
    MatNf  dbg_Id_th_ = MatNf::Zero();
    MatNf  dbg_Id_cam_ = MatNf::Zero();
    MatNf  dbg_Id_L_ = MatNf::Zero();
    int    last_beam_staged_ = 0;
    int    last_gap_rays_ = 0;
};

}  // namespace rc::ai::engine
