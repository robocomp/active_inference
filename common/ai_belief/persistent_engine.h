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
 *   - the GnTrace diagnostic (one row per GN step, and a finite-difference check of every channel's gradient);
 *   - the offer gain (spec shared-engine G5): predicted_gain, the θ-marginal entropy drop of a hypothetical camera look from
 *     a standpoint under this same joint covariance update, and last_delivered_nats, the same quantity for the last update.
 *
 * The MODEL (concept PersistentModel) supplies the shape: the camera-point hooks of recursive_laplace.h, its camera-only and
 * θ-only extra terms (each with its energy, the GN acceptance objective), its beam parts, how a robot pose error moves θ,
 * the class prior, the constraints. Conventions the engine relies on: θ(0), θ(1) are the room-frame centre (cx, cy) — the
 * lever is taken there and the birth position covariance lands there — and θ(M::kYaw) is an angle (wrapped; −1 = none).
 *
 * All solves in double; fixed-size Eigen throughout. The solver's tolerances are solver constants (SolverConfig), and the only
 * case split on a latent is σ_prev = 0 (a new error source). NOT threshold-free: three ⚠ THRESHOLDs are flagged in place, each
 * with what it decides and why no model-level term replaces it yet (CLAUDE.md):
 *   - the wall-identity gate, same_physical_wall: χ² ≤ the quantile at P(|z| ≤ 3) for the DOF count (transport δ vs a new source;
 *     whether the attachment priors re-enter);
 *   - the GN-cap switch in update(): an update ending at the iteration cap carries P̄ instead of the Laplace Σ;
 *   - the Σ ≤ Σ₀ cap in predict_theta_on: the predict never lets a θ variance exceed the class prior's (G1x closed list V5).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "beam_channel.h"
#include "foreign_box.h"
#include "offer_view.h"
#include "persistent_update.h"
#include "viewpoint.h"

namespace rc::ai::engine
{

using rc::ai::beam::ForeignBox;   // a neighbour's published belief (foreign_box.h; spec shared-engine G3)
using rc::ai::offer::Standpoint;  // an offered standpoint and the camera it puts to work (offer_view.h; G5)
using rc::ai::offer::OfferSensors;

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
    // The camera's room position (floor plane) at this update (G5): where the offer gain's ρ path starts. Absent ⇒ robot_xy.
    std::optional<Eigen::Vector2f> camera_xy;
    // F1 (task-F1): the sensors' viewpoints about the object and the latents' anchors BEFORE this update, as the fitter staged
    // them (rc::ai::view::view_stage). The engine commits them with this update's ρ for the offer (predicted_gain). off ⇒ the
    // offer's ρ is the straight path from the last update's position, as before F1.
    rc::ai::view::ViewState view;
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

// Inferred process noise (the HISTORY stage, MODEL_HISTORY.md §3; plan rev 2 F1.2), optional model hook volatility(). Between
// updates θ takes random-walk steps w ~ N(0, Q), Q = diag(e^ω) unknown and per object. Q's conjugate running estimate is the
// mean of the EM statistic of the steps (see update()), with the configured Q₀ (the model's process_noise_diag) as a prior
// worth n₀ = 1/σ_ω² pseudo-observations and a memory λ per measured update. infer = false: Q = Q₀ throughout.
// With a permanence prior (jump_var, jump_prob below) q₀ = π·M is the class's mean per-step motion variance and Q is inferred
// AROUND it; the Gaussian statistic alone would make a small q near-absorbing (gain (q/Σ̄)²), the spike-and-slab one does not.
struct VolatilityConfig
{
    bool   infer = false;
    double sigma_omega = 2.0;            // the hyperprior's width; its strength n₀ = 1/σ_ω² in pseudo-observations
    double memory = 0.995;               // λ per measured update (~200 updates): long enough to earn stiffness, short to recant
    // PERMANENCE (owner ruling 11-25; MODEL_HISTORY.md §3, CONCEPT_AGENT_LIFECYCLE.md §4.1 "stiffness is never a lock"). A passive
    // object of the class is MOVED now and then: per predict step with probability jump_prob, by a step of variance jump_var(i)
    // per θ component (a spike-and-slab process noise, the class manifest's [prior.permanence]). Its mean per-step variance is
    // the hyperprior mean q₀ = jump_prob · jump_var (= the model's process_noise_diag). When jump_var has N entries the EM
    // statistic is the spike-and-slab one (learn_volatility): each update weighs "moved in this interval" against "stayed" by
    // the innovation's likelihood under both, so a shove is recognised at ANY q — the Gaussian statistic's gain (q/Σ̄)² makes a
    // small q near-absorbing — and a static object pulls Q back toward q₀ through the hyperprior. Empty ⇒ the Gaussian
    // statistic below (item 2), for models with no permanence prior.
    std::vector<double> jump_var;
    double jump_prob = 0.0;
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
//   prior_cov_diag(), process_noise_diag()     Σ₀ and the CONFIGURED Q₀ diagonals (the engine's Q is Q₀, or e^ω once the
//                                              volatility learns: process_noise());
//   state(), set_state(s), covariance(), set_covariance(Σ)   the model's θ and its float N×N mirror of P_θθ (the model's
//                                              own terms may read the mirror: the engine sets it to the predicted P̄_θθ
//                                              before the first linearisation, and to the posterior after);
//   shape_spread()                             Δ, the class's layer spread per θ component (spec shared-engine G4): the
//                                              range over which the layer a sensor sees may lie; the shape misfit is
//                                              uniform over it, so its marginal is Δ²/12 (0 ⇒ no layering on that DOF).
// Optional (detected by requires-expressions; absent ⇒ the stated default):
//   accumulate_camera_extra(s, f, I, b, E)     the camera channel's non-point terms, same energy convention (absent: none);
//   solver()                                   SolverConfig (absent: SolverConfig{});
//   volatility()                               VolatilityConfig (absent: infer = false, Q = Q₀ throughout);
//   kWallLatents = 1, wall_anchor(f) -> WallAnchor, accumulate_wall_prior(s, f, I, b, E)   the wall latent and the wall priors
//                                              it carries (see WallAnchor; absent: no wall latent, K unchanged);
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

// Optional (G5): the context a look's extra terms read, kept per update instead of a copy of the whole frame.
//   typename M::OfferContext                    what the camera-extra and θ-only terms read from a frame besides its points;
//   offer_context(f) -> OfferContext            taken from each measured frame;
//   offer_frame(ctx) -> Frame                   a frame carrying only that context (built once per planning cycle);
//   offer_look(Frame&, points, R)               put one look's points (R per point) into it, with whatever the model derives
//                                               from them (countable: lidar_only = no points).
// Absent ⇒ the offer has no extra terms.
template <class M>
concept HasOfferContext =
    requires { typename M::OfferContext; }
    and requires(const M& cm, const typename M::Frame& f, typename M::Frame& g, const typename M::OfferContext& c,
                 std::span<const Eigen::Vector3f> p, std::span<const float> r) {
        { cm.offer_context(f) } -> std::same_as<typename M::OfferContext>;
        { cm.offer_frame(c) } -> std::same_as<typename M::Frame>;
        cm.offer_look(g, p, r);
    };
// ── The wall latent (plan rev 2 F1.4; Fable order review §2.2; coordinator rulings 11-24) ──────────────────────────────────────
// A model with M::kWallLatents = 1 carries ONE more ρ = 1 latent δ: the error of the staged wall line along its inward normal n
// (the true wall is the staged line + δ·n), marginal σ_wall² from the room, born when the instance first associates with a wall
// edge. Moving the wall by δ·n equals moving the object by −δ·n relative to it, so the model's wall PRIORS (flush / stand-off,
// parallel; NO separate no-cross probit: a stand-off prior N(gap − δ; μ ≥ 0, σ) already says "at or off the wall", and a
// probit on top counted σ_wall twice — coordinator ruling 11-24) read it through the selector column (−n_x, −n_y, 0, …) on
// (cx, cy): they are evaluated at θ + B_w b. They enter ONCE, into the joint prior, at that first association (re-linearised by that update's GN, then frozen
// in P); afterwards only data (points, rays) update θ and δ. A fixed relation re-entered as a measurement on every update is
// what pinned the parked toilet to the millimetre (engine1 diagnosis item 2).
// Identity (F1 round 6.3): GEOMETRIC, not the outline's edge index (room_concept may renumber its outline, and the nearest edge
// may flip between two walls): two staged lines are the same physical wall when they are in the same room, face the same way and
// agree within 3σ in (offset along n, angle): Δo²/(σ_a² + σ_b²) + Δφ²/((σ_a/L_a)² + (σ_b/L_b)²) ≤ 9 — the "a few σ_wall" gate
// of the ruling, on the lines' own published σ. The same wall as δ's: its line move is TRANSPORTED (δ's mean shifted by the move
// along n, no information). Another wall: a new δ source (the old one marginalised out), and the wall priors enter only if that
// wall is not already in the instance's set of walls entered (matched the same way; its stored line is transported too) — so
// the priors enter ONCE per physical wall, through renumbering and through A → B → A alternation.
// KNOWN APPROXIMATION (option A, coordinator ruling): the camera mixture's wall component and the beam's foreign density depend
// on δ directly, not through θ; they are evaluated at the wall shifted by δ's CURRENT posterior mean (wall_offset_mean), so
// they ignore δ's variance. Full coupling (a ∂E/∂δ column in those channels) is deferred.
// Models without kWallLatents: K unchanged, bit-identical.
struct WallAnchor
{
    bool ok = false;                                   // a wall is staged in this frame
    bool keyed = false;                                // (room, edge) identify it
    std::uint64_t room = 0;
    int edge = -1;
    Eigen::Vector2d p = Eigen::Vector2d::Zero();       // a point on the staged line (room frame)
    Eigen::Vector2d n = Eigen::Vector2d::UnitX();      // unit inward normal
    double sigma_m = 0.0;                              // the room's std of the line along n (the latent's marginal)
    double length_m = 0.0;                             // the edge's length (its direction is known to σ/L; 0 ⇒ direction unknown)
};

// The wall-identity gate's χ² quantile at P(|z| ≤ 3) = erf(3/√2) for 1 or 2 DOF (closed forms: 3² and −2 ln erfc(3/√2)).
inline double wall_identity_chi2_gate(int dof)
{
    return dof == 1 ? 9.0 : -2.0 * std::log(std::erfc(3.0 / std::sqrt(2.0)));
}

// The same physical wall (see above): same room (when both are known), facing the same way, within 3σ in offset and angle.
// ⚠ THRESHOLD (flagged, CLAUDE.md; final review I2). WHAT IT DECIDES: a YES transports δ (its mean shifted by the staged line's
// move, no information) and, for walls_entered_, means the attachment priors (flush / stand-off, parallel) do NOT re-enter; a NO
// starts a new δ source (the old one marginalised out) and, for a wall not yet entered, enters the priors again. WHY KEPT: the
// model-level alternative — a mixture over "same wall / another wall" with the identity's posterior weighting both the transport
// and the priors — needs a per-update hypothesis over the wall set that the ρ = 1 δ latent does not carry; post-12-02.
// THE QUANTILE: the statistic d² is χ² with 1 DOF (offset only: a length unknown) or 2 DOF (offset + angle). BASE used 9 for both
// — 3σ (99.73 %) for 1 DOF but 98.9 % for 2. It is now the χ² quantile at ONE probability, P(|z| ≤ 3) = erf(3/√2) = 0.99730, for
// the actual DOF count: 9 (1 DOF, exactly) and −2 ln(1 − P) = 11.83 (2 DOF). Replay of the four graded box-class dumps
// (p1r8_bed, p1r8_nightstand, k1_fridge, k2b_chair) is bit-identical before / after (final-fix-wave-report.md).
inline bool same_physical_wall(const WallAnchor& a, const WallAnchor& b)
{
    if (a.room != 0 and b.room != 0 and a.room != b.room) return false;
    const double c = a.n.dot(b.n);
    if (not (c > 0.0)) return false;
    const double dphi = std::atan2(a.n.x() * b.n.y() - a.n.y() * b.n.x(), c);
    const double doff = (a.p - b.p).dot(b.n);
    const double so2 = a.sigma_m * a.sigma_m + b.sigma_m * b.sigma_m;
    double d2 = so2 > 0.0 ? doff * doff / so2 : (doff == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    int dof = 1;
    if (a.length_m > 0.0 and b.length_m > 0.0)
    {
        const double sp2 = std::pow(a.sigma_m / a.length_m, 2) + std::pow(b.sigma_m / b.length_m, 2);
        d2 += sp2 > 0.0 ? dphi * dphi / sp2 : (dphi == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        dof = 2;
    }
    return d2 <= wall_identity_chi2_gate(dof);
}

namespace detail
{
// The move evidence of one update (F1 round 6c; round 7 A2: information form). The data of the update are a Gaussian factor on x
// about the predicted joint state x̄ = (θ̄, b̄), with information Λ and gradient η (every channel; linearised at the stayed
// posterior mode, round 7 A1, see update()). Under "stayed" x ~ N(x̄, P̄); under "moved" the θ block also takes a step v ~ N(0, M),
// i.e. x ~ N(x̄, P̄_m), P̄_m = P̄ + E M Eᵀ (E = the θ rows). Each hypothesis's evidence is
//   log Z(P) = ½ ηᵀ P (𝟙 + ΛP)⁻¹ η − ½ log det(𝟙 + ΛP),
// so log BF = log Z(P̄_m) − log Z(P̄), p = π BF / (π BF + 1 − π), and under "moved"
//   E[v | data] = M Eᵀ (𝟙 + ΛP̄_m)⁻¹ η,   Cov[v | data] = M − M Eᵀ (𝟙 + ΛP̄_m)⁻¹ Λ E M.
// 𝟙 + ΛP has eigenvalues ≥ 1 for PSD Λ, P (it is similar to 𝟙 + P^½ΛP^½), so nothing singular is ever inverted: a direction the
// data do not inform (Λ's null space) contributes exactly its η there, which a consistent linearisation makes 0, and a direction
// with no prior variance contributes 0. Round 6c's g-space form solved S_s = A P̄ Aᵀ + W; on bed_5 update 140 (p1_bed, a LiDAR-only
// frame with the depth h wholly uninformed) S_s had a 1e-91 eigenvalue, S_s⁻¹W came out at 1e30, log BF 1.3e62 and ω → +134,
// "DEGENERATE joint P". Same Bayes factor, conditioned the other way round.
template <int N>
struct JumpEvidence
{
    double log_bf = 0.0, p = 0.0;
    Eigen::Matrix<double, N, 1> v_mean = Eigen::Matrix<double, N, 1>::Zero();
    Eigen::Matrix<double, N, N> v_cov = Eigen::Matrix<double, N, N>::Zero();
};
template <int N, int NX>
inline JumpEvidence<N> jump_evidence(const Eigen::Matrix<double, NX, NX>& Lambda, const Eigen::Matrix<double, NX, 1>& eta,
                                     const Eigen::Matrix<double, NX, NX>& P_bar, const Eigen::Matrix<double, N, 1>& m_var, double pi)
{
    using MatX = Eigen::Matrix<double, NX, NX>;
    using VecX = Eigen::Matrix<double, NX, 1>;
    JumpEvidence<N> out;
    const MatX L = 0.5 * (Lambda + Lambda.transpose());
    const auto logZ = [&](const MatX& P, Eigen::PartialPivLU<MatX>& lu)
    {
        lu.compute(MatX::Identity() + L * P);
        double logdet = 0.0;
        for (int i = 0; i < NX; ++i) logdet += std::log(std::abs(lu.matrixLU()(i, i)));
        const VecX y = lu.solve(eta);                                   // (𝟙 + ΛP)⁻¹ η
        return 0.5 * eta.dot(P * y) - 0.5 * logdet;
    };
    MatX Pm = P_bar;
    for (int i = 0; i < N; ++i) Pm(i, i) += m_var(i);
    Eigen::PartialPivLU<MatX> lu_s, lu_m;
    const double ls = logZ(P_bar, lu_s), lm = logZ(Pm, lu_m);
    out.log_bf = lm - ls;
    const double lo = std::log(pi) + out.log_bf, lst = std::log1p(-pi), mx = std::max(lo, lst);
    out.p = std::exp(lo - mx) / (std::exp(lo - mx) + std::exp(lst - mx));
    const VecX ym = lu_m.solve(eta);
    out.v_mean = m_var.cwiseProduct(ym.template head<N>());
    const MatX YL = lu_m.solve(L);                                     // (𝟙 + ΛP̄_m)⁻¹ Λ
    const Eigen::Matrix<double, N, N> Mm = m_var.asDiagonal();
    out.v_cov = Mm - Mm * YL.template topLeftCorner<N, N>() * Mm;
    return out;
}

// K2a-L: the model's shape PROPORTIONS (M::kProportions; absent ⇒ 0). See the class comment at kProp.
template <class M> constexpr int proportion_latents()
{
    if constexpr (requires { { M::kProportions } -> std::convertible_to<int>; }) return M::kProportions;
    else return 0;
}
template <class M> constexpr int wall_latents()
{
    if constexpr (requires { { M::kWallLatents } -> std::convertible_to<int>; }) return M::kWallLatents;
    else return 0;
}
struct NoOfferContext {};
template <class M, bool = HasOfferContext<M>> struct OfferContextOf { using type = NoOfferContext; };
template <class M> struct OfferContextOf<M, true> { using type = typename M::OfferContext; };
}  // namespace detail

template <PersistentModel M>
class PersistentEngine
{
public:
    static constexpr int N = M::N;
    static constexpr int W = detail::wall_latents<M>();             // 0 or 1 wall latents (F1.4)
    static_assert(W == 0 or W == 1, "one wall latent at most");
    // K2a-L: NP shape-PROPORTION latents (M::kProportions), appended after the wall latent. A proportion is a fraction of the
    // box a part boundary sits at (the toilet's bowl top as a fraction of H): q = q₀ + b_q, q₀ = the class range's midpoint
    // (proportion_mean), b_q born at Δ²/12 (Δ = proportion_spread, the class range hi − lo; spec G4's uniform spread) the first
    // time a channel measures, then carried with ρ = 1 (the view never changes the object's proportions). It enters the
    // geometry DIRECTLY, not as a θ offset: each sensor channel carries (u, q) information (persistent_update.h Channel<N, K, NP>)
    // from the model's ∂sdf/∂q (sdf_jacobian_q) and its beam parts' Jq. Shape, not pose: no process noise (predict_theta is θ
    // only) and outside the move statistic (jump_evidence moves θ only). Δ = 0 keeps the latent exactly 0 with zero variance.
    // Required hooks when NP > 0: proportion_mean(), proportion_spread() (Eigen P-vectors, double), set_proportions(q) const
    // (the model evaluates its SDF, beam parts and carve at q from then on), proportions() (the q it evaluates at),
    // sdf_jacobian_q(p, s, prim) (float P-vector), proportion_extent() (H − z0 at the model's state: the band a proportion
    // scales), and M::kHeight (the θ component whose common-mode size std and view ρ the q common mode takes).
    static constexpr int NP = detail::proportion_latents<M>();
    // Per proportion THREE latents (fix round 1, I1): the object's q (shared by the sensors, ρ = 1) and, per sensor channel, a
    // per-view COMMON MODE of that channel on q — re-drawn with the channel's view ρ exactly as b_cam / b_L are for θ — so a
    // channel measures q + b_q,c: many frames from one view cannot pin q below that view's common mode, and a new view adds.
    static constexpr int K = 3 + 6 * N + W + 3 * NP;
    static constexpr int kWall = 3 + 6 * N;                          // the wall latent's index inside b (when W = 1)
    static constexpr int kProp = 3 + 6 * N + W;                      // the first proportion latent's index inside b (when NP > 0)
    static constexpr int kPropCam = kProp + NP, kPropL = kProp + 2 * NP;   // the camera's / LiDAR's common mode on q
    static constexpr int NC = N + NP;                                 // a sensor channel's coordinates: θ + B b, then q
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
    using VecPd = Eigen::Matrix<double, NP, 1>;
    using Chan  = rc::ai::persistent::Channel<N, K, NP>;
    // A sensor channel's information on q (NP > 0): the (u, q) cross block, the q block, and the q gradient (bd's sign convention).
    struct QInfo
    {
        Eigen::Matrix<double, N, NP> Iuq = Eigen::Matrix<double, N, NP>::Zero();
        Eigen::Matrix<double, NP, NP> Iqq = Eigen::Matrix<double, NP, NP>::Zero();
        Eigen::Matrix<double, NP, 1> gq  = Eigen::Matrix<double, NP, 1>::Zero();
    };

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
        set_q(m, b_);                                                     // the model's proportions follow b: q = q₀ (review I4)
        first_update_done_ = false;                                       // the class prior re-enters once, at the class mean
        vol_init_ = false;                                                // the volatility estimate is suspect too: back to Q₀
        wall_set_ = false;                                                // the wall priors re-enter once, at the next association
        walls_entered_.clear();
        m.set_covariance(Sigma_theta());
        return true;
    }

    // θ's predict on the θθ block of P: Σ ← Σ + q_scale·Q (F = I, static), then the clamp_to_prior congruence on θ's rows AND
    // columns of P, so the θ–b cross blocks scale with it and P stays PSD. The latents are predicted only on the update that
    // measures them (their ρ spans the time since their channel's last update).
    void predict(M& m, float q_scale)
    {
        ensure_joint(m.covariance());
        steps_since_update_ += double(std::max(0.0f, q_scale));     // the predict steps the next measured update's interval spans
        predict_theta(m, q_scale);
        m.set_covariance(Sigma_theta());
    }

    // δ's current posterior mean for a staged wall: transported to `a`'s line when it is the same edge; 0 for a wall the instance
    // has no latent for (a new source starts at 0). What the camera mixture and the beam's foreign density read (option A).
    double wall_offset_mean(const WallAnchor& a) const
    {
        if constexpr (W == 0) return 0.0;
        else
        {
            if (not (a.ok and wall_set_ and same_wall(a))) return 0.0;
            return b_(kWall) - (a.p - wall_.p).dot(a.n);
        }
    }
    bool wall_latent_set() const { return wall_set_; }
    const WallAnchor& wall_anchor() const { return wall_; }

    // Q per predict step: e^ω once the volatility has learned from an update, else the model's configured Q₀.
    VecNf process_noise(const M& m) const
    {
        if (not vol_init_) return m.process_noise_diag();
        return vol_omega_.array().exp().matrix().template cast<float>();
    }
    // ω = log Q per θ component (log Q₀ until the first update the volatility learns from).
    VecNd log_volatility(const M& m) const
    {
        if (vol_init_) return vol_omega_;
        return m.process_noise_diag().template cast<double>().array().log().matrix();
    }

    // Step 1 on the latents (spec 7c.7 a, b). e is predicted whenever a channel measures: a NEW SOURCE when it has no lever
    // yet or its previous Σ_pose was zero (ρ = 0 for that transition) — the lever is then taken at (θ's centre − robot);
    // otherwise the lever moves with the robot, ℓ ← ρ(ℓ − Δr) + (1 − ρ)(c − r), BEFORE the predict, and e is carried by
    // F = chain_transport(Δr) with correlation ρ_chain and fresh variance (1 − ρ²) Σ_pose. b_cam / b_L: F = I, per-DOF ρ,
    // Σ_now = diag(σ_now²), new source where σ_prev = 0, predicted only when their channel measures. u_cam / u_L: re-drawn
    // every update (rev 3.3.13). s_cam / s_L: ρ = 1, born at diag(Δ²/12) with Δ = shape_spread(), when their channel measures.
    void predict_latents(const M& m, const FrameLatents<N>& fl)
    {
        predict_latents_on(m, fl, P_, b_, sig_prev_, lever_, last_sigma_pose_);
    }
    // The same on explicit latent state (the offer gain predicts a copy of exactly these, G5).
    static void predict_latents_on(const M& m, const FrameLatents<N>& fl, MatJd& P_, VecKd& b_, VecKd& sig_prev_,
                                   rc::ai::view::ChainLever& lever_, Eigen::Matrix3d& last_sigma_pose_)
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
        if constexpr (NP > 0)
        {   // the shape proportions (K2a-L): born at Δ²/12 (a new source while σ_prev = 0), then ρ = 1 — the predict of a ρ = 1,
            // F = I latent leaves its mean and every row of P unchanged, so a carried proportion is simply not touched. Both
            // channels measure the SAME proportion (one object), so it is born whichever channel measures first.
            const VecPd spread = m.proportion_spread();
            for (int j = 0; j < NP; ++j)
            {
                const int r = N + kProp + j;
                if (sig_prev_(kProp + j) != 0.0 or not (spread(j) > 0.0)) continue;
                P_.row(r).setZero(); P_.col(r).setZero();
                P_(r, r) = spread(j) * spread(j) / 12.0;
                x(r) = 0.0;
                sig_prev_(kProp + j) = spread(j) / std::sqrt(12.0);
            }
            // the per-view common mode of each channel on q (fix round 1, I1). A common-mode error of the channel's SIZE along
            // the vertical, std σ_size,cm = √sig2_c[kHeight] (the channel's existing common-mode size std, what b_c carries on
            // H), moves a part boundary z = z0 + q·(H − z0) by that much, i.e. q by σ_q,cm = σ_size,cm / (H − z0). It is
            // predicted like b_c on θ: F = I, ρ = the channel's view ρ on that DOF, a new source while σ_prev = 0. A class with
            // Δ = 0 keeps it at 0 (no proportion: nothing to carry).
            const double ext = m.proportion_extent();
            const auto cm_block = [&](int off, const std::array<float, N>& rho, const std::array<float, N>& s2)
            {
                const double sg2 = std::max(0.0, double(s2[M::kHeight])) / (ext * ext);
                Eigen::VectorXd rv(NP), sv(NP);
                bool all_new = true;
                for (int j = 0; j < NP; ++j)
                {
                    const bool on = spread(j) > 0.0 and ext > 0.0;
                    const bool nw = sig_prev_(off + j) == 0.0;
                    rv(j) = nw ? 0.0 : double(rho[M::kHeight]);
                    sv(j) = on ? sg2 : 0.0;
                    all_new = all_new and nw;
                }
                pu::predict_block<N, K>(P_, x, off, NP, Eigen::MatrixXd::Identity(NP, NP), rv, sv.asDiagonal().toDenseMatrix(), all_new);
                for (int j = 0; j < NP; ++j) sig_prev_(off + j) = std::sqrt(sv(j));
            };
            if (camera) cm_block(kPropCam, fl.rho_cam, fl.sig2_cam);
            if (lidar)  cm_block(kPropL,   fl.rho_L,   fl.sig2_L);
        }
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
    // P > 0: the model evaluates at the proportions last set (set_q), and qc / qL receive the channels' q information.
    void accumulate_channels(const M& m, const State& s_th, const State& s_cam, const State& s_L, const Frame& f, bool camera,
                             bool lidar, MatNf& Id_cam, VecNf& bd_cam, MatNf& Id_L, VecNf& bd_L, MatNf& Id_th, VecNf& bd_th,
                             QInfo* qc = nullptr, QInfo* qL = nullptr, const VecKd* bb = nullptr) const
    {
        Id_cam.setZero(); bd_cam.setZero(); Id_L.setZero(); bd_L.setZero(); Id_th.setZero(); bd_th.setZero();
        if (qc) *qc = QInfo{};
        if (qL) *qL = QInfo{};
        if (camera)
        {
            if (bb) set_q_channel(m, *bb, true);                       // the camera measures q + b_q,cam
            accumulate_points(m, s_cam, f, Id_cam, bd_cam, qc);
            camera_extra(m, s_cam, f, Id_cam, bd_cam, nullptr);
        }
        if (bb and lidar) set_q_channel(m, *bb, false);                  // the rays measure q + b_q,L
        if (not lidar_override(m, s_L, f, &Id_L, &bd_L, nullptr) and lidar)
        {
            if constexpr (NP == 0)
                beam_.accumulate(key_of(m, s_L), [&] { return m.beam_parts(s_L); }, m.beam_rays(f), m.beam_config(), &Id_L, &bd_L);
            else
            {   // the beam on (θ, q): the parts carry Jq; split the NC-dimensional information into the θ and q blocks
                Eigen::Matrix<float, NC, NC> Ic = Eigen::Matrix<float, NC, NC>::Zero();
                Eigen::Matrix<float, NC, 1>  bc = Eigen::Matrix<float, NC, 1>::Zero();
                beam_.accumulate(key_of(m, s_L), [&] { return m.beam_parts(s_L); }, m.beam_rays(f), m.beam_config(), &Ic, &bc);
                Id_L += Ic.template topLeftCorner<N, N>();
                bd_L += bc.template head<N>();
                if (qL)
                {
                    qL->Iuq += Ic.template topRightCorner<N, NP>().template cast<double>();
                    qL->Iqq += Ic.template bottomRightCorner<NP, NP>().template cast<double>();
                    qL->gq  += bc.template tail<NP>().template cast<double>();
                }
            }
        }
        m.accumulate_extra(s_th, f, Id_th, bd_th, nullptr);
    }
    // The model's proportions at the latent mean bb (NP > 0): the object's q = q₀ + b_q. No-op for NP = 0.
    void set_q(const M& m, const VecKd& bb) const
    {
        if constexpr (NP > 0) m.set_proportions(VecPd(m.proportion_mean() + bb.template segment<NP>(kProp)));
    }
    // What channel c measures: q₀ + b_q + b_q,c.
    void set_q_channel(const M& m, const VecKd& bb, bool camera) const
    {
        if constexpr (NP > 0)
            m.set_proportions(VecPd(m.proportion_mean() + bb.template segment<NP>(kProp)
                                    + bb.template segment<NP>(camera ? kPropCam : kPropL)));
    }
    // A sensor channel on (θ + B b, q): its q information and Bq (the q rows of H_c select b_q).
    void add_q(Chan& c, const QInfo& qi, bool camera) const
    {
        if constexpr (NP > 0)
        {
            c.Iuq = qi.Iuq; c.Iqq = qi.Iqq; c.gq = qi.gq;
            c.Bq.setZero();
            for (int j = 0; j < NP; ++j) { c.Bq(j, kProp + j) = 1.0; c.Bq(j, (camera ? kPropCam : kPropL) + j) = 1.0; }
        }
    }

    // The energy whose gradients accumulate_channels builds, term by term, with the same per-frame gates: the camera points'
    // −log mixture likelihood and the camera extra terms at s_cam, the beam energy (or the override's) at s_L, the θ-only
    // terms at s_th. ∇E = −(bd_cam + bd_L + bd_th) up to the state dependence of the terms' weights, which the GN information
    // also holds fixed.
    double data_energy(const M& m, const State& s_cam, const State& s_L, const State& s_th, const Frame& f, bool camera,
                       bool lidar, const VecKd* bb = nullptr) const
    {
        MatNf Id = MatNf::Zero();   // the terms' information is not wanted here
        VecNf bd = VecNf::Zero();
        double E = 0.0;
        if (camera)
        {
            if (bb) set_q_channel(m, *bb, true);
            const float sig2 = m.sigma2();
            for (std::size_t i = 0; i < f.points.size(); ++i)
            {
                const float R = R_of(f, i, sig2);
                E += neighbour_mix(m.point_energy(f.points[i], s_cam, R), f.points[i], R, nullptr);
            }
            camera_extra(m, s_cam, f, Id, bd, &E);
        }
        if (bb and lidar) set_q_channel(m, *bb, false);
        if (not lidar_override(m, s_L, f, nullptr, nullptr, &E) and lidar)
            E += beam_.accumulate(key_of(m, s_L), [&] { return m.beam_parts(s_L); }, m.beam_rays(f), m.beam_config(), nullptr, nullptr);
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
        ensure_joint(m.covariance());
        const double H_before = half_logdet_theta(P_);                     // G5: the published Σθ this update starts from
        last_delivered_nats_ = std::numeric_limits<double>::quiet_NaN();   // NaN until an update measures

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

        // F1.4: the wall latent — transported when the staged edge is the same one, a new source (and its priors' one entry) when
        // the instance associates with an edge for the first time.
        bool wall_enter = false;
        Eigen::Matrix<double, N, K> B_w = Eigen::Matrix<double, N, K>::Zero();
        if constexpr (W > 0)
        {
            const WallAnchor a = m.wall_anchor(f);
            if (a.ok and a.sigma_m > 0.0)
            {
                if (wall_set_ and same_wall(a))
                    b_(kWall) -= (a.p - wall_.p).dot(a.n);                // the same wall, its staged line moved: transport
                else
                {
                    const int j = N + kWall;                               // a new source: the old δ is marginalised out
                    P_.row(j).setZero(); P_.col(j).setZero();
                    P_(j, j) = a.sigma_m * a.sigma_m;
                    b_(kWall) = 0.0;
                    sig_prev_(kWall) = a.sigma_m;
                    // the priors enter once per PHYSICAL wall: not again for a wall already entered (its stored line transported)
                    const auto it = std::ranges::find_if(walls_entered_, [&](const WallAnchor& e) { return same_physical_wall(a, e); });
                    if (it == walls_entered_.end()) { walls_entered_.push_back(a); wall_enter = true; }
                    else *it = a;
                }
                wall_ = a;
                wall_set_ = true;
            }
            if (wall_set_) { B_w(0, kWall) = -wall_.n.x(); B_w(1, kWall) = -wall_.n.y(); }
        }
        last_wall_entered_ = wall_enter;
        const bool had_posterior = first_update_done_;                     // the volatility learns from a posterior's step only
        const VecNf q_used = process_noise(m);                             // the Q of this update's (and the interval's) predicts
        FrameLatents<N> fl_used = fl;
        fl_used.lidar_measured = lidar;                                    // the latents follow the channels that measure
        predict_latents(m, fl_used);                                       // step 1, latents
        const typename rc::ai::beam::BeamChannel<NC>::Scope cache_scope(beam_);   // the beam cache lives for this update
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

        std::vector<Chan> channels;
        pu::FreshInfo<N> theta_only;
        MatN Id_cam, Id_L, Id_th; VecN bd_cam, bd_L, bd_th;
        QInfo q_cam, q_L;
        const auto build = [&](const VecNd& th, const VecKd& bb)
        {
            accumulate_channels(m, S(th), S(th + B_cam * bb), S(th + B_L * bb), f, camera, lidar, Id_cam, bd_cam, Id_L, bd_L, Id_th, bd_th,
                                &q_cam, &q_L, &bb);
            channels.clear();
            if (camera) { Chan c; c.info = pu::marginalise_fresh<N>(Id_cam, bd_cam, Sf_cam); c.B = B_cam; add_q(c, q_cam, true); channels.push_back(c); }
            if (lidar)  { Chan c; c.info = pu::marginalise_fresh<N>(Id_L,   bd_L,   Sf_L);   c.B = B_L;   add_q(c, q_L, false);  channels.push_back(c); }
            theta_only.I = Id_th.template cast<double>();
            theta_only.g = bd_th.template cast<double>();
            if constexpr (W > 0)
                if (wall_enter)
                {   // the wall priors, ONCE: a channel measuring θ + B_w b (the object relative to the TRUE wall)
                    MatN Iw = MatN::Zero(); VecN bw = VecN::Zero();
                    m.accumulate_wall_prior(S(th + B_w * bb), f, Iw, bw, nullptr);
                    Chan c; c.info = pu::marginalise_fresh<N>(Iw, bw, MatN::Zero()); c.B = B_w; channels.push_back(c);
                }
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
            double E = data_energy(m, S(th + B_cam * bb), S(th + B_L * bb), S(th), f, camera, lidar, &bb);
            if constexpr (W > 0)
                if (wall_enter)
                {
                    MatN Iw = MatN::Zero(); VecN bw = VecN::Zero();
                    m.accumulate_wall_prior(S(th + B_w * bb), f, Iw, bw, &E);
                }
            return E;
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
        const auto chans = [&] { return std::span<const Chan>(channels); };
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
                const auto st0 = pu::joint_step_damped<N, K, NP>(x_tilde, x_bar, P_bar, chans(), theta_only, 0.0);
                const VecNd d0 = theta_step(x_tilde, x0n, st0);
                const MatJd P_lin = pu::joint_cov<N, K, NP>(P_bar, chans(), theta_only);
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
            const auto st = pu::joint_step_damped<N, K, NP>(x_tilde, x_bar, P_bar, chans(), theta_only, lambda, /*damp_latents*/ consistent);
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
                    pu::detail::information<N, K, NP>(chans(), theta_only, Wj, Gj);
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
                    const auto st0 = pu::joint_step_damped<N, K, NP>(x_tilde, x_bar, P_bar, chans(), theta_only, 0.0);
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
        // (zero if none): a frame that stopped while still moving carries its last move as uncertainty. Unconverged: the prior
        // moment (F1 round 9 b, below).
        build(theta_post, consistent ? b_post : b_);
        // The jump statistic's view of this update's data (learn_volatility, F1 round 7 A1): the data as a Gaussian factor on the
        // joint x, LINEARISED AT THE STAYED POSTERIOR MODE x̂ = (θ̂, b̂) — the Laplace point of the stayed evidence — and written as a
        // factor about x̄: information Λ̂ and gradient η̂ + Λ̂(x̂ − x̄). For data that are quadratic in x this is the factor at x̄ bit
        // for bit (the evidence does not depend on how far the prior let the GN follow a shove). For data that are not, the stayed
        // evidence is only right at its own mode, and the factor at x̄ extrapolates: bed_5 (p1_bed) cycles 1708/1709, LiDAR-only
        // while driving, 14 rays crossing the box's top side edge 1–3.4 σ_s deep and ending on the wall behind (refuted, E up to
        // 7.6 each). Their gradient is large and their curvature small (a hit/miss tail, I_d ≪ g²), so the factor at x̄ put the
        // data's mean ~0.8 m / 0.9 rad away — reachable only by the 0.5 m slab — and log BF = 36 / 22 (P(moved) 1.0 / 0.94). The
        // stayed fit frees them inside its own prior (the LiDAR's ego-motion latent u_L, σ 6 cm / 4°): at x̂ the same factor gives
        // log BF −3.3 / −2.5.
        MatJd Lambda_data; VecXd eta_data;
        pu::detail::information<N, K, NP>(chans(), theta_only, Lambda_data, eta_data);
        {
            VecXd x_hat; x_hat << theta_post, (consistent ? b_post : b_);
            eta_data += Lambda_data * wrapped(x_hat - x_bar);
        }
        P_ = pu::joint_cov<N, K, NP>(P_bar, chans(), theta_only);
        // F1 round 9 (b): an update that did NOT reach a mode carries the PRIOR moment, not the Laplace one. joint_cov's P is
        // (𝟙 + P̄W)⁻¹P̄, the inverse curvature of the energy AT THE POINT REACHED; it is a posterior covariance only where that
        // point is the energy's minimum (the stop rule's condition, `converged`). At an unconverged point (the iteration cap) it
        // is the curvature on a slope, and it collapsed Σ around a fit the solver had not finished: k2c_table's table_2 ran 25
        // steps unconverged and took σ_cx 0.30 → 0.033 and σ_h 0.572 → 0.036, after which same-view frames (ρ ≈ 1) added
        // almost nothing and the bad fit froze. The update therefore keeps the point it reached (x̂ is lower in energy than x̄:
        // every accepted step lowered J, so it is the better start for the next frame) and the predicted covariance P̄: the
        // frame's information is not counted as absorbed until a frame's fit reaches its mode, and the next frame of the same
        // view can still move the estimate. Measured alternative, rejected: P + δ₀δ₀ᵀ with δ₀ the remaining undamped step (the
        // local quadratic's second moment about x̂) — at a non-mode δ₀ is the quadratic's extrapolation, metres on p1r8_bed's
        // capped updates (refuted rays), and the bed ran away (cx −0.31 → −2.15). Converged updates are unchanged (P plus the
        // last accepted step's outer product on θθ, round 6).
        // ⚠ THRESHOLD (flagged, CLAUDE.md "a threshold that is truly unavoidable"; F1 round 9 fix round 2 ruling): this is a binary
        // switch on the solve's stop rule (converged ⇔ the undamped step ≤ tol_sigma·σ on θ). tol_sigma decides whether a frame's
        // information is absorbed (Laplace Σ) or not at all (P̄), and x̂ keeps the frame's pull while P̄ says it added nothing.
        // No continuous model-level term was found (task-F1r9-report.md, fix round 1). The tempered Laplace family
        // P_α = (P̄⁻¹ + αW)⁻¹ failed on every measured variant, on p1r8_bed / k2c_table:
        //  · α = exp(−½δ₀ᵀP⁻¹δ₀) (density ratio): ≈ 0 at any capped point (R in posterior σ), the mean returns to x̄, progress
        //    lost (unit test: cx 1.25 → 1.06 → back to 1.25; the "carry x̄, P̄" diagnostic capped 739 of 2272 bed updates);
        //  · α = A/(A+R) with the tempered mean x̄ + αP_αη': the local quadratic extrapolates (table_2 ran to h = 19.2 m, DEGENERATE);
        //  · α = A/(A+R) with a mean anchored at x̂: overconfident (unit test α 0.966 at 69 % of the distance, mode 4.6σ away;
        //    bed d² 3.53, NEES 3.35, no gain over BASE 3.81 / 3.49).
        // This switch: bed_11 d² 3.81 → 0.04, NEES 3.49 → 1.82; nightstand, fridge and chair bit-identical (they never cap).
        // The cap binds at most 5 times per graded dump. ROOT FIX (post-12-02 owner list): remove why fits cap at all —
        // kinked energies (mixture / beam panel boundaries) and λ escalation in the damped GN.
        if (not converged) P_ = P_bar;
        const VecNd post_var = P_.diagonal().template head<N>();          // the model's posterior, before the solver's last-step term
        if (converged) P_.template topLeftCorner<N, N>() += d_last * d_last.transpose();
        if (cfg.gn_trace >= 2 or (cfg.gn_trace == 1 and not converged))
        {   // GnTrace: does the solver's G / W match the energy J it accepts against? Per channel, the raw −bd, the
            // marginalised −g, central differences of data_energy with only that channel's state perturbed, and the
            // information diagonal against the second difference. Channel 3: the joint θ gradient against J along θ.
            const auto fd_check = [&](const VecNd& th, const VecKd& bb, int at)
            {
                MatN Ic, IL, It; VecN gc, gL, gt;
                const VecNd sc0 = th + B_cam * bb, sl0 = th + B_L * bb;
                accumulate_channels(m, S(th), S(sc0), S(sl0), f, camera, lidar, Ic, gc, IL, gL, It, gt, nullptr, nullptr, &bb);
                const auto mc = pu::marginalise_fresh<N>(Ic, gc, Sf_cam);
                const auto mL = pu::marginalise_fresh<N>(IL, gL, Sf_L);
                const double E0 = data_energy(m, S(sc0), S(sl0), S(th), f, camera, lidar, &bb);
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
                            return data_energy(m, S(a), S(l), S(t), f, camera, lidar, &bb);
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
                pu::detail::information<N, K, NP>(chans(), theta_only, Wj, Gj);
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
        set_q(m, b_);                                                      // the model evaluates at the posterior proportions
        m.set_covariance(Sigma_theta());
        if (had_posterior) learn_volatility(m, q_used, x_bar.template head<N>(), P_bar, post_var, Lambda_data, eta_data, theta_post);
        steps_since_update_ = 0.0;
        dbg_Id_th_ = Id_th; dbg_Id_cam_ = Id_cam; dbg_Id_L_ = Id_L;
        // G5: what this update delivered (the θ-marginal entropy drop of the published Σ), and what the next offer's
        // hypothetical look starts from: the camera's position, how many points it gave and their mean variance, b_cam's
        // marginal, and the robot's position (e's transport).
        last_delivered_nats_ = H_before - half_logdet_theta(P_);
        last_robot_xy_ = fl.robot_xy;
        last_sig2_cam_ = fl.sig2_cam;
        last_sig2_L_   = fl.sig2_L;
        if (camera)
        {
            last_cam_xy_ = fl.camera_xy.value_or(fl.robot_xy);
            last_cam_n_ = f.points.size();
        }
        if (lidar) last_lidar_origin_ = m.beam_rays(f).origin;
        // F1: the anchors after this update (the fitter commits the same on acceptance), where the offer's ρ is measured from.
        if (fl.view.on)
        {
            float rho_H = fl.rho_L[0];
            if constexpr (requires { { M::kHeight } -> std::convertible_to<int>; }) rho_H = fl.rho_L[M::kHeight];
            last_view_ = rc::ai::view::view_commit(fl.view, camera, lidar, fl.rho_cam[0], fl.rho_L[0], rho_H);
        }
        if constexpr (HasOfferContext<M>)
            last_ctx_ = m.offer_context(f);                            // the offer's extra terms' context (G5), not the frame
        offer_cache_.valid = false;                                    // Ω at the last camera position is recomputed once
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

    // ── The offer gain (spec shared-engine G5) ──────────────────────────────────────────────────────────────────────────
    // What a look from standpoint `s` would measure (offer_view.h): the camera samples the standpoint sees, each standing
    // for weight_k = n·dΩ_k/Ω_last mask points (n = the last camera update's count, Ω_last = what the same camera model sees
    // from that update's position: the look's point count scales with the solid angle it covers, and a grazing face gets
    // few), all at R = the model's variance for a STANDING look at the samples' mean range (standing_point_variance; absent
    // ⇒ sigma2()); and, with a LiDAR in `sn`, the rays it would stage. Before any camera update: n = the sample count.
    struct OfferObservation
    {
        std::vector<Eigen::Vector3f> points;
        std::vector<float> weight;
        float R = 0.0f;
        rc::ai::beam::BeamRays rays;
        bool lidar = false;                                            // the rays measure (beam_readiness kMeasured)
    };
    OfferObservation offer_observation(const M& m, const Standpoint& s, const OfferSensors& sn) const
    {
        namespace of = rc::ai::offer;
        const std::vector<rc::ai::beam::BeamPart>& parts = offer_parts(m);
        const Eigen::Vector3f cam3 = of::camera_at(s, sn);
        OfferObservation o;
        const of::View v = of::visible_surface(parts, cam3, s.yaw, sn);
        if (not v.points.empty() and v.solid_angle > 0.0)
        {
            double n = double(v.points.size()), omega = v.solid_angle;
            if (last_cam_xy_.has_value() and last_cam_n_ > 0)
            {
                if (not offer_cache_.omega_last.has_value())
                {
                    const Eigen::Vector2f c(m.state().vec()(0), m.state().vec()(1)), to_c = c - *last_cam_xy_;
                    const Eigen::Vector3f last3(last_cam_xy_->x(), last_cam_xy_->y(), cam3.z());
                    offer_cache_.omega_last = of::visible_surface(parts, last3, std::atan2(to_c.y(), to_c.x()), sn).solid_angle;
                }
                if (*offer_cache_.omega_last > 0.0) { n = double(last_cam_n_); omega = *offer_cache_.omega_last; }
            }
            double range = 0.0;
            for (std::size_t k = 0; k < v.points.size(); ++k) range += v.d_omega[k] * double((v.points[k] - cam3).norm());
            o.R = standing_R(m, float(range / v.solid_angle));
            o.points = v.points;
            o.weight.resize(v.points.size());
            for (std::size_t k = 0; k < v.points.size(); ++k) o.weight[k] = float(n * v.d_omega[k] / omega);
        }
        if (sn.lidar.has_value())
        {
            // the select box's anchor: the look's mask centroid (points spread by solid angle), as live; θ's centre without one
            Eigen::Vector2f centroid(m.state().vec()(0), m.state().vec()(1));
            if (not v.points.empty() and v.solid_angle > 0.0)
            {
                Eigen::Vector2d acc = Eigen::Vector2d::Zero();
                for (std::size_t k = 0; k < v.points.size(); ++k) acc += v.d_omega[k] * v.points[k].head<2>().template cast<double>();
                centroid = (acc / v.solid_angle).template cast<float>();
            }
            o.rays = of::synth_rays(parts, of::lidar_origin(s, *sn.lidar), *sn.lidar, sn, cam3, s.yaw, centroid);
            o.lidar = rc::ai::beam::beam_readiness(o.rays, m.beam_config()) == rc::ai::beam::Readiness::kMeasured;
        }
        return o;
    }

    // The θ-marginal entropy drop ½ log det Σθ_now − ½ log det Σθ_after of a hypothetical look from standpoint `s`, under
    // this engine's own update with the same channel set — so it is capped by the same shared errors the update is:
    //   - θ predicted as update() predicts it (Q, the clamp to the class P; the class prior at the class mean before the
    //     first update);
    //   - the latents predicted on a copy of exactly (P, b, σ_prev, lever, Σ_pose): ρ_cam = rho_camera of the straight path
    //     from the last camera update's position to the standpoint's camera (10 equal chords of accumulate_path), z and w⊥
    //     there; ρ_L = rho_lidar_lateral / rho_lidar_height of the straight path from the last LiDAR origin, g = the widest
    //     ring gap (height on M::kHeight when the model names it) — with F1 viewpoints staged (FrameLatents::view), the
    //     argument is instead the look's view's separation from the latent's anchor beyond its frontier, as the update reads it;
    //     ρ_chain = 1 (no localiser correction predicted), e carried
    //     by the straight robot displacement; the latents' marginals as at the last update; no fresh ego-motion share (the
    //     look is taken standing at the standpoint);
    //   - the channels the update uses: the camera points (offer_observation) and the model's camera extra terms at
    //     θ + B_cam b, the synthesised rays at θ + B_L b, the θ-only terms at θ — the extra terms read the last measured
    //     frame's context with the look's points in it (none before the first update);
    //   - joint_cov through the channels' selectors, exactly as a measured frame's.
    // A standpoint whose look measures nothing promises 0.
    // ⚠ SINGLE-THREAD (review I4): const, but NOT read-only — it moves the model's proportions (set_q / set_q_channel write
    // through CountableEngineModel::set_proportions' const_cast into the belief's q) and fills the mutable offer_cache_. Call it
    // only from the thread that owns the belief (the countable's compute thread, which also runs update()), never concurrently
    // with another offer or an update of the same belief. The RAII guard below puts q back at the posterior (q₀ + b_q) on EVERY
    // exit path, early returns and exceptions included.
    double predicted_gain(const M& m, const Standpoint& s, const OfferSensors& sn) const
    {
        struct RestoreProportions
        {
            const PersistentEngine& e; const M& m;
            ~RestoreProportions() { e.set_q(m, e.b_); }
        } const restore_q{*this, m};
        namespace vw = rc::ai::view;
        namespace pu = rc::ai::persistent;
        namespace of = rc::ai::offer;
        MatJd P = P_;
        VecKd b = b_, sp = sig_prev_;
        vw::ChainLever lev = lever_;
        Eigen::Matrix3d lsp = last_sigma_pose_;
        if (not joint_init_)
        {   // ensure_joint on the copy
            P.setZero();
            P.template topLeftCorner<N, N>() = m.covariance().template cast<double>();
            b.setZero(); sp.setZero(); lev = {}; lsp.setZero();
        }
        const double H_before = half_logdet_theta(P);
        predict_theta_on(P, process_noise(m), m.prior_cov_diag(), 1.0f);

        const OfferObservation obs = offer_observation(m, s, sn);
        const bool camera = not obs.points.empty(), lidar = obs.lidar;
        if (not camera and not lidar) return 0.0;
        const VecNf th = m.state().vec();
        const Eigen::Vector2f c(th(0), th(1));
        const std::vector<rc::ai::beam::BeamPart>& parts = offer_parts(m);
        const Eigen::Vector3f cam3 = of::camera_at(s, sn);
        const auto straight = [&](const Eigen::Vector2f& from, const Eigen::Vector2f& to)
        {
            vw::ViewpointPath path;
            const Eigen::Vector2f step = (to - from) / float(kOfferPathSteps);
            for (int k = 0; k < kOfferPathSteps; ++k) vw::accumulate_path(path, from + float(k) * step, step, c);
            return path;
        };
        // F1: with the fitter's viewpoints staged, the look's ρ is its view's separation from the latent's anchor beyond the
        // frontier (as the update's), the view reached by the same straight 10-chord path from the last update's sensor position
        // (signed increments, rc::ai::view::advance_viewpoint). Without them (older callers), the straight path as before.
        const auto view_from = [&](vw::Viewpoint x, const Eigen::Vector2f& from, const Eigen::Vector2f& to)
        {
            const Eigen::Vector2f step = (to - from) / float(kOfferPathSteps);
            // advance_viewpoint takes the sensor's position AFTER the step (the step ends at from + (k + 1)·step)
            for (int k = 0; k < kOfferPathSteps; ++k) x = vw::advance_viewpoint(x, from + float(k + 1) * step, step, c);
            return x;
        };
        const bool f1_cam = last_view_.has_value() and last_view_->cam_xy.has_value() and last_view_->a_cam.set;
        const bool f1_L   = last_view_.has_value() and last_view_->L_xy.has_value() and last_view_->a_lat.set;
        FrameLatents<N> fl;
        dbg_offer_rho_ = {};
        dbg_offer_rho_.f1_cam = f1_cam; dbg_offer_rho_.f1_L = f1_L;
        float rho_cam = 1.0f;
        if (camera and last_cam_xy_.has_value())
        {
            const float z = (cam3.head<2>() - c).norm(), w_perp = of::extent_across(parts, cam3.head<2>(), c);
            if (z > 0.0f and w_perp > 0.0f)
            {
                const vw::ViewpointPath path = f1_cam
                    ? vw::anchor_separation(last_view_->a_cam, view_from(last_view_->cam, *last_view_->cam_xy, cam3.head<2>()))
                    : straight(*last_cam_xy_, cam3.head<2>());
                rho_cam = vw::rho_camera(path, z, w_perp);
                dbg_offer_rho_.cam = path;
            }
        }
        fl.rho_cam.fill(rho_cam);
        fl.rho_L.fill(1.0f);
        if (lidar and last_lidar_origin_.has_value())
        {
            const Eigen::Vector3f o = obs.rays.origin.template cast<float>();
            const vw::Viewpoint xL = f1_L ? view_from(last_view_->L, *last_view_->L_xy, o.head<2>()) : vw::Viewpoint{};
            const vw::ViewpointPath path = f1_L ? vw::anchor_separation(last_view_->a_lat, xL)
                                                : straight(last_lidar_origin_->head<2>().template cast<float>(), o.head<2>());
            const vw::ViewpointPath path_H = f1_L ? vw::anchor_separation(last_view_->a_H, xL) : path;
            const float g = of::max_ring_gap(sn.lidar->ring_elev_rad);
            fl.rho_L.fill(vw::rho_lidar_lateral(path, g));
            dbg_offer_rho_.lat = path; dbg_offer_rho_.H = path_H;
            if constexpr (requires { { M::kHeight } -> std::convertible_to<int>; })
            {   // the ring that grazes the top moves over it with range
                double top = -std::numeric_limits<double>::infinity();
                for (const auto& P0 : parts) top = std::max(top, P0.c.z() + P0.half.z());
                const float zl = (o.head<2>() - c).norm();
                if (zl > 0.0f) fl.rho_L[M::kHeight] = vw::rho_lidar_height(path_H, zl, float(std::atan2(top - o.z(), zl)), g);
            }
        }
        dbg_offer_rho_.rho_cam = fl.rho_cam; dbg_offer_rho_.rho_L = fl.rho_L;
        fl.sig2_cam = last_sig2_cam_;
        fl.sig2_L = last_sig2_L_;
        fl.rho_chain = 1.0f;
        fl.sigma_pose = lsp;
        fl.chain_dr = s.xy - last_robot_xy_.value_or(s.xy);
        fl.robot_xy = s.xy;
        fl.camera_xy = cam3.head<2>();
        fl.camera_measured = camera;
        fl.lidar_measured = lidar;
        predict_latents_on(m, fl, P, b, sp, lev, lsp);
        set_q(m, b);                                                       // the look is linearised at the predicted proportions

        MatJd P_bar = P;
        if (not first_update_done_)
        {   // as update(): the class prior at the class mean, counted once, plus the birth spread
            P_bar.template topRows<N>().setZero();
            P_bar.template leftCols<N>().setZero();
            P_bar.diagonal().template head<N>() = m.prior_cov_diag().template cast<double>();
            P_bar.template topLeftCorner<2, 2>() += birth_pos_excess_.template cast<double>();
        }

        const Eigen::Matrix<double, N, K> B_cam = selector_at(m, true, lev), B_L = selector_at(m, false, lev);
        const State s_th  = State::from_vec(th);
        const State s_cam = State::from_vec((th.template cast<double>() + B_cam * b).template cast<float>());
        const State s_L   = State::from_vec((th.template cast<double>() + B_L * b).template cast<float>());
        // the look's frame for the model's extra terms: the cycle's context frame with this look's points in it (R/weight each)
        Frame* ctx = nullptr;
        if constexpr (HasOfferContext<M>)
            if (last_ctx_.has_value())
            {
                if (not offer_cache_.frame.has_value()) offer_cache_.frame.emplace(m.offer_frame(*last_ctx_));
                offer_cache_.R.resize(obs.points.size());
                for (std::size_t k = 0; k < obs.points.size(); ++k) offer_cache_.R[k] = obs.R / obs.weight[k];
                m.offer_look(*offer_cache_.frame, obs.points, offer_cache_.R);
                ctx = &*offer_cache_.frame;
            }
        std::vector<Chan> chans;
        if (camera)
        {
            MatNf Id = MatNf::Zero();
            VecNf bd = VecNf::Zero();
            QInfo qi;
            set_q_channel(m, b, true);
            accumulate_offer_points(m, s_cam, obs.points, obs.weight, obs.R, Id, &qi);
            if (ctx != nullptr) camera_extra(m, s_cam, *ctx, Id, bd, nullptr);
            Chan ch;
            ch.info = pu::marginalise_fresh<N>(Id, VecNf::Zero(), MatNf::Zero());
            ch.B = B_cam;
            add_q(ch, qi, true);
            chans.push_back(ch);
        }
        if (lidar)
        {
            set_q_channel(m, b, false);
            Eigen::Matrix<double, NC, NC> W = Eigen::Matrix<double, NC, NC>::Zero();
            Eigen::Matrix<double, NC, 1> g = Eigen::Matrix<double, NC, 1>::Zero();
            const std::vector<rc::ai::beam::BeamPart> parts_L = m.beam_parts(s_L);
            rc::ai::beam::accumulate_beam_rays<NC>(parts_L, obs.rays, m.beam_config(), W, g, nullptr);   // information only
            Chan ch;
            ch.info = pu::marginalise_fresh<N>(MatNf(W.template topLeftCorner<N, N>().template cast<float>()), VecNf::Zero(), MatNf::Zero());
            ch.B = B_L;
            if constexpr (NP > 0)
            {
                QInfo qi;
                qi.Iuq = W.template topRightCorner<N, NP>();
                qi.Iqq = W.template bottomRightCorner<NP, NP>();
                add_q(ch, qi, false);
            }
            chans.push_back(ch);
        }
        pu::FreshInfo<N> theta_only;
        if (ctx != nullptr)
        {
            MatNf Id_th = MatNf::Zero();
            VecNf bd_th = VecNf::Zero();
            m.accumulate_extra(s_th, *ctx, Id_th, bd_th, nullptr);
            theta_only.I = Id_th.template cast<double>();
        }
        const MatJd P_after = pu::joint_cov<N, K, NP>(P_bar, std::span<const Chan>(chans), theta_only);
        return H_before - half_logdet_theta(P_after);                     // restore_q: back to the posterior proportions
    }
    // The θ-marginal entropy drop ½ log det Σθ_before − ½ log det Σθ_after of the last update(): what it delivered, the same
    // quantity predicted_gain promises. NaN when the last update measured nothing (predict-only).
    double last_delivered_nats() const { return last_delivered_nats_; }

    // ── Diagnostics of the last update ───────────────────────────────────────────────────────────────────────────────
    const std::vector<VecNf>&           dbg_iterates()  const { return dbg_iterates_; }   // ACCEPTED θ iterates
    const std::vector<double>&          dbg_gn_energy() const { return dbg_gn_energy_; }  // J at θ₀ and after every accepted step
    const std::vector<GnTraceStep<N>>&  dbg_gn_steps()  const { return dbg_gn_steps_; }
    const std::vector<GnTraceFd>&       dbg_gn_fd()     const { return dbg_gn_fd_; }
    long   gn_update_serial()  const { return gn_update_serial_; }
    bool   last_wall_entered() const { return last_wall_entered_; }
    double last_jump_log_bf() const { return last_jump_log_bf_; }   // log BF(moved : stayed) of the last update (NaN: none)
    double last_jump_p() const { return last_jump_p_; }             // P(moved | data) of the last update   // F1.4: the last update entered the wall priors
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
    static constexpr int kOfferPathSteps = 10;                       // G5: chords of the straight path to a standpoint

    // A standing look's per-point variance at range d: the model's standing_point_variance(d) (optional hook: its R with
    // the ego-motion term 0), else sigma2().
    static float standing_R(const M& m, float d)
    {
        if constexpr (requires { { m.standing_point_variance(d) } -> std::convertible_to<float>; })
            return m.standing_point_variance(d);
        else
            return m.sigma2();
    }
    // The model's beam parts at its current state, built once per state for all the standpoints of one planning cycle; the
    // cache (and Ω at the last camera position with it) is dropped when the state changes and after every update.
    const std::vector<rc::ai::beam::BeamPart>& offer_parts(const M& m) const
    {
        const VecNf key = m.state().vec();
        if (not offer_cache_.valid or offer_cache_.key != key)
        {
            offer_cache_.parts = m.beam_parts(m.state());
            offer_cache_.key = key;
            offer_cache_.omega_last.reset();
            offer_cache_.frame.reset();                                 // the context frame is rebuilt from the new update's context
            offer_cache_.valid = true;
        }
        return offer_cache_.parts;
    }

    // The same physical wall as δ's (same_physical_wall, F1 round 6.3).
    bool same_wall(const WallAnchor& a) const { return same_physical_wall(a, wall_); }

    // The model's optional volatility settings: VolatilityConfig{} (infer = false) when absent.
    static VolatilityConfig volatility_of(const M& m)
    {
        if constexpr (requires { { m.volatility() } -> std::convertible_to<VolatilityConfig>; })
            return m.volatility();
        else
            return VolatilityConfig{};
    }

    // The EM statistic of the process steps over one update interval (plan rev 2 F1.2; Fable order review §2.3). Per θ component,
    // with q the per-step Q the interval was predicted with, Σ̄ the predicted (prior) variance this update started from, Σ⁺ the
    // posterior and δθ the posterior mean minus the predicted mean, each of the interval's random-walk steps w has
    //     E[w | data] = (q/Σ̄)·δθ ,   Var[w | data] = q − (q/Σ̄)²·(Σ̄ − Σ⁺) ,
    // so E[w² | data] = q + (q/Σ̄)²·(δθ² − (Σ̄ − Σ⁺)). It is ≥ 0 whenever q ≤ Σ̄ (Σ̄ holds at least one q), so it takes no
    // rectifier, and its fixed point is "the innovations are as large as the filter expected", which is what the right Q means.
    // The weight (q/Σ̄)² is the share of the prior variance motion accounts for: a large correction of a wide prior (a seed
    // collapsing onto data) reads as the prior being corrected, a shove of a converged object (Σ̄ a few q) as motion.
    // Running mean, conjugate in the per-step variance: n ← λn + 1, S ← λS + E[w²], Q = (n₀Q₀ + S)/(n₀ + n), ω = log Q; Q is a
    // weighted mean of Q₀ and of non-negative statistics, so it stays positive with no clamp.
    void learn_volatility(const M& m, const VecNf& q_used, const VecNd& theta_pred, const MatJd& P_bar, const VecNd& post_var,
                          const MatJd& Lambda_data, const VecXd& eta_data, const VecNd& theta_post)
    {
        const VolatilityConfig vc = volatility_of(m);
        if (not vc.infer) return;
        const VecNd q0 = m.process_noise_diag().template cast<double>();
        if (not vol_init_)
        {
            vol_omega_ = q0.array().log().matrix();
            vol_s_.setZero();
            vol_n_ = 0.0;
            vol_init_ = true;
        }
        const double n0 = 1.0 / (vc.sigma_omega * vc.sigma_omega);
        const bool jumps = static_cast<int>(vc.jump_var.size()) == N;
        const double k_steps = std::max(1.0, steps_since_update_);      // the update's own predict is always one
        // P(at least one move in the interval) = 1 − (1 − π)^k
        const double pi_int = jumps ? -std::expm1(k_steps * std::log1p(-std::min(vc.jump_prob, 1.0))) : 0.0;
        // Spike-and-slab, JOINT over θ (one move moves the object as a whole): the data of this update as a factor (Λ, η) on the
        // joint x at x̄; "stayed" x ~ N(x̄, P̄), "moved" adds v ~ N(0, M) on θ (M = diag jump_var). p = P(moved | data) by Bayes with
        // prior π_int (detail::jump_evidence, information form); per component E[w_i²] = p·E[v_i² | moved, data], ÷ k per step.
        VecNd ew2_jump = VecNd::Zero();
        if (jumps)
        {
            Eigen::Matrix<double, N, N> Mj = Eigen::Matrix<double, N, N>::Zero();
            for (int i = 0; i < N; ++i) Mj(i, i) = vc.jump_var[std::size_t(i)];
            // the Bayes factor in information form on the joint x (detail::jump_evidence, round 7 A2)
            const auto je = detail::jump_evidence<N, N + K>(Lambda_data, eta_data, P_bar, VecNd(Mj.diagonal()), pi_int);
            for (int i = 0; i < N; ++i) ew2_jump(i) = je.p * (je.v_mean(i) * je.v_mean(i) + je.v_cov(i, i)) / k_steps;
            last_jump_log_bf_ = je.log_bf; last_jump_p_ = je.p;
        }
        vol_n_ = vc.memory * vol_n_ + 1.0;
        for (int i = 0; i < N; ++i)
        {
            double dth = theta_post(i) - theta_pred(i);
            if (i == kYaw) dth = std::remainder(dth, 2.0 * M_PI);
            const double q = double(q_used(i)), sbar = P_bar(i, i), spost = post_var(i);
            double ew2;
            if (not jumps)
            {   // (unreachable for the countable classes: CountableEngineModel::volatility() always sets jump_var, so `jumps` holds)
                // Gaussian random walk (item 2): E[w²] = q + (q/Σ̄)²(δθ² − (Σ̄ − Σ⁺))
                const double k = q / sbar;
                ew2 = q + k * k * (dth * dth - (sbar - spost));
            }
            else ew2 = ew2_jump(i);
            vol_s_(i) = vc.memory * vol_s_(i) + ew2;
            vol_omega_(i) = std::log((n0 * q0(i) + vol_s_(i)) / (n0 + vol_n_));
        }
    }

    // ½ log det of P's θθ block (LDLT; a non-positive pivot ⇒ NaN, so a degenerate Σ cannot pass as a number).
    static double half_logdet_theta(const MatJd& P)
    {
        const Eigen::LDLT<Eigen::Matrix<double, N, N>> ldlt(Eigen::Matrix<double, N, N>(P.template topLeftCorner<N, N>()));
        double h = 0.0;
        for (int i = 0; i < N; ++i) h += 0.5 * std::log(ldlt.vectorD()(i));
        return h;
    }

    // θ's predict on P (Σ += q_scale·Q, then the clamp_to_prior congruence; see predict()).
    void predict_theta(const M& m, float q_scale) { predict_theta_on(P_, process_noise(m), m.prior_cov_diag(), q_scale); }
    // ⚠ THRESHOLD (flagged, CLAUDE.md; G1x closed prior-term list violation V5; no behaviour change). WHAT IT DECIDES: after the
    // process noise, a θ variance above the class prior's Σ₀,ii is rescaled back to it (symmetrically, so P stays PSD) — a hard
    // ceiling Σ ≤ Σ₀ on every θ DOF, applied on every predict: the class prior (counted once, at birth) is used again as a clamp,
    // for ever (V5: not argued on the closed list). WHY KEPT: it is recursive_laplace.h's clamp_to_prior, moved here unchanged —
    // without it Q (authored per frame) makes an unobserved static object lose its prior at σ ∝ √frames (live table_9, ~2 min
    // unseen: std_w 0.030 → 0.170 m, std_yaw 0.101 → 0.348 rad) and the next stray mask re-cuts it. The model-level replacement is
    // a stationary process (Ornstein–Uhlenbeck toward the class prior, Σ∞ = Σ₀, the claim recursive_laplace.h makes for the clamp
    // but does not implement) in place of the random walk + ceiling — post-12-02 owner list. No behaviour change here.
    static void predict_theta_on(MatJd& P_, const VecNf& q, const VecNf& p0, float q_scale)
    {
        for (int i = 0; i < N; ++i) P_(i, i) += double(std::max(0.0f, q_scale)) * double(q(i));
        for (int i = 0; i < N; ++i)
            if (p0(i) > 0.0f and P_(i, i) > double(p0(i)))
            {
                const double sc = std::sqrt(double(p0(i)) / P_(i, i));   // symmetric rescale keeps P PSD
                P_.row(i) *= sc;
                P_.col(i) *= sc;
            }
    }

    // The hypothetical look's camera information (G5): accumulate_points over synthesised points, point k standing for
    // weight[k] mask points of variance R (the per-point skip of accumulate_points is on one point's r/R, as there).
    void accumulate_offer_points(const M& m, const State& s, std::span<const Eigen::Vector3f> pts, std::span<const float> weight,
                                 float R, MatNf& Id, QInfo* qi = nullptr) const
    {
        const int n_prims = m.n_prims();
        for (std::size_t k = 0; k < pts.size(); ++k)
        {
            const Eigen::Vector3f& p = pts[k];
            const float wk = weight[k];
            const auto r = m.responsibilities(p, s, R);
            for (int prim = 0; prim < n_prims; ++prim)
            {
                const float w = r[prim] / R;
                if (w < 1e-6f) continue;
                const VecNf J = m.sdf_jacobian(p, s, prim);
                Id.noalias() += (wk * w) * (J * J.transpose());
                if constexpr (NP > 0)
                    if (qi)
                    {
                        const Eigen::Matrix<double, NP, 1> Jq = m.sdf_jacobian_q(p, s, prim).template cast<double>();
                        qi->Iuq.noalias() += double(wk * w) * (J.template cast<double>() * Jq.transpose());
                        qi->Iqq.noalias() += double(wk * w) * (Jq * Jq.transpose());
                    }
            }
        }
    }
    // The beam cache's key: θ, and the proportions the model evaluates at (NP > 0).
    static typename rc::ai::beam::BeamChannel<NC>::Key key_of(const M& m, const State& s)
    {
        const VecNf v = s.vec();
        typename rc::ai::beam::BeamChannel<NC>::Key k{};
        for (int i = 0; i < N; ++i) k[i] = v(i);
        if constexpr (NP > 0) for (int j = 0; j < NP; ++j) k[N + j] = float(m.proportions()(j));
        return k;
    }

    // The camera points' GN information (the recursive-Laplace accumulate loop, unsaturated). With neighbours each point's
    // responsibilities are scaled by the model's share ω of the neighbour mixture: the EM identity of neighbour_mix's energy.
    void accumulate_points(const M& m, const State& s, const Frame& f, MatNf& Id, VecNf& bd, QInfo* qi = nullptr) const
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
                if constexpr (NP > 0)
                    if (qi)
                    {   // ∂d/∂q beside ∂d/∂θ: the same Gauss-Newton term on the (u, q) coordinates
                        const Eigen::Matrix<double, NP, 1> Jq = m.sdf_jacobian_q(f.points[i], s, prim).template cast<double>();
                        qi->Iuq.noalias() += double(w) * (J.template cast<double>() * Jq.transpose());
                        qi->Iqq.noalias() += double(w) * (Jq * Jq.transpose());
                        qi->gq.noalias()  += -double(w) * double(d) * Jq;
                    }
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
    Eigen::Matrix<double, N, K> selector(const M& m, bool camera) const { return selector_at(m, camera, lever_); }
    static Eigen::Matrix<double, N, K> selector_at(const M& m, bool camera, const rc::ai::view::ChainLever& lever)
    {
        Eigen::Matrix<double, N, K> B = Eigen::Matrix<double, N, K>::Zero();
        B.template block<N, 3>(0, kChain) = m.chain_jacobian(m.state(), lever.l);
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
    rc::ai::beam::BeamChannel<NC> beam_;
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
    // the wall latent (F1.4): the edge δ belongs to, whether it is set, and whether the last update entered its priors
    WallAnchor wall_;
    bool   wall_set_ = false;
    double last_jump_log_bf_ = std::numeric_limits<double>::quiet_NaN();   // the last update's move statistic (diagnostic)
    double last_jump_p_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<WallAnchor> walls_entered_;          // the physical walls whose priors have entered (once each)
    bool   last_wall_entered_ = false;
    // the inferred volatility (learn_volatility): ω = log Q per θ component, the running statistic S and count n
    bool   vol_init_ = false;
    double steps_since_update_ = 0.0;                // predict steps (Σ q_scale) since the last measured update
    VecNd  vol_omega_ = VecNd::Zero();
    VecNd  vol_s_ = VecNd::Zero();
    double vol_n_ = 0.0;
    MatNf  dbg_Id_th_ = MatNf::Zero();
    MatNf  dbg_Id_cam_ = MatNf::Zero();
    MatNf  dbg_Id_L_ = MatNf::Zero();
    int    last_beam_staged_ = 0;
    int    last_gap_rays_ = 0;
    // G5: the last update's delivery, and where the next hypothetical look starts from
    double last_delivered_nats_ = std::numeric_limits<double>::quiet_NaN();
    std::optional<Eigen::Vector2f> last_cam_xy_;      // the camera's position at the last camera update
    std::optional<Eigen::Vector2f> last_robot_xy_;    // the robot's position at the last measured update
    std::size_t last_cam_n_ = 0;                      // that camera update's point count
    std::array<float, N> last_sig2_cam_{};            // the latents' marginals at the last measured update
    std::array<float, N> last_sig2_L_{};
    std::optional<Eigen::Vector3d> last_lidar_origin_;   // the LiDAR origin at the last LiDAR update
public:
    // The last predicted_gain's latent ρ and their kernel arguments (test hook, review F1-I2: the promise must use the ρ the update
    // then applies). f1_*: the anchors were read (a view was staged); else the straight path from the last update's position.
    struct OfferRhoDbg
    {
        bool f1_cam = false, f1_L = false;
        rc::ai::view::ViewpointPath cam, lat, H;
        std::array<float, N> rho_cam{}, rho_L{};
    };
    const OfferRhoDbg& dbg_offer_rho() const { return dbg_offer_rho_; }
private:
    mutable OfferRhoDbg dbg_offer_rho_;
    std::optional<rc::ai::view::ViewState> last_view_;   // F1: viewpoints + anchors after the last update (off ⇒ none staged)
    std::optional<typename detail::OfferContextOf<M>::type> last_ctx_;   // the extra terms' context at the last measured update
    struct OfferCache
    {
        bool valid = false;
        VecNf key = VecNf::Zero();
        std::vector<rc::ai::beam::BeamPart> parts;
        std::optional<double> omega_last;
        std::optional<Frame> frame;                   // the context frame of this cycle's looks (offer_frame), points reassigned
        std::vector<float> R;
    };
    mutable OfferCache offer_cache_;                  // offer_parts / Ω_last, per state and per update
};

}  // namespace rc::ai::engine
