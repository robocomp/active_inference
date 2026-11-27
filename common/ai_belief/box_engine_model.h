/*
 * common/ai_belief/box_engine_model.h  —  the shared persistent engine's model for the box beliefs cloned from the
 * refrigerator (refrigerator_concept, hood_concept; spec docs/superpowers/specs/2026-11-23-shared-engine-design.md, Task 10).
 *
 * Both beliefs carry θ = [cx, cy, H, w, h, yaw] (N = 6) and ONE box primitive; they differ only in where the box stands
 * (the refrigerator's spans z ∈ [0, H], the hood's hangs at z ∈ [H − extent, H]) and in their class means. So the engine's
 * hooks are the same forwarding layer for both; what differs lives in the belief, behind the names this adapter calls:
 *
 *   B::State, B::Frame (Frame::points, Frame::R, Frame::persist : PersistentInputs<6>)
 *   sdf_prim, sdf_jacobian, responsibilities, n_prims, sigma2, apply_constraints, canonicalize,
 *   prior_cov_diag, process_noise_diag                                   (the recursive-Laplace hooks the legacy path uses)
 *   point_energy(p, s, R)            one camera point's −log mixture likelihood (box + clutter + wall), double;
 *   persistent_camera_extra(s, f, I, b, E)   the camera channel's non-point terms (the robust top, the depth_unobs shares);
 *   persistent_theta_only(s, f, I, b, E)     the θ-only channel (the wall prior once, the one-sided constraints);
 *   beam_parts(s)                    the box as the beam model's convex part;
 *   class_prior_mean(seed)           the first update's prior mean (seed position and yaw, the class size);
 *   params().beam, params().layer_spread_m;
 *   state_, Sigma_                   the belief's θ and its float mirror of P_θθ (the adapter is a friend).
 *
 * The countable belief keeps its own adapter (CountableEngineModel): it has a room-axis anchor, LiDAR-only frames, the
 * first-crossing test hook and the solver knobs, none of which these two classes have.
 */
#pragma once

#include <array>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "beam_ray.h"
#include "foreign_box.h"
#include "persistent_engine.h"
#include "viewpoint.h"

namespace rc::ai::engine
{

// What the fitter stages on a frame for the persistent path (READ ONLY there; the legacy path never reads it). Order of the
// N-vectors: θ's. The defaults (ρ = 0, zero variances, no rays, no neighbours) make every latent a zero-variance new source.
template <int N>
struct PersistentInputs
{
    std::array<float, N> rho_cam{}, rho_L{};          // lag-1 correlation of b_cam / b_L since that channel's last update
    std::array<float, N> sig2_cam{}, sig2_L{};        // the latents' marginal variances at this frame
    std::array<float, N> fresh_cam{}, fresh_L{};      // the ego-motion (fresh) variances, re-drawn every update
    float           rho_chain  = 0.0f;                // e's correlation with its last update
    Eigen::Matrix3d sigma_pose = Eigen::Matrix3d::Zero();   // the robot's pose covariance in the room (x, y, ψ)
    Eigen::Vector2f chain_dr   = Eigen::Vector2f::Zero();   // the robot's room displacement since the last update
    Eigen::Vector2f robot_xy   = Eigen::Vector2f::Zero();   // the robot's room position at the capture stamp
    std::optional<Eigen::Vector2f> camera_xy;               // the camera's room position (the offer's ρ path start)
    rc::ai::view::ViewState view;                           // F1: viewpoints + anchors before this update (off ⇒ none)
    rc::ai::beam::BeamRays beam;                            // the staged LiDAR rays (the beam channel)
    std::vector<rc::ai::beam::ForeignBox> foreign_boxes;    // G3: the neighbours' previous-cycle published beliefs
    double mask_error_rate = 0.0;                           // ε, their prior weight (Model.MaskPixelErrorRate)
};

// The engine's per-update latent inputs from a frame's staged inputs and its channels' "measured" flags.
template <int N>
FrameLatents<N> frame_latents(const PersistentInputs<N>& in, bool camera, bool lidar)
{
    FrameLatents<N> fl;
    fl.rho_cam = in.rho_cam;   fl.rho_L = in.rho_L;
    fl.sig2_cam = in.sig2_cam; fl.sig2_L = in.sig2_L;
    fl.fresh_c = in.fresh_cam; fl.fresh_L = in.fresh_L;
    fl.rho_chain  = in.rho_chain;
    fl.sigma_pose = in.sigma_pose;
    fl.chain_dr   = in.chain_dr;
    fl.robot_xy   = in.robot_xy;
    fl.camera_xy  = in.camera_xy;
    fl.view       = in.view;
    fl.camera_measured = camera;
    fl.lidar_measured  = lidar;
    fl.foreign_boxes   = in.foreign_boxes;
    fl.mask_error_rate = in.mask_error_rate;
    return fl;
}

// A view on one box belief B (see the file header). The engine calls these hooks, which read (and, for the state and its
// covariance mirror, write) that belief.
template <class B>
struct BoxEngineModel
{
    static constexpr int N = 6;
    static constexpr int kYaw = 5;                 // θ = [cx, cy, H, w, h, yaw]
    // H: the DOF the LiDAR's top-grazing ring decorrelates (ρ_L height, PersistentEngine::predicted_gain). ★ASSUMES A
    // FLOOR-STANDING box whose TOP lies above the Helios and is grazed from below-and-beside, as the refrigerator's. The hood
    // HANGS above the sensor: its rings strike the underside (z = H − extent), not the top, so ρ_L on H is the wrong geometry
    // for it. Left as is for the belief port; the hood's fitter task (10d) must give the hood its own height decorrelation.
    static constexpr int kHeight = 2;
    using State = typename B::State;
    using Frame = typename B::Frame;
    using Mat6 = Eigen::Matrix<float, 6, 6>;
    using Vec6 = Eigen::Matrix<float, 6, 1>;

    B* b = nullptr;

    float sdf_prim(const Eigen::Vector3f& p, const State& s, int prim) const { return b->sdf_prim(p, s, prim); }
    Vec6  sdf_jacobian(const Eigen::Vector3f& p, const State& s, int prim) const { return b->sdf_jacobian(p, s, prim); }
    std::array<float, 3> responsibilities(const Eigen::Vector3f& p, const State& s, float R) const
    { return b->responsibilities(p, s, R); }
    int    n_prims() const { return b->n_prims(); }
    float  sigma2() const { return b->sigma2(); }
    double point_energy(const Eigen::Vector3f& p, const State& s, float R) const { return b->point_energy(p, s, R); }
    void accumulate_camera_extra(const State& s, const Frame& f, Mat6& Id, Vec6& bd, double* E) const
    { b->persistent_camera_extra(s, f, Id, bd, E); }
    void accumulate_extra(const State& s, const Frame& f, Mat6& Id, Vec6& bd, double* E = nullptr) const
    { b->persistent_theta_only(s, f, Id, bd, E); }
    std::vector<rc::ai::beam::BeamPart> beam_parts(const State& s) const { return b->beam_parts(s); }
    const rc::ai::beam::BeamRays&   beam_rays(const Frame& f) const { return f.persist.beam; }
    const rc::ai::beam::BeamConfig& beam_config() const { return b->params_.beam; }
    // e measured through B_chain = chain_B(ℓ) (spec 7c.7 b); θ does not enter it
    Eigen::Matrix<double, 6, 3> chain_jacobian(const State&, const Eigen::Vector2f& lever) const
    { return rc::ai::view::chain_B(lever); }
    void apply_constraints(State& s) const { b->apply_constraints(s); }
    void canonicalize(State& s) const { b->canonicalize(s); }
    Vec6 shape_spread() const { return Eigen::Map<const Vec6>(b->params_.layer_spread_m.data()); }
    Vec6 prior_cov_diag() const { return b->prior_cov_diag(); }
    Vec6 process_noise_diag() const { return b->process_noise_diag(); }
    Eigen::Matrix<double, 6, 1> class_prior_mean(const State& seed, const Frame&) const { return b->class_prior_mean(seed); }
    const State& state() const { return b->state_; }
    void  set_state(const State& s) { b->state_ = s; }
    Mat6  covariance() const { return b->Sigma_; }
    void  set_covariance(const Mat6& S) { b->Sigma_ = S; }
};

}  // namespace rc::ai::engine
