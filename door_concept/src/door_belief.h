/*
 * door_belief.h  —  AI2 door belief (single wall-anchored panel, on the shared rc::ai engine)
 *
 * A door is a THIN RIGID PANEL that lives IN a wall of the room. So — unlike the chair this file was
 * scaffolded from — the belief does NOT estimate a free 6-DOF pose. The containing wall (a segment of the
 * room polygon) fixes the door's yaw, its lateral off-wall position, and its floor datum; what remains is
 * WHERE along the wall it sits and HOW BIG it is. The belief therefore works in the WALL FRAME:
 *
 *   State θ = [s, w, h]   (3 DOF)
 *     s = along-wall offset of the door's NEAR edge from the wall's near corner O   (localised — broad prior)
 *     w = panel width  (along the wall)                                             (strong prior ~0.70 m)
 *     h = panel height (vertical, base pinned to the floor)                         (strong prior ~2.00 m)
 *
 * The wall frame (near corner O, along-wall unit u, fixed thickness T) is authored by the FITTER each
 * cycle from the room-polygon wall this door is associated to (params_.wall_O / wall_u / thickness). The
 * single primitive is one thin box: centre_room = O + (s + w/2)·u  (xy)  + (floor_z + h/2)·ẑ, oriented by
 * the wall (local x = u along the wall, local y = n across it, local z up), half-extents [w/2, T/2, h/2].
 *
 * ONE panel primitive ⇒ n_prims()=1, responsibilities → [panel, clutter]. There is NO orientation
 * ambiguity (the wall fixes yaw), so the chair's 4-way resolve_orientation / flip accumulator / mode
 * posterior are GONE. Inference is the shared recursive variational-Laplace filter (3×3 Σ). Pure Eigen,
 * no torch, no DSR — unit-testable in isolation (self_test()).
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"

namespace rc
{

struct DoorBeliefState
{
    float s = 0.0f;    // along-wall offset of the door's near edge from the wall corner O (m)
    float w = 0.70f;   // panel width  (m, along the wall)
    float h = 2.00f;   // panel height (m, vertical; base pinned to the floor)

    Eigen::Matrix<float, 3, 1> vec() const
    { return (Eigen::Matrix<float, 3, 1>() << s, w, h).finished(); }
    static DoorBeliefState from_vec(const Eigen::Matrix<float, 3, 1>& v)
    { return {v(0), v(1), v(2)}; }
};

struct DoorBeliefParams
{
    // ── Wall frame (authored by the fitter from the associated room-polygon wall). The door lies IN this
    //    wall plane; s is measured along u from the near corner O; the panel is T thick across the wall. ──
    Eigen::Vector2f wall_O = {0.0f, 0.0f};   // near corner of the wall (room frame, m)
    Eigen::Vector2f wall_u = {1.0f, 0.0f};   // along-wall unit direction (room frame; MUST be normalised)
    float thickness = 0.05f;                 // panel thickness T (m, across the wall — fixed, not a DOF)
    float floor_z   = 0.0f;                  // room-frame floor: the door base is PINNED here (not a DOF)
    float floor_std = 0.03f;                 // floor-height uncertainty (m) → published z covariance

    // Observation model
    float sigma_base_m    = 0.03f;
    float clutter_frac    = 0.10f;
    float clutter_scale_m = 0.12f;

    // ── Priors (seed Σ diagonal). s is the localised DOF (BROAD); w,h are STRONG template priors. Because
    //    the shared engine's prior is TEMPORAL (it pulls each frame toward the PREVIOUS state, not a fixed
    //    template), a strong w/h prior is realised as a tight seed Σ + tiny process noise: Σ_ww/Σ_hh stay
    //    small so the per-frame prior keeps re-anchoring w,h near their seeded 0.70/2.00, and only sustained
    //    consistent evidence moves them. The per-frame common-mode on w,h (below) caps how far any single
    //    mask can push them. No gate — the arbitration is a covariance ratio. ──
    float prior_s_std = 0.60f;   // broad: the fit localises s
    float prior_w_std = 0.06f;   // strong: standard door leaf ≈ 0.70 m
    float prior_h_std = 0.08f;   // strong: standard door leaf ≈ 2.00 m

    // FIXED template anchor: a genuine non-drifting Gaussian prior N(tpl_w,prior_w_std²) / N(tpl_h,prior_h_std²)
    // applied EVERY frame via accumulate_extra. The shared engine's prior is TEMPORAL (it pulls toward the
    // previous state), so w,h have no fixed anchor and a persistently over-/under-segmented mask can slowly
    // random-walk them away from a standard leaf (observed live: w → 1.12 m). This factor pins them to the
    // template — it costs free energy to move w,h, and only sustained, precise evidence pays for it. No gate.
    float tpl_w = 0.70f;
    float tpl_h = 2.00f;

    // Associated wall segment length (m), authored by the fitter. The panel must lie WITHIN its wall, so
    // apply_constraints clamps s ∈ [0, wall_len − w] when this is > 0 (a door can't run past the corner).
    float wall_len = 0.0f;

    // Temporal transition (static furniture). s tracks localisation jitter; w,h barely move.
    float process_std_s = 0.005f;
    float process_std_w = 0.001f;
    float process_std_h = 0.001f;

    // Per-frame COMMON-MODE error (shared by all points of a mask → doesn't average out; Woodbury cap in the
    // engine, so N correlated points can't collapse σ). chain_cov_s (pose-chain along-wall var) adds to s.
    float common_mode_s_std  = 0.03f;
    // A mask's WIDTH/HEIGHT carries a large per-frame SHARED error (the segmentation includes/excludes the
    // frame, jamb, or a neighbour consistently across all its points), so a single frame must not be able to
    // out-vote the template anchor. Keep this wide (cap 1/std² BELOW the anchor precision 1/prior_{w,h}_std²)
    // so one over-segmented mask only nudges w,h — the fix for the live w→1.12 m drift. Continuous, no gate.
    float common_mode_wh_std = 0.35f;

    // Optimiser
    int   gn_iters = 4;
    float fd_eps   = 1e-3f;
};

// One fitted frame's evidence (room-frame points + per-point measurement variance R). chain_cov_s carries
// the SHARED pose-chain + static-range variance the fitter computes for the along-wall position (added to
// the s common-mode). Dimensions (w,h) have no localisation-chain term.
struct DoorFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;
    float chain_cov_s = 0.0f;   // along-wall position variance from the pose chain (m²)
};

class DoorBelief
{
public:
    static constexpr int N = 3;
    using State = DoorBeliefState;
    using Frame = DoorFrame;

    DoorBelief() = default;
    DoorBelief(const DoorBeliefState& s, const DoorBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const DoorBeliefState&             state()      const { return state_; }
    const Eigen::Matrix<float, 3, 3>&  covariance() const { return Sigma_; }
    const DoorBeliefParams&            params()     const { return params_; }
    void set_state(const DoorBeliefState& s) { state_ = s; }
    void set_params(const DoorBeliefParams& p) { params_ = p; }
    // Author the wall frame this door is anchored to (the fitter calls this each cycle from the associated
    // room-polygon wall). u is normalised defensively so the SDF's along/across split stays orthonormal.
    void set_wall(const Eigen::Vector2f& O, const Eigen::Vector2f& u, float len = 0.0f)
    {
        params_.wall_O = O;
        const float n = u.norm();
        params_.wall_u = (n > 1e-6f) ? (u / n).eval() : Eigen::Vector2f(1.0f, 0.0f);
        params_.wall_len = std::max(0.0f, len);
    }

    // ── Geometry read-back for the fitter write-back / scene-graph publish ──────────
    float width()     const { return state_.w; }
    float height()    const { return state_.h; }
    float thickness() const { return params_.thickness; }
    float cz()        const { return params_.floor_z; }                       // pinned floor datum (base)
    float center_z()  const { return params_.floor_z + 0.5f * state_.h; }     // panel centre height
    // Room-frame (x,y) of the panel centre: along the wall by s+w/2 from the near corner O.
    Eigen::Vector2f center_xy() const { return params_.wall_O + (state_.s + 0.5f * state_.w) * params_.wall_u; }
    // Room-frame yaw = the wall tangent direction (the door's local +x runs along the wall).
    float yaw() const { return std::atan2(params_.wall_u.y(), params_.wall_u.x()); }

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const DoorFrame& frame) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()                      { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    // Age the belief with NO measurement: Σ ← FΣFᵀ + Q·(dt/dt_nominal), mean held. The fitter calls this
    // when a door's mask stream is stale/dead so Σ grows on the agent's clock instead of freezing.
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    { ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s); }
    Eigen::Matrix<float, 3, 3> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }

    // ── Generative-model hooks (engine interface + SDF API) ───────────────────
    // Single thin-box panel SDF in the wall frame, evaluated at room point p for state s. 0 on the surface;
    // >0 outside; <0 inside.
    float sdf_panel(const Eigen::Vector3f& p, const DoorBeliefState& s) const;
    float sdf_prim (const Eigen::Vector3f& p, const DoorBeliefState& s, int prim) const;  // prim==0 → panel
    Eigen::Matrix<float, 3, 1> sdf_jacobian(const Eigen::Vector3f& p, const DoorBeliefState& s, int prim) const;
    std::array<float, 2> responsibilities(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const;
    // Fixed template prior on w,h folded into the SAME GN normal equations as the data (engine detects this
    // hook via a C++23 requires). Pulls w→tpl_w, h→tpl_h with precision 1/prior_{w,h}_std² each frame; s is
    // untouched (it carries only the broad temporal prior). This is what makes the strong size prior stick.
    void accumulate_extra(const DoorBeliefState& s, const DoorFrame& frame,
                          Eigen::Matrix<float, 3, 3>& Id, Eigen::Matrix<float, 3, 1>& bd) const;

    // Mean per-point mixture NEGATIVE LOG-LIKELIHOOD of `pts` (includes clutter) — the multi-instance
    // ASSOCIATION evidence. A far / wrong-instance slice (all points → flat clutter) → tiny likelihood →
    // HIGH nll → not claimed (unlike mean_energy, which scores it ~0).
    float association_nll(const std::vector<Eigen::Vector3f>& pts, float R) const;
    // Mean per-point mixture NLL at an ARBITRARY state — the proper likelihood used e.g. by the fitter.
    float mixture_nll(const std::vector<Eigen::Vector3f>& pts, const DoorBeliefState& s, float R) const;
    // Mean per-point data energy (responsibility-weighted SDF), at an ARBITRARY state.
    float mean_energy(const std::vector<Eigen::Vector3f>& pts, const DoorBeliefState& s, float R) const;
    // Mean clutter responsibility over the frame's points (diagnostic; off-model / contamination fraction).
    float clutter_fraction(const std::vector<Eigen::Vector3f>& pts, float R) const;

    // The wall fixes yaw, so there is no discrete-orientation ambiguity to report: the reported covariance
    // IS the 3×3 posterior Σ over [s,w,h]. Kept as a named accessor so publish/log call sites read cleanly.
    Eigen::Matrix<float, 3, 3> covariance_reported() const { return Sigma_; }

    void  apply_constraints(DoorBeliefState& s) const;
    void  canonicalize(DoorBeliefState&) const {}   // wall fixes yaw ⇒ no symmetry fold

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 1; }             // one thin panel (clutter is the +1)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 3, 3> transition() const { return Eigen::Matrix<float, 3, 3>::Identity(); }  // static
    Eigen::Matrix<float, 3, 1> process_noise_diag() const;
    Eigen::Matrix<float, 3, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 3, 1> common_mode_inv_diag(const DoorFrame& frame) const;

    static bool self_test();

private:
    // Unnormalized mixture terms [panel, clutter] at p (likelihood×prior). Shared by responsibilities()
    // (normalizes) and association_nll() (sums).
    std::array<float, 2> mixture_unnorm(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const;

    DoorBeliefState            state_;
    DoorBeliefParams           params_;
    Eigen::Matrix<float, 3, 3> Sigma_ = Eigen::Matrix<float, 3, 3>::Identity();
    Eigen::Matrix<float, 3, 1> prior_mean_ = Eigen::Matrix<float, 3, 1>::Zero();
};

}  // namespace rc
