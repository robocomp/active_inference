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
#include <optional>
#include <limits>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"
#include "../../common/ai_belief/lidar_ray_factor.h"   // rc::LidarRays + lidar_ray_cost
#include "door_geometry.h"      // rc::door:: Aperture / LeafState / LeafPose — the single geometry source

namespace rc
{

struct DoorBeliefState
{
    float s = 0.0f;    // along-wall offset of the door's near edge from the wall corner O (m)
    float w = 0.70f;   // panel width  (m, along the wall)
    float h = 2.00f;   // panel height (m, vertical; base pinned to the floor)
    // ★Task 5b fix round 1: the LEAF PLANE's depth beyond the wall line (m, positive = OUT of the room, i.e. along
    // −across_u, which associate_wall points INTO the room). A per-door state, estimated with s, w, h; see
    // DoorBeliefParams::recess_prior_m for its prior and door::Aperture::depth for who renders it.
    float d = 0.0f;

    Eigen::Matrix<float, 4, 1> vec() const
    { return (Eigen::Matrix<float, 4, 1>() << s, w, h, d).finished(); }
    static DoorBeliefState from_vec(const Eigen::Matrix<float, 4, 1>& v)
    { return {v(0), v(1), v(2), v(3)}; }
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

    // ── Leaf articulation (openable door, M0) ────────────────────────────────────────────────────
    // The hinged panel's opening angle + which aperture edge it hangs from. NOT part of theta (theta is [s, w, h, d]; d, the leaf plane's depth, since Task 5b)
    // and the shared inference engine is untouched. M0 pins phi = 0 (leaf flush in the aperture, so every
    // geometry consumer reduces exactly to the old wall-plane behaviour); phi becomes a fitted DOF in M1
    // and hinge/swing become discrete hypotheses in M2. Authored by DoorFitter::make_belief_params.
    door::LeafState leaf{};

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
    // Length of the associated wall run (informational). s is NOT clamped to it: see apply_constraints.
    float wall_len = 0.0f;

    // Temporal transition (static furniture). s tracks localisation jitter; w,h barely move.
    // ★A DOORWAY DOES NOT MIGRATE ALONG ITS WALL, so the process noise on `s` is ZERO. This is not a
    // freeze and not a gate: it is the generative model saying what kind of thing an aperture is. With
    // no noise injected, the posterior precision on `s` only ever grows, the Kalman gain on it shrinks
    // in proportion, and the mean stops being dragged — continuously, with no switch anywhere and no
    // moment at which behaviour changes.
    // ★IT IS THE DEGENERATE DIRECTION, which is why it matters. `s` and the leaf angle phi explain the
    // same observations: the fit slides the aperture along the wall to keep a SWUNG leaf on the mask.
    // Measured: r = -1.000 between the slide and the swing over 27 rows, a 34 cm slide against a 41 deg
    // swing — one degree of freedom estimated twice by two procedures chasing each other along a
    // degenerate valley. A 5 mm/cycle random walk on `s` is exactly the licence to do that. Removing it
    // leaves phi as the only parameter that can absorb a swing, which is the one that physically moves.
    // ⚠A wrong aperture is then corrected by EVIDENCE against a finite Σ — slowly, in proportion to how
    // confident we are — or removed by the existence channel, which is the designed path for "this door
    // is not where we think". It is never corrected by injected noise, because injected noise cannot
    // tell a real correction from a swing it should not be absorbing.
    float process_std_s = 0.0f;
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

    // ★THE LEAF PLANE'S DEPTH IS A STATE OF THE DOOR (Task 5b, D1 + D2; fix round 1). The model used to draw the leaf ON
    // the room polygon's wall line. On both recorded tours the shut DOOR_0 leaf's mask points sit 0.10-0.15 m off that
    // line — on the ROOM side of it (associate_wall's normal points into the room; the camera sat at +2.7 m along it).
    // No DOF of [s, w, h] explains a shared across-wall offset, and Gauss-Newton turned it into motion along whatever DOF
    // had the faintest lever on it: w through −sin φ at φ = 0.03 (tourR2 door_2: w 1.01 → 2.51 m in one frame, 36 σ, then
    // removed for a silhouette 2.5 m wide), s through float noise in the finite-difference Jacobian (tourR1 door_4: a
    // 3.66 m first update that explained none of its own points). The leaf hangs at ONE depth: d is the fourth DOF of
    // the belief, estimated by the same Gauss-Newton as s, w, h, with no process noise (a doorway does not move in its
    // wall); a frame's shared across error is the SAME common mode its s already carries (pose chain + range + motion,
    // plus the leaf angle's lever, common_mode_inv_diag). Every consumer renders the leaf on that plane (Aperture::depth).
    // ★THE PRIOR IS N(0, T²/12 + σ_b²) — GAUSSIAN, UNBOUNDED, CENTRED ON THE LINE — not a [0, T] support, and the data say why.
    //   · Centred on the line: room_concept traces the polygon from FREE SPACE, and at a shut doorway the first thing free
    //     space meets is the leaf, so the line passes through the leaf, not in front of a recess. Measured: tourR1/R2 DOOR_0
    //     −0.10..−0.15 m (room side), the synthetic fixtures ≈ 0. A T/2 mean (the leaf mid-wall behind the face) was tried
    //     and is 2 σ from DOOR_0; started there, the low LiDAR's leaf angle took the offset as a swing (φ 0.20 on a shut
    //     door in test_door_core_fit) before the masks could move d.
    //   · The spread: where in the wall's thickness T the leaf hangs relative to the wall beside it (uniform: T²/12) plus the
    //     polygon's own registration error σ_b (Existence.BoundarySigmaM, 0.10 m) — the same T and σ_b as the wall band.
    //   · Unbounded: d is measured from a line that carries registration error, so no physical bound applies to it; a bounded
    //     parameter would pin at its bound and leave the rest of the offset to the levers again — D1 re-created.
    // Authored by DoorFit::make_belief_params.
    float recess_prior_m   = 0.0f;   // prior mean of d (m)
    float recess_prior_std = 0.0f;   // prior σ of d (m); 0 ⇒ d is pinned at its prior (the pre-5b leaf on the line)

    // Optimiser
    int   gn_iters = 4;
    float fd_eps   = 1e-3f;
};

// One fitted frame's evidence (room-frame points + per-point measurement variance R). chain_cov_s carries
// the SHARED pose-chain + static-range variance the fitter computes for the along-wall position (added to
// the s common-mode). Dimensions (w,h) have no localisation-chain term.
// ★Task D-C review C1: WHERE THE MASK'S SIDES ARE, AS BEARINGS. The ZED mask's support points carry the depth map's
// range error along the viewing ray; from an oblique view that error moves s and the leaf plane d TOGETHER, and the fit
// slid DOOR_0's aperture 0.35-0.4 m along its wall (rtD1i, ~200 cycles with masks present) — the frame channel then read
// the misplacement as absence and removed a door YOLO was detecting. The mask's left/right image columns are a
// DEPTH-FREE observation: the bearings, from the camera, of the outermost parts of the door — its jambs on the wall line,
// or the leaf's free edge where a swung leaf stands outside them. Censored (NaN) where the mask touches the image border.
struct MaskBearings
{
    Eigen::Vector2f cam = Eigen::Vector2f::Zero();   // the camera, room frame (x, y)
    float ref = 0.0f;                                 // the camera's forward yaw (room frame): angles are taken relative to it
    float lo = std::numeric_limits<float>::quiet_NaN(), hi = std::numeric_limits<float>::quiet_NaN();   // relative bearings
    float var = 0.0f;                                 // bearing variance (rad²)
};
struct DoorFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;
    float chain_cov_s = 0.0f;   // along-wall position variance from the pose chain (m²)
    std::optional<MaskBearings> bearings;   // Task D-C: the mask's sides (empty = not measured)
};

class DoorBelief
{
public:
    static constexpr int N = 4;   // [s, w, h, d] — d the leaf plane's depth (Task 5b)
    // ★WHAT THE BELIEF REPORTS is [s, w, h] — the aperture. d is the belief's own nuisance-like state (recess(),
    // recess_sd()). Every consumer of a reported matrix (Σ, the NBV information, the DOF table, the dashboard, the RT
    // covariance) is sized by THIS, never by N: a DOF table sized N indexed the 3×3 Σ at (3,3) and killed the agent live
    // (rtD1g, 2026-10-21, EpistemicPlanner::compute → adequacy_gap_nats).
    static constexpr int kReported = 3;
    using State = DoorBeliefState;
    using Frame = DoorFrame;

    DoorBelief() = default;
    DoorBelief(const DoorBeliefState& s, const DoorBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const DoorBeliefState&             state()      const { return state_; }
    // The posterior over [s, w, h] — what every consumer outside the belief reads (the RT covariance, the planner, the
    // along-wall channel's s); the leaf plane's depth d is the belief's own (covariance_full, recess_sd).
    Eigen::Matrix<float, 3, 3>         covariance() const { return Sigma_.topLeftCorner<3, 3>(); }
    const Eigen::Matrix<float, 4, 4>&  covariance_full() const { return Sigma_; }
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
    // NOTE: center_xy()/yaw() describe the APERTURE — the static hole in the wall. They are what the DSR RT
    // edge, resolve_wall, merge, ghost identity and the room-containment prior key on, precisely because
    // they cannot be dragged by a swinging leaf. For the LEAF's current pose use leaf_pose() below.
    float width()     const { return state_.w; }
    float height()    const { return state_.h; }
    float thickness() const { return params_.thickness; }
    float cz()        const { return params_.floor_z; }                       // pinned floor datum (base)
    float center_z()  const { return params_.floor_z + 0.5f * state_.h; }     // panel centre height
    // Room-frame (x,y) of the APERTURE centre: along the wall by s+w/2 from the near corner O.
    Eigen::Vector2f center_xy() const { return params_.wall_O + (state_.s + 0.5f * state_.w) * params_.wall_u; }
    // Room-frame yaw of the APERTURE = the wall tangent (the aperture's local +x runs along the wall).
    float yaw() const { return std::atan2(params_.wall_u.y(), params_.wall_u.x()); }

    // ── Aperture / leaf geometry (door_geometry.h is the single source of truth) ──────────
    // The aperture at an arbitrary state (the SDF needs this for the finite-difference Jacobian), and at
    // the current one. `leaf` carries the articulation; with phi pinned at 0 the leaf IS the aperture.
    door::Aperture aperture_at(const DoorBeliefState& s) const
    {
        door::Aperture a;
        a.wall_O = params_.wall_O; a.wall_u = params_.wall_u;
        a.s = s.s; a.w = s.w; a.h = s.h;
        a.floor_z = params_.floor_z; a.thickness = params_.thickness;
        a.depth = -s.d;   // d is OUT of the room; across_u points INTO it (associate_wall)
        return a;
    }
    // The leaf plane's depth beyond the wall line (m) and its posterior σ (Task 5b).
    float recess()    const { return state_.d; }
    float recess_sd() const { return std::sqrt(std::max(0.0f, Sigma_(3, 3))); }
    door::Aperture aperture()   const { return aperture_at(state_); }
    door::LeafPose leaf_pose_at(const DoorBeliefState& s) const { return door::leaf_pose(aperture_at(s), params_.leaf); }
    door::LeafPose leaf_pose()  const { return leaf_pose_at(state_); }
    // ★M1: phi is no longer a pinned construction constant. It is estimated per cycle (DoorFitter::
    // estimate_phi) from silhouette agreement, with the robot's OWN actuation command as its prior —
    // so a door the robot asked to be opened is ANTICIPATED rather than experienced as the object
    // disappearing. It stays outside theta: the belief is [s, w, h, d] (d since Task 5b) and the shared inference
    // engine is untouched, exactly as door_geometry.h's header anticipated.
    void  set_leaf_phi(float phi) { params_.leaf.phi = phi; }
    void  set_leaf_hinge(door::HingeSide h) { params_.leaf.hinge = h; }
    void  set_leaf_swing(float swing) { params_.leaf.swing = swing; }   // Task 12 fix D: the MAP swing side (±1)
    // Task 15: fuse a Gaussian observation of the aperture's slide along the wall, innovation `ds` (m, relative to the
    // current s) with variance `var` — a 1-D Kalman update on [s, w, h] through Σ's first column (w and h follow their
    // correlation with s). var ≤ 0 or non-finite ⇒ nothing.
    void  fuse_slide(float ds, float var)
    {
        if (not std::isfinite(ds) or not std::isfinite(var) or not (var > 0.f)) return;
        const float S = Sigma_(0, 0) + var;
        if (not (S > 0.f)) return;
        const Eigen::Matrix<float, N, 1> K = Sigma_.col(0) / S;
        state_ = DoorBeliefState::from_vec(state_.vec() + K * ds);
        const Eigen::Matrix<float, N, N> Sn = Sigma_ - K * Sigma_.row(0);
        Sigma_ = 0.5f * (Sn + Sn.transpose());
    }
    // try96 (door_opening_edges.h): the LiDAR's along-wall measurement observes s PLUS the scan's registration offset b,
    // a state the caller carries (mean `bias`, variance `Pb`, cross-covariance `c` with [s, w, h]). A probabilistic-data-
    // association update of the augmented [s, w, h, b] with H = [1 0 0 1], innovation `nu`, measurement variance `R` and
    // association weight `beta` ∈ [0, 1] (the PDA covariance: shrunk by β·KSKᵀ, re-grown by β(1−β)·Kν²Kᵀ). Writes Σ,
    // the state, and the caller's bias / Pb / c back.
    void  fuse_along_wall(float nu, float R, float beta, float& bias, float& Pb, Eigen::Matrix<float, 4, 1>& c)
    {
        if (not std::isfinite(nu) or not std::isfinite(R) or not (beta > 0.f)) return;
        using MA = Eigen::Matrix<float, N + 1, N + 1>;
        using VA = Eigen::Matrix<float, N + 1, 1>;
        MA A = MA::Zero();
        A.topLeftCorner<N, N>() = Sigma_;
        A.block<N, 1>(0, N) = c;
        A.block<1, N>(N, 0) = c.transpose();
        A(N, N) = Pb;
        VA H = VA::Zero(); H(0) = 1.f; H(N) = 1.f;
        const float S = H.dot(A * H) + std::max(0.f, R);
        if (not (S > 0.f)) return;
        const VA K = A * H / S;
        const float bw = std::min(1.f, beta);
        state_ = DoorBeliefState::from_vec(state_.vec() + bw * nu * K.head<N>());
        bias += bw * K(N) * nu;
        MA An = A - bw * S * (K * K.transpose()) + bw * (1.f - bw) * nu * nu * (K * K.transpose());
        An = 0.5f * (An + An.transpose());
        Sigma_ = An.topLeftCorner<N, N>();
        c = An.block<N, 1>(0, N);
        Pb = std::max(0.f, An(N, N));
    }

    // ★try96: THE MASK SEES THE SAME REGISTRATION OFFSET. Camera and LiDAR are registered to the room by ONE pose, so a mask
    // frame observes s + b exactly as the LiDAR's opening does (door_opening_edges.h). This runs the engine's update (the
    // GN-Laplace mask fit, unchanged) and then re-expresses what it learned — its information J = Σ₁⁻¹ − Σ₀⁻¹ and the
    // matching information vector — as a measurement of [s + b, w, h] on the augmented [s, w, h, b], so b, its variance Pb
    // and its cross-covariance c with [s, w, h] are updated with it. Without this a persistent mask-vs-LiDAR conflict (a
    // wrong leaf model, which is not a pose error) would be absorbed into b and the LiDAR would lose every argument to the
    // mask (test_door_phi_stability (g)). The caller removes the pose chain from the frame's common mode: b carries it.
    float update_sharing_offset(const DoorFrame& frame, float& bias, float& Pb, Eigen::Matrix<float, 4, 1>& c)
    {
        using MN = Eigen::Matrix<float, N, N>;
        using MA = Eigen::Matrix<float, N + 1, N + 1>;
        using VA = Eigen::Matrix<float, N + 1, 1>;
        const Eigen::Matrix<float, N, 1> x0 = state_.vec();
        MN Sp = Sigma_;
        if (Sp.isZero()) Sp.diagonal() = prior_cov_diag();
        Sp.diagonal() += process_noise_diag();                 // the engine's own predict step (F = I)
        const float e = update(frame);
        if (not (Pb > 1e-10f)) return e;                       // no offset to share: the engine's result stands
        const MN L1 = Sigma_.inverse(), L0 = Sp.inverse();
        MN J = L1 - L0;
        J = 0.5f * (J + J.transpose());
        const Eigen::Matrix<float, N, 1> g = L1 * state_.vec() - L0 * x0;
        for (int i = 0; i < N; ++i)
        {
            const float lim = 0.99f * std::sqrt(std::max(0.f, Sp(i, i) * Pb));
            c(i) = std::clamp(c(i), -lim, lim);
        }
        MA A = MA::Zero();
        A.topLeftCorner<N, N>() = Sp;
        A.block<N, 1>(0, N) = c;
        A.block<1, N>(N, 0) = c.transpose();
        A(N, N) = Pb;
        // the mask measures [s + b, w, h, d]: H is the identity on [s, w, h, d] with b added to s
        Eigen::Matrix<float, N, N + 1> H = Eigen::Matrix<float, N, N + 1>::Zero();
        H.leftCols<N>().setIdentity(); H(0, N) = 1.f;
        const MA Lam = A.inverse();
        VA xa; xa << x0, bias;
        const MA Lam1 = Lam + H.transpose() * J * H;
        const VA eta1 = Lam * xa + H.transpose() * g;
        MA A1 = Lam1.inverse();
        if (not A1.allFinite()) return e;
        A1 = 0.5f * (A1 + A1.transpose());
        const VA x1 = A1 * eta1;
        if (not x1.allFinite()) return e;
        DoorBeliefState st = DoorBeliefState::from_vec(x1.head<N>());
        apply_constraints(st);
        state_ = st;
        Sigma_ = A1.topLeftCorner<N, N>();
        c = A1.block<N, 1>(0, N);
        Pb = std::max(0.f, A1(N, N));
        bias = x1(N);
        return e;
    }

    // ── THE HINGE BRANCH: free energy as a function of the leaf angle alone ──────────────────────
    // ★THIS IS THE SAME MINIMISATION THE AGENT ALREADY DOES, RESTRICTED TO ONE COORDINATE. Nothing new
    // is being introduced: the model has a hinge, phi is the coordinate of that hinge, and this is the
    // surface term of the same free energy evaluated along it with [s, w, h] held. The aperture is a
    // hole in a wall and cannot move, so holding it is not an approximation — it is the statement that
    // only one thing in this object is articulated.
    // ★AND IT IS THE CURE FOR THE DEGENERACY, structurally. `s` and phi explain the same observations:
    // the fit slides the aperture along the wall to keep a swung leaf on the mask (measured r = -1.000,
    // 34 cm of slide against 41 deg of swing). Optimising them jointly lets the pair wander that valley;
    // optimising phi with `s` held leaves the swing nowhere to go but into the parameter that swings.
    // Returns 0.5 * SUM d^2 / (sigma^2 + R_i) — the Gaussian surface term, in nats, comparable across
    // angles for the SAME frame. Points are room-frame; `R` is per-point measurement variance.
    // ⚠It can only answer about points the leaf could explain. A cloud with nothing on the leaf yields a
    // flat curve, which is the honest output of a measurement that was not taken — not a vote for phi = 0.
    [[nodiscard]] float phi_free_energy(const DoorFrame& f, float phi) const;

    // ★THE RAY VERSION, AND IT IS THE ONE THAT CAN SEE A DOORWAY. An aperture is a HOLE: a closed leaf
    // stops the ray at the wall plane; an open one lets it fly through into the next room. That is a fact
    // about WHERE RAYS STOP, including the ones that do not, and no cost built from return points can
    // express it — the through-rays' endpoints are in the next room, so a point-based selection discards
    // exactly the evidence that distinguishes open from closed. Uses the sensor ORIGIN, which a point
    // cloud does not carry.
    [[nodiscard]] float phi_ray_free_energy(const rc::ai::LidarRays& rays, float phi) const;

    // ── THE MEASUREMENT BEFORE THE MODEL CHANGE ──────────────────────────────────────────────────
    // ★HOW MANY OF THESE RAYS DOES THE MODEL ACTUALLY EXPLAIN, at a given angle? With a leaf-only model
    // a CLOSED door should already be decisive: nearly every ray in the aperture column should stop ON
    // the leaf, giving a large free-energy gap against any open hypothesis. Measured 2026-09-13 it was
    // flat instead — which means the selected rays are NOT stopping on the modelled leaf, and that is a
    // selection or geometry fault that no change to the likelihood can repair. Suspects, unverified: the
    // bpearl is a DOWNWARD DOME (ROBOT_GEOMETRY.md), so most of its rays toward a door 2-3 m away may hit
    // the floor first; the aperture's s or floor_z may be out by more than sigma; the real doorway may
    // have a step. `helios`, the plane that sees the leaf at mid-height where the lever arm from the
    // hinge is the full width, is not used for phi at all.
    // ★`mean_signed` says WHERE they stop instead: negative ⇒ the return is SHORTER than the model
    // predicts (something in front — floor, step, frame), positive ⇒ the ray flew PAST the leaf. Those
    // are different faults with different fixes, and a count alone cannot tell them apart.
    struct RayExplain { int n_hit = 0; int n_explained = 0; float mean_signed_m = 0.0f; };
    [[nodiscard]] RayExplain phi_ray_explain(const rc::ai::LidarRays& rays, float phi, float tol_m) const;
    [[nodiscard]] std::vector<std::pair<float, float>>
    phi_ray_likelihood(const rc::ai::LidarRays& rays, float phi_min, float phi_max, int nstep) const;

    // ── THE FULL-FIELD RESPONSE (2026-09-16) — what phi_ray_likelihood could not do ───────────────
    // ★THE BEAM MODEL ABOVE CANNOT SEE AN OPEN DOOR, and this is a property of the MODEL, not a tuning
    // fault. Its only surface is the leaf (n_prims()==1, there is no wall). So a ray crossing an open
    // doorway scores w_u*U under "open" (no predicted surface) and w_u*U under "shut" as well once the
    // return lands beyond the predicted leaf — identical terms, hence a FLAT curve. Measured live
    // 2026-09-16 with the door standing open and the robot in the doorway: cv = 0.038 across 25
    // hypotheses (1/25 is exactly flat) and w_sdf = 0.0393, inside the 0.038..0.047 band the synthetic
    // predicted. phi then carried nothing but our own actuation command.
    //
    // The cure is to model the FIELD the sensor actually reports — every ray in the doorway, whether it
    // stops or flies through — against a wall WITH AN OPENING plus the leaf:
    //   · ray meets a modelled surface at t : (1 - w_u - eps) N(rho; t, sigma) + w_u U[0,t) + eps U(t,cap]
    //   · ray leaves through the opening    : (1 - w_u) U(t_ap, cap] + w_u U[0, t_ap)
    // Now "shut" PREDICTS a return at the leaf and is REFUTED by a ray that flew 2 m past it, which the
    // leaf-only model had no way to express.
    //
    // ★SUMMED, NOT AVERAGED — and that is only safe WITH the nuisance below. Summing is what lets
    // evidence accumulate (the per-ray mean is why sigma_phi GREW from 0.199 to 0.647 rad as the ray
    // count rose 71 -> 918: inert rays diluted the informative ones, so more evidence FLATTENED the
    // curve). But a sum turns any unmodelled structure into certainty: the synthetic picked the wrong
    // angle by 38 nats until the nuisance was added.
    //
    // ★THE NUISANCE IS MEASURED, NOT ASSUMED. The live residual against the modelled leaf is a tight
    // +0.112 m (n=618 shut-consistent rows, sd 0.031, p25..p95 = +0.110..+0.118) — the returns stop
    // ~11 cm NEARER than the model, consistent with a leaf modelled 5 cm thick sitting mid-wall in a
    // ~22 cm wall. The prototype's {0, 3, 6} cm grid could not reach it. `depth` spans that offset and
    // is marginalised out; `lat` does the same for the aperture's along-wall placement.
    struct FieldNuisance
    {
        float depth_lo = -0.20f, depth_hi = 0.20f;   // wall/leaf offset across the wall (m)
        int   depth_n  = 9;
        float lat_lo   = -0.06f, lat_hi   = 0.06f;   // aperture placement along the wall (m)
        int   lat_n    = 3;
        float w_unexp  = 0.10f;   // P(return short of the model: a person, clutter, the frame)
        float eps_beyond = 0.03f; // P(return beyond it: a miss, a reflective surface)
        int   max_rays = 400;     // stride-subsampled; cost is nstep * depth_n * lat_n * rays
        float ray_icc  = -1.0f;   // phi_field_loglik only: < 0 ⇒ derived from the nuisance grid (see there)
        // phi_field_loglik only: the PRIOR over the slide — N(0, lat_sigma²), the aperture placement's own posterior
        // width. <= 0 ⇒ uniform over the grid (the old behaviour). Uniform let a SHUT leaf slide half a door-width onto
        // the solid wall beside the doorway and explain it perfectly (try44: every phi collapse had lat = -0.55 m).
        float lat_sigma = -1.0f;
        // phi_field_loglik only (Task 11): ALSO score the leaf hung from the OTHER jamb (FieldResult::loglik_other).
        // The hinge is a discrete hidden variable of the generative model, not a config fact: DOOR_1's real hinge is
        // on the jamb the config pin (Near) does not name, and with only one hinge in the model the open leaf could be
        // explained only by sliding the aperture one width along the wall (try69: 0.88 m) or by over-swinging it
        // (118 deg for a leaf at 90). The two hypotheses share the frame's nuisances and its tempering.
        bool other_hinge = false;
        // phi_field_loglik only (Task 20): ALSO score the leaf swung to the OTHER side of the wall (FieldResult::loglik_swing,
        // and loglik_both with other_hinge). The swing is a discrete hidden variable exactly as the hinge is: DOOR_1 opens
        // into room_1 while the config pin (Openable.SwingDir) put it into room_2, and with one swing in the model an open
        // leaf on the far side can only be "explained" by over-swinging it behind the wall (room_1: 117 deg for 90) or by a
        // half-shut leaf (try87 room_2: 60 deg, P(passable) 0 for an open door ⇒ `open` re-offered and timed out 22 times).
        bool other_swing = false;
        // ★ THE ROOM THE SENSOR IS IN IS PART OF THE SCENE (try97, 2026-10-19). Vertical wall segments (room frame, from
        //   the floor up) — the room polygon's edges other than the door's own wall (rc::door::room_walls_beside) — are a
        //   modelled surface under EVERY leaf hypothesis: a return on them is the wall's, so a leaf swung toward the sensor
        //   is credited only with returns the room cannot explain, and is refuted by any ray that flew past where it would
        //   stand. Before this the field had no background structure, so returns short of the doorway plane were DROPPED by
        //   the selection (select_doorway_rays, 2026-10-18: the corridor wall beside DOOR_1 read as a leaf at 97° on a shut
        //   door) — which blinded the model to every leaf on the sensor's side and biased the swing AWAY from the observer
        //   (try97: +2309 nats for the wrong swing from room_1, carried to room_2, phi 73°/49° for a leaf at 90°).
        //   Empty ⇒ no background (the field as before, used by the synthetic harnesses).
        std::vector<std::array<Eigen::Vector2f, 2>> room_walls;
        // phi_field_loglik only (Task D-C, rtD1i door_5): ★THE ROOM'S WALLS ARE PLACED WITH THE SAME UNCERTAINTY AS THE
        //   DOOR'S OWN WALL. Each room wall carries a per-frame offset along its normal, a COMMON MODE shared by every return
        //   on it, prior N(0, room_wall_sigma²) — the room polygon's face uncertainty (Existence.BoundarySigmaM, the σ the
        //   wall-band prior already reads) — marginalised on the door wall's own depth grid [depth_lo, depth_hi] × depth_n.
        //   Before this the door's leaf had two placement nuisances (depth, slide) and the room walls had none: a partition
        //   wall 2.7 cm off its polygon edge, seen from the far side of a SHUT DOOR_0 with the doorway itself occluded, was
        //   fitted better by a leaf at 90° slid 0.17 m onto that wall than by the wall it is (17-35 rays, +2..+10 nats a
        //   frame ⇒ phi 0.18 → 1.55 in 0.8 s, then the image channels refuted the "open" leaf: L +4 → −4). One side was
        //   scored at its optimum. <= 0 ⇒ rigid room walls (the old field, exactly).
        float room_wall_sigma = 0.0f;
        // phi_field_loglik only (Task D-C): ★THE APERTURE LIES ON ITS OWN WALL. The along-wall extent [wall_seg_lo,
        //   wall_seg_hi] (m from Aperture::wall_O along wall_u) of the polygon edge(s) the doorway is cut in, placed with the
        //   same polygon-face σ (room_wall_sigma). The slide's prior gains log Φ((s+lat − lo)/σ) + log Φ((hi − s−lat−w)/σ):
        //   an aperture slid past the end of its wall — into the perpendicular wall at that corner — is improbable in
        //   proportion to how far past it is. rtD1i door_5: the 90° leaf won by sliding the aperture +0.20 m (3.3σ of the
        //   placement) until the leaf lay on the partition wall that meets DOOR_0's wall 7 cm from its jamb, i.e. with the
        //   hinge 0.2 m inside that partition. Non-finite ⇒ unbounded (the old prior).
        float wall_seg_lo = std::numeric_limits<float>::quiet_NaN(), wall_seg_hi = std::numeric_limits<float>::quiet_NaN();
    };
    [[nodiscard]] std::vector<std::pair<float, float>>
    phi_field_likelihood(const rc::ai::LidarRays& rays, float phi_min, float phi_max, int nstep,
                         const FieldNuisance& nz) const;

    // ── THE SAME FIELD MODEL, TEMPERED BY WHAT THE FRAME ACTUALLY OBSERVES (2026-10-13) ───────────
    // ★WHY phi_field_likelihood ALONE FLIPPED THE CROSS AFFORDANCE. Its per-ray log densities are SUMMED
    // over every selected ray as if each were an independent observation. They are not: every ray in a
    // doorway shares the same structural model error (the aperture's placement, the wall's thickness,
    // the reveal and casing the model does not have), so a misfit common to all of them is counted
    // hundreds of times. Live (try36-try40, the crossed door carried into room_2 with its leaf at ~95 deg
    // OUT of that room) the curve came out as a DELTA every frame — dbg_phi_cv = 4.90 = sqrt(24), all of
    // the weight on ONE of 25 hypotheses — on a viewpoint-dependent angle, so P(passable) went 1 -> 0 -> 1.
    // Two changes, both in the generative model:
    //  1. ★THE APERTURE'S PLACEMENT IS A NUISANCE OVER ITS HONEST SUPPORT, NOT +-6 cm. The carried mirror
    //     sat 0.23-0.40 m along the wall from the real doorway (actuator log, "nearest by place", every
    //     room_2 request). With +-6 cm the model had NO hypothesis that fits the frame, so whichever
    //     angle best covered the real wall inside the misplaced opening won by hundreds of nats. The
    //     caller now sets lat_lo/lat_hi to +-w/2 (a doorway misplaced by more than half its own width is a
    //     different doorway); the jamb rays pin the offset, so a right aperture is unaffected.
    //  2. ★THE FRAME'S EVIDENCE COUNTS WHAT IT DISCRIMINATES, ONCE PER INDEPENDENT ERROR. A ray's density
    //     that does not change with phi (it hits the wall beside the doorway, it cannot reach the leaf under
    //     ANY hypothesis) is no observation of the leaf: n_inf = sum_i (1 - exp(-D_i)), D_i = the spread of
    //     ray i's log density across the phi grid at the MAP nuisance — 0 for a ray no hypothesis can tell
    //     apart, -> 1 for one that separates them by nats. And n_inf rays sharing a common-mode error of
    //     intraclass correlation rho carry n_eff = n_inf / (1 + (n_inf - 1) rho) independent observations
    //     (Kish's design effect; the Woodbury common mode this codebase already uses on mask points). The
    //     joint log-likelihood is raised to beta = n_eff / n_inf before the nuisances are marginalised —
    //     a power likelihood, continuous in both counts, no gate. A frame that sees a sliver of the leaf
    //     carries a sliver of evidence; a frame that sees it whole is decisive, but saturates at 1/rho
    //     observations instead of growing without bound with the ray count.
    //     rho is NOT tuned: the nuisances are marginalised on a GRID, so every ray carries the same
    //     residual offset of up to half a grid step, variance step^2/12 per nuisance axis — the minimum
    //     common mode the model itself guarantees: rho = s_c^2 / (s_c^2 + sigma^2), sigma = range noise.
    //     (FieldNuisance::ray_icc >= 0 overrides it.)
    struct FieldResult
    {
        std::vector<std::pair<float, float>> loglik;   // (phi, tempered log-likelihood), max == 0
        float n_inf    = 0.0f;   // rays that discriminate between angles (soft count)
        float beta     = 1.0f;   // power applied to the joint log-likelihood
        float rho      = 0.0f;   // the intraclass correlation used
        float lat_map  = 0.0f;   // MAP aperture slide along the wall (m) — a placement error, if large
        float depth_map = 0.0f;  // MAP wall/leaf offset across the wall (m)
        int   n_rays   = 0;      // rays scored (after stride subsampling)
        // FieldNuisance::other_hinge only: the same curve with the leaf on the OTHER jamb, on the SAME scale as
        // `loglik` (their joint max is 0, so log Σ exp of each is that hinge's marginal evidence up to one constant).
        // Empty when not asked for; then `loglik` alone has max 0, as before.
        std::vector<std::pair<float, float>> loglik_other;
        // FieldNuisance::other_swing only (Task 20), same scale: the believed hinge with the leaf on the OTHER side of the
        // wall, and (with other_hinge too) both flipped. Empty when not asked for.
        std::vector<std::pair<float, float>> loglik_swing, loglik_both;
        // Task 15: the SLIDE's own evidence — per slide cell (offset from the belief's s, m), the tempered log-likelihood
        // marginalised over the leaf (every angle, both hinges when scored) and the depth, with the slide's prior NOT
        // included. Where the WALL'S HOLE is, from the wall's ends and the beams through it, whatever the leaf is doing.
        std::vector<std::pair<float, double>> lat_loglik;
    };
    [[nodiscard]] FieldResult
    phi_field_loglik(const rc::ai::LidarRays& rays, float phi_min, float phi_max, int nstep,
                     const FieldNuisance& nz) const;

    // Argmin over a grid, plus the whole curve so a consumer can marginalise instead of conditioning on
    // the winner. Weights are exp(-(F - F_min)): a proper likelihood over phi, normalised by the caller.
    [[nodiscard]] std::vector<std::pair<float, float>>
    phi_likelihood(const DoorFrame& f, float phi_min, float phi_max, int nstep) const;
    float leaf_phi() const { return params_.leaf.phi; }
    Eigen::Vector2f leaf_centre_xy() const { return leaf_pose().centre_xy; }
    float           leaf_yaw()       const { return leaf_pose().yaw(); }

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const DoorFrame& frame) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()                      { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    // Age the belief with NO measurement: Σ ← FΣFᵀ + Q·(dt/dt_nominal), mean held. The fitter calls this
    // when a door's mask stream is stale/dead so Σ grows on the agent's clock instead of freezing.
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    { ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s); }
    // The Fisher information on [s, w, h] that observing `pts` would add, with d MARGINALISED: the Schur complement of the
    // 4×4 information, d regularised by its own current posterior (Σ_dd). Not the raw top-left block — that books, as
    // information on [s, w, h], every lever the view shares with d, which it cannot tell from d (the NBV would overrate
    // exactly the views D1/D2 were made of).
    Eigen::Matrix<float, 3, 3> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    {
        const Eigen::Matrix<float, N, N> I = ai::predicted_information<N>(*this, state_, pts, R);
        const float prec_d = 1.0f / std::max(1e-12f, Sigma_(3, 3));
        const Eigen::Matrix<float, 3, 1> c = I.block<3, 1>(0, 3);
        return I.topLeftCorner<3, 3>() - (c * c.transpose()) / (I(3, 3) + prec_d);
    }

    // ── Generative-model hooks (engine interface + SDF API) ───────────────────
    // Single thin-box panel SDF in the wall frame, evaluated at room point p for state s. 0 on the surface;
    // >0 outside; <0 inside.
    float sdf_panel(const Eigen::Vector3f& p, const DoorBeliefState& s) const;
    float sdf_prim (const Eigen::Vector3f& p, const DoorBeliefState& s, int prim) const;  // prim==0 → panel
    Eigen::Matrix<float, 4, 1> sdf_jacobian(const Eigen::Vector3f& p, const DoorBeliefState& s, int prim) const;
    std::array<float, 3> responsibilities(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const;   // [believed leaf, other jamb's leaf, clutter]
    // Fixed template prior on w,h folded into the SAME GN normal equations as the data (engine detects this
    // hook via a C++23 requires). Pulls w→tpl_w, h→tpl_h with precision 1/prior_{w,h}_std² each frame; s is
    // untouched (it carries only the broad temporal prior). This is what makes the strong size prior stick.
    [[nodiscard]] std::pair<float, float> predicted_bearing_extent(const DoorBeliefState& s, const MaskBearings& mb) const;
    void accumulate_extra(const DoorBeliefState& s, const DoorFrame& frame,
                          Eigen::Matrix<float, 4, 4>& Id, Eigen::Matrix<float, 4, 1>& bd) const;

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
    Eigen::Matrix<float, 3, 3> covariance_reported() const { return Sigma_.topLeftCorner<3, 3>(); }

    void  apply_constraints(DoorBeliefState& s) const;
    void  canonicalize(DoorBeliefState&) const {}   // wall fixes yaw ⇒ no symmetry fold

    int   gn_iters() const { return params_.gn_iters; }
    // Task 15: TWO panel primitives — the leaf hung from the believed jamb, and from the other (weight P(believed hinge),
    // set_hinge_prob) — so the mask's points are explained under the HINGE MARGINAL, never by sliding the aperture a width
    // to make the believed hinge fit (try75). P = 1 ⇒ the other leaf has no weight and no surface: the old model, exactly.
    int   n_prims()  const { return 2; }             // two hinge hypotheses (clutter is the +1)
    void  set_hinge_prob(float p) { p_hinge_ = std::isfinite(p) ? std::clamp(p, 0.f, 1.f) : 1.f; }
    // Task 15: the leaf ANGLE's posterior σ (rad). The mask places the aperture only THROUGH the leaf panel at the believed φ,
    // so an error δφ moves the s that explains it by ≈ (w/2)·|sin φ|·δφ: that is added to the frame's along-wall common mode
    // (common_mode_inv_diag), and a mask frame can no longer move s further than the leaf's own uncertainty lets it say
    // where the hole is. 0 (default) ⇒ the old frame weight.
    void  set_leaf_phi_sigma(float sd) { leaf_phi_sigma_ = std::isfinite(sd) ? std::max(0.f, sd) : 0.f; }
    float hinge_prob() const { return p_hinge_; }
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 4, 4> transition() const { return Eigen::Matrix<float, 4, 4>::Identity(); }  // static
    Eigen::Matrix<float, 4, 1> process_noise_diag() const;
    Eigen::Matrix<float, 4, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 4, 1> common_mode_inv_diag(const DoorFrame& frame) const;

    static bool self_test();

private:
    // Unnormalized mixture terms [panel, clutter] at p (likelihood×prior). Shared by responsibilities()
    // (normalizes) and association_nll() (sums).
    std::array<float, 3> mixture_unnorm(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const;
    float p_hinge_ = 1.f;   // Task 15: P(the believed hinge); 1 = the single-hinge model
    float leaf_phi_sigma_ = 0.f;   // Task 15: the leaf angle's σ (rad), for the mask frame's along-wall common mode

    DoorBeliefState            state_;
    DoorBeliefParams           params_;
    Eigen::Matrix<float, 4, 4> Sigma_ = Eigen::Matrix<float, 4, 4>::Identity();
    Eigen::Matrix<float, 4, 1> prior_mean_ = Eigen::Matrix<float, 4, 1>::Zero();
};

}  // namespace rc
