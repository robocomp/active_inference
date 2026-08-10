/*
 * cabinet_wall_run_belief.h — Stage 1 of the kitchen-of-runs model (Fable design): a WALL-ANCHORED RUN as an
 * INTERVAL on a wall, not a free box. State θ = [t0, t1, d, z0, z1] (N=5) in the WALL CHART of one cell's
 * wall (origin A, unit direction u along the wall, INWARD normal n; back face ON the wall line, so the box
 * occupies t ∈ [t0,t1] along the wall, s ∈ [0,d] inward, z ∈ [z0,z1]).
 *
 * WHAT'S FIXED (chart, unrepresentable otherwise): yaw ≡ wall direction; lateral placement ≡ back face on the
 * wall; DOMAINS t ∈ [0,W] and z ∈ [0,H_room] (⇒ through-wall growth and ceiling-touching CANNOT be expressed).
 * WHAT'S TIGHT-PRIOR (estimated, σ covers the GT spread): d, z0, z1 (per tier; base z0≈0). FREE: t0, t1.
 *
 * Reuses the shared recursive-Laplace engine (common/ai_belief) and CabinetFrame/CabinetBeliefParams. The
 * flush/parallel/room-axis factors are GONE (the chart makes them identities). Kept + transplanted: the
 * censored along-wall EXTENT (now directly on t0/t1, clamped to [0,W]), the FACE-GATED free-space CARVE (end
 * exit → t, top exit → z1, depth/bottom exit → skip), and the ego-motion→common-mode STILLNESS.
 *
 * Validated by wall_run_self_test().
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "cabinet_belief.h"                         // CabinetFrame, CabinetBeliefParams, rc::ai::LidarRays
#include "../../common/ai_belief/recursive_laplace.h"
#include "../../common/existence_belief/existence_belief.h"   // rc::exist:: SensorModel / Evidence / saturate

namespace rc {

struct WallRunState
{
    float t0 = 0.0f, t1 = 1.0f;    // interval along the wall (m from corner A)
    float d  = 0.55f;              // carcass depth off the wall (m)
    float z0 = 0.0f, z1 = 0.85f;   // bottom / top height (m)

    Eigen::Matrix<float, 5, 1> vec() const
    { return (Eigen::Matrix<float, 5, 1>() << t0, t1, d, z0, z1).finished(); }
    static WallRunState from_vec(const Eigen::Matrix<float, 5, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4)}; }
};

// A cell's wall (origin A, unit direction u, inward normal n, length W) + the room ceiling height.
struct WallChart
{
    Eigen::Vector2f A{0, 0}, u{1, 0}, n{0, 1};
    float W = 3.0f;        // wall length (t domain upper bound)
    float H_room = 2.6f;   // ceiling (z domain upper bound)
    // World XY → wall coords (t along wall, s inward from the wall line).
    void to_wall(const Eigen::Vector3f& p, float& t, float& s) const
    { const Eigen::Vector2f q(p.x() - A.x(), p.y() - A.y()); t = q.dot(u); s = q.dot(n); }
};

// Per-tier standardized-but-physical priors (tight, σ covers the GT spread) — NOT hard constants.
struct WallTierPrior
{
    float d_mean = 0.55f, d_std = 0.10f;
    float z0_mean = 0.0f, z0_std = 0.02f;    // base sits on the floor (near-fixed); upper ~1.45±0.10
    float z1_mean = 0.85f, z1_std = 0.06f;
};

class WallRunBelief
{
public:
    static constexpr int N = 5;
    using State = WallRunState;
    using Frame = CabinetFrame;

    WallRunBelief() = default;
    WallRunBelief(const WallRunState& s, const CabinetBeliefParams& p, const WallChart& chart, const WallTierPrior& tier)
        : state_(s), params_(p), chart_(chart), tier_(tier)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); prior_mean_ = prior_mean_vec(); }

    // ── State access ────────────────────────────────────────────────────────────────────────────
    const WallRunState&              state()      const { return state_; }
    const Eigen::Matrix<float, 5, 5>& covariance() const { return Sigma_; }
    void set_state(const WallRunState& s) { state_ = s; }
    const WallChart& chart() const { return chart_; }

    // Corner-fill flags (set by KitchenManager each cycle): extend this run's low (t0→0) and/or high (t1→W)
    // end to the wall endpoint = the shared room corner, so an L of two perpendicular runs MEETS there instead
    // of leaving a hole. A hard structural prior, applied one-sided (fill only, never shrink).
    void set_corner_fill(bool low, bool high) { cf_low_ = low; cf_high_ = high; }

    // Derived ROOM-frame box for DSR publishing (the chart hides yaw/lateral; expose them here).
    void room_box(float& cx, float& cy, float& yaw, float& L, float& d, float& z0, float& z1) const
    {
        const float tc = 0.5f * (state_.t0 + state_.t1);
        const Eigen::Vector2f c = chart_.A + tc * chart_.u + 0.5f * state_.d * chart_.n;   // footprint centre
        cx = c.x(); cy = c.y(); yaw = std::atan2(chart_.u.y(), chart_.u.x());
        L = std::max(0.0f, state_.t1 - state_.t0); d = state_.d; z0 = state_.z0; z1 = state_.z1;
    }

    // ── Engine hooks ────────────────────────────────────────────────────────────────────────────
    int   n_prims() const { return 1; }
    float sigma2()  const { return params_.sigma_base_m * params_.sigma_base_m; }

    // Oriented-box SDF, evaluated in the WALL CHART (t along wall, s inward, z up). The back face is at s=0.
    float sdf_prim(const Eigen::Vector3f& p, const WallRunState& s, int) const
    {
        float t, sl; chart_.to_wall(p, t, sl);
        const float half_t = 0.5f * std::max(kMinExtent, s.t1 - s.t0);
        const float half_s = 0.5f * std::max(kMinExtent, s.d);
        const float half_z = 0.5f * std::max(kMinExtent, s.z1 - s.z0);
        const Eigen::Vector3f q(t - 0.5f * (s.t0 + s.t1), sl - 0.5f * s.d, p.z() - 0.5f * (s.z0 + s.z1));
        const Eigen::Vector3f e = q.cwiseAbs() - Eigen::Vector3f(half_t, half_s, half_z);
        return e.cwiseMax(0.0f).norm() + std::min(e.maxCoeff(), 0.0f);
    }
    float sdf_box(const Eigen::Vector3f& p, const WallRunState& s) const { return sdf_prim(p, s, 0); }

    Eigen::Matrix<float, 5, 1> sdf_jacobian(const Eigen::Vector3f& p, const WallRunState& s, int prim) const
    {
        Eigen::Matrix<float, 5, 1> J;
        const Eigen::Matrix<float, 5, 1> v = s.vec();
        const float eps = params_.fd_eps;
        for (int k = 0; k < 5; ++k)
        {
            Eigen::Matrix<float, 5, 1> vp = v, vm = v; vp(k) += eps; vm(k) -= eps;
            J(k) = std::clamp((sdf_prim(p, WallRunState::from_vec(vp), prim)
                             - sdf_prim(p, WallRunState::from_vec(vm), prim)) / (2.0f * eps), -kJClamp, kJClamp);
        }
        return J;
    }

    float mixture_unnormalized(const Eigen::Vector3f& p, const WallRunState& s, float R, std::array<float, 2>& u) const
    {
        const float eps  = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
        const float dist = sdf_box(p, s);
        const float Rc   = std::max(1e-9f, R);
        const float inv  = 1.0f / std::sqrt(2.0f * std::numbers::pi_v<float> * Rc);
        u[0] = (1.0f - eps) * inv * std::exp(-0.5f * dist * dist / Rc);
        u[1] = eps / std::max(1e-6f, params_.clutter_scale_m);
        return u[0] + u[1];
    }
    std::array<float, 2> responsibilities(const Eigen::Vector3f& p, const WallRunState& s, float R) const
    {
        std::array<float, 2> u{};
        const float sum = mixture_unnormalized(p, s, R, u);
        if (not(sum > 0.0f) or not std::isfinite(sum)) return {0.0f, 1.0f};
        return {u[0] / sum, u[1] / sum};
    }

    // Domains (the "unrepresentable" guarantees): t ∈ [0,W]; positive extents; z ∈ [0, H_room]; base z0≈0.
    void apply_constraints(WallRunState& s) const
    {
        s.t0 = std::clamp(s.t0, 0.0f, chart_.W);
        s.t1 = std::clamp(s.t1, 0.0f, chart_.W);
        if (s.t1 < s.t0 + kMinExtent) { const float m = 0.5f * (s.t0 + s.t1); s.t0 = m - 0.5f * kMinExtent; s.t1 = m + 0.5f * kMinExtent; }
        s.d  = std::max(kMinExtent, s.d);
        s.z0 = std::max(0.0f, s.z0);
        s.z1 = std::min(chart_.H_room, std::max(s.z0 + kMinExtent, s.z1));
    }
    void canonicalize(WallRunState&) const {}   // no C2v ambiguity: yaw is the wall direction

    // Transition (static furniture) + process/prior/common-mode diagonals — mirrors CabinetBelief.
    Eigen::Matrix<float, 5, 5> transition() const { return Eigen::Matrix<float, 5, 5>::Identity(); }
    Eigen::Matrix<float, 5, 1> process_noise_diag() const
    { const float pm = params_.process_std_m * params_.process_std_m; Eigen::Matrix<float, 5, 1> q; q << pm, pm, pm, pm, pm; return q; }
    Eigen::Matrix<float, 5, 1> prior_cov_diag() const
    { Eigen::Matrix<float, 5, 1> p; p << params_.prior_L_std * params_.prior_L_std, params_.prior_L_std * params_.prior_L_std,
             tier_.d_std * tier_.d_std, tier_.z0_std * tier_.z0_std, tier_.z1_std * tier_.z1_std; return p; }
    Eigen::Matrix<float, 5, 1> prior_mean_vec() const
    { Eigen::Matrix<float, 5, 1> m; m << state_.t0, state_.t1, tier_.d_mean, tier_.z0_mean, tier_.z1_mean; return m; }
    Eigen::Matrix<float, 5, 1> common_mode_inv_diag(const CabinetFrame& f) const
    {
        const float egp = std::max(0.0f, f.ego_motion_pos_var);
        const float pos  = params_.common_mode_pos_std  * params_.common_mode_pos_std + egp + std::max(0.0f, f.chain_cov_xx);
        const float size = params_.common_mode_size_std * params_.common_mode_size_std + std::max(0.0f, f.chain_cov_size) + egp;
        Eigen::Matrix<float, 5, 1> c;
        c << 1.0f / std::max(1e-9f, pos), 1.0f / std::max(1e-9f, pos),
             1.0f / std::max(1e-9f, size), 1.0f / std::max(1e-9f, size), 1.0f / std::max(1e-9f, size);
        return c;
    }
    int gn_iters() const { return params_.gn_iters; }

    // Extra structural factors: censored EXTENT + face-gated free-space CARVE + corner-fill + object exclusion.
    void accumulate_extra(const WallRunState& s, const CabinetFrame& f,
                          Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const
    { accumulate_extent(s, f, Id, bd); accumulate_freespace(s, f, Id, bd); accumulate_corner_fill(s, Id, bd);
      accumulate_object_exclusion(s, f, Id, bd); accumulate_tier_prior(s, Id, bd); }

    // STANDING tier prior on the three TIGHT-PRIOR DOFs (see this file's header: "WHAT'S TIGHT-PRIOR
    // (estimated, σ covers the GT spread): d, z0, z1. FREE: t0, t1"). The engine applies that prior
    // ONCE, at construction, and then re-anchors to the running posterior — so the model declares a
    // 0.60±0.04 carcass and then lets the state walk to 0.906 unopposed. This asserts it every update.
    // See CabinetBeliefParams::tier_prior_gain for the live measurement that motivated it.
    //
    // t0/t1 deliberately get NOTHING: a run's along-wall extent is genuinely free and is already
    // shaped by the censored-extent factor and the free-space carve.
    void accumulate_tier_prior(const WallRunState& s,
                               Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const
    {
        const float g = params_.tier_prior_gain;
        if (g <= 0.0f) return;
        const auto push = [&](int idx, float e, float std_dev)
        {
            if (not (std_dev > 0.0f)) return;                     // an undeclared σ asserts nothing
            const float w = g / (std_dev * std_dev);
            Eigen::Matrix<float, 5, 1> J = Eigen::Matrix<float, 5, 1>::Zero(); J(idx) = 1.0f;
            Id.noalias() += w * (J * J.transpose()); bd.noalias() += -w * J * e;
        };
        push(2, s.d  - tier_.d_mean,  tier_.d_std);
        push(3, s.z0 - tier_.z0_mean, tier_.z0_std);
        push(4, s.z1 - tier_.z1_mean, tier_.z1_std);
    }

    // Scene-object non-penetration: a run may not cross another agent's furniture. For each object whose
    // room-frame OBB conflicts with the run in BOTH depth and z, retract the PENETRATING along-wall END onto
    // the object boundary (nearest-face, like the free-space carve). Retract-only; the "engulfed" case (run
    // centre inside the object) is left to KitchenManager's engulfment retirement (a face pick would oscillate).
    void accumulate_object_exclusion(const WallRunState& s, const CabinetFrame& f,
                                     Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const
    {
        if (params_.object_exclusion_precision <= 0.0f or f.scene_objects.empty()) return;
        const float lam    = params_.object_exclusion_precision;
        const float margin = params_.object_exclusion_margin_m;
        const float tc     = 0.5f * (s.t0 + s.t1);
        const auto push = [&](int idx, float e, float w)
        { Eigen::Matrix<float, 5, 1> J = Eigen::Matrix<float, 5, 1>::Zero(); J(idx) = 1.0f;
          Id.noalias() += w * (J * J.transpose()); bd.noalias() += -w * J * e; };

        for (const auto& o : f.scene_objects)
        {
            // OBB footprint → EXACT 1-D shadows on the wall axes (u = along, n = depth): min/max of the 4
            // projected corners is the exact interval a rotated rectangle spans on each axis (any object yaw).
            const Eigen::Vector2f ex(std::cos(o.yaw), std::sin(o.yaw)), ey(-std::sin(o.yaw), std::cos(o.yaw));
            const Eigen::Vector2f c(o.cx, o.cy);
            float o_t0 = 1e30f, o_t1 = -1e30f, o_s0 = 1e30f, o_s1 = -1e30f;
            for (float sx : {-0.5f, 0.5f}) for (float sy : {-0.5f, 0.5f})
            {
                const Eigen::Vector2f q = c + sx * o.w * ex + sy * o.d * ey - chart_.A;
                const float t = q.dot(chart_.u), sl = q.dot(chart_.n);
                o_t0 = std::min(o_t0, t); o_t1 = std::max(o_t1, t);
                o_s0 = std::min(o_s0, sl); o_s1 = std::max(o_s1, sl);
            }
            // 3-D conflict gate: the object must occupy the run's depth band [0,d] AND z band [z0,z1] AND
            // overlap it along the wall — else no penetration (object in front/behind, above/below, or aside).
            if (o_s1 <= 0.0f or o_s0 >= s.d)   continue;
            if (o.z1 <= s.z0 or o.z0 >= s.z1)  continue;
            if (o_t1 <= s.t0 or o_t0 >= s.t1)  continue;
            // Retract the penetrating END to the boundary (Δ(idx) = −e drives the coord to its target).
            if (tc <= o_t0)          // run centre LOW of the object ⇒ its HIGH end t1 crossed in ⇒ pull t1 down
            {
                const float e = s.t1 - (o_t0 - margin);
                if (e > 0.0f) push(1, e, lam);
            }
            else if (tc >= o_t1)     // run centre HIGH of the object ⇒ its LOW end t0 crossed in ⇒ push t0 up
            {
                const float e = s.t0 - (o_t1 + margin);
                if (e < 0.0f) push(0, e, lam);
            }
            // else tc strictly inside [o_t0,o_t1]: engulfed — defer to engulfment retirement.
        }
    }

    // Corner-fill: a HARD prior pulling a flagged end to the wall endpoint (the shared room corner), so two
    // perpendicular runs meeting there fill the inside corner instead of leaving a hole. One-sided (grow only):
    // cf_high pushes t1 UP to W, cf_low pulls t0 DOWN to 0; never shrinks a run that already spans past.
    void accumulate_corner_fill(const WallRunState& s,
                                Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const
    {
        constexpr float lam = 40.0f;   // strong: the corner is empty of mask points, so nothing opposes the fill
        if (cf_high_)
        {
            const float e = s.t1 - chart_.W;                 // <0 ⇒ t1 short of the wall end ⇒ push UP to W
            if (e < 0.0f) { Eigen::Matrix<float, 5, 1> J = Eigen::Matrix<float, 5, 1>::Zero(); J(1) = 1.0f;
                            Id.noalias() += lam * (J * J.transpose()); bd.noalias() += -lam * J * e; }
        }
        if (cf_low_)
        {
            const float e = s.t0;                            // >0 ⇒ t0 short of the wall start ⇒ pull DOWN to 0
            if (e > 0.0f) { Eigen::Matrix<float, 5, 1> J = Eigen::Matrix<float, 5, 1>::Zero(); J(0) = 1.0f;
                            Id.noalias() += lam * (J * J.transpose()); bd.noalias() += -lam * J * e; }
        }
    }

    // ── EVIDENCE OF ABSENCE (the retirement channel) ────────────────────────────────────────────
    // The free-space carve above RESHAPES a run (it retracts faces); it can never say "this run is not there".
    // This does: it classifies the sweep's beams against the run's box IN THE WALL CHART, where the physics of a
    // solid wall-anchored carcass is exact, and returns the per-cycle occupancy/absence log-likelihood ratio.
    //
    //   · a return on an OBSERVABLE FACE (front s≈d, top z≈z1)          ⇒ OCCUPANCY  (the run is there)
    //   · a beam that traverses the whole box and exits the far side    ⇒ FREE SPACE (the run is not there) —
    //     for a wall run the dominant case is a beam that reaches the WALL BEHIND it, which is exactly the
    //     observation that refutes a phantom;
    //   · a return DEEP INSIDE the box (the floor under a phantom base run, the back wall) is NOT occupancy: a
    //     solid carcass cannot return from its own interior. The surface membership `face_w` weights it out —
    //     this is what stops a phantom from feeding itself the very floor returns that refute it;
    //   · a beam that stops SHORT of the box (occluded, p_reached→0) contributes NOTHING: absence of evidence is
    //     not evidence of absence, so an occluded run simply holds.
    //
    // absence_conf ∈ [0,1] scales the FREE half only (beams get sparse with range ⇒ a distant pass-through is
    // weak evidence of removal); occupancy always counts at full weight. n_reached==0 ⇒ not probed ⇒ caller HOLDs.
    rc::exist::Evidence lidar_existence_evidence(const rc::ai::LidarRays& rays, const rc::exist::SensorModel& sm,
                                                 float absence_conf = 1.0f) const
    {
        rc::exist::Evidence ev;
        if (rays.endpoints.empty()) return ev;
        const WallRunState& s = state_;
        const float lo[3] = {s.t0, 0.0f, s.z0};
        const float hi[3] = {s.t1, s.d,  s.z1};
        // Surface blur = sensor σ ⊕ common-mode registration σ ⊕ the belief's own DEPTH σ. Depth (Σ_dd) is the
        // right DOF: it is where the FRONT SURFACE sits along the beam, so it is exactly the uncertainty that
        // decides "returned on the face" vs "went through". The along-wall ends (Σ_t0/Σ_t1) carry a deliberately
        // broad prior and do not move the surface the beam meets — folding them in would blur the test to
        // uselessness (a metre-scale σ makes every beam read as a pass-through, incl. on a real cabinet).
        const float sigma_surf = std::sqrt(sm.sensor_sigma_m * sm.sensor_sigma_m
                                         + params_.common_mode_pos_std * params_.common_mode_pos_std
                                         + std::max(0.0f, Sigma_(2, 2)));
        const auto phi = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); };
        const auto to3 = [&](const Eigen::Vector3f& p) { float t, sl; chart_.to_wall(p, t, sl); return Eigen::Vector3f(t, sl, p.z()); };
        const Eigen::Vector3f O = to3(rays.origin);
        const float pd = std::clamp(sm.detection_prob, 1e-3f, 1.0f - 1e-3f);
        const float pc = std::clamp(sm.clutter_prob,   1e-3f, 1.0f - 1e-3f);
        const float llr_occ  = std::log(pd / pc);                    // >0: surface return   ⇒ still there
        const float llr_free = std::log((1.0f - pd) / (1.0f - pc));  // <0: passed through   ⇒ gone

        float E_occ = 0.0f, E_free = 0.0f, reached_mass = 0.0f;
        for (const auto& ep : rays.endpoints)
        {
            const Eigen::Vector3f E = to3(ep);
            const Eigen::Vector3f dloc = E - O;
            const float len = dloc.norm();
            if (len < 1e-4f) continue;
            float t_near = 0.0f, t_far = std::numeric_limits<float>::max();
            bool miss = false;
            for (int a = 0; a < 3 and not miss; ++a)
            {
                if (std::abs(dloc(a)) < 1e-6f) { if (O(a) < lo[a] or O(a) > hi[a]) miss = true; continue; }
                const float s1 = (lo[a] - O(a)) / dloc(a), s2 = (hi[a] - O(a)) / dloc(a);
                t_near = std::max(t_near, std::min(s1, s2));
                t_far  = std::min(t_far,  std::max(s1, s2));
            }
            if (miss or t_near > t_far or t_far < 0.0f) continue;    // the beam never crosses this box
            const float sigma_t   = sigma_surf / len;
            const float p_reached = phi((1.0f - t_near) / sigma_t);  // got at least to the near face
            const float p_through = phi((1.0f - t_far)  / sigma_t);  // ...and all the way past the far face
            if (p_reached < 1e-3f) continue;                         // stopped short ⇒ occluded ⇒ NO evidence
            // Surface membership of the RETURN: distance from the endpoint to the run's OBSERVABLE faces.
            const float df = std::min(std::abs(E(1) - s.d), std::abs(E(2) - s.z1));
            const float face_w = std::exp(-0.5f * df * df / (sigma_surf * sigma_surf));
            E_occ  += (p_reached - p_through) * face_w;
            E_free += p_through;
            reached_mass += p_reached;
        }
        ev.e_occ = E_occ; ev.e_free = E_free * std::clamp(absence_conf, 0.0f, 1.0f);
        ev.n_reached = static_cast<int>(std::lround(reached_mass));
        if (ev.n_reached == 0) return ev;                            // not probed this cycle ⇒ HOLD
        // Common-mode saturation: the cycle's beams share ONE registration error, so N of them are not N
        // independent observations — cap the cycle's |ΔL| at one confident observation's worth (shared tanh).
        ev.log_odds_delta = rc::exist::saturate(ev.e_occ * llr_occ + ev.e_free * llr_free, llr_occ);
        return ev;
    }

    float update(const CabinetFrame& frame)  { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()                          { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }

    static bool self_test();

private:
    static constexpr float kMinExtent = 0.05f;
    static constexpr float kJClamp    = 4.0f;

    // Censored along-wall extent: the observed t-span is a LOWER bound on [t0,t1] (occlusion only shortens).
    // r0 = max(0, t0 − t_min) pulls t0 DOWN to reach the leftmost point; r1 = max(0, t_max − t1) pulls t1 UP.
    // Silent once the interval contains the cloud. Depth/z of the point gate membership softly (cross-section).
    void accumulate_extent(const WallRunState& s, const CabinetFrame& f,
                           Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const
    {
        if (params_.extent_precision <= 0.0f or f.points.empty()) return;
        const float sig = std::max(1e-3f, params_.sigma_base_m);
        const float half_s = 0.5f * std::max(kMinExtent, s.d), sc = 0.5f * s.d;
        const float half_z = 0.5f * std::max(kMinExtent, s.z1 - s.z0), zc = 0.5f * (s.z0 + s.z1);
        // Collect the along-wall t of every point inside the box cross-section (depth/z gate), then take a ROBUST
        // quantile for the run's extremes — the raw min/max let ONE stray mask point ratchet the end outward, and
        // with grow-only + persistence the box then elongates without bound past the mask evidence.
        std::vector<float> ts; ts.reserve(f.points.size());
        for (const auto& p : f.points)
        {
            float t, sl; chart_.to_wall(p, t, sl);
            const float dl = std::max(0.0f, std::abs(sl - sc) - half_s);
            const float dz = std::max(0.0f, std::abs(p.z() - zc) - half_z);
            if (std::exp(-0.5f * (dl * dl + dz * dz) / (sig * sig)) < 1e-3f) continue;
            ts.push_back(t);
        }
        if (ts.size() < 4) return;
        std::sort(ts.begin(), ts.end());
        const std::size_t n = ts.size();
        constexpr float kQ = 0.02f;                                     // ignore the extreme 2% (outliers)
        const float t_min = ts[static_cast<std::size_t>(kQ * (n - 1))];
        const float t_max = ts[static_cast<std::size_t>((1.0f - kQ) * (n - 1))];

        // GROW-ONLY (censored lower bound): a mask span is a LOWER bound on the true extent — occlusion / a narrow
        // view only shorten what we SEE, never lengthen it. So the extent is the UNION (max) of observed spans and
        // is NEVER shrunk by a partial view (that is absence of evidence, not evidence of absence). The ONLY force
        // that retracts an end is the free-space carve (a ray THROUGH the run = evidence of absence). The robust
        // quantile above keeps mask NOISE from inflating the bound; grow-only keeps a static run from breathing.
        const float lam = params_.extent_precision;
        const auto push = [&](int idx, float e, float w)
        { Eigen::Matrix<float, 5, 1> J = Eigen::Matrix<float, 5, 1>::Zero(); J(idx) = 1.0f;
          Id.noalias() += w * (J * J.transpose()); bd.noalias() += -w * J * e; };
        const float e0 = s.t0 - t_min;   // >0 ⇒ t0 above the leftmost point ⇒ grow t0 DOWN; <0 ⇒ HOLD (partial view)
        if (e0 > 0.0f) push(0, e0, lam);
        const float e1 = s.t1 - t_max;   // <0 ⇒ t1 below the rightmost point ⇒ grow t1 UP;  >0 ⇒ HOLD (partial view)
        if (e1 < 0.0f) push(1, e1, lam);
    }

    // OCCUPANCY-BALANCED face-gated free-space carve. A through-beam exiting an END face (t) shrinks t0/t1; the
    // TOP (z1) shrinks; DEPTH/bottom are SKIPPED (wall owns depth, floor owns z0). CRUCIAL: the carve is BOUNDED
    // by POSITIVE occupancy — the along-wall (and z) extent of LiDAR returns that land ON the box. A far end seen
    // at a grazing angle is sampled sparsely, so a few beams SLIP PAST the real surface and read as false empty;
    // without the bound they carve into a real cabinet. Requiring the carve to stop at the furthest surface HIT
    // means the many beams that hit the cabinet VETO the few that slip past — hits are the positive evidence.
    void accumulate_freespace(const WallRunState& s, const CabinetFrame& f,
                              Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const
    {
        if (params_.free_space_precision <= 0.0f or f.lidar_freespace.endpoints.empty()) return;
        const rc::ai::LidarRays& rays = f.lidar_freespace;
        const float lo[3] = {s.t0, 0.0f, s.z0};
        const float hi[3] = {s.t1, s.d,  s.z1};
        const float sigma_surf = std::sqrt(params_.sigma_base_m * params_.sigma_base_m
                                         + params_.common_mode_pos_std * params_.common_mode_pos_std);
        const float m = std::max(0.05f, 2.0f * sigma_surf);   // surface-hit membership margin
        const auto phi = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); };
        const auto to3 = [&](const Eigen::Vector3f& p) { float t, sl; chart_.to_wall(p, t, sl); return Eigen::Vector3f(t, sl, p.z()); };
        const Eigen::Vector3f O = to3(rays.origin);

        // Pass 1 — POSITIVE occupancy: the along-wall span [h_min,h_max] and top height h_zmax of returns landing
        // on the FRONT FACE (s ≈ d). A return deeper than the front face (s < d−m, incl. the back WALL at s≈0) is
        // NOT a cabinet hit — it is a beam that reached past where the front should be ⇒ evidence of ABSENCE, so
        // it must NOT bound the carve. Restricting to the front face is what lets the carve remove an over-grown
        // end while a densely-seen front face vetoes the sparse grazing beams that slip past it.
        float h_min = 1e30f, h_max = -1e30f, h_zmax = -1e30f;
        for (const auto& ep : rays.endpoints)
        {
            const Eigen::Vector3f E = to3(ep);
            if (std::abs(E(1) - s.d) > m)             continue;   // not on the front face ⇒ not a cabinet surface hit
            if (E(2) < s.z0 - m or E(2) > s.z1 + m)   continue;   // outside the z band
            if (E(0) < s.t0 - m or E(0) > s.t1 + m)   continue;   // not near the along-wall span (avoid neighbours)
            h_min = std::min(h_min, E(0)); h_max = std::max(h_max, E(0)); h_zmax = std::max(h_zmax, E(2));
        }

        const auto push = [&](int idx, float e, float w)
        { Eigen::Matrix<float, 5, 1> J = Eigen::Matrix<float, 5, 1>::Zero(); J(idx) = 1.0f;
          Id.noalias() += w * (J * J.transpose()); bd.noalias() += -w * J * e; };

        // Pass 2 — through-beams carve, bounded by the Pass-1 hits (max(·,h_max) / min(·,h_min) / max(·,h_zmax);
        // when there are no hits the bounds are ±inf-neutral, so this reduces to the plain carve).
        for (const auto& ep : rays.endpoints)
        {
            const Eigen::Vector3f E = to3(ep);
            const Eigen::Vector3f dloc = E - O;
            const float len = dloc.norm();
            if (len < 1e-4f) continue;
            float t_near = 0.0f, t_far = std::numeric_limits<float>::max();
            int far_axis = -1; bool far_pos = true, miss = false;
            for (int a = 0; a < 3 and not miss; ++a)
            {
                if (std::abs(dloc(a)) < 1e-6f) { if (O(a) < lo[a] or O(a) > hi[a]) miss = true; continue; }
                const float s1 = (lo[a] - O(a)) / dloc(a), s2 = (hi[a] - O(a)) / dloc(a);
                t_near = std::max(t_near, std::min(s1, s2));
                const float ex = std::max(s1, s2);
                if (ex < t_far) { t_far = ex; far_axis = a; far_pos = dloc(a) > 0.0f; }
            }
            if (miss or far_axis < 0 or t_near > t_far or t_far < 0.0f or t_near >= 1.0f) continue;
            const float p_through = phi((1.0f - t_far) / (sigma_surf / len));
            if (p_through < 1e-3f) continue;
            const Eigen::Vector3f pf = O + 0.5f * (t_near + t_far) * dloc;   // wall-coord empty witness
            const float w = params_.free_space_precision * p_through;
            if (far_axis == 0)   // exit an END face ⇒ retract the nearer end toward the witness, NOT past a hit
            {
                if (std::abs(pf(0) - s.t1) < std::abs(pf(0) - s.t0))
                { const float e = s.t1 - std::max(pf(0), h_max); if (e > 0.0f) push(1, e, w); }   // t1 down (≥ furthest hit)
                else
                { const float e = s.t0 - std::min(pf(0), h_min); if (e < 0.0f) push(0, e, w); }   // t0 up  (≤ nearest hit)
            }
            else if (far_axis == 2 and far_pos)   // exit the TOP ⇒ retract z1, NOT below the highest surface hit
            {
                const float e = s.z1 - std::max(pf(2), h_zmax); if (e > 0.0f) push(4, e, w);
            }
            // else: exit a DEPTH face (axis 1) or the bottom ⇒ no carve.
        }
    }

    WallRunState                state_;
    CabinetBeliefParams         params_;
    WallChart                   chart_;
    WallTierPrior               tier_;
    Eigen::Matrix<float, 5, 5>  Sigma_      = Eigen::Matrix<float, 5, 5>::Identity();
    Eigen::Matrix<float, 5, 1>  prior_mean_ = Eigen::Matrix<float, 5, 1>::Zero();
    bool                        cf_low_  = false;   // corner-fill: extend t0→0 (set per-cycle by KitchenManager)
    bool                        cf_high_ = false;   // corner-fill: extend t1→W
};

// Chart along +x: world (x,y,z) == wall (t,s,z), so a front face at s=d is at world y=d — makes the tests read.
inline bool WallRunBelief::self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* w) { if (not c) { std::printf("[wall_run::self_test] FAIL: %s\n", w); ok = false; } };
    WallChart ch; ch.A = {0, 0}; ch.u = {1, 0}; ch.n = {0, 1}; ch.W = 3.0f; ch.H_room = 2.6f;
    const WallTierPrior base;                       // d 0.55±0.10, z0 0±0.02, z1 0.85±0.06
    CabinetBeliefParams pr; pr.free_space_precision = 800.0f; pr.object_exclusion_precision = 800.0f;

    // Front-face (s=d) + top point cloud over t ∈ [t0,t1].
    const auto cloud = [&](float t0, float t1, float d, float z1) {
        std::vector<Eigen::Vector3f> pts;
        for (int i = 0; i <= 40; ++i) { const float t = t0 + (t1 - t0) * i / 40.0f;
            for (int j = 0; j <= 5; ++j) pts.emplace_back(t, d, z1 * j / 5.0f);   // front face at world y=d
            pts.emplace_back(t, 0.5f * d, z1); }                                  // top
        return pts;
    };

    // (a) EXTENT grows to the observed span, and NEVER past the wall end W (through-wall unrepresentable).
    {
        WallRunState s0; s0.t0 = 1.4f; s0.t1 = 1.6f; s0.d = 0.55f; s0.z0 = 0; s0.z1 = 0.85f;
        WallRunBelief b(s0, pr, ch, base);
        const auto p1 = cloud(0.5f, 2.5f, 0.55f, 0.85f);
        for (int it = 0; it < 30; ++it) { CabinetFrame f; f.points = p1; b.update(f); }
        check(b.state().t0 < 0.7f and b.state().t1 > 2.3f, "extent grows to the observed span");
        const auto p2 = cloud(0.5f, 3.5f, 0.55f, 0.85f);   // points run PAST the wall end
        for (int it = 0; it < 30; ++it) { CabinetFrame f; f.points = p2; b.update(f); }
        check(b.state().t1 <= 3.001f, "t1 never grows past the wall end W (through-wall impossible)");
    }
    // (b) FRONT FACE alone determines d (the old d/lateral null direction is gone), and σ_d shrinks.
    {
        WallRunState s0; s0.t0 = 0.5f; s0.t1 = 2.5f; s0.d = 0.48f; s0.z0 = 0; s0.z1 = 0.85f;
        WallRunBelief b(s0, pr, ch, base);
        const float sd0 = std::sqrt(b.covariance()(2, 2));
        const auto pts = cloud(0.5f, 2.5f, 0.55f, 0.85f);   // front face at s = 0.55
        for (int it = 0; it < 30; ++it) { CabinetFrame f; f.points = pts; b.update(f); }
        check(std::abs(b.state().d - 0.55f) < 0.05f, "d is set by the front-face position (0.55)");
        check(std::sqrt(b.covariance()(2, 2)) < sd0,  "σ_d shrinks (depth identifiable)");
    }
    // (c) NO CEILING-TOUCHING: the domain caps z1 ≤ H_room, and a far-above point cannot drag z1 up.
    {
        WallRunState s0; s0.t0 = 0.5f; s0.t1 = 2.5f; s0.z1 = 5.0f; WallRunBelief b(s0, pr, ch, base);
        const auto pts = cloud(0.5f, 2.5f, 0.55f, 0.85f);
        for (int it = 0; it < 5; ++it) { CabinetFrame f; f.points = pts; b.update(f); }
        check(b.state().z1 <= 2.601f, "z1 is capped at the ceiling H_room (over-tall unrepresentable)");
        WallRunState s1; s1.t0 = 0.5f; s1.t1 = 2.5f; s1.z1 = 0.85f; WallRunBelief b2(s1, pr, ch, base);
        std::vector<Eigen::Vector3f> hi; for (int i = 0; i <= 20; ++i) hi.emplace_back(0.5f + 2.0f * i / 20.0f, 0.55f, 1.60f);
        for (int it = 0; it < 30; ++it) { CabinetFrame f; f.points = hi; b2.update(f); }
        check(b2.state().z1 < 0.98f, "a far-above point does not drag z1 up (prior/clutter dominance)");
    }
    // (d) FACE-GATED CARVE: a DEPTH-exiting beam never changes d; an END-exiting beam retracts t1.
    {
        WallRunState s0; s0.t0 = 0.0f; s0.t1 = 3.0f; s0.d = 0.55f; s0.z0 = 0; s0.z1 = 0.85f;
        const auto carve = [&](const Eigen::Vector3f& o, const Eigen::Vector3f& e, int n) {
            WallRunBelief b(s0, pr, ch, base);
            const auto pts = cloud(0.0f, 1.5f, 0.55f, 0.85f);   // real run only x∈[0,1.5]
            for (int it = 0; it < n; ++it) { CabinetFrame f; f.points = pts;
                f.lidar_freespace.origin = o; f.lidar_freespace.endpoints = {e}; b.update(f); }
            return b.state();
        };
        const auto sd = carve({1.0f, -3.0f, 0.4f}, {1.0f, 3.0f, 0.4f}, 20);        // straight through depth (+y)
        check(std::abs(sd.d - 0.55f) < 0.05f, "carve: a DEPTH-exiting beam leaves d unchanged (the d-collapse fix)");
        const auto se = carve({1.8f, 0.1f, 0.4f}, {3.5f, 0.3f, 0.4f}, 25);         // through the empty +t region, exits +t
        check(se.t1 < 2.9f, "carve: an END-exiting beam retracts t1 toward the support");
        check(std::abs(se.d - 0.55f) < 0.05f, "carve: the END-exiting beam still leaves d unchanged");
    }
    // (d2) OCCUPANCY-BALANCED CARVE: the real scenario for "closer/grazing views shrink the cabinet". An
    // established run (t1=2.5) is now seen CLOSE: the camera masks only its near part [0,1.5] (so the grow-only
    // extent can't defend the far end), and a grazing LiDAR beam slips past the far end. WITHOUT front-face hits
    // the slip-past carves t1 in; WITH the LiDAR front-face returns (which reach t=2.5 even when the camera does
    // not) the hits VETO it and the run holds. This is the fix: positive occupancy (hits) bounds the carve.
    {
        WallRunState s0; s0.t0 = 0.0f; s0.t1 = 2.5f; s0.d = 0.55f; s0.z0 = 0; s0.z1 = 0.85f;
        const Eigen::Vector3f O(1.6f, 0.3f, 0.4f);                  // grazes along the wall toward the far end
        const auto run_carve = [&](bool with_hits) {
            WallRunBelief b(s0, pr, ch, base);
            const auto pts = cloud(0.0f, 1.5f, 0.55f, 0.85f);      // CLOSE view: masks only the near part [0,1.5]
            for (int it = 0; it < 25; ++it)
            {
                CabinetFrame f; f.points = pts; f.lidar_freespace.origin = O;
                if (with_hits)                                     // LiDAR front-face returns reach the true end t=2.5
                    for (int i = 0; i <= 20; ++i) f.lidar_freespace.endpoints.emplace_back(2.5f * i / 20.0f, 0.55f, 0.4f);
                f.lidar_freespace.endpoints.emplace_back(3.0f, 0.3f, 0.4f);   // one grazing slip-past, exits the +t end
                b.update(f);
            }
            return b.state();
        };
        check(run_carve(true).t1  > 2.35f, "carve VETOED by front-face hits (a grazing slip-past cannot shrink a seen run)");
        check(run_carve(false).t1 < 2.20f, "...with NO front-face hits the same beam DOES carve the end (bug reproduced)");
    }
    // (e) OBJECT EXCLUSION: a run may not cross another agent's furniture — the penetrating END retracts to the
    // object boundary, and an object clear in DEPTH or in Z does not constrain it.
    {
        WallRunState s0; s0.t0 = 0.0f; s0.t1 = 3.0f; s0.d = 0.55f; s0.z0 = 0; s0.z1 = 0.85f;
        const auto excl = [&](const SceneObjectBox& o) {
            WallRunBelief b(s0, pr, ch, base);
            const auto pts = cloud(0.0f, 1.5f, 0.55f, 0.85f);        // real run only x∈[0,1.5]
            for (int it = 0; it < 25; ++it) { CabinetFrame f; f.points = pts; f.scene_objects = {o}; b.update(f); }
            return b.state();
        };
        // Object straddling the wall at t∈[2.2,2.8], overlapping the run in depth+z ⇒ retract t1 to ~2.2−margin.
        const auto sh = excl({2.5f, 0.30f, 0.0f, 0.6f, 0.6f, 0.0f, 1.8f});
        check(sh.t1 < 2.25f and sh.t1 > 2.05f, "exclusion: high end retracts to the object near face");
        check(sh.t0 < 0.10f,                    "exclusion: the far (low) end is untouched");
        check(std::abs(sh.d - 0.55f) < 0.06f,   "exclusion: depth unchanged (END face only)");
        // DEPTH-clear object (its footprint is in front of the run, s>d) ⇒ no retraction.
        check(excl({2.5f, 1.2f, 0.0f, 0.6f, 0.6f, 0.0f, 1.8f}).t1 > 2.9f, "exclusion: object clear in depth does not retract");
        // Z-clear object (above the base run's z band) ⇒ no retraction.
        check(excl({2.5f, 0.30f, 0.0f, 0.6f, 0.6f, 1.5f, 2.1f}).t1 > 2.9f, "exclusion: object above the run's z band does not retract");
    }
    // (f) SYMMETRIC low-end exclusion: an object below the run centre pushes t0 up to its far boundary.
    {
        WallRunState s0; s0.t0 = 0.0f; s0.t1 = 3.0f; s0.d = 0.55f; s0.z0 = 0; s0.z1 = 0.85f;
        WallRunBelief b(s0, pr, ch, base);
        const auto pts = cloud(1.5f, 3.0f, 0.55f, 0.85f);           // real run x∈[1.5,3]
        const SceneObjectBox o{0.5f, 0.30f, 0.0f, 0.6f, 0.6f, 0.0f, 1.8f};   // object at t∈[0.2,0.8]
        for (int it = 0; it < 25; ++it) { CabinetFrame f; f.points = pts; f.scene_objects = {o}; b.update(f); }
        check(b.state().t0 > 0.75f and b.state().t0 < 0.95f, "exclusion: low end retracts up to the object far face");
        check(b.state().t1 > 2.9f,                            "exclusion: the far (high) end is untouched");
    }
    // (g) EVIDENCE OF ABSENCE (the retirement channel). The chart is +x, so the wall line is y=0 and a base run
    // occupies y∈[0,0.55]. LiDAR sits in the room at y=3 looking back at the wall.
    {
        WallRunState s0; s0.t0 = 0.5f; s0.t1 = 2.5f; s0.d = 0.55f; s0.z0 = 0; s0.z1 = 0.85f;
        WallRunBelief b(s0, pr, ch, base);
        rc::exist::SensorModel sm;                                   // 0.03 m, pd .85, pc .05
        const Eigen::Vector3f O(1.5f, 3.0f, 0.60f);                  // in the room, at run height
        // (g1) PHANTOM: every beam reaches the WALL BEHIND the run (y≈0) ⇒ pure free space ⇒ ΔL < 0.
        rc::ai::LidarRays wall_hits; wall_hits.origin = O;
        for (int i = 0; i <= 20; ++i) wall_hits.endpoints.emplace_back(0.5f + 2.0f * i / 20.0f, 0.0f, 0.60f);
        const auto ev_gone = b.lidar_existence_evidence(wall_hits, sm);
        check(ev_gone.n_reached > 0,          "absence: wall-behind beams probe the run");
        check(ev_gone.log_odds_delta < -1.0f, "absence: beams reaching the wall BEHIND the run refute it (ΔL<0)");
        // (g2) REAL: the same beams stop on the run's FRONT face (y≈d) ⇒ occupancy ⇒ ΔL > 0.
        rc::ai::LidarRays face_hits; face_hits.origin = O;
        for (int i = 0; i <= 20; ++i) face_hits.endpoints.emplace_back(0.5f + 2.0f * i / 20.0f, 0.55f, 0.60f);
        check(b.lidar_existence_evidence(face_hits, sm).log_odds_delta > 1.0f,
              "absence: front-face returns are occupancy (ΔL>0) — a seen run is never removed");
        // (g3) INTERIOR returns are NOT occupancy: the floor under a PHANTOM base run (z≈0, deep inside) must not
        // feed the phantom. Same rays, endpoints on the floor mid-footprint ⇒ no positive drive.
        rc::ai::LidarRays floor_hits; floor_hits.origin = O;
        for (int i = 0; i <= 20; ++i) floor_hits.endpoints.emplace_back(0.5f + 2.0f * i / 20.0f, 0.25f, 0.0f);
        check(b.lidar_existence_evidence(floor_hits, sm).log_odds_delta <= 0.01f,
              "absence: a return DEEP INSIDE the box is not occupancy (a solid carcass has no interior returns)");
        // (g4) OCCLUDED: beams stop well SHORT of the run ⇒ no evidence at all (HOLD, never removal).
        rc::ai::LidarRays short_hits; short_hits.origin = O;
        for (int i = 0; i <= 20; ++i) short_hits.endpoints.emplace_back(0.5f + 2.0f * i / 20.0f, 2.0f, 0.60f);
        check(b.lidar_existence_evidence(short_hits, sm).n_reached == 0,
              "absence: beams stopping short of the run give NO evidence (occlusion ⇒ HOLD)");
        // (g5) OUT OF THE Z BAND: beams passing above the base run's top never probe it.
        rc::ai::LidarRays above; above.origin = Eigen::Vector3f(1.5f, 3.0f, 1.60f);
        for (int i = 0; i <= 20; ++i) above.endpoints.emplace_back(0.5f + 2.0f * i / 20.0f, 0.0f, 1.60f);
        check(b.lidar_existence_evidence(above, sm).n_reached == 0, "absence: beams above the run's z band do not probe it");
        // (g6) absence_conf scales the FREE half only ⇒ a far (low-confidence) view refutes more weakly.
        const float dl_full = b.lidar_existence_evidence(wall_hits, sm, 1.0f).log_odds_delta;
        const float dl_far  = b.lidar_existence_evidence(wall_hits, sm, 0.1f).log_odds_delta;
        check(dl_far > dl_full, "absence: range-degraded confidence weakens the refutation");
    }
    if (ok) std::printf("[wall_run::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
