/*
 * door_belief.cpp  —  AI2 door belief (single wall-anchored thin panel; state θ=[s,w,h,d] in the wall frame; d = the leaf plane's depth off the wall line, Task 5b).
 */

#include "door_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <random>

namespace rc
{

// ─── SDF: the LEAF, wherever it currently is ────────────────────────────────────────────────────────────
//
// Delegates to rc::door::leaf_sdf — the single source of truth (see door_geometry.h). The state [s,w,h]
// gives the APERTURE; params_.leaf gives the leaf's articulation (phi / hinge / swing). With phi pinned at
// 0 (M0) the leaf is flush in the aperture and this reduces, term for term and BIT-EXACTLY, to the old
// wall-plane expression this function used to spell out inline. Routing the fit through the same function
// as the silhouette / split / mesh / planner is what stops those consumers drifting apart again.
float DoorBelief::sdf_panel(const Eigen::Vector3f& p, const DoorBeliefState& s) const
{
    return door::leaf_sdf(leaf_pose_at(s), p);
}

float DoorBelief::sdf_prim(const Eigen::Vector3f& p, const DoorBeliefState& s, int prim) const
{
    if (prim == 0) return sdf_panel(p, s);
    // Task 15: the same leaf hung from the OTHER jamb. A hypothesis with no probability has no surface at all (so the
    // engine's nearest-primitive information picks it never): the single-hinge model is recovered exactly at P = 1.
    if (not (p_hinge_ < 1.f)) return 1e6f;
    door::LeafState other = params_.leaf;
    other.hinge = other.hinge == door::HingeSide::Near ? door::HingeSide::Far : door::HingeSide::Near;
    return door::leaf_sdf(door::leaf_pose(aperture_at(s), other), p);
}

// ─── Mixture responsibilities [believed leaf, other jamb's leaf, clutter] ─────────────────────
std::array<float, 3> DoorBelief::mixture_unnorm(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const
{
    const float eps     = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pi_surf = 1.0f - eps;
    const float inv2R   = 0.5f / std::max(1e-9f, R);

    std::array<float, 3> u{};
    const float d0 = sdf_panel(p, s);
    u[0] = pi_surf * p_hinge_ * std::exp(-d0 * d0 * inv2R);                       // the believed leaf
    if (p_hinge_ < 1.f)
    {
        const float d1 = sdf_prim(p, s, 1);
        u[1] = pi_surf * (1.f - p_hinge_) * std::exp(-d1 * d1 * inv2R);          // the other jamb's leaf
    }
    const float cs = params_.clutter_scale_m;
    u[2] = eps * std::exp(-cs * cs * inv2R);                                     // clutter
    return u;   // UNNORMALIZED
}

std::array<float, 3> DoorBelief::responsibilities(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const
{
    std::array<float, 3> u = mixture_unnorm(p, s, R);
    const float sum = u[0] + u[1] + u[2];
    if (sum <= 0.0f) return {0.0f, 0.0f, 1.0f};
    for (float& x : u) x /= sum;
    return u;
}

float DoorBelief::mean_energy(const std::vector<Eigen::Vector3f>& pts, const DoorBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double e = 0.0;
    for (const auto& p : pts)
    {
        const auto  r = responsibilities(p, s, R);
        const float d = sdf_panel(p, s);
        e += 0.5 * r[0] * d * d / R;
        if (r[1] > 0.f) { const float d1 = sdf_prim(p, s, 1); e += 0.5 * r[1] * d1 * d1 / R; }
    }
    return static_cast<float>(e / static_cast<double>(pts.size()));
}

// Mean per-point mixture NLL (includes clutter) — the association evidence. A far / wrong-instance slice
// (all points → flat clutter) has a tiny mixture likelihood → HIGH nll → not claimed. The common Gaussian
// normaliser cancels in the argmin, so the raw unnormalised sum suffices.
float DoorBelief::mixture_nll(const std::vector<Eigen::Vector3f>& pts, const DoorBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double nll = 0.0;
    for (const auto& p : pts)
    {
        const auto u = mixture_unnorm(p, s, R);
        nll += -std::log(std::max(1e-30, static_cast<double>(u[0] + u[1] + u[2])));
    }
    return static_cast<float>(nll / static_cast<double>(pts.size()));
}

float DoorBelief::association_nll(const std::vector<Eigen::Vector3f>& pts, float R) const
{
    return mixture_nll(pts, state_, R);
}

float DoorBelief::clutter_fraction(const std::vector<Eigen::Vector3f>& pts, float R) const
{
    if (pts.empty()) return 0.0f;
    double c = 0.0;
    for (const auto& p : pts) c += responsibilities(p, state_, R)[2];   // index 2 = clutter
    return static_cast<float>(c / static_cast<double>(pts.size()));
}

// ─── Jacobian (central finite difference over the 3 DOFs [s,w,h]) ────────────────
Eigen::Matrix<float, 4, 1> DoorBelief::sdf_jacobian(const Eigen::Vector3f& p, const DoorBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 4, 1> J;
    const Eigen::Matrix<float, 4, 1> base = s.vec();
    const float e = params_.fd_eps;
    for (int j = 0; j < N; ++j)
    {
        Eigen::Matrix<float, 4, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        J(j) = (sdf_prim(p, DoorBeliefState::from_vec(vp), prim) -
                sdf_prim(p, DoorBeliefState::from_vec(vm), prim)) / (2.0f * e);
    }
    return J;
}

// ─── Fixed template prior on w,h (non-drifting; folded into the GN normal equations each frame) ──
void DoorBelief::accumulate_extra(const DoorBeliefState& s, const DoorFrame& /*frame*/,
                                  Eigen::Matrix<float, 4, 4>& Id, Eigen::Matrix<float, 4, 1>& bd) const
{
    // GN normal-equation contribution of a residual r=(x−x0) with unit Jacobian and precision iw:
    //   Id += iw · JJᵀ = iw (on the diagonal),   bd += −iw · J · r = −iw·(x − x0).
    const float iw = 1.0f / std::max(1e-9f, params_.prior_w_std * params_.prior_w_std);
    const float ih = 1.0f / std::max(1e-9f, params_.prior_h_std * params_.prior_h_std);
    Id(1, 1) += iw; bd(1) += -iw * (s.w - params_.tpl_w);
    Id(2, 2) += ih; bd(2) += -ih * (s.h - params_.tpl_h);
}

// ─── Constraints (physical positivity + the panel stays within its wall segment; s otherwise free) ───
void DoorBelief::apply_constraints(DoorBeliefState& s) const
{
    s.w = std::max(0.05f, s.w);   // a panel cannot have non-positive extent
    s.h = std::max(0.05f, s.h);
    // ★ NO CLAMP OF s TO THE WALL RUN (2026-10-18). This used to force the span [s, s+w] inside [0, wall_len], on
    //   the argument that a door lies within its wall. In an outline traced from free space the doorway is exactly
    //   where the wall is ABSENT: the outline bends into the opening at the jambs, so the door spans the GAP between
    //   two runs, never the inside of one. Live try50: room_2's carried mirror of door_1 sat at (0.08,0.03), the
    //   run began at x = −0.11, the clamp put s = 0 and the door centre at x = 0.39 (std_s 0.58 m — the belief did
    //   not know, the clamp decided); the cross axis ran 0.35 m off the opening and the robot drove into the jamb.
    //   s is now decided by the evidence and its prior; the wall-band prior still keeps the aperture IN the wall.
}

// ─── Engine hooks: prior cov, process noise, common-mode ─────────────────────────
// d's σ is floored at 0.1 mm: recess_prior_std 0 means "pinned at the prior", and Σ must stay invertible (P0 = Σ⁻¹).
Eigen::Matrix<float, 4, 1> DoorBelief::prior_cov_diag() const
{
    const float sd_d = std::max(1e-4f, params_.recess_prior_std);
    return (Eigen::Matrix<float, 4, 1>() <<
            params_.prior_s_std * params_.prior_s_std,
            params_.prior_w_std * params_.prior_w_std,
            params_.prior_h_std * params_.prior_h_std,
            sd_d * sd_d).finished();
}

// d: no process noise — a doorway's leaf does not move in its wall (the same statement process_std_s = 0 makes for s).
Eigen::Matrix<float, 4, 1> DoorBelief::process_noise_diag() const
{
    return (Eigen::Matrix<float, 4, 1>() <<
            params_.process_std_s * params_.process_std_s,
            params_.process_std_w * params_.process_std_w,
            params_.process_std_h * params_.process_std_h,
            0.0f).finished();
}

Eigen::Matrix<float, 4, 1> DoorBelief::common_mode_inv_diag(const DoorFrame& frame) const
{
    // Task 15: + the leaf's angle uncertainty, carried into where the mask says the aperture is (see set_leaf_phi_sigma).
    const float leaf_s = 0.5f * state_.w * std::abs(std::sin(params_.leaf.phi)) * leaf_phi_sigma_;
    const float s2  = params_.common_mode_s_std  * params_.common_mode_s_std + leaf_s * leaf_s;
    const float wh2 = params_.common_mode_wh_std * params_.common_mode_wh_std;
    // Task 17 add-on: the mask reads the WIDTH through the leaf panel as well — its along-wall extent w·cos φ moves by
    // w·|sin φ|·δφ under an angle error — so the same leaf-angle uncertainty enters the width's common mode (try78: σ_w 5 mm).
    const float leaf_w = state_.w * std::abs(std::sin(params_.leaf.phi)) * leaf_phi_sigma_;
    // Task 5b: the leaf plane's depth d. A frame's points share the SAME pose-chain / range / motion error across the wall as
    // along it (frame.chain_cov_s, common_mode_s_std: one registration), and an angle error δφ moves the leaf's mean
    // across-wall position by (w/2)·|cos φ|·δφ — so at a poorly known φ a frame cannot say where the plane is.
    const float leaf_n = 0.5f * state_.w * std::abs(std::cos(params_.leaf.phi)) * leaf_phi_sigma_;
    const float n2 = params_.common_mode_s_std * params_.common_mode_s_std + leaf_n * leaf_n;
    return (Eigen::Matrix<float, 4, 1>() <<
            1.0f / std::max(1e-9f, s2 + frame.chain_cov_s),
            1.0f / std::max(1e-9f, wh2 + leaf_w * leaf_w),
            1.0f / std::max(1e-9f, wh2),
            1.0f / std::max(1e-9f, n2 + frame.chain_cov_s)).finished();
}

// ─── Self-test ───────────────────────────────────────────────────────────────────
bool DoorBelief::self_test()
{
    std::mt19937 rng(2026);
    std::normal_distribution<float> noise(0.0f, 0.006f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    // A wall running along +x, near corner O at the origin, room interior on the +y side (n = +y).
    DoorBeliefParams P;
    P.wall_O = {0.0f, 0.0f};
    P.wall_u = {1.0f, 0.0f};
    const DoorBeliefState gt{1.20f, 0.72f, 2.05f};   // s, w, h

    // Sample the panel surface in room coordinates (front/back faces + the two vertical edges + top).
    const Eigen::Vector2f u = P.wall_u, nrm(-P.wall_u.y(), P.wall_u.x());
    const auto to_world = [&](float along, float across, float up) -> Eigen::Vector3f
    {
        const Eigen::Vector2f c = P.wall_O + (gt.s + 0.5f * gt.w) * u;
        const Eigen::Vector2f xy = c + along * u + across * nrm;
        return {xy.x(), xy.y(), P.floor_z + 0.5f * gt.h + up};
    };
    std::vector<Eigen::Vector3f> pts;
    const float hw = 0.5f * gt.w, ht = 0.5f * P.thickness, hh = 0.5f * gt.h;
    for (int i = 0; i < 1400; ++i)   // front & back faces (across = ±T/2)
    {
        const float a = U(rng) * hw, up = U(rng) * hh, side = (i % 2 == 0) ? ht : -ht;
        pts.push_back(to_world(a, side + noise(rng), up));
    }
    for (int i = 0; i < 400; ++i)    // the two vertical side edges (along = ±w/2)
    {
        const float side = (i % 2 == 0) ? hw : -hw, up = U(rng) * hh, ac = U(rng) * ht;
        pts.push_back(to_world(side + noise(rng), ac, up));
    }
    for (int i = 0; i < 200; ++i)    // top edge (up = +h/2)
        pts.push_back(to_world(U(rng) * hw, U(rng) * ht, hh + noise(rng)));

    std::vector<Eigen::Vector3f> clutter;
    for (int i = 0; i < 150; ++i)    // off-model floor/room points on the interior side
        clutter.push_back(to_world(U(rng) * 2.0f, 0.5f + U01(rng) * 1.0f, -hh + U01(rng) * 0.05f));
    std::vector<Eigen::Vector3f> all = pts;
    all.insert(all.end(), clutter.begin(), clutter.end());

    bool ok = true;
    auto check = [&](bool cond, const char* msg) { if (!cond) { ok = false; std::printf("  FAIL: %s\n", msg); } };

    // (a) SDF sanity
    {
        DoorBelief b(gt, P);
        const Eigen::Vector2f c = P.wall_O + (gt.s + 0.5f * gt.w) * u;
        check(b.sdf_panel({c.x(), c.y(), P.floor_z + 0.5f * gt.h}, gt) < -1e-3f, "panel centre SDF should be inside");
        check(std::abs(b.sdf_panel(to_world(0.2f * gt.w, ht, 0.1f * gt.h), gt)) < 1e-2f, "on-face SDF ~0");
        check(b.sdf_panel({c.x() + 3.0f, c.y(), P.floor_z + 0.5f * gt.h}, gt) > 1.0f, "far point SDF large");
    }

    // (b) [s,w,h] recovery from a coarse seed (s off by 0.4 m, w,h at the template prior).
    DoorBelief belief(DoorBeliefState{gt.s + 0.4f, 0.70f, 2.00f}, P);
    DoorFrame frame; frame.points = all;
    float e = 0.0f;
    for (int it = 0; it < 40; ++it) e = belief.update(frame);
    const auto& st = belief.state();
    std::printf("  recovered: s=%.3f w=%.3f h=%.3f  (E=%.4f)\n", st.s, st.w, st.h, e);
    std::printf("  truth:     s=%.3f w=%.3f h=%.3f\n", gt.s, gt.w, gt.h);
    check(std::abs(st.s - gt.s) < 0.03f, "s not recovered");
    check(std::abs(st.w - gt.w) < 0.05f, "w not recovered");
    check(std::abs(st.h - gt.h) < 0.06f, "h not recovered");

    // (c) association_nll: a far slice must score HIGH (unclaimable); own slice low.
    {
        DoorBelief b(gt, P);
        const float R = P.sigma_base_m * P.sigma_base_m;
        const float e_own = b.association_nll(pts, R);
        std::vector<Eigen::Vector3f> far;
        for (const auto& q : pts) far.push_back({q.x() + 2.0f, q.y(), q.z()});   // shift 2 m along the wall
        const float e_far = b.association_nll(far, R);
        std::printf("  association_nll: own=%.2f far(2m)=%.2f (far must be >> own)\n", e_own, e_far);
        check(e_far > e_own + 3.0f, "association_nll does not reject a far slice");
    }

    // (d) STRONG w/h prior: a badly under-segmented mask (only the left half seen) must NOT collapse w to
    // half — the tight prior + common-mode cap keep it near the standard 0.70 m.
    {
        DoorBelief b(DoorBeliefState{gt.s, 0.70f, 2.00f}, P);
        std::vector<Eigen::Vector3f> half;
        for (const auto& q : pts)
        {
            const Eigen::Vector2f d(q.x() - P.wall_O.x(), q.y() - P.wall_O.y());
            if (d.dot(u) < gt.s + 0.5f * gt.w) half.push_back(q);   // keep only the near half along the wall
        }
        DoorFrame hf; hf.points = half;
        for (int it = 0; it < 40; ++it) b.update(hf);
        std::printf("  half-mask w: %.3f (prior 0.70, must stay > 0.60)\n", b.state().w);
        check(b.state().w > 0.60f, "strong width prior collapsed on an under-segmented mask");
    }

    // (e) WIDE contamination: a mask that over-segments to 1.6 m wide (grabbing wall/jamb beyond the leaf)
    // must NOT drag w wide — the fixed template anchor + wide size common-mode keep it near 0.70 (the live
    // w→1.12 m drift). This is the regression test for the accumulate_extra anchor.
    {
        DoorBelief b(DoorBeliefState{gt.s, 0.70f, 2.00f}, P);
        std::vector<Eigen::Vector3f> wide = pts;
        for (int i = 0; i < 1000; ++i)   // extra front/back-face points out to ±0.8 m along the wall
        {
            const float a = (U(rng) < 0.0f ? -1.0f : 1.0f) * (hw + U01(rng) * (0.8f - hw));
            const float up = U(rng) * hh, side = (i % 2 == 0) ? ht : -ht;
            wide.push_back(to_world(a, side + noise(rng), up));
        }
        DoorFrame wf; wf.points = wide;
        for (int it = 0; it < 80; ++it) b.update(wf);
        std::printf("  wide-mask(1.6m) w: %.3f (anchor 0.70, must stay < 0.90)\n", b.state().w);
        check(b.state().w < 0.90f, "template width anchor overwhelmed by an over-segmented mask (w drifted wide)");
    }

    // (f) NO segment clamp (2026-10-18): a door spans the GAP between two wall runs of a free-space outline, so a
    //     span past the run's end is the normal case at a doorway — apply_constraints must leave s alone.
    {
        DoorBeliefParams Pc = P; Pc.wall_len = 1.0f;
        DoorBelief b(DoorBeliefState{0.90f, 0.70f, 2.00f}, Pc);   // s+w = 1.6 > len = 1.0: the door straddles the run end
        DoorBeliefState s = b.state(); b.apply_constraints(s); b.set_state(s);
        std::printf("  no clamp: s=%.3f (wall_len=1.0, w=0.70 → s must stay 0.90)\n", b.state().s);
        check(std::abs(b.state().s - 0.90f) < 1e-4f, "apply_constraints moved s: a doorway span must not be clamped into one run");
    }

    // (g) APERTURE / LEAF geometry (door_geometry.h). Asserts that phi = 0 reproduces the old wall-plane
    // expressions BIT-EXACTLY, that a swing pins the hinge, and that the aperture never moves with phi —
    // the regression net that keeps M1/M2 from re-scattering the geometry. One test entry point for the agent.
    check(door::self_test(), "door_geometry self_test failed (see the FAIL lines above)");

    const auto& S = belief.covariance();
    std::printf("  Σ diag (std): s=%.3f w=%.3f h=%.3f\n",
                std::sqrt(std::max(0.f, S(0, 0))), std::sqrt(std::max(0.f, S(1, 1))), std::sqrt(std::max(0.f, S(2, 2))));
    std::printf("DoorBelief::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}


// ─── the hinge branch ────────────────────────────────────────────────────────────────────────────
float DoorBelief::phi_free_energy(const DoorFrame& f, float phi) const
{
    if (f.points.empty())
        return 0.0f;
    // A const method that must evaluate the SDF at a DIFFERENT leaf angle: copy the params rather than
    // mutate ours. The copy is three floats and a pose; the alternative is a mutable member that makes
    // every caller wonder whether the object changed underneath them.
    DoorBelief probe(state_, params_);
    probe.set_leaf_phi(phi);
    const float s2 = probe.sigma2();
    double acc = 0.0;
    for (std::size_t i = 0; i < f.points.size(); ++i)
    {
        const float d = probe.sdf_panel(f.points[i], probe.state_);
        const float r = (i < f.R.size()) ? f.R[i] : 0.0f;
        acc += 0.5 * static_cast<double>(d) * d / std::max(1e-6f, s2 + r);
    }
    // ★PER POINT, NOT SUMMED — 1600 RAYS ON ONE LEAF ARE NOT 1600 INDEPENDENT OBSERVATIONS. They are one
    // flat surface sampled 1600 times, and the rays are as correlated as the surface is rigid. Summing
    // multiplies the evidence by however many happened to land, which is a property of the sensor's
    // geometry and the robot's distance, not of the door.
    // ★MEASURED, AND IT BROKE THE ESTIMATOR IN A VERY SPECIFIC WAY. With the sum, the free-energy
    // difference between NEIGHBOURING angles ran to hundreds of nats, so exp(-(F - F_min)) was 1 at the
    // argmin and numerically 0 everywhere else: the likelihood was a delta function. The mixture then had
    // no middle — either the SDF's argmin survived the prior and won outright, or its weight at the
    // chosen angle was exactly zero and the mask channel took over and pulled the leaf back to flush.
    // That is the bimodal estimate seen live: 476 rows near 0 deg, 106 piled against the 120 deg clamp,
    // almost nothing between, with w_sdf logged as either ~0.9 or exactly 0.0000 and never in between.
    // It got WORSE as the robot approached and the ray count rose 1183 -> 1660.
    // Dividing by the count makes the curve's sharpness a property of how well the surface FITS rather
    // than of how many rays hit it, which is the honest reading when they all lie on one plane.
    // ⚠THIS IS THE CHEAP CORRECTION, NOT THE COMPLETE ONE. The principled treatment is the common-mode
    // marginalisation this codebase already applies to correlated mask points (Woodbury, not a sigma
    // floor): the shared surface error belongs in a common-mode term, and what remains per point is the
    // genuinely independent part. The mean is that idea taken at its crudest — n_eff = 1 — which is
    // conservative in the right direction: it under-claims evidence rather than over-claiming it.
    return static_cast<float>(acc / static_cast<double>(f.points.size()));
}

std::vector<std::pair<float, float>>
DoorBelief::phi_likelihood(const DoorFrame& f, float phi_min, float phi_max, int nstep) const
{
    std::vector<std::pair<float, float>> out;
    if (nstep < 2 or f.points.empty())
        return out;   // no evidence ⇒ an EMPTY curve, which the caller must read as "not measured"
    out.reserve(static_cast<std::size_t>(nstep));
    std::vector<float> F;
    F.reserve(static_cast<std::size_t>(nstep));
    float fmin = std::numeric_limits<float>::max();
    for (int i = 0; i < nstep; ++i)
    {
        const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(i) / (nstep - 1);
        const float fe  = phi_free_energy(f, phi);
        F.push_back(fe);
        fmin = std::min(fmin, fe);
    }
    // exp(-(F - F_min)) — shifted so the best hypothesis is 1 and the rest fall off by their excess free
    // energy in nats. The shift is a normalisation, not a threshold: it cancels in any ratio the consumer
    // forms, and it keeps the exponential from underflowing on a cloud with many points.
    for (int i = 0; i < nstep; ++i)
    {
        const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(i) / (nstep - 1);
        out.emplace_back(phi, std::exp(-(F[static_cast<std::size_t>(i)] - fmin)));
    }
    return out;
}


float DoorBelief::phi_ray_free_energy(const rc::ai::LidarRays& rays, float phi) const
{
    DoorBelief probe(state_, params_);
    probe.set_leaf_phi(phi);
    // Per RAY (the shared cost divides), for the same reason phi_free_energy is per point: how many rays
    // happen to fall on a door is a property of the sensor and the stand-off, not of the door.
    return rc::ai::lidar_ray_cost<DoorBelief::N, DoorBelief, DoorBeliefState>(probe, probe.state_, rays);
}

std::vector<std::pair<float, float>>
DoorBelief::phi_ray_likelihood(const rc::ai::LidarRays& rays, float phi_min, float phi_max, int nstep) const
{
    std::vector<std::pair<float, float>> out;
    if (nstep < 2 or rays.endpoints.empty() or rays.precision <= 0.0f)
        return out;   // no rays ⇒ an EMPTY curve = NOT MEASURED, never a vote for any angle
    std::vector<float> F; F.reserve(static_cast<std::size_t>(nstep));
    float fmin = std::numeric_limits<float>::max();
    for (int i = 0; i < nstep; ++i)
    {
        const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(i) / (nstep - 1);
        const float fe  = phi_ray_free_energy(rays, phi);
        F.push_back(fe); fmin = std::min(fmin, fe);
    }
    out.reserve(static_cast<std::size_t>(nstep));
    for (int i = 0; i < nstep; ++i)
        out.emplace_back(phi_min + (phi_max - phi_min) * static_cast<float>(i) / (nstep - 1),
                         std::exp(-(F[static_cast<std::size_t>(i)] - fmin)));
    return out;
}


namespace
{
    // ── Ray vs ORIENTED BOX (the leaf), slab method. Returns the ENTRY distance, or <0 for no hit. ──
    // Analytic on purpose: the field likelihood evaluates nstep * depth_n * lat_n hypotheses per cycle,
    // and sphere-tracing an SDF per ray per hypothesis is two orders of magnitude too slow for 10 Hz.
    float ray_obb_entry(const Eigen::Vector3f& o, const Eigen::Vector3f& u, const door::LeafPose& L)
    {
        const Eigen::Vector3f ex(L.ex.x(), L.ex.y(), 0.0f);
        const Eigen::Vector3f ey(L.ey.x(), L.ey.y(), 0.0f);
        const Eigen::Vector3f ez(0.0f, 0.0f, 1.0f);
        const Eigen::Vector3f c(L.centre_xy.x(), L.centre_xy.y(), L.centre_z);
        const Eigen::Vector3f d = c - o;
        const float half[3] = {L.half_w, L.half_t, L.half_h};
        const Eigen::Vector3f axis[3] = {ex, ey, ez};
        float tmin = 0.0f, tmax = std::numeric_limits<float>::max();
        for (int i = 0; i < 3; ++i)
        {
            const float e = axis[i].dot(d), f = axis[i].dot(u);
            if (std::abs(f) > 1e-6f)
            {
                float t1 = (e - half[i]) / f, t2 = (e + half[i]) / f;
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return -1.0f;
            }
            else if (-e - half[i] > 0.0f or -e + half[i] < 0.0f)
                return -1.0f;                     // parallel and outside this slab
        }
        return tmin > 0.0f ? tmin : -1.0f;
    }

    // ── Ray vs the ROOM's walls (FieldNuisance::room_walls): vertical segments standing on the floor. Returns the first
    //    hit distance along the unit 3-D ray u, or <0 for none. A crossing below the floor is not a wall hit (the floor
    //    stopped the ray first). Static: the same under every leaf hypothesis and every door nuisance.
    float ray_room_walls(const Eigen::Vector3f& o, const Eigen::Vector3f& u,
                         const std::vector<std::array<Eigen::Vector2f, 2>>& walls, float floor_z)
    {
        float best = -1.0f;
        const Eigen::Vector2f o2 = o.head<2>(), u2 = u.head<2>();
        for (const auto& [a, b] : walls)
        {
            const Eigen::Vector2f e = b - a, d = a - o2;
            const float den = u2.x() * e.y() - u2.y() * e.x();
            if (std::abs(den) < 1e-9f) continue;                      // parallel to the wall
            const float t = (d.x() * e.y() - d.y() * e.x()) / den;    // along the ray (3-D distance: u is unit)
            const float s = (d.x() * u2.y() - d.y() * u2.x()) / den;  // along the wall segment, [0, 1]
            if (t <= 0.0f or s < 0.0f or s > 1.0f) continue;
            if (o.z() + t * u.z() < floor_z) continue;
            if (best < 0.0f or t < best) best = t;
        }
        return best;
    }
}   // namespace

std::vector<std::pair<float, float>>
DoorBelief::phi_field_likelihood(const rc::ai::LidarRays& rays, float phi_min, float phi_max, int nstep,
                                 const FieldNuisance& nz) const
{
    std::vector<std::pair<float, float>> out;
    if (nstep < 2 or rays.endpoints.empty())
        return out;   // no rays ⇒ EMPTY curve = NOT MEASURED, never a vote for any angle

    const float cap   = std::max(0.5f, rays.max_range_m);
    const float sigma = std::max(0.01f, rays.robust_c_m);        // range noise, not a robust scale
    const float w_u   = std::clamp(nz.w_unexp, 0.0f, 0.5f);
    const float w_b   = std::clamp(nz.eps_beyond, 0.0f, 0.5f);
    const float w_s   = std::max(0.0f, 1.0f - w_u - w_b);
    const float inv_sqrt2pi_sig = 1.0f / (sigma * std::sqrt(2.0f * std::numbers::pi_v<float>));

    // Stride-subsample: the cost is (hypotheses x rays) and a doorway is oversampled at close range.
    const std::size_t n_all = rays.endpoints.size();
    const std::size_t stride = nz.max_rays > 0 and n_all > static_cast<std::size_t>(nz.max_rays)
                             ? (n_all + nz.max_rays - 1) / static_cast<std::size_t>(nz.max_rays) : 1;

    const int   dn = std::max(1, nz.depth_n), ln = std::max(1, nz.lat_n);
    const auto  grid = [](float lo, float hi, int n, int i)
                       { return n <= 1 ? 0.5f * (lo + hi) : lo + (hi - lo) * static_cast<float>(i) / (n - 1); };

    // log L(phi, depth, lat) summed over rays, for every hypothesis; then marginalise the nuisances.
    std::vector<double> best(static_cast<std::size_t>(nstep), -std::numeric_limits<double>::infinity());
    std::vector<std::vector<double>> ll(static_cast<std::size_t>(nstep));

    for (int k = 0; k < nstep; ++k)
    {
        const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(k) / (nstep - 1);
        ll[static_cast<std::size_t>(k)].reserve(static_cast<std::size_t>(dn * ln));
        for (int di = 0; di < dn; ++di)
        {
            const float depth = grid(nz.depth_lo, nz.depth_hi, dn, di);
            for (int li = 0; li < ln; ++li)
            {
                const float lat = grid(nz.lat_lo, nz.lat_hi, ln, li);

                DoorBeliefState st = state_;
                st.s += lat;                       // aperture slides along the wall
                const door::Aperture ap = aperture_at(st);
                door::LeafState lf = params_.leaf;
                lf.phi = phi;
                door::LeafPose L = door::leaf_pose(ap, lf);

                // The nuisance moves the wall AND the leaf together: they are one rigid structure.
                const Eigen::Vector2f n2 = ap.across_u();
                const Eigen::Vector3f n3(n2.x(), n2.y(), 0.0f);
                L.centre_xy += depth * n2;
                L.hinge_xy  += depth * n2;
                const Eigen::Vector2f wallO = ap.wall_O + depth * n2;   // the WALL stays on the polygon line; the leaf L hangs on its own plane (Aperture::depth, 5b)
                const Eigen::Vector3f wallP(wallO.x(), wallO.y(), 0.0f);

                double acc = 0.0;
                for (std::size_t i = 0; i < n_all; i += stride)
                {
                    const Eigen::Vector3f& ep = rays.endpoints[i];
                    const Eigen::Vector3f dir = ep - rays.origin;
                    const float rho = dir.norm();
                    if (rho < 1e-3f) continue;
                    const Eigen::Vector3f u = dir / rho;

                    // ── the WALL plane, with the aperture cut out of it ──────────────────────────
                    float t_wall = -1.0f, t_ap = -1.0f;
                    const float den = u.dot(n3);
                    if (std::abs(den) > 1e-6f)
                    {
                        const float t = (wallP - rays.origin).dot(n3) / den;
                        if (t > 0.0f)
                        {
                            const Eigen::Vector3f P = rays.origin + t * u;
                            const float along = (P.head<2>() - wallO).dot(ap.wall_u);
                            const bool in_open = along >= ap.s and along <= ap.s + ap.w
                                             and P.z() >= ap.floor_z and P.z() <= ap.floor_z + ap.h;
                            if (in_open) t_ap = t;                 // the ray goes through the hole
                            else if (P.z() >= ap.floor_z) t_wall = t;   // solid wall (floor is not ours)
                        }
                    }
                    // ── the ROOM's walls (try97): a wall met before the door's wall/hole stops the ray there ─────
                    if (const float t_room = ray_room_walls(rays.origin, u, nz.room_walls, ap.floor_z);
                        t_room > 0.0f and (t_wall <= 0.0f or t_room < t_wall) and (t_ap <= 0.0f or t_room < t_ap))
                    { t_wall = t_room; t_ap = -1.0f; }
                    // ── the LEAF ────────────────────────────────────────────────────────────────
                    const float t_leaf = ray_obb_entry(rays.origin, u, L);

                    float t_model = -1.0f;
                    if (t_wall > 0.0f and t_leaf > 0.0f) t_model = std::min(t_wall, t_leaf);
                    else if (t_wall > 0.0f)              t_model = t_wall;
                    else if (t_leaf > 0.0f)              t_model = t_leaf;

                    // ★EVERY RAY IS SCORED UNDER EVERY HYPOTHESIS, against a density over the SAME support.
                    // (Fixed 2026-09-16, after the live crossing.) This used to `continue` when the model
                    // predicted no surface, on the theory that such a ray "cancels". It cancels only if it is
                    // skipped under ALL hypotheses — but whether a ray meets the leaf DEPENDS ON phi. So each
                    // angle was scored on a different subset of rays, and an angle that predicted FEWER hits
                    // collected fewer penalties and won for that reason alone. Measured live with the robot
                    // 0.26 m from the door centre, i.e. with the sensor INSIDE the aperture: 0 of 3061 rays
                    // explained by a shut leaf, and phi still read 5 deg at full confidence. The synthetic
                    // never exposed it because its sensor always stood well outside the doorway.
                    // ★AND EACH UNIFORM COMPONENT APPLIES ONLY ON ITS OWN SIDE of the prediction. Both used to
                    // be added to every ray. That is not a density, and it rewarded the wrong hypothesis in
                    // exactly the doorway case: a leaf predicted 0.1 m away earned w_u/t = 1.0 of free credit
                    // on rays that flew 2.4 m past it.
                    // All three cases below integrate to 1 over [0, cap], so their values are comparable.
                    const float r = std::min(rho, cap - 1e-3f);   // a no-return reads as "at max range"
                    double p = 0.0;
                    if (t_model > 0.0f and t_model < cap)
                    {
                        // A surface is predicted at t: a return ON it, SHORT of it, or BEYOND it.
                        const float z = (r - t_model) / sigma;
                        p = static_cast<double>(w_s) * inv_sqrt2pi_sig * std::exp(-0.5f * z * z);
                        if (r < t_model) p += static_cast<double>(w_u) / std::max(1e-3f, t_model);
                        else             p += static_cast<double>(w_b) / std::max(1e-3f, cap - t_model);
                    }
                    else if (t_ap > 0.0f and t_ap < cap)
                    {
                        // ★THE RAY LEAVES THROUGH THE OPENING — the term the leaf-only model lacked.
                        // "shut" predicts a surface here, so a return far beyond the aperture REFUTES it
                        // instead of scoring the same constant under both hypotheses.
                        if (r >= t_ap) p = static_cast<double>(1.0f - w_u) / std::max(1e-3f, cap - t_ap);
                        else           p = static_cast<double>(w_u) / std::max(1e-3f, t_ap);
                    }
                    else
                    {
                        // Nothing modelled along this ray: the return may be anywhere in range. A proper
                        // density, NOT an omission — this is what keeps the hypotheses comparable.
                        p = 1.0 / static_cast<double>(cap);
                    }

                    acc += std::log(std::max(1e-30, p));
                }
                ll[static_cast<std::size_t>(k)].push_back(acc);
                best[static_cast<std::size_t>(k)] = std::max(best[static_cast<std::size_t>(k)], acc);
            }
        }
    }

    // Marginalise the nuisances (log-sum-exp, uniform prior over the grid), then normalise the curve to
    // its own maximum so the caller sees exp(-(F - F_min)) exactly as it did from phi_ray_likelihood.
    std::vector<double> m(static_cast<std::size_t>(nstep), -std::numeric_limits<double>::infinity());
    double mmax = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < nstep; ++k)
    {
        const auto& v = ll[static_cast<std::size_t>(k)];
        const double b = best[static_cast<std::size_t>(k)];
        if (not std::isfinite(b)) continue;
        double acc = 0.0;
        for (const double x : v) acc += std::exp(x - b);
        m[static_cast<std::size_t>(k)] = b + std::log(std::max(1e-300, acc / static_cast<double>(v.size())));
        mmax = std::max(mmax, m[static_cast<std::size_t>(k)]);
    }
    out.reserve(static_cast<std::size_t>(nstep));
    for (int k = 0; k < nstep; ++k)
    {
        const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(k) / (nstep - 1);
        const double v = m[static_cast<std::size_t>(k)];
        out.emplace_back(phi, std::isfinite(v) ? static_cast<float>(std::exp(v - mmax)) : 0.0f);
    }
    return out;
}


// ─── phi_field_loglik: the same field density, restructured for a wide nuisance and tempered ──────
// The density per ray is IDENTICAL to phi_field_likelihood's (test_door_phi_stability checks it to float
// precision with beta forced to 1). What changes is (a) the cost, so the aperture slide can span +-w/2 at
// 5 cm, and (b) the power on the joint log-likelihood — see FieldResult in the header for the model.
// ★WHY IT IS FAST ENOUGH. The leaf changes the density of a ray only if it is met BEFORE the wall, so the
// "wall with a hole, no leaf" density is computed once per (depth, slide) and the leaf adds a correction
// only for the rays it actually intercepts. The ray–leaf slab test is done in the leaf's own axes, where
// the nuisance moves the box by a scalar per axis: its direction terms are per (angle, ray), not per
// (angle, depth, slide, ray).
DoorBelief::FieldResult
DoorBelief::phi_field_loglik(const rc::ai::LidarRays& rays, float phi_min, float phi_max, int nstep,
                             const FieldNuisance& nz) const
{
    FieldResult out;
    if (nstep < 2 or rays.endpoints.empty())
        return out;   // no rays ⇒ EMPTY = NOT MEASURED, never a vote for any angle

    const float cap   = std::max(0.5f, rays.max_range_m);
    const float sigma = std::max(0.01f, rays.robust_c_m);
    const float w_u   = std::clamp(nz.w_unexp, 0.0f, 0.5f);
    const float w_b   = std::clamp(nz.eps_beyond, 0.0f, 0.5f);
    const float w_s   = std::max(0.0f, 1.0f - w_u - w_b);
    const float inv_sqrt2pi_sig = 1.0f / (sigma * std::sqrt(2.0f * std::numbers::pi_v<float>));

    // The per-ray density, exactly as phi_field_likelihood writes it. t_model: first modelled surface
    // (wall or leaf), <= 0 if none; t_ap: where the ray leaves through the opening, <= 0 if it does not.
    const auto log_density = [&](float t_model, float t_ap, float r) -> double
    {
        double p;
        if (t_model > 0.0f and t_model < cap)
        {
            const float z = (r - t_model) / sigma;
            p = static_cast<double>(w_s) * inv_sqrt2pi_sig * std::exp(-0.5f * z * z);
            if (r < t_model) p += static_cast<double>(w_u) / std::max(1e-3f, t_model);
            else             p += static_cast<double>(w_b) / std::max(1e-3f, cap - t_model);
        }
        else if (t_ap > 0.0f and t_ap < cap)
        {
            if (r >= t_ap) p = static_cast<double>(1.0f - w_u) / std::max(1e-3f, cap - t_ap);
            else           p = static_cast<double>(w_u) / std::max(1e-3f, t_ap);
        }
        else
            p = 1.0 / static_cast<double>(cap);
        return std::log(std::max(1e-30, p));
    };

    // ── rays (stride-subsampled exactly as phi_field_likelihood does) ──
    const std::size_t n_all  = rays.endpoints.size();
    const std::size_t stride = nz.max_rays > 0 and n_all > static_cast<std::size_t>(nz.max_rays)
                             ? (n_all + nz.max_rays - 1) / static_cast<std::size_t>(nz.max_rays) : 1;
    std::vector<Eigen::Vector3f> U;
    std::vector<float> Rr;
    U.reserve(n_all / stride + 1); Rr.reserve(n_all / stride + 1);
    for (std::size_t i = 0; i < n_all; i += stride)
    {
        const Eigen::Vector3f dir = rays.endpoints[i] - rays.origin;
        const float rho = dir.norm();
        if (rho < 1e-3f) continue;
        U.push_back(dir / rho);
        Rr.push_back(std::min(rho, cap - 1e-3f));   // a no-return reads as "at max range"
    }
    const int M = static_cast<int>(U.size());
    out.n_rays = M;
    if (M == 0)
        return out;

    const int dn = std::max(1, nz.depth_n), ln = std::max(1, nz.lat_n);
    const auto grid = [](float lo, float hi, int n, int i)
                      { return n <= 1 ? 0.5f * (lo + hi) : lo + (hi - lo) * static_cast<float>(i) / (n - 1); };
    const int NN = dn * ln;

    const door::Aperture ap0 = aperture_at(state_);
    const Eigen::Vector2f n2 = ap0.across_u();
    const Eigen::Vector2f uw = ap0.wall_u;
    const Eigen::Vector3f n3(n2.x(), n2.y(), 0.0f);
    const Eigen::Vector3f& o = rays.origin;

    // ── the wall with its hole, per (depth, slide, ray): the effective wall hit and the base density ──
    // Index: (di * ln + li) * M + i.
    std::vector<float>  t_wall_eff(static_cast<std::size_t>(NN) * M, -1.0f);
    std::vector<float>  t_ap_v(static_cast<std::size_t>(NN) * M, -1.0f);
    std::vector<double> base(static_cast<std::size_t>(NN) * M, 0.0);
    // The room's walls (FieldNuisance::room_walls, try97): static, so once per ray.
    std::vector<float> t_room(static_cast<std::size_t>(M), -1.0f);
    if (not nz.room_walls.empty())
        for (int i = 0; i < M; ++i)
            t_room[static_cast<std::size_t>(i)] = ray_room_walls(o, U[static_cast<std::size_t>(i)], nz.room_walls, ap0.floor_z);
    for (int di = 0; di < dn; ++di)
    {
        const float depth = grid(nz.depth_lo, nz.depth_hi, dn, di);
        const Eigen::Vector2f wallO = ap0.wall_O + depth * n2;   // the wall on the polygon line (the leaf: Aperture::depth, 5b)
        const Eigen::Vector3f wallP(wallO.x(), wallO.y(), 0.0f);
        for (int i = 0; i < M; ++i)
        {
            const Eigen::Vector3f& u = U[static_cast<std::size_t>(i)];
            float t_plane = -1.0f, along = 0.0f, pz = 0.0f;
            const float den = u.dot(n3);
            if (std::abs(den) > 1e-6f)
            {
                const float t = (wallP - o).dot(n3) / den;
                if (t > 0.0f)
                {
                    const Eigen::Vector3f P = o + t * u;
                    t_plane = t;
                    along   = (P.head<2>() - wallO).dot(uw);
                    pz      = P.z();
                }
            }
            for (int li = 0; li < ln; ++li)
            {
                const float lat = grid(nz.lat_lo, nz.lat_hi, ln, li);
                const float s   = ap0.s + lat;
                float t_wall = -1.0f, t_ap = -1.0f;
                if (t_plane > 0.0f)
                {
                    const bool in_open = along >= s and along <= s + ap0.w
                                     and pz >= ap0.floor_z and pz <= ap0.floor_z + ap0.h;
                    if (in_open) t_ap = t_plane;
                    else if (pz >= ap0.floor_z) t_wall = t_plane;
                }
                // A room wall met before the door's wall / hole stops the ray there (the same rule as phi_field_likelihood).
                if (const float tr = t_room[static_cast<std::size_t>(i)];
                    tr > 0.0f and (t_wall <= 0.0f or tr < t_wall) and (t_ap <= 0.0f or tr < t_ap))
                { t_wall = tr; t_ap = -1.0f; }
                const std::size_t idx = static_cast<std::size_t>(di * ln + li) * M + i;
                t_wall_eff[idx] = t_wall;
                t_ap_v[idx]     = t_ap;
                const double b  = log_density(t_wall, t_ap, Rr[static_cast<std::size_t>(i)]);
                base[idx] = b;
            }
        }
    }

    // ── the leaf, per angle: a correction on the rays it intercepts before the wall ──
    // In the leaf's own axes (ex, ey, ez) the nuisance moves the box by a scalar per axis:
    //   e_a = a·(c_k − o) + depth (a·n) + lat (a·u_w),   f_a = a·u_i (independent of the nuisance).
    struct LeafK { float half[3]; float exn, exu, eyn, eyu; float cz; Eigen::Vector2f ex, ey, c; };
    // Hypothesis index kk = hyp * nstep + k over the (hinge, swing) hypotheses asked for: hyp 0 = the believed leaf, then
    // the other jamb (other_hinge), the other swing (other_swing, Task 20), and both flipped (when both are asked for).
    struct Hyp { bool flip_hinge, flip_swing; };
    std::vector<Hyp> hyps{{false, false}};
    if (nz.other_hinge) hyps.push_back({true, false});
    if (nz.other_swing) hyps.push_back({false, true});
    if (nz.other_hinge and nz.other_swing) hyps.push_back({true, true});
    const int H  = static_cast<int>(hyps.size());
    const int KK = H * nstep;
    std::vector<LeafK> leaves(static_cast<std::size_t>(KK));
    for (int kk = 0; kk < KK; ++kk)
    {
        const int k = kk % nstep;
        const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(k) / (nstep - 1);
        door::LeafState lf = params_.leaf;
        lf.phi = phi;
        const Hyp& hy = hyps[static_cast<std::size_t>(kk / nstep)];
        if (hy.flip_hinge)
            lf.hinge = lf.hinge == door::HingeSide::Near ? door::HingeSide::Far : door::HingeSide::Near;
        if (hy.flip_swing)
            lf.swing = -lf.swing;
        const door::LeafPose L = door::leaf_pose(ap0, lf);
        LeafK& q = leaves[static_cast<std::size_t>(kk)];
        q.half[0] = L.half_w; q.half[1] = L.half_t; q.half[2] = L.half_h;
        q.ex = L.ex; q.ey = L.ey; q.c = L.centre_xy; q.cz = L.centre_z;
        q.exn = L.ex.dot(n2); q.exu = L.ex.dot(uw);
        q.eyn = L.ey.dot(n2); q.eyu = L.ey.dot(uw);
    }
    // Slab entry, same semantics as ray_obb_entry (parallel-and-outside ⇒ miss; inside at t=0 ⇒ miss).
    const auto slab = [](const float e[3], const float f[3], const float half[3]) -> float
    {
        float tmin = 0.0f, tmax = std::numeric_limits<float>::max();
        for (int a = 0; a < 3; ++a)
        {
            if (std::abs(f[a]) > 1e-6f)
            {
                float t1 = (e[a] - half[a]) / f[a], t2 = (e[a] + half[a]) / f[a];
                if (t1 > t2) std::swap(t1, t2);
                tmin = std::max(tmin, t1);
                tmax = std::min(tmax, t2);
                if (tmin > tmax) return -1.0f;
            }
            else if (-e[a] - half[a] > 0.0f or -e[a] + half[a] < 0.0f)
                return -1.0f;
        }
        return tmin > 0.0f ? tmin : -1.0f;
    };
    // Per angle: the leaf's offset from the sensor in its own axes (shared by every ray — one origin) and
    // each ray's direction cosines on the leaf's in-plane axes.
    std::vector<float> fx(static_cast<std::size_t>(M)), fy(static_cast<std::size_t>(M));
    float ex0 = 0.0f, ey0 = 0.0f, ez0 = 0.0f;
    const auto prepare = [&](int k)
    {
        const LeafK& q = leaves[static_cast<std::size_t>(k)];
        const Eigen::Vector2f d = q.c - o.head<2>();
        ex0 = q.ex.dot(d); ey0 = q.ey.dot(d); ez0 = q.cz - o.z();
        for (int i = 0; i < M; ++i)
        {
            const Eigen::Vector3f& u = U[static_cast<std::size_t>(i)];
            fx[static_cast<std::size_t>(i)] = q.ex.x() * u.x() + q.ex.y() * u.y();
            fy[static_cast<std::size_t>(i)] = q.ey.x() * u.x() + q.ey.y() * u.y();
        }
    };
    // Density of ray i under (k, di, li): the leaf's correction if it intercepts the ray before the wall,
    // otherwise the wall-with-a-hole density computed once above.
    const auto ray_term = [&](int k, int di, int li, int i, float depth, float lat) -> double
    {
        const LeafK& q = leaves[static_cast<std::size_t>(k)];
        const std::size_t si = static_cast<std::size_t>(i);
        const float e[3] = {ex0 + depth * q.exn + lat * q.exu,
                            ey0 + depth * q.eyn + lat * q.eyu,
                            ez0};
        const float f[3] = {fx[si], fy[si], U[si].z()};
        const float t_leaf = slab(e, f, q.half);
        const std::size_t idx = static_cast<std::size_t>(di * ln + li) * M + si;
        const float t_wall = t_wall_eff[idx];
        if (t_leaf > 0.0f and (t_wall <= 0.0f or t_leaf < t_wall))
            return log_density(t_leaf, t_ap_v[idx], Rr[si]);
        return base[idx];
    };

    // The slide's log-prior per grid cell (normalised over the grid): flat unless lat_sigma says where the aperture is.
    std::vector<double> lprior(static_cast<std::size_t>(ln), 0.0);
    if (nz.lat_sigma > 0.0f)
    {
        double z = -std::numeric_limits<double>::infinity();
        for (int li = 0; li < ln; ++li)
        {
            const double r = grid(nz.lat_lo, nz.lat_hi, ln, li) / nz.lat_sigma;
            lprior[static_cast<std::size_t>(li)] = -0.5 * r * r;
            z = std::max(z, lprior[static_cast<std::size_t>(li)]);
        }
        double acc = 0.0;
        for (const double v : lprior) acc += std::exp(v - z);
        const double lz = z + std::log(acc / static_cast<double>(ln));   // so a flat prior is exactly 0
        for (double& v : lprior) v -= lz;
    }

    std::vector<double> ll(static_cast<std::size_t>(KK) * NN, 0.0);
    double best = -std::numeric_limits<double>::infinity();
    int bd = 0, bl = 0;
    for (int k = 0; k < KK; ++k)
    {
        prepare(k);
        for (int di = 0; di < dn; ++di)
        {
            const float depth = grid(nz.depth_lo, nz.depth_hi, dn, di);
            for (int li = 0; li < ln; ++li)
            {
                const float lat = grid(nz.lat_lo, nz.lat_hi, ln, li);
                double acc = 0.0;
                for (int i = 0; i < M; ++i)
                    acc += ray_term(k, di, li, i, depth, lat);
                ll[static_cast<std::size_t>(k) * NN + di * ln + li] = acc;
                // The MAP nuisance and n_inf below are the BELIEVED leaf's (k < nstep), exactly as before the hinge
                // was marginalised: they measure how much this frame says about the ANGLE, and the other hinge's
                // hypotheses are scored under the same tempering. Counting the rays that only separate the two
                // hinges as well would temper the angle by evidence it never received (test (c): P 0.95 -> 0.78).
                if (k < nstep and acc + lprior[static_cast<std::size_t>(li)] > best)
                { best = acc + lprior[static_cast<std::size_t>(li)]; bd = di; bl = li; }
            }
        }
    }

    // ── how many rays discriminate, at the MAP nuisance ──
    {
        const float depth = grid(nz.depth_lo, nz.depth_hi, dn, bd);
        const float lat   = grid(nz.lat_lo, nz.lat_hi, ln, bl);
        std::vector<double> vmin(static_cast<std::size_t>(M), std::numeric_limits<double>::infinity());
        std::vector<double> vmax(static_cast<std::size_t>(M), -std::numeric_limits<double>::infinity());
        for (int k = 0; k < nstep; ++k)
        {
            prepare(k);
            for (int i = 0; i < M; ++i)
            {
                const double v = ray_term(k, bd, bl, i, depth, lat);
                vmin[static_cast<std::size_t>(i)] = std::min(vmin[static_cast<std::size_t>(i)], v);
                vmax[static_cast<std::size_t>(i)] = std::max(vmax[static_cast<std::size_t>(i)], v);
            }
        }
        double n_inf = 0.0;
        for (int i = 0; i < M; ++i)
            n_inf += 1.0 - std::exp(-(vmax[static_cast<std::size_t>(i)] - vmin[static_cast<std::size_t>(i)]));
        out.n_inf = static_cast<float>(n_inf);
        out.lat_map = lat;
        out.depth_map = depth;
    }

    // ── the common mode: rho from the nuisance grid's own quantisation, unless overridden ──
    float rho = nz.ray_icc;
    if (rho < 0.0f)
    {
        const float dstep = dn > 1 ? (nz.depth_hi - nz.depth_lo) / static_cast<float>(dn - 1) : 0.0f;
        const float lstep = ln > 1 ? (nz.lat_hi - nz.lat_lo) / static_cast<float>(ln - 1) : 0.0f;
        const float sc2 = (dstep * dstep + lstep * lstep) / 12.0f;
        rho = sc2 / (sc2 + sigma * sigma);
    }
    out.rho  = rho;
    out.beta = 1.0f / (1.0f + std::max(0.0f, out.n_inf - 1.0f) * rho);
    const double beta = out.beta;

    // ── power likelihood, then marginalise the nuisances (log-mean-exp, uniform prior over the grid) ──
    std::vector<double> m(static_cast<std::size_t>(KK), -std::numeric_limits<double>::infinity());
    double mmax = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < KK; ++k)
    {
        const double* v = &ll[static_cast<std::size_t>(k) * NN];
        double b = -std::numeric_limits<double>::infinity();
        // The slide prior is NOT tempered: beta discounts correlated EVIDENCE, the prior is not evidence.
        const auto lp = [&](int j) { return lprior[static_cast<std::size_t>(j % ln)]; };
        for (int j = 0; j < NN; ++j) b = std::max(b, beta * v[j] + lp(j));
        double acc = 0.0;
        for (int j = 0; j < NN; ++j) acc += std::exp(beta * v[j] + lp(j) - b);
        m[static_cast<std::size_t>(k)] = b + std::log(std::max(1e-300, acc / static_cast<double>(NN)));
        mmax = std::max(mmax, m[static_cast<std::size_t>(k)]);
    }
    // Task 15: the slide's marginal (over the leaf — every angle and hinge scored — and the depth), prior excluded.
    {
        out.lat_loglik.reserve(static_cast<std::size_t>(ln));
        double lmax = -std::numeric_limits<double>::infinity();
        std::vector<double> lv(static_cast<std::size_t>(ln));
        for (int li = 0; li < ln; ++li)
        {
            double b = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < KK; ++k)
                for (int di = 0; di < dn; ++di)
                    b = std::max(b, beta * ll[static_cast<std::size_t>(k) * NN + di * ln + li]);
            double acc = 0.0;
            for (int k = 0; k < KK; ++k)
                for (int di = 0; di < dn; ++di)
                    acc += std::exp(beta * ll[static_cast<std::size_t>(k) * NN + di * ln + li] - b);
            lv[static_cast<std::size_t>(li)] = b + std::log(std::max(1e-300, acc / static_cast<double>(KK * dn)));
            lmax = std::max(lmax, lv[static_cast<std::size_t>(li)]);
        }
        for (int li = 0; li < ln; ++li)
            out.lat_loglik.emplace_back(grid(nz.lat_lo, nz.lat_hi, ln, li), lv[static_cast<std::size_t>(li)] - lmax);
    }
    out.loglik.reserve(static_cast<std::size_t>(nstep));
    for (int kk = 0; kk < KK; ++kk)
    {
        const int k = kk % nstep;
        const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(k) / (nstep - 1);
        const Hyp& hy = hyps[static_cast<std::size_t>(kk / nstep)];
        auto& dst = hy.flip_swing ? (hy.flip_hinge ? out.loglik_both : out.loglik_swing)
                                  : (hy.flip_hinge ? out.loglik_other : out.loglik);
        dst.emplace_back(phi, static_cast<float>(m[static_cast<std::size_t>(kk)] - mmax));
    }
    return out;
}


DoorBelief::RayExplain DoorBelief::phi_ray_explain(const rc::ai::LidarRays& rays, float phi, float tol_m) const
{
    RayExplain out;
    if (rays.endpoints.empty())
        return out;
    DoorBelief probe(state_, params_);
    probe.set_leaf_phi(phi);
    const float tcap = rays.max_range_m;
    double acc = 0.0; int n = 0;
    for (const auto& ep : rays.endpoints)
    {
        const Eigen::Vector3f dir = ep - rays.origin;
        const float rho = dir.norm();
        if (rho < 1e-3f) continue;
        const Eigen::Vector3f u = dir / rho;
        float t = 0.0f; bool hit = false;
        for (int it = 0; it < rays.max_steps and t < tcap; ++it)
        {
            const float d = probe.sdf_prim(rays.origin + t * u, probe.state_, 0);
            if (d < rays.surf_eps_m) { hit = true; break; }
            t += std::max(d, rays.surf_eps_m);
        }
        if (not hit) continue;              // the model predicts nothing here: not a hit to explain
        ++out.n_hit;
        const float e = t - rho;            // + ⇒ the ray flew PAST; - ⇒ it stopped SHORT of the model
        acc += e; ++n;
        if (std::abs(e) < tol_m) ++out.n_explained;
    }
    out.mean_signed_m = n > 0 ? static_cast<float>(acc / n) : 0.0f;
    return out;
}

}  // namespace rc
