/*
 * door_belief.cpp  —  AI2 door belief (single wall-anchored thin panel; state θ=[s,w,h] in the wall frame).
 */

#include "door_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

float DoorBelief::sdf_prim(const Eigen::Vector3f& p, const DoorBeliefState& s, int /*prim*/) const
{
    return sdf_panel(p, s);   // one primitive
}

// ─── Mixture responsibilities [panel, clutter] ──────────────────────────────────
std::array<float, 2> DoorBelief::mixture_unnorm(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const
{
    const float eps     = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pi_surf = 1.0f - eps;
    const float inv2R   = 0.5f / std::max(1e-9f, R);

    const float d = sdf_panel(p, s);
    std::array<float, 2> u{};
    u[0] = pi_surf * std::exp(-d * d * inv2R);                                   // panel
    const float cs = params_.clutter_scale_m;
    u[1] = eps * std::exp(-cs * cs * inv2R);                                     // clutter
    return u;   // UNNORMALIZED
}

std::array<float, 2> DoorBelief::responsibilities(const Eigen::Vector3f& p, const DoorBeliefState& s, float R) const
{
    std::array<float, 2> u = mixture_unnorm(p, s, R);
    const float sum = u[0] + u[1];
    if (sum <= 0.0f) return {0.0f, 1.0f};
    u[0] /= sum; u[1] /= sum;
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
        nll += -std::log(std::max(1e-30, static_cast<double>(u[0] + u[1])));
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
    for (const auto& p : pts) c += responsibilities(p, state_, R)[1];   // index 1 = clutter
    return static_cast<float>(c / static_cast<double>(pts.size()));
}

// ─── Jacobian (central finite difference over the 3 DOFs [s,w,h]) ────────────────
Eigen::Matrix<float, 3, 1> DoorBelief::sdf_jacobian(const Eigen::Vector3f& p, const DoorBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 3, 1> J;
    const Eigen::Matrix<float, 3, 1> base = s.vec();
    const float e = params_.fd_eps;
    for (int j = 0; j < 3; ++j)
    {
        Eigen::Matrix<float, 3, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        J(j) = (sdf_prim(p, DoorBeliefState::from_vec(vp), prim) -
                sdf_prim(p, DoorBeliefState::from_vec(vm), prim)) / (2.0f * e);
    }
    return J;
}

// ─── Fixed template prior on w,h (non-drifting; folded into the GN normal equations each frame) ──
void DoorBelief::accumulate_extra(const DoorBeliefState& s, const DoorFrame& /*frame*/,
                                  Eigen::Matrix<float, 3, 3>& Id, Eigen::Matrix<float, 3, 1>& bd) const
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
    // A door lies WITHIN its wall: its span [s, s+w] must fit the segment [0, wall_len]. Clamp the near
    // edge so it can't run past either corner (physical, from the wall geometry — not a tuning threshold).
    // This constrains the APERTURE, which is rigid in the wall. A swung LEAF legitimately protrudes past
    // the wall plane — never clamp the leaf footprint here.
    if (params_.wall_len > 0.0f)
        s.s = std::clamp(s.s, 0.0f, std::max(0.0f, params_.wall_len - s.w));
}

// ─── Engine hooks: prior cov, process noise, common-mode ─────────────────────────
Eigen::Matrix<float, 3, 1> DoorBelief::prior_cov_diag() const
{
    return (Eigen::Matrix<float, 3, 1>() <<
            params_.prior_s_std * params_.prior_s_std,
            params_.prior_w_std * params_.prior_w_std,
            params_.prior_h_std * params_.prior_h_std).finished();
}

Eigen::Matrix<float, 3, 1> DoorBelief::process_noise_diag() const
{
    return (Eigen::Matrix<float, 3, 1>() <<
            params_.process_std_s * params_.process_std_s,
            params_.process_std_w * params_.process_std_w,
            params_.process_std_h * params_.process_std_h).finished();
}

Eigen::Matrix<float, 3, 1> DoorBelief::common_mode_inv_diag(const DoorFrame& frame) const
{
    const float s2  = params_.common_mode_s_std  * params_.common_mode_s_std;
    const float wh2 = params_.common_mode_wh_std * params_.common_mode_wh_std;
    return (Eigen::Matrix<float, 3, 1>() <<
            1.0f / std::max(1e-9f, s2 + frame.chain_cov_s),
            1.0f / std::max(1e-9f, wh2),
            1.0f / std::max(1e-9f, wh2)).finished();
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

    // (f) segment clamp: with a short wall the panel must not run past a corner — s clamped to [0, len−w].
    {
        DoorBeliefParams Pc = P; Pc.wall_len = 1.0f;
        DoorBelief b(DoorBeliefState{0.90f, 0.70f, 2.00f}, Pc);   // seed s so s+w=1.6 > len=1.0 (pokes out)
        DoorBeliefState s = b.state(); b.apply_constraints(s); b.set_state(s);
        std::printf("  clamp: s=%.3f (wall_len=1.0, w=0.70 → s must be ≤ 0.30)\n", b.state().s);
        check(b.state().s <= 0.30f + 1e-4f and b.state().s >= 0.0f, "along-wall clamp did not keep the panel within its wall");
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

}  // namespace rc
