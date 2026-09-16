/*
 * door_belief.cpp  —  AI2 door belief (single wall-anchored thin panel; state θ=[s,w,h] in the wall frame).
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
                const Eigen::Vector2f wallO = ap.wall_O + depth * n2;
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
