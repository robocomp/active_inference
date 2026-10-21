// test_door_leaf_recess — Task 5b, defects D1 + D2 (tapes tourR2 / tourR1): ONE mask update must not move the aperture
// by more than its own prior predicts when the shut leaf it sees is RECESSED into the wall (the across-wall offset the
// model's leaf, drawn on the room-side polygon face, cannot explain).
//
// ★WHAT THE TAPES SHOWED. On both tapes the shut DOOR_0 leaf's mask points sit 0.10-0.15 m beyond the room-side wall
// line (a leaf hangs inside the wall's thickness: Existence.WallThicknessM, door_room_band.h). The mask update holds the
// leaf angle phi fixed and draws the leaf ON the face line, so every point carries a shared ~0.12 m residual that no
// DOF of [s, w, h] can explain — and Gauss-Newton converts it into motion along whatever DOF has the faintest lever:
//   · D1 (tourR2 door_2, 7th update, cycle ~3603): phi = 0.029 rad, far-jamb hinge ⇒ dd/dw = −sin phi = −0.028 per
//     point; 28937 points × d 0.116 m ⇒ a Newton step of +1.45 m on w against a prior σ_w of 0.041 m (36 σ). The w
//     common mode w·|sin phi|·σφ vanishes exactly there (phi → 0), so the frame had full authority. w 1.01 → 2.51; the
//     silhouette then predicted a 2.5 m door over a 1 m doorway ⇒ free ≫ occ ⇒ REMOVE at 3629.
//   · D2 (tourR1 door_4, first update, cycle 3417): phi = 0, every point INSIDE the panel ⇒ the true dd/ds is 0, but the
//     finite-difference Jacobian reads ±6.7e-5 (float rounding of the leaf centre, coherent over 22k points) × d 0.143 m
//     ⇒ a gradient of 10 nats/m against only the prior precision 1/0.6² ⇒ a 3.6 m step at the LAST Gauss-Newton
//     iteration, unverified (clutter 100 % after it). Which way it lands flips with ±2 % of the ego variance (task-4b).
// ★THE MODEL TERM. The leaf plane's across-wall position within the wall is a nuisance n ~ N(0, T²) shared by every point
// of the frame, marginalised exactly (Schur complement of the [θ, n] normal equations, DoorBelief::accumulate_extra):
// an across residual is then explained by n, and only what n cannot explain — the jambs, the lintel — informs [s, w, h].
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_leaf_recess.cpp src/door_belief.cpp -o /tmp/test_door_leaf_recess && /tmp/test_door_leaf_recess
// LEAF_RECESS_TRACE=1 prints every case.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <random>
#include <vector>

#include "../src/door_belief.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what)
{
    ++g_checks;
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}
const bool kTrace = std::getenv("LEAF_RECESS_TRACE") != nullptr;

// The tapes' wall: room_1's y = 2.538 face, run along −x (O and u as associate_wall authored them on tourR1/R2).
rc::DoorBeliefParams tape_params()
{
    rc::DoorBeliefParams p;
    p.wall_O = {4.031f, 2.538f};
    p.wall_u = Eigen::Vector2f(-1.0f, -0.0005f).normalized();
    p.sigma_base_m = 0.03f; p.clutter_frac = 0.10f; p.clutter_scale_m = 0.12f;
    p.thickness = 0.05f;
    p.prior_s_std = 0.60f; p.prior_w_std = 0.06f; p.prior_h_std = 0.08f;
    p.tpl_w = 1.0f; p.tpl_h = 2.1f;
    p.process_std_s = 0.005f; p.process_std_w = 0.0005f; p.process_std_h = 0.0005f;
    p.common_mode_s_std = 0.03f;
    p.common_mode_wh_std = 0.02f;   // the live config (AI2CommonModeSizeStd)
    p.gn_iters = 4;
    return p;
}

// A shut leaf spanning [a0, a0 + w] along the wall, `recess` m along the wall normal n = (−u.y, u.x), 0..h tall.
std::vector<Eigen::Vector3f> leaf_points(const rc::DoorBeliefParams& p, float a0, float w, float h, float recess,
                                         int n, std::mt19937& rng)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> N(0.f, 0.008f);
    const Eigen::Vector2f u = p.wall_u, nn(-u.y(), u.x());
    std::vector<Eigen::Vector3f> pts;
    pts.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f xy = p.wall_O + (a0 + U(rng) * w) * u + (recess + N(rng)) * nn;
        pts.emplace_back(xy.x(), xy.y(), U(rng) * h);
    }
    return pts;
}

struct Outcome { rc::DoorBeliefState pre, post; float sd_s0 = 0, sd_w0 = 0; };

Outcome one_update(rc::DoorBeliefParams p, const rc::DoorBeliefState& s0, float phi, float phi_sigma, float p_hinge,
                   const std::vector<Eigen::Vector3f>& pts, float R, float chain_cov_s)
{
    p.leaf.phi = phi;
    rc::DoorBelief b(s0, p);
    b.set_hinge_prob(p_hinge);
    b.set_leaf_phi_sigma(phi_sigma);
    Outcome o;
    o.pre = b.state();
    // the prior the update predicts from: the seed Σ plus one step of process noise (recursive_laplace predict)
    o.sd_s0 = std::sqrt(b.covariance()(0, 0) + p.process_std_s * p.process_std_s);
    o.sd_w0 = std::sqrt(b.covariance()(1, 1) + p.process_std_w * p.process_std_w);
    rc::DoorFrame f;
    f.points = pts;
    f.R.assign(pts.size(), R);
    f.chain_cov_s = chain_cov_s;
    b.update(f);
    o.post = b.state();
    return o;
}
}  // namespace

int main()
{
    std::mt19937 rng(5);

    // ── (a) D1: a recessed shut leaf at a small phi must not stretch the width ────────────────────────────────────
    // tourR2 door_2's 7th update: s 3.063, w 1.014, phi 0.029 rad (σ 0.146), P(hinge) σ(1.41) = 0.80, R 0.00235,
    // chain 0.0122. Every hinge / swing / recess side is tried: the lever is on whichever side brings the swung leaf
    // toward the points, and a fix must hold on all of them.
    std::println("(a) D1 — recessed shut leaf, phi = 0.029: |Δw| and |Δs| within 3 prior σ");
    {
        float worst_w = 0.f, worst_s = 0.f;
        for (const auto hinge : {rc::door::HingeSide::Near, rc::door::HingeSide::Far})
            for (const float swing : {+1.f, -1.f})
                for (const float recess : {+0.115f, -0.115f})
                {
                    auto p = tape_params();
                    p.leaf.hinge = hinge; p.leaf.swing = swing;
                    const auto pts = leaf_points(p, 3.06f, 1.0f, 2.07f, recess, 6000, rng);
                    const auto o = one_update(p, {3.063f, 1.014f, 2.074f}, 0.029f, 0.146f, 0.80f, pts, 0.00235f, 0.0122f);
                    const float zw = std::abs(o.post.w - o.pre.w) / o.sd_w0, zs = std::abs(o.post.s - o.pre.s) / o.sd_s0;
                    worst_w = std::max(worst_w, zw); worst_s = std::max(worst_s, zs);
                    if (kTrace)
                        std::println("    hinge {} swing {:+.0f} recess {:+.3f}: s {:.3f}->{:.3f} w {:.3f}->{:.3f} (z_s {:.2f} z_w {:.2f})",
                                     hinge == rc::door::HingeSide::Near ? "near" : "far", swing, recess,
                                     o.pre.s, o.post.s, o.pre.w, o.post.w, zs, zw);
                }
        std::println("    worst z_w {:.2f}, worst z_s {:.2f}", worst_w, worst_s);
        check(worst_w <= 3.f, "D1: one recessed-leaf frame moved w by more than 3 prior sigma");
        check(worst_s <= 3.f, "D1: one recessed-leaf frame moved s by more than 3 prior sigma");
    }

    // ── (b) D2: the FIRST update of a recessed shut leaf, every point inside the panel ─────────────────────────────
    // tourR1 door_4 at birth: phi = 0 (σ 2.09), P(hinge) 0.5, R 0.0045, chain 0.0153, the leaf spanning [2.92, 3.90]
    // recessed 0.143 m, the seed at the mask centroid. The jump is chaotic in the seed (float rounding decides its
    // sign), so a sweep of seeds is the test: none may leave the prior's 3 σ.
    std::println("(b) D2 — first update, recessed shut leaf at phi = 0: |Δs| within 3 prior σ over a sweep of seeds");
    {
        // ★And the update must not walk OFF its own evidence: at every seed the panel contains the mask (centroid
        // 3.41 m along the wall); an aperture that no longer contains it explains none of the frame (clutter 100 %,
        // as the tape's 6.48 did), so it cannot be the posterior mode of that frame.
        float worst = 0.f;
        int n_cases = 0, n_off = 0;
        for (const float recess : {+0.143f, -0.143f})
        {
            auto p = tape_params();
            const auto pts = leaf_points(p, 2.92f, 0.98f, 2.1f, recess, 6000, rng);
            const float centroid = 2.92f + 0.5f * 0.98f;
            for (int k = 0; k <= 24; ++k)
            {
                const float s0 = 2.80f + 0.005f * static_cast<float>(k);
                const auto o = one_update(p, {s0, 1.0f, 2.1f}, 0.0f, 2.094f, 0.5f, pts, 0.0045f, 0.0153f);
                const float z = std::abs(o.post.s - o.pre.s) / o.sd_s0;
                const bool off = not (o.post.s <= centroid and centroid <= o.post.s + o.post.w);
                worst = std::max(worst, z);
                n_off += off ? 1 : 0;
                ++n_cases;
                if (kTrace and (z > 1.f or off))
                    std::println("    recess {:+.3f} seed {:.3f}: s -> {:.3f} (z {:.2f}), w {:.3f}{}", recess, s0, o.post.s, z,
                                 o.post.w, off ? "  OFF THE MASK" : "");
            }
        }
        std::println("    {} seeds, worst z_s {:.2f}, {} landed off the mask", n_cases, worst, n_off);
        check(worst <= 3.f, "D2: a first update moved s by more than 3 prior sigma");
        check(n_off == 0, "D2: a first update walked the aperture off the mask it was fitting");
    }

    // ── (c) the fix must not blind the fit: the jambs still localise the aperture ──────────────────────────────────
    // The same recessed leaf, seeded 0.25 m off along the wall (and 0.08 m too narrow), with the robot still: the
    // update must still pull s most of the way to the truth. (Marginalising the across offset removes only what it
    // can explain; the leaf's along-wall extent is untouched.)
    std::println("(c) the along-wall extent still localises s");
    {
        auto p = tape_params();
        p.common_mode_wh_std = 0.35f;
        const auto pts = leaf_points(p, 3.00f, 1.0f, 2.1f, 0.12f, 6000, rng);
        rc::DoorBelief b({3.25f, 0.92f, 2.1f}, p);
        b.set_hinge_prob(1.f);
        rc::DoorFrame f;
        f.points = pts;
        f.R.assign(pts.size(), 0.0025f);
        f.chain_cov_s = 0.0005f;
        for (int k = 0; k < 5; ++k) b.update(f);
        std::println("    s 3.250 -> {:.3f} (truth 3.000), w 0.920 -> {:.3f} (truth 1.000)", b.state().s, b.state().w);
        check(std::abs(b.state().s - 3.0f) < 0.08f, "the along-wall extent no longer localises s");
    }

    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
