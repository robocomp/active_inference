// test_door_leaf_recess — Task 5b (D1 + D2) and its fix round 1: the leaf plane's DEPTH d off the wall line is a state of
// the door, estimated with [s, w, h], and the fit is unbiased whatever that depth is.
//
// ★WHAT THE TAPES SHOWED. On both recorded tours the shut DOOR_0 leaf's mask points sit 0.10-0.15 m off the room polygon's
// wall line, on the ROOM side of it (associate_wall's normal points into the room; the camera sat 2.7 m along it). The fit
// drew the leaf ON the line, so every point carried a shared ~0.12 m residual no DOF of [s, w, h] could explain, and
// Gauss-Newton converted it into motion along the faintest lever:
//   · D1 (tourR2 door_2, 7th update): phi 0.029 rad, the far-jamb hinge ⇒ dd/dw = −sin phi = −0.028 per point; 28937
//     points × 0.116 m ⇒ w +1.45 m in one frame (36 σ); the silhouette then predicted a 2.5 m door ⇒ REMOVE at 3629.
//   · D2 (tourR1 door_4, first update): every point inside the panel ⇒ dd/ds is 0, but the finite-difference Jacobian
//     reads ±6.7e-5 (float rounding, coherent over 22k points) × 0.143 m ⇒ a 3.6 m step against the prior alone.
// ★THE MODEL. d ~ N(0, T²/12 + σ_b²) beyond the line (DoorBeliefParams::recess_prior_m) is the fourth DOF; every
// consumer renders the leaf on its plane (door::Aperture::depth).
//   (a) shut leaf, d = ∓0.12 m (room side, as on the tapes, and beyond the line), phi 0 and 0.029: after a run of frames
//       w, s and d within 2 σ of the truth (why not 1 σ on w: see the section); and the two single-frame failures stay inside their prior (D1: z ≤ 3; D2: a
//       sweep of 50 seeds, none leaving 3 σ or walking off its own mask).
//   (b) OPEN leaf, phi 0.6 and 1.2, d = 0 and 0.12: s and w within 2 σ, and the leaf angle the frame's free energy
//       prefers within one grid step of the truth — with the recess on as with it off.
//   (c) the along-wall extent still localises s (the model term removes nothing it should keep).
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
constexpr float kT = 0.20f, kSigmaB = 0.10f;   // Existence.WallThicknessM / BoundarySigmaM (the live config)

// The tapes' wall: room_1's y = 2.538 face, run along −x; across_u = (−u.y, u.x) points INTO the room.
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
    p.recess_prior_m = 0.0f;        // DoorFit::make_belief_params
    p.recess_prior_std = std::sqrt(kT * kT / 12.0f + kSigmaB * kSigmaB);
    return p;
}

// Points on BOTH faces of the true leaf (aperture [s, s+w] at depth d beyond the line, angle phi), 0..h tall.
// mid_plane: every point on the leaf's mid-plane (what a mask's depth gives at a few mm noise — the tapes' D1/D2 frames).
std::vector<Eigen::Vector3f> leaf_points(const rc::DoorBeliefParams& p, float s, float w, float h, float d, float phi,
                                         int n, std::mt19937& rng, bool mid_plane = false)
{
    rc::door::Aperture a;
    a.wall_O = p.wall_O; a.wall_u = p.wall_u; a.s = s; a.w = w; a.h = h; a.thickness = p.thickness; a.depth = -d;
    rc::door::LeafState ls = p.leaf;
    ls.phi = phi;
    const rc::door::LeafPose L = rc::door::leaf_pose(a, ls);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> N(0.f, 0.006f);
    std::vector<Eigen::Vector3f> pts;
    pts.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const float lx = (U(rng) - 0.5f) * w;
        const float ly = (mid_plane ? 0.f : (i % 2 == 0 ? 1.f : -1.f) * L.half_t) + N(rng);
        pts.push_back(rc::door::leaf_point(L, lx, ly, U(rng) * h));
    }
    return pts;
}

rc::DoorFrame frame_of(const std::vector<Eigen::Vector3f>& pts, float R, float chain)
{
    rc::DoorFrame f;
    f.points = pts;
    f.R.assign(pts.size(), R);
    f.chain_cov_s = chain;
    return f;
}

float sd(const rc::DoorBelief& b, int i) { return std::sqrt(std::max(0.f, b.covariance_full()(i, i))); }
}  // namespace

int main()
{
    std::mt19937 rng(5);

    // ── (a) shut leaf off the line: unbiased after a run of frames; single frames inside their prior ────────────────────
    // ★TWO σ, NOT ONE, ON w — and why. Under the live size common mode (AI2CommonModeSizeStd 0.02, against the code's
    // documented 0.35) twelve frames leave σ_w ≈ 9 mm, and the slab SDF fed surface points reads w ~1 cm short even with
    // the leaf exactly ON the line (printed: "on the line" 0.992), a pre-5b property of the edge model, not of the plane.
    // With the plane a state the off-line fits read 0.983-0.985 (z ≤ 1.9); with the leaf pinned on the line (the pre-5b
    // geometry) they read 7.8 σ off. One σ would test the size common mode's over-confidence, which this task does not
    // change (report, concerns).
    std::println("(a) shut leaf at d = -0.12 (room side, the tapes) and +0.12: w, s and d within 2 σ of the truth");
    {
        const auto fit = [](rc::door::HingeSide hinge, float swing, float d_true, float phi, int& seed, float* jump)
        {
            auto p = tape_params();
            p.leaf.hinge = hinge; p.leaf.swing = swing; p.leaf.phi = phi;
            rc::DoorBelief b({3.10f, 1.0f, 2.1f, p.recess_prior_m}, p);
            b.set_hinge_prob(1.0f);
            b.set_leaf_phi_sigma(0.146f);
            std::mt19937 r(static_cast<unsigned>(seed));
            for (int k = 0; k < 12; ++k)
            {
                const auto pts = leaf_points(p, 3.06f, 1.0f, 2.07f, d_true, phi, 3000, r);
                const float w0 = b.state().w, sw0 = sd(b, 1);
                b.update(frame_of(pts, 0.00235f, 0.0122f));
                if (jump) *jump = std::max(*jump, std::abs(b.state().w - w0) / sw0);
            }
            return b;
        };
        float worst_zw = 0.f, worst_zs = 0.f, worst_zd = 0.f, worst_jump = 0.f, w_line = 0.f;
        int seed = 11;
        for (const auto hinge : {rc::door::HingeSide::Near, rc::door::HingeSide::Far})
            for (const float swing : {+1.f, -1.f})
                for (const float phi : {0.0f, 0.029f})
                {
                    ++seed;
                    const auto b0 = fit(hinge, swing, 0.0f, phi, seed, nullptr);
                    w_line = b0.state().w;
                    for (const float d_true : {-0.12f, +0.12f})
                    {
                        const auto b = fit(hinge, swing, d_true, phi, seed, &worst_jump);
                        const auto& st = b.state();
                        const float zw = std::abs(st.w - 1.0f) / sd(b, 1), zs = std::abs(st.s - 3.06f) / sd(b, 0);
                        const float zd = std::abs(st.d - d_true) / sd(b, 3);
                        worst_zw = std::max(worst_zw, zw); worst_zs = std::max(worst_zs, zs); worst_zd = std::max(worst_zd, zd);
                        if (kTrace)
                            std::println("    hinge {} swing {:+.0f} d {:+.2f} phi {:.3f}: s {:.3f} (z {:.2f}) w {:.3f}±{:.3f} vs {:.3f} on the line (z {:.2f}) d {:+.3f}±{:.3f} (z {:.2f})",
                                         hinge == rc::door::HingeSide::Near ? "near" : "far", swing, d_true, phi,
                                         st.s, zs, st.w, sd(b, 1), b0.state().w, zw, st.d, sd(b, 3), zd);
                    }
                }
        std::println("    worst z: w {:.2f} (the same fit on the line reads {:.3f}), s {:.2f}, d {:.2f}; worst one-frame w step {:.2f} σ",
                     worst_zw, w_line, worst_zs, worst_zd, worst_jump);
        check(worst_zw <= 2.f, "(a) w biased by more than 2 sigma");
        check(worst_zs <= 2.f, "(a) s biased by more than 2 sigma");
        check(worst_zd <= 2.f, "(a) the leaf plane's depth is not recovered (2 sigma)");
        check(worst_jump <= 3.f, "(a) one frame moved w by more than 3 prior sigma");
    }
    // D1, the tape's exact frame: phi 0.029, far hinge, the posterior it had after six frames, one 6000-point frame.
    {
        float worst = 0.f;
        for (const float swing : {+1.f, -1.f})
            for (const float d_true : {-0.115f, +0.115f})
            {
                auto p = tape_params();
                p.leaf.hinge = rc::door::HingeSide::Far; p.leaf.swing = swing; p.leaf.phi = 0.029f;
                rc::DoorBelief b({3.063f, 1.014f, 2.074f, p.recess_prior_m}, p);
                b.set_hinge_prob(0.8f);
                b.set_leaf_phi_sigma(0.146f);
                const float w0 = b.state().w, sw0 = sd(b, 1);
                b.update(frame_of(leaf_points(p, 3.06f, 1.0f, 2.07f, d_true, 0.029f, 6000, rng, true), 0.00235f, 0.0122f));
                worst = std::max(worst, std::abs(b.state().w - w0) / sw0);
            }
        std::println("    D1 frame: worst w step {:.2f} prior σ", worst);
        check(worst <= 3.f, "(a) D1: the tape's frame moved w by more than 3 prior sigma");
    }

    // ── D2: the FIRST update, every point inside the panel, a sweep of seeds ─────────────────────────────────────────────
    std::println("(a) D2 — first update at phi = 0 (σφ 2.09): |Δs| within 3 prior σ and the aperture stays on its own mask");
    {
        float worst = 0.f;
        int n_cases = 0, n_off = 0;
        for (const float d_true : {-0.143f, +0.143f})
        {
            auto p = tape_params();
            const auto pts = leaf_points(p, 2.92f, 0.98f, 2.1f, d_true, 0.0f, 6000, rng, true);
            const float centroid = 2.92f + 0.5f * 0.98f;
            for (int k = 0; k <= 24; ++k)
            {
                const float s0 = 2.80f + 0.005f * static_cast<float>(k);
                rc::DoorBelief b({s0, 1.0f, 2.1f, p.recess_prior_m}, p);
                b.set_hinge_prob(0.5f);
                b.set_leaf_phi_sigma(2.094f);
                const float sd0 = std::sqrt(sd(b, 0) * sd(b, 0) + p.process_std_s * p.process_std_s);
                b.update(frame_of(pts, 0.0045f, 0.0153f));
                const float z = std::abs(b.state().s - s0) / sd0;
                const bool off = not (b.state().s <= centroid and centroid <= b.state().s + b.state().w);
                worst = std::max(worst, z);
                n_off += off ? 1 : 0;
                ++n_cases;
                if (kTrace and (z > 1.f or off))
                    std::println("    d {:+.3f} seed {:.3f}: s -> {:.3f} (z {:.2f}){}", d_true, s0, b.state().s, z, off ? " OFF THE MASK" : "");
            }
        }
        std::println("    {} seeds, worst z_s {:.2f}, {} off the mask", n_cases, worst, n_off);
        check(worst <= 3.f, "D2: a first update moved s by more than 3 prior sigma");
        check(n_off == 0, "D2: a first update walked the aperture off the mask it was fitting");
    }

    // ── (b) OPEN leaf: s, w and the preferred angle unbiased, recess on and off ──────────────────────────────────────────
    std::println("(b) open leaf, phi 0.6 / 1.2, d 0 / 0.12: s, w within 2 σ; the frame's preferred phi within 0.02 rad");
    {
        float worst_s = 0.f, worst_w = 0.f, worst_phi = 0.f;
        for (const float phi : {0.6f, 1.2f})
            for (const float d_true : {0.0f, 0.12f})
                for (const float swing : {+1.f, -1.f})
                {
                    auto p = tape_params();
                    p.leaf.hinge = rc::door::HingeSide::Near; p.leaf.swing = swing; p.leaf.phi = phi;
                    p.common_mode_wh_std = 0.35f;
                    rc::DoorBelief b({3.10f, 0.97f, 2.1f, p.recess_prior_m}, p);
                    b.set_hinge_prob(1.f);
                    b.set_leaf_phi_sigma(0.05f);
                    std::vector<Eigen::Vector3f> last;
                    for (int k = 0; k < 12; ++k)
                    {
                        last = leaf_points(p, 3.06f, 1.0f, 2.07f, d_true, phi, 3000, rng);
                        b.update(frame_of(last, 0.0025f, 0.002f));
                    }
                    const auto f = frame_of(last, 0.0025f, 0.002f);
                    float best = 1e30f, best_phi = 0.f;
                    for (int k = 0; k <= 160; ++k)
                    {
                        const float ph = 0.01f * static_cast<float>(k);
                        const float fe = b.phi_free_energy(f, ph);
                        if (fe < best) { best = fe; best_phi = ph; }
                    }
                    const float zs = std::abs(b.state().s - 3.06f) / sd(b, 0), zw = std::abs(b.state().w - 1.0f) / sd(b, 1);
                    worst_s = std::max(worst_s, zs); worst_w = std::max(worst_w, zw);
                    worst_phi = std::max(worst_phi, std::abs(best_phi - phi));
                    if (kTrace)
                        std::println("    phi {:.1f} d {:.2f} swing {:+.0f}: s {:.3f} (z {:.2f}) w {:.3f} (z {:.2f}) d {:+.3f}±{:.3f} phi* {:.2f}",
                                     phi, d_true, swing, b.state().s, zs, b.state().w, zw, b.state().d, sd(b, 3), best_phi);
                }
        std::println("    worst z: s {:.2f}, w {:.2f}; worst |phi* - phi| {:.3f}", worst_s, worst_w, worst_phi);
        check(worst_s <= 2.f, "(b) open leaf: s biased by more than 2 sigma");
        check(worst_w <= 2.f, "(b) open leaf: w biased by more than 2 sigma");
        check(worst_phi <= 0.02f + 1e-4f, "(b) open leaf: the frame prefers another leaf angle");
    }

    // ── (c) the along-wall extent still localises s ─────────────────────────────────────────────────────────────────────
    std::println("(c) the along-wall extent still localises s");
    {
        auto p = tape_params();
        p.common_mode_wh_std = 0.35f;
        rc::DoorBelief b({3.25f, 0.92f, 2.1f, p.recess_prior_m}, p);
        b.set_hinge_prob(1.f);
        const auto f = frame_of(leaf_points(p, 3.00f, 1.0f, 2.1f, -0.12f, 0.0f, 6000, rng), 0.0025f, 0.0005f);
        for (int k = 0; k < 5; ++k) b.update(f);
        std::println("    s 3.250 -> {:.3f} (truth 3.000), w 0.920 -> {:.3f}, d -> {:+.3f} (truth -0.12)", b.state().s, b.state().w, b.state().d);
        check(std::abs(b.state().s - 3.0f) < 0.05f, "the along-wall extent no longer localises s");
    }

    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
