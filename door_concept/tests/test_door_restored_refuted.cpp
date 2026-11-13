// test_door_restored_refuted.cpp — chat_e2e13 (2026-11-12): the real DOOR_0 restored as door_22 was removed by ONE mask frame,
// and a phantom door_24 was born in room_1's partition on blank wall. Two model defects, each reproduced here on the run's own
// numbers, with the agent's own code.
//
// (B) THE SUPPORT CHANNEL SCORED THE LEAF'S POINT ESTIMATE, NOT THE PLACE. door_22 carried phi 81° σ 25° (LiDAR only, no mask
//     for 25 min). Its first mask lay on the CLOSED leaf plane: 255 points, the fit's clutter share x = 1.00 at the 81° point
//     estimate ⇒ `sup dl −1.95` ⇒ REMOVE. x is the existence channel's question "does a door AT THIS PLACE explain its support",
//     and the leaf angle is a nuisance of it: DoorBelief::clutter_fraction_leaf_marginal marginalises it over the φ posterior.
//       (B1) bimodal posterior (most mass open, some near shut), mask on the shut plane: the marginal x is far below the point x,
//            and the support log-likelihood ratio is no longer a refutation of the place.
//       (B2) a SHARP posterior at the true angle reproduces clutter_fraction (no change where the leaf belief is right).
//       (B3) a mask that lies nowhere a leaf of this door can be (1.5 m along the wall, on plain wall): x stays ≈ 1 under ANY
//            leaf posterior — the channel still refutes a fit that landed on the wrong place.
// (C) THE WALL A DETECTION IS FRAMED ON. door_1's mask, its depth on an occluder 2.2 m out (centroid (−0.63, 1.84), z 0.92-1.67),
//     voted for its NEAREST wall — room_1's partition, 0.6 m to the side — because no wall lay between camera and point. The line
//     of sight through that point meets the TOP wall beside door_1 (door_standpoint.h associate_wall, seen_at).
//       (C1) occluder points: framed on the top wall (y ≈ 4.6), seen_at within 0.3 m of door_1 along it — not the partition.
//       (C2) a closed leaf ON its wall and (C3) a doorway seen THROUGH (points in the next room): the same wall as before, and
//            seen_at on the opening (the seen-through case: the perpendicular foot of the centroid is 0.5 m off, the ray is not).
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_restored_refuted.cpp src/door_belief.cpp -o /tmp/test_door_restored_refuted && /tmp/test_door_restored_refuted
#include <clocale>
#include <cmath>
#include <format>
#include <numbers>
#include <print>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_standpoint.h"
#include "../src/core/detector_llr.h"

using V2 = Eigen::Vector2f;
using V3 = Eigen::Vector3f;
static int g_fail = 0;
static void check(bool ok, const std::string& what) { std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what); if (not ok) ++g_fail; }

// room_1 as door_concept loaded it in chat_e2e13 (residual_concept/etc/residual_room_poly.csv, 32 vertices).
static const std::vector<V2> kRoom1 = {
    {1.0175f,-4.55f},{1.0215f,-3.836f},{0.0189f,-3.8088f},{0.0569f,-2.1228f},{-0.0629f,-2.1302f},{-0.0649f,-3.8132f},
    {-0.9535f,-3.806f},{-0.9305f,-4.56f},{-4.0285f,-4.449f},{-4.2535f,-4.167f},{-4.2515f,-4.018f},{-3.8565f,-4.018f},
    {-3.8565f,-3.418f},{-4.2405f,-3.418f},{-4.1385f,2.462f},{-1.6785f,2.474f},{-1.6665f,2.535f},{-2.0825f,2.541f},
    {-2.0765f,3.398f},{-1.6305f,3.379f},{-1.6175f,4.628f},{0.0399f,4.6051f},{-0.0601f,0.3961f},{0.0591f,0.3819f},
    {0.0991f,2.5359f},{4.0315f,2.538f},{4.2535f,2.264f},{4.0755f,-3.434f},{3.7425f,-3.434f},{3.7425f,-4.034f},
    {4.0575f,-4.034f},{4.0385f,-4.628f}};

// DOOR_0's aperture as the fit held it at the refuting frame (ai2_log.csv cycle 49702): centre (0.666, 2.536), wall along −x.
static rc::DoorBelief door22(float phi)
{
    rc::DoorBeliefParams p;
    p.thickness = 0.05f; p.floor_z = 0.0f;
    p.wall_u = V2(-1.0f, 0.0f);
    p.wall_O = V2(4.03f, 2.536f);
    p.leaf = {phi, rc::door::HingeSide::Far, +1.0f};   // the log: "hinge far swing +1 (opens INTO this room)"
    const float s = (4.03f - 0.666f) - 0.5f;          // centre = O + (s + w/2)·u
    rc::DoorBelief b({s, 0.986f, 2.1f}, p);
    b.set_hinge_prob(1.0f / (1.0f + std::exp(-12.8f)));   // hinge_llr 12.8 at that frame
    return b;
}

// The mask: points on the CLOSED leaf plane (y = 2.50, 3.6 cm short of the aperture line), x in [0.55, 1.15] (the frame's
// centroid was (0.905, 2.484)), z in [0.3, 2.0], with ~2 cm depth noise.
static std::vector<V3> closed_leaf_mask(float x0, float x1, int n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(x0, x1), uz(0.3f, 2.0f);
    std::normal_distribution<float> ny(0.0f, 0.02f);
    std::vector<V3> v;
    for (int i = 0; i < n; ++i) v.emplace_back(ux(rng), 2.50f + ny(rng), uz(rng));
    return v;
}

static std::vector<std::pair<float, float>> curve(std::initializer_list<std::pair<float, float>> modes, float sd)
{   // a φ posterior on the agent's grid (0..120°, 5°): a mixture of Gaussians (mean rad, weight)
    std::vector<std::pair<float, float>> c;
    for (int k = 0; k < 25; ++k)
    {
        const float phi = 2.0944f * float(k) / 24.0f;
        float w = 0.0f;
        for (const auto& [m, a] : modes) w += a * std::exp(-0.5f * (phi - m) * (phi - m) / (sd * sd));
        c.emplace_back(phi, w);
    }
    return c;
}

int main()
{
    std::setlocale(LC_ALL, "");
    const float R = 0.00511114f;   // the frame's R (ai2_log.csv cycle 49702)
    const rc::door::core::SupportModel sm{};   // the compiled support densities (door_detectors.csv carries the same family)
    const float d2r = std::numbers::pi_v<float> / 180.0f;

    std::print("== (B1) door_22's refuting frame: leaf believed 81°, posterior bimodal, mask on the shut plane\n");
    {
        const auto b = door22(81.0f * d2r);
        const auto mask = closed_leaf_mask(0.55f, 1.15f, 255, 1);
        const float x_point = b.clutter_fraction(mask, R);
        const auto post = curve({{88.0f * d2r, 0.9f}, {8.0f * d2r, 0.1f}}, 6.0f * d2r);   // mean ≈ 80°, σ ≈ 25°
        const float x_marg = b.clutter_fraction_leaf_marginal(mask, R, post);
        const float l_point = rc::door::core::support_llr(x_point, sm), l_marg = rc::door::core::support_llr(x_marg, sm);
        std::print("    x at the point estimate {:.3f} (ℓ_sup {:+.2f});  x leaf-marginal {:.3f} (ℓ_sup {:+.2f})\n", x_point, l_point, x_marg, l_marg);
        check(x_point > 0.95f, "reproduced: the point estimate explains none of the mask (live x = 1.00)");
        check(x_marg < 0.5f, "leaf-marginal: a door at this place explains most of its support");
        check(l_marg > -0.5f, "…so the support channel no longer refutes the place (live ℓ refuted it)");
    }

    std::print("== (B2) a sharp posterior at the true angle: unchanged\n");
    {
        for (const float phi_deg : {0.0f, 45.0f, 85.0f})
        {
            const auto b = door22(phi_deg * d2r);
            const auto mask = closed_leaf_mask(0.55f, 1.15f, 200, 2);
            const auto post = curve({{phi_deg * d2r, 1.0f}}, 0.2f * d2r);   // a delta on the grid node
            const float a = b.clutter_fraction(mask, R), m = b.clutter_fraction_leaf_marginal(mask, R, post);
            std::print("    phi {:>3.0f}°: clutter_fraction {:.4f}, marginal {:.4f}\n", phi_deg, a, m);
            check(std::abs(a - m) < 1e-3f, std::format("phi {:.0f}°: a sharp posterior reproduces clutter_fraction", phi_deg));
        }
        const auto b = door22(30.0f * d2r);
        const auto mask = closed_leaf_mask(0.55f, 1.15f, 200, 3);
        check(std::abs(b.clutter_fraction(mask, R) - b.clutter_fraction_leaf_marginal(mask, R, {})) < 1e-6f,
              "an empty curve IS clutter_fraction");
    }

    std::print("== (B3) a mask on plain wall 1.5 m along it: refuted under any leaf posterior\n");
    {
        const auto b = door22(81.0f * d2r);
        const auto mask = closed_leaf_mask(2.1f, 2.7f, 255, 4);   // the aperture spans x ∈ [0.17, 1.16]
        const auto flat = curve({{60.0f * d2r, 1.0f}}, 200.0f * d2r);   // ≈ uniform over the grid
        const float x = b.clutter_fraction_leaf_marginal(mask, R, flat);
        std::print("    x leaf-marginal (uniform φ) {:.3f}, ℓ_sup {:+.2f}\n", x, rc::door::core::support_llr(x, sm));
        check(x > 0.95f, "no leaf angle reaches the wall 1 m past the jamb: x ≈ 1");
        check(rc::door::core::support_llr(x, sm) < -3.0f, "the support channel still refutes a fit that is not on its place");
    }

    std::print("== (C1) door_1's mask, depth on an occluder 2.2 m out (chat_e2e13 birth of door_24)\n");
    const V2 cam(-0.84f, -0.69f);   // the sensor at the birth (ai2_log_phi.csv door_24's first row)
    {
        std::vector<V2> pts;
        for (int i = 0; i < 40; ++i)   // centroid (−0.63, 1.84), a 0.3 m wide patch facing the camera
        {
            const float a = -0.15f + 0.3f * float(i) / 39.0f;
            pts.emplace_back(-0.63f + a, 1.84f + 0.02f * float((i % 3) - 1));
        }
        const auto w = rc::door::associate_wall(kRoom1, pts, cam, 0.12f);
        const bool top = w.ok and std::abs(w.u.y()) < 0.1f and std::abs(w.O.y() - 4.61f) < 0.1f;
        std::print("    wall O({:+.2f},{:+.2f}) u({:+.2f},{:+.2f}) len {:.2f}; seen_at ({:+.2f},{:+.2f})\n", w.O.x(), w.O.y(),
                   w.u.x(), w.u.y(), w.len, w.seen_at ? w.seen_at->x() : NAN, w.seen_at ? w.seen_at->y() : NAN);
        check(top, "framed on the TOP wall (the line of sight's wall), not the partition 0.6 m to the side");
        check(w.seen_at.has_value() and std::abs(w.seen_at->x() - (-0.57f)) < 0.3f,
              "seen_at within 0.3 m of door_1 (−0.57, 4.61) along it — where the mask's light came from");
    }

    std::print("== (C2) DOOR_0 closed, its leaf on the y = 2.54 wall, seen from below\n");
    {
        std::vector<V2> pts;
        for (int i = 0; i < 40; ++i) pts.emplace_back(0.20f + 0.9f * float(i) / 39.0f, 2.55f + 0.01f * float((i % 3) - 1));
        const V2 c2(0.9f, -0.3f);
        const auto w = rc::door::associate_wall(kRoom1, pts, c2, 0.12f);
        check(w.ok and std::abs(w.u.y()) < 0.1f and std::abs(w.O.y() - 2.537f) < 0.05f, "on the y ≈ 2.54 wall");
        check(w.seen_at.has_value() and std::abs(w.seen_at->x() - 0.65f) < 0.05f and std::abs(w.seen_at->y() - 2.54f) < 0.03f,
              "seen_at on the leaf (x ≈ 0.65)");
    }

    std::print("== (C3) DOOR_0 open, seen THROUGH obliquely: points on room_3's far wall 2 m beyond\n");
    {
        const V2 c3(2.6f, 0.3f);
        std::vector<V2> pts;
        for (int i = 0; i < 40; ++i)   // rays through the opening x ∈ [0.25, 1.05] on y = 2.54, ending 2 m further on
        {
            const V2 at(0.25f + 0.8f * float(i) / 39.0f, 2.54f);
            const V2 dir = (at - c3).normalized();
            pts.push_back(at + dir * (2.0f / dir.y()));
        }
        V2 centroid = V2::Zero(); for (const auto& p : pts) centroid += p; centroid /= float(pts.size());
        const auto w = rc::door::associate_wall(kRoom1, pts, c3, 0.12f);
        std::print("    centroid ({:+.2f},{:+.2f}); seen_at ({:+.2f},{:+.2f})\n", centroid.x(), centroid.y(),
                   w.seen_at ? w.seen_at->x() : NAN, w.seen_at ? w.seen_at->y() : NAN);
        check(w.ok and std::abs(w.u.y()) < 0.1f and std::abs(w.O.y() - 2.537f) < 0.05f, "on the y ≈ 2.54 wall");
        check(w.seen_at.has_value() and std::abs(w.seen_at->x() - 0.65f) < 0.05f, "seen_at in the opening (x ≈ 0.65)");
        check(std::abs(centroid.x() - 0.65f) > 0.4f, "(while the centroid's perpendicular foot is > 0.4 m off it)");
    }

    std::print("\n{} failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
