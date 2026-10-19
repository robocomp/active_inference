// Task 11 — tests for door_anchor_open.h: the anchor's own beams as evidence about the leaf (fix 2), and the pose the
// not-seen wait-turn is handed (fix 4).
//
// Build + run (from door_concept/, Q=/usr/include/x86_64-linux-gnu/qt5):
//   g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore tests/test_door_anchor_open.cpp \
//       -o build/test_door_anchor_open -lfastdds -lfastcdr -lQt5Core && ./build/test_door_anchor_open
#include <clocale>
#include <cmath>
#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

#include "door_anchor_open.h"

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

using V3 = Eigen::Vector3f;

// The robot at the origin facing +y; a wall at y = D with an opening |x| < W/2 (the anchor at (0, D), yaw 0, so the
// anchor's along-wall axis is the robot's x). Beyond it the next room's far wall at y = D + 3. A wall-band sweep,
// 0.25° steps over the forward half-plane, 1 cm range noise.
static std::vector<V3> sweep(bool open, float D, float W, std::mt19937& rng, float occl_frac = 0.f)
{
    std::normal_distribution<float> n(0.f, 0.01f);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::vector<V3> pts;
    for (int a = 0; a <= 720; ++a)
    {
        const float az = (0.25f * a) * float(M_PI) / 180.f;
        const Eigen::Vector2f u(std::cos(az), std::sin(az));
        if (u.y() < 0.05f) continue;
        float t = D / u.y();
        const float xw = t * u.x();
        if (open and std::abs(xw) < 0.5f * W) t = (D + 3.f) / u.y();   // through the opening, to the far wall
        if (U(rng) < occl_frac) t = 0.6f + U(rng) * 0.5f;              // a person / clutter in front
        t = std::min(t + n(rng), 7.9f);
        pts.emplace_back(t * u.x(), t * u.y(), 0.5f);
    }
    return pts;
}

int main()
{
    std::setlocale(LC_ALL, "");
    std::mt19937 rng(11);
    const V3 origin(0.f, 0.f, 0.5f);
    const float D = 1.8f, W = 1.0f, T = 0.11f;
    rc::interaction::RelPose post;
    post.x = {0.f, D, 0.f};
    post.P = Eigen::Vector3f(0.02f * 0.02f, 0.02f * 0.02f, 0.01f * 0.01f).asDiagonal();
    const float p_view = 0.74f;   // try69 leg 2's published p_view while it was publishing

    std::print("(1) the leaf estimator collapsed (pi = 0.02) but the anchor's beams pass through the opening\n");
    {
        float fused = 0.02f;
        int first = -1;
        for (int f = 0; f < 3; ++f)
        {
            const auto ev = rc::door::anchor_open_llr(post, W, T, origin, sweep(true, D, W, rng), p_view);
            fused = rc::door::fuse_open(0.02f, ev.llr);   // the door's own pi stays 0.02: only the latest frame is added
            std::print("  frame {}: Lambda {:+.2f} (raw {:+.1f}, n_inf {:.0f}, beta {:.3f}, {} rays) -> P(open) {:.3f}\n",
                       f, ev.llr, ev.raw, ev.n_inf, ev.beta, ev.n_rays, fused);
            if (first < 0 and fused > 0.8f) first = f;
        }
        check(first >= 0 and first < 3, std::format("fused P(open) > 0.8 within 3 frames (frame {})", first));
    }

    std::print("(2) a truly SHUT door: the fused P(open) stays < 0.1\n");
    for (const float pi : {0.02f, 0.5f})
    {
        float worst = 0.f;
        for (int f = 0; f < 5; ++f)
        {
            const auto ev = rc::door::anchor_open_llr(post, W, T, origin, sweep(false, D, W, rng), p_view);
            worst = std::max(worst, rc::door::fuse_open(pi, ev.llr));
            if (f == 0) std::print("  pi {:.2f}: Lambda {:+.2f} (raw {:+.1f}, n_inf {:.0f})\n", pi, ev.llr, ev.raw, ev.n_inf);
        }
        check(worst < 0.1f, std::format("shut door, pi {:.2f}: fused P(open) max {:.4f} < 0.1", pi, worst));
    }

    std::print("(3) a shut door with clutter in front (30 % of beams short) still reads shut, an open one still open\n");
    {
        const auto es = rc::door::anchor_open_llr(post, W, T, origin, sweep(false, D, W, rng, 0.3f), p_view);
        const auto eo = rc::door::anchor_open_llr(post, W, T, origin, sweep(true, D, W, rng, 0.3f), p_view);
        std::print("  shut Lambda {:+.2f}, open Lambda {:+.2f}\n", es.llr, eo.llr);
        check(rc::door::fuse_open(0.5f, es.llr) < 0.1f and rc::door::fuse_open(0.02f, eo.llr) > 0.8f,
              "occluders are no evidence either way");
    }

    std::print("(4) the anchor posed on the WALL beside the real opening (0.9 m off): no open evidence\n");
    {
        rc::interaction::RelPose off = post;
        off.x.x() = -0.9f;   // its 'opening' is solid wall; the real one is open
        const auto ev = rc::door::anchor_open_llr(off, W, T, origin, sweep(true, D, W, rng), p_view);
        std::print("  Lambda {:+.2f}\n", ev.llr);
        check(rc::door::fuse_open(0.5f, ev.llr) < 0.1f, "a misplaced anchor cannot read the wall as open");
    }

    std::print("(5) a door out of view (p_view 0) says nothing\n");
    {
        const auto ev = rc::door::anchor_open_llr(post, W, T, origin, sweep(true, D, W, rng), 0.f);
        check(ev.llr == 0.0 and rc::door::fuse_open(0.02f, ev.llr) == 0.02f, "Lambda = 0, pi unchanged");
    }

    std::print("(6) fix 4: the born-stale wait pose is the last SEEN pose, else the room-initialised one — never the posterior\n");
    {
        rc::interaction::RelPose seen, init, posterior;
        seen.x = {0.1f, 1.5f, 0.02f};
        init.x = {0.3f, 2.0f, 0.1f};
        posterior.x = {1.2f, 0.4f, 1.5f};   // the unpublished posterior, dragged 86 deg by odometry in a wedge
        const auto& a = rc::door::wait_pose(true, seen, init);
        const auto& b = rc::door::wait_pose(false, seen, init);
        check(&a == &seen and &b == &init, "seen_once ⇒ seen_pose; not yet seen ⇒ the init pose");
        check(&a != &posterior and &b != &posterior, "the unpublished posterior is never the wait pose");
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
