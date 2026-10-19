// Task 11 — tests for door_anchor_open.h: the anchor's own beams as evidence about the LEAF ANGLE (fix 2, fix round 1),
// P(passable) in log space (door_passability.h passable_logit), and the not-seen wait pose (fix 4).
//
// The pipeline is estimate_phi's: the anchor's φ curve (both hinges) → phi_hinge_marginal → phi_posterior_step →
// passable_logit. Here the anchor curve is the ONLY likelihood, so the test says what the anchor's beams alone decide.
//
// Build + run (from door_concept/, Q=/usr/include/x86_64-linux-gnu/qt5):
//   g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore tests/test_door_anchor_open.cpp \
//       src/door_belief.cpp -o build/test_door_anchor_open -lfastdds -lfastcdr -lQt5Core && ./build/test_door_anchor_open
#include <clocale>
#include <cmath>
#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

#include "door_anchor_open.h"
#include "door_phi_posterior.h"
#include "door_passability.h"

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

using V3 = Eigen::Vector3f;
using V2 = Eigen::Vector2f;

constexpr float D = 1.8f, W = 1.0f, T = 0.11f, NEEDED = 0.70f;   // NEEDED: the body + margin (test_door_phi_stability)
constexpr float PHI_MAX = 2.0944f, PHI_STEP = 0.0873f;

// Robot at the origin facing +y; a wall at y = D with an opening |x| < W/2 — the anchor at (0, D), yaw 0, so the anchor's
// along-wall axis is the robot's x and its +y is AWAY from the robot. Beyond: the next room's far wall at y = D + 3.
// The leaf: hinged at (−W/2, D), length W, swung by phi AWAY (+y) or TOWARD the robot (−y). 0.25° sweep, 1 cm noise.
static float ray_seg(const V2& o, const V2& u, const V2& a, const V2& b)
{
    const V2 d = b - a;
    const float den = u.x() * (-d.y()) - u.y() * (-d.x());
    if (std::abs(den) < 1e-9f) return -1.f;
    const V2 w = a - o;
    const float t = (w.x() * (-d.y()) - w.y() * (-d.x())) / den, q = (u.x() * w.y() - u.y() * w.x()) / den;
    return (t > 0.f and q >= 0.f and q <= 1.f) ? t : -1.f;
}
static std::vector<V3> sweep(float phi, float swing, std::mt19937& rng, float occl = 0.f)
{
    std::normal_distribution<float> n(0.f, 0.01f);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const V2 o(0.f, 0.f), h(-0.5f * W, D), e = h + W * V2(std::cos(phi), swing * std::sin(phi));
    std::vector<V3> pts;
    for (int a = 0; a <= 720; ++a)
    {
        const float az = (0.25f * a) * float(M_PI) / 180.f;
        const V2 u(std::cos(az), std::sin(az));
        if (u.y() < 0.05f) continue;
        float t = (D + 3.f) / u.y();
        for (const auto& [p, q] : {std::pair{V2(-6.f, D), V2(-0.5f * W, D)}, std::pair{V2(0.5f * W, D), V2(6.f, D)}, std::pair{h, e}})
            if (const float s = ray_seg(o, u, p, q); s > 0.f) t = std::min(t, s);
        if (U(rng) < occl) t = 0.6f + U(rng) * 0.5f;
        t = std::min(t + n(rng), 7.9f);
        pts.emplace_back(t * u.x(), t * u.y(), 0.5f);
    }
    return pts;
}

struct Chain { float mean = 0.f, sigma = PHI_MAX; bool first = true; float hinge_llr = 0.f; double logit = 0.0; float p = 0.f; };
// One cycle, estimate_phi's way, with the anchor's curve as the likelihood. `believed_right` false = the anchor's
// LeafPrior names the other jamb (the try69 inheritance).
static void step(Chain& c, const std::vector<V3>& pts, float swing, bool believed_right, float p_view = 0.74f)
{
    rc::interaction::RelPose post;
    post.x = {0.f, D, 0.f};
    post.P = Eigen::Vector3f(0.02f * 0.02f, 0.02f * 0.02f, 0.01f * 0.01f).asDiagonal();
    rc::door::LeafPrior leaf;
    leaf.known = true; leaf.len = W; leaf.swing_y = swing;
    leaf.hinge_u = (believed_right ? -0.5f : 0.5f) * W;
    leaf.dir_u   = believed_right ? 1.f : -1.f;
    const auto g = rc::door::phi_grid(PHI_MAX, PHI_STEP);
    const auto ac = rc::door::anchor_phi_loglik(post, W, T, leaf, V3(0.f, 0.f, 0.5f), pts, p_view, g.lo, g.hi, g.n);
    std::vector<std::pair<float, float>> ll_this, ll_other;
    for (int k = 0; k < int(ac.ll_this.size()); ++k)
    {
        const float phi = g.lo + (g.hi - g.lo) * k / (g.n - 1);
        ll_this.emplace_back(phi, ac.ll_this[k]);
        ll_other.emplace_back(phi, ac.ll_other[k]);
    }
    const float sp = rc::door::phi_prior_sigma(c.sigma, 2.0f, c.first ? std::numeric_limits<float>::infinity() : 0.1f, PHI_STEP, PHI_MAX);
    c.first = false;
    auto hm = rc::door::phi_hinge_marginal(ll_this, ll_other, c.hinge_llr, c.mean, sp);
    c.hinge_llr = hm.hinge_llr;
    const auto post_step = rc::door::phi_posterior_step(hm.loglik, g.lo, g.hi, g.n, c.mean, sp);
    c.mean = post_step.mean; c.sigma = post_step.sigma;
    c.logit = rc::door::passable_logit(post_step.log_curve, W, NEEDED).value_or(0.0);
    c.p = float(rc::door::sigmoid(c.logit));
}

int main()
{
    std::setlocale(LC_ALL, "");
    std::mt19937 rng(11);

    std::print("(1) AJAR doors: P(passable) follows clear_width(w, phi) >= body ({:.2f} m), first look, 5 frames\n", NEEDED);
    for (const float swing : {+1.f, -1.f})
        for (const float deg : {0.f, 20.f, 30.f, 45.f, 60.f, 90.f, 110.f})
        {
            const float phi = deg * float(M_PI) / 180.f;
            Chain c;
            for (int f = 0; f < 5; ++f) step(c, sweep(phi, swing, rng), swing, true);
            const float clear = rc::door::clear_width(W, phi);
            const bool passable = clear >= NEEDED;
            std::print("  leaf {} {:3.0f} deg: clear {:.2f} m → P(passable) {:.3f} (logit {:+.1f}), phi est {:.0f} deg\n",
                       swing > 0 ? "AWAY  " : "TOWARD", deg, clear, c.p, c.logit, c.mean * 57.2958f);
            if (passable)
                check(c.p > 0.8f, std::format("{} {:.0f} deg (span {:.2f} ≥ body): P {:.3f} > 0.8", swing > 0 ? "away" : "toward", deg, clear, c.p));
            else
                check(c.p < 0.2f, std::format("{} {:.0f} deg (span {:.2f} < body): P {:.3f} < 0.2", swing > 0 ? "away" : "toward", deg, clear, c.p));
        }

    std::print("(2) the anchor's leaf prior names the WRONG jamb (try69): the hinge is learned, P follows the truth\n");
    for (const float deg : {30.f, 90.f})
    {
        const float phi = deg * float(M_PI) / 180.f;
        Chain c;
        for (int f = 0; f < 5; ++f) step(c, sweep(phi, +1.f, rng), +1.f, false);
        std::print("  {:.0f} deg: P {:.3f}, hinge log-odds (believed vs other) {:+.1f}\n", deg, c.p, c.hinge_llr);
        check(c.hinge_llr < 0.f, "the evidence favours the true jamb");
        check(deg < 70.f ? c.p < 0.2f : c.p > 0.8f, std::format("{:.0f} deg read right with the wrong jamb believed", deg));
    }

    std::print("(3) clutter: 30 % of beams short — shut stays shut, open stays open\n");
    {
        Chain s, o;
        for (int f = 0; f < 5; ++f) { step(s, sweep(0.f, 1.f, rng, 0.3f), 1.f, true); step(o, sweep(1.5708f, 1.f, rng, 0.3f), 1.f, true); }
        std::print("  shut P {:.3f}, open P {:.3f}\n", s.p, o.p);
        check(s.p < 0.1f and o.p > 0.8f, "occluders do not decide it");
    }

    std::print("(4) p_view = 0: the anchor curve is empty (no evidence)\n");
    {
        rc::interaction::RelPose post; post.x = {0.f, D, 0.f}; post.P = Eigen::Matrix3f::Identity() * 1e-4f;
        rc::door::LeafPrior leaf; leaf.known = true; leaf.len = W; leaf.hinge_u = -0.5f; leaf.dir_u = 1.f; leaf.swing_y = 1.f;
        const auto ac = rc::door::anchor_phi_loglik(post, W, T, leaf, V3(0, 0, 0.5f), sweep(1.5708f, 1.f, rng), 0.f, 0.f, PHI_MAX, 25);
        check(ac.ll_this.empty() and ac.ll_other.empty(), "nothing measured");
    }

    std::print("(5) pi exactly 0 is not absorbing: sigma_phi 0.01 rad at phi ~ 0, then an OPEN door's beams\n");
    {
        // The carried posterior after a long shut stretch: mean 0.01 rad, sigma 0.01 rad (try69's collapse).
        const auto g = rc::door::phi_grid(PHI_MAX, PHI_STEP);
        std::vector<std::pair<float, float>> flat;   // unmeasured frame: the prior alone
        const auto shut = rc::door::phi_posterior_step(flat, g.lo, g.hi, g.n, 0.01f, 0.01f);
        const auto l0 = rc::door::passable_logit(shut.log_curve, W, NEEDED);
        float p_lin = rc::door::passable_prob(shut.curve, W, NEEDED).value_or(-1.f);
        std::print("  collapsed: linear float P {:.3g} (absorbing), log-odds {:+.1f} (finite)\n", p_lin, l0.value_or(0.0));
        check(l0.has_value() and std::isfinite(*l0), "the log-odds of a collapsed curve is finite");
        Chain c; c.mean = 0.01f; c.sigma = 0.01f; c.first = false;
        int first = -1;
        for (int f = 0; f < 10; ++f)
        {
            step(c, sweep(1.5708f, 1.f, rng), 1.f, true);
            if (first < 0 and c.p > 0.8f) first = f;
        }
        std::print("  after the open beams: P {:.3f} (> 0.8 from frame {})\n", c.p, first);
        check(first >= 0 and first <= 3, "strong opposing evidence moves it within 3 frames");
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

    std::print("(7) fix 4: the odometry carry is EXCLUSIVE — init carried while unseen, frozen after the first sighting\n");
    {
        rc::interaction::RelPose seen, init;
        seen.x = {0.1f, 1.5f, 0.0f}; init.x = {0.3f, 2.0f, 0.1f};
        rc::rt::Twist tw; tw.linear = {0.f, 0.3f, 0.f}; tw.angular = {0.f, 0.f, 0.2f};
        const Eigen::Vector3f var = Eigen::Vector3f::Zero();
        const auto s0 = seen.x, i0 = init.x;
        rc::door::carry_wait_poses(false, seen, init, tw, var, 0.5f);
        check(seen.x == s0 and not (init.x == i0), "unseen: the init pose moves, seen_pose does not");
        const auto i1 = init.x, s1 = seen.x;
        rc::door::carry_wait_poses(true, seen, init, tw, var, 0.5f);
        check(init.x == i1 and not (seen.x == s1), "seen: seen_pose moves, the init pose is frozen");
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
