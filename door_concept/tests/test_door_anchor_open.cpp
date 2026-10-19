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
static void step(Chain& c, const std::vector<V3>& pts, float swing, bool believed_right, float p_view = 0.74f,
                 double* swing_llr = nullptr)
{
    rc::interaction::RelPose post;
    post.x = {0.f, D, 0.f};
    post.P = Eigen::Vector3f(0.02f * 0.02f, 0.02f * 0.02f, 0.01f * 0.01f).asDiagonal();
    rc::door::LeafPrior leaf;
    leaf.known = true; leaf.len = W; leaf.swing_y = swing;
    leaf.hinge_u = (believed_right ? -0.5f : 0.5f) * W;
    leaf.dir_u   = believed_right ? 1.f : -1.f;
    const auto g = rc::door::phi_grid(PHI_MAX, PHI_STEP);
    leaf.phi = c.mean; leaf.phi_sigma = c.sigma; leaf.phi_known = not c.first;
    const auto ac = rc::door::anchor_phi_loglik(post, W, T, leaf, V3(0.f, 0.f, 0.5f), pts, p_view, g.lo, g.hi, g.n);
    if (swing_llr != nullptr) *swing_llr += ac.swing_llr;
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

    std::print("(8) Task 12 fix D: the BELIEVED SWING is wrong (a config pin / an inherited side) — both swings are scored\n");
    for (const float swing_true : {+1.f, -1.f})
        for (const float deg : {20.f, 30.f, 45.f, 60.f, 90.f})
        {
            const float phi = deg * float(M_PI) / 180.f;
            Chain c;
            double sllr = 0.0;
            for (int f = 0; f < 5; ++f) step(c, sweep(phi, swing_true, rng), -swing_true, true, 0.74f, &sllr);
            const float clear = rc::door::clear_width(W, phi);
            std::print("  leaf truly {} {:3.0f} deg, believed the other way: P(passable) {:.3f} (logit {:+.1f}), phi est {:.0f} deg, "
                       "swing log-odds (believed vs flipped) {:+.1f}\n", swing_true > 0 ? "AWAY  " : "TOWARD", deg, c.p, c.logit,
                       c.mean * 57.2958f, sllr);
            if (clear >= NEEDED)
                check(c.p > 0.8f, std::format("wrong swing, {:.0f} deg (span {:.2f} ≥ body): P {:.3f} > 0.8", deg, clear, c.p));
            else
                check(c.p < 0.2f, std::format("wrong swing, {:.0f} deg (span {:.2f} < body): P {:.3f} < 0.2", deg, clear, c.p));
            if (deg >= 30.f) check(sllr < 0.0, std::format("wrong swing, {:.0f} deg: the evidence favours the TRUE swing ({:+.1f})", deg, sllr));
        }
    {
        Chain c;
        double sllr = 0.0;
        for (int f = 0; f < 5; ++f) step(c, sweep(0.f, 1.f, rng), -1.f, true, 0.74f, &sllr);
        std::print("  a SHUT door: swing log-odds {:+.3f}\n", sllr);
        check(std::abs(sllr) < 1.0, "a shut leaf says (almost) nothing about the swing");
    }

    std::print("(9) fix round 1 (I-3): ALIASED frames (the pose 60–86° off, try70 leg 5) do not flip a swing believed at log-odds 10\n");
    {
        // The truth: the leaf hinged at −W/2, swung AWAY 90°. The door believes it, swing log-odds +10.
        float swing_llr = 10.f, swing_llr_ungated = 10.f;
        const float swing = 1.f;
        for (int f = 0; f < 5; ++f)
        {
            const auto pts = sweep(1.5708f, swing, rng);
            rc::interaction::RelPose alias;
            const float yaw = (60.f + 6.5f * float(f)) * float(M_PI) / 180.f;   // 60 … 86°
            alias.x = {0.25f, D - 0.3f, yaw};
            alias.P = Eigen::Vector3f(0.02f * 0.02f, 0.02f * 0.02f, 0.02f * 0.02f).asDiagonal();
            rc::door::LeafPrior leaf;
            leaf.known = true; leaf.len = W; leaf.swing_y = swing; leaf.hinge_u = -0.5f * W; leaf.dir_u = 1.f;
            leaf.phi = 1.5708f; leaf.phi_sigma = 0.1f; leaf.phi_known = true;
            const auto view = rc::door::door_view(alias, W, T, -3.f, 3.f, V3(0.f, 0.f, 0.5f), pts, {}, {}, 1.f, leaf);
            const auto g = rc::door::phi_grid(PHI_MAX, PHI_STEP);
            const auto ac = rc::door::anchor_phi_loglik(alias, W, T, leaf, V3(0.f, 0.f, 0.5f), pts, std::max(view.p_view, 0.74f),
                                                        g.lo, g.hi, g.n);
            swing_llr_ungated += float(ac.swing_llr);
            if (rc::door::anchor_frame_is_leaf_evidence(view.p_view, 0.5f)) swing_llr += float(ac.swing_llr);
            std::print("  aliased frame yaw {:.0f} deg: p_view {:.2f}, the frame's swing log-odds {:+.1f}\n", yaw * 57.2958f,
                       view.p_view, ac.swing_llr);
        }
        std::print("  swing log-odds after 5 aliased frames: {:+.1f} (counted only as published), {:+.1f} if every frame counted\n",
                   swing_llr, swing_llr_ungated);
        check(swing_llr >= 10.f, "the believed swing is not flipped by aliased frames");
        // …and the gate passes a frame that IS seen.
        check(rc::door::anchor_frame_is_leaf_evidence(0.74f, 0.5f) and not rc::door::anchor_frame_is_leaf_evidence(0.46f, 0.5f),
              "a published frame counts, a refused one does not");
    }

    std::print("(10) fix round 1 (I-2): the retry's room fusion is weighted by P(out of the doorway band)\n");
    {
        rc::interaction::RelPose belief, room;
        belief.x = {0.1f, 0.3f, 0.05f}; belief.P = Eigen::Vector3f(4e-4f, 4e-4f, 1e-4f).asDiagonal();
        room.x = {0.4f, 0.3f, -0.15f};  room.P = Eigen::Vector3f(0.09f, 0.04f, 0.01f).asDiagonal();
        const Eigen::Matrix3f model = Eigen::Vector3f(0.09f, 0.04f, 0.01f).asDiagonal();
        const auto in_band = rc::door::retry_fusion(belief, room, model, 0.f);
        const auto half    = rc::door::retry_fusion(belief, room, model, 0.5f);
        const auto out     = rc::door::retry_fusion(belief, room, model, 1.f);
        std::print("  lateral moved: in the band {:.4f} m, half {:.3f} m, out of it {:.3f} m\n", (in_band.x - belief.x).head<2>().norm(),
                   (half.x - belief.x).head<2>().norm(), (out.x - belief.x).head<2>().norm());
        check(in_band.x == belief.x and in_band.P.isApprox(belief.P), "inside the band the belief stands (no broadening, no pull)");
        check((half.x - belief.x).head<2>().norm() < (out.x - belief.x).head<2>().norm() and (half.x - belief.x).head<2>().norm() > 0.f,
              "the pull grows continuously with P(out of band)");
        // The weight: refused only where BOTH estimates put the robot in the doorway (try73: belief s −0.07, room s −1.7).
        rc::interaction::RelPose b_in, r_in, r_out;
        b_in.x = {0.f, 0.07f, 0.f};  b_in.P = Eigen::Vector3f(4e-4f, 4e-4f, 1e-4f).asDiagonal();
        r_in.x = {0.f, 0.30f, 0.f};  r_in.P = Eigen::Vector3f(0.09f, 0.04f, 0.01f).asDiagonal();
        r_out.x = {0.f, 1.70f, 0.f}; r_out.P = r_in.P;
        const float w_both = rc::door::retry_weight_out_of_band(b_in, r_in, 0.8f), w_disagree = rc::door::retry_weight_out_of_band(b_in, r_out, 0.8f);
        std::print("  weight out of band: both in the doorway {:.3f}; belief in, room 1.7 m out {:.3f}\n", w_both, w_disagree);
        check(w_both < 0.05f, "both in the doorway: the room is refused (try70 leg 5)");
        check(w_disagree > 0.95f, "the room disagrees about being in the doorway: it is fused (try73 leg 1)");
    }

    std::print("(11) Task 17: a belief far off the room initialisation, beyond their covariance, is the suspect\n");
    {
        const Eigen::Matrix3f model = Eigen::Vector3f(0.09f, 0.04f, 0.01f).asDiagonal();
        auto run = [&](float off, float yaw_deg) {
            rc::interaction::RelPose belief, room;
            belief.x = {0.f, 2.f, 0.f}; belief.P = Eigen::Vector3f(0.03f * 0.03f, 0.03f * 0.03f, 0.01f * 0.01f).asDiagonal();
            room.x = {off, 2.f, yaw_deg / 57.2958f};
            room.P = Eigen::Vector3f(0.0905f, 0.0405f, 0.0101f).asDiagonal();   // try78's P0: belief + localisation + σ_model
            const auto e = rc::door::belief_lost(belief, room, 0.9f);
            const auto out = rc::door::retry_reinit(belief, room, model, 0.f, e.p_lost);   // inside the band: w_band 0
            std::print("  room − belief {:.2f} m / {:.0f}°: Mahalanobis² {:.1f}, P(lost) {:.3f} → moved {:.3f} m toward the room ({:.3f} m left), σ_lat {:.3f}\n",
                       off, yaw_deg, e.d2, e.p_lost, out.x.x(), off - out.x.x(), std::sqrt(out.P(0, 0)));
            return std::pair{e.p_lost, off - out.x.x()};
        };
        const auto [p05, r05] = run(0.05f, 0.f);
        check(p05 < 0.05f and std::abs(r05 - 0.05f) < 0.01f, "0.05 m off: the belief is KEPT");
        const auto [p08, r08] = run(0.8f, 10.f);
        const auto [p097, r097] = run(0.97f, 9.5f);   // try78's second retry
        const auto [p15, r15] = run(1.5f, 20.f);
        check(r15 < 0.2f, "1.5 m / 20° off: re-initialised toward the room, within 0.2 m");
        check(p08 > p05 and p097 > p08 and p15 > p097, "P(lost) grows continuously with the disagreement");
        (void)r08; (void)r097;
    }

    std::print("(12) Task 17 add-on: the belief's own visibility joins the evidence that it is lost\n");
    {
        const Eigen::Matrix3f model = Eigen::Vector3f(0.09f, 0.04f, 0.01f).asDiagonal();
        const double tau = 3.0;   // kAnchorPredictMs (interaction_anchor.h), in s
        auto run = [&](float off, float yaw_deg, float p_view, double seconds) {
            double llr = 0.0;
            for (double t = 0.0; t < seconds; t += 0.1) llr += rc::door::view_lost_llr_step(p_view, 0.1, tau);   // 10 Hz frames
            rc::interaction::RelPose belief, room;
            belief.x = {0.f, 2.f, 0.f}; belief.P = Eigen::Vector3f(0.03f * 0.03f, 0.03f * 0.03f, 0.01f * 0.01f).asDiagonal();
            room.x = {off, 2.f, yaw_deg / 57.2958f};
            room.P = Eigen::Vector3f(0.0905f, 0.0405f, 0.0101f).asDiagonal();
            const auto e = rc::door::belief_lost(belief, room, 0.9f, llr);
            const auto out = rc::door::retry_reinit(belief, room, model, 0.f, e.p_lost);
            std::print("  {:.2f} m / {:.0f}° off, {:.0f} s at p_view {:.2f}: view log-odds {:+.1f}, P(lost) {:.3f} → {:.3f} m from the room\n",
                       off, yaw_deg, seconds, p_view, llr, e.p_lost, off - out.x.x());
            return off - out.x.x();
        };
        check(run(0.8f, 10.f, 0.15f, 60.0) < 0.2f, "0.8 m / 10° off AND 60 s unpublished at p_view 0.15: re-initialised within 0.2 m");
        check(std::abs(run(0.05f, 0.f, 0.30f, 2.5) - 0.05f) < 0.01f, "a correct belief briefly unpublished (2.5 s at 0.30): kept");
        check(std::abs(run(0.05f, 0.f, 0.85f, 60.0) - 0.05f) < 0.01f, "a correct belief seen for 60 s: kept");
        check(rc::door::view_lost_llr_step(0.0f, 0.1, tau) <= std::log(3.0) * 0.1 / tau + 1e-12, "one frame is never decisive (bounded)");
    }

    std::print("(13) Task 19: an approach claim creates no anchor; a cross claim does\n");
    {
        check(not rc::door::anchor_creation_claim(false, true), "approach claimed (door not yet open): no anchor");
        check(rc::door::anchor_creation_claim(true, false), "cross claimed (offered only on a passable door): the anchor is created");
        check(rc::door::anchor_creation_claim(true, true), "cross claimed while an approach is still claimed: the cross creates it");
        check(not rc::door::anchor_creation_claim(false, false), "nothing claimed: no anchor");
    }

    std::print("(14) Task 20: the (hinge x swing) marginal — the decision that unblocks try87's reopen\n");
    {
        // A 25-point grid over [0, 120 deg]; Gaussian log-likelihood bumps (nats).
        constexpr int N = 25;
        constexpr float PMAX = 2.0944f;
        const auto bump = [&](float mu, float sd, float depth)
        {
            std::vector<std::pair<float, float>> c;
            for (int k = 0; k < N; ++k)
            {
                const float phi = PMAX * k / (N - 1);
                c.emplace_back(phi, -depth * (1.f - std::exp(-0.5f * (phi - mu) * (phi - mu) / (sd * sd))));
            }
            return c;
        };
        const std::vector<std::pair<float, float>> none;
        // (a) only the two hinges scored: identical to phi_hinge_marginal (the swing prior cancels)
        {
            const auto a = bump(1.0f, 0.2f, 8.f), o = bump(1.6f, 0.2f, 12.f);
            const auto hm = rc::door::phi_hinge_marginal(a, o, 0.7f, 1.2f, 0.4f);
            const auto lm = rc::door::phi_leaf_marginal(a, o, none, none, 0.7f, 3.0f, 1.2f, 0.4f);
            float worst = 0.f;
            for (std::size_t k = 0; k < a.size(); ++k) worst = std::max(worst, std::abs(hm.loglik[k].second - lm.loglik[k].second));
            check(worst < 1e-4f and std::abs(hm.hinge_llr - lm.hinge_llr) < 1e-4f and lm.swing_llr == 3.0f,
                  std::format("hinges only: == phi_hinge_marginal (max |dL| {:.1e}), swing log-odds untouched", worst));
        }
        // (b) try87: the believed swing explains the far-side leaf only as a half-shut 60 deg leaf (deep miss elsewhere);
        //     the OTHER swing explains it at 90 deg, by more nats. The marginal must move to the other swing and read passable.
        {
            const auto bb = bump(1.04f, 0.08f, 30.f), bs = bump(1.5708f, 0.08f, 60.f);   // bs: 30 nats better at its peak
            // put the other swing's peak above the believed one's: its curve is 0 at 90 deg, the believed one's is -18 there
            auto bb_low = bb; for (auto& [phi, l] : bb_low) l -= 18.f;
            const auto lm = rc::door::phi_leaf_marginal(bb_low, none, bs, none, 0.0f, 0.0f, 1.3f, 0.6f);
            const auto pin = rc::door::passable_logit(std::vector<std::pair<float, double>>(bb_low.begin(), bb_low.end()), 1.0f, 0.70f);
            std::vector<std::pair<float, double>> mixd(lm.loglik.begin(), lm.loglik.end());
            const auto mix = rc::door::passable_logit(mixd, 1.0f, 0.70f);
            std::print("    believed swing alone: P(passable) log-odds {:+.1f}; swing-marginal {:+.1f}; swing log-odds {:+.1f}\n",
                       pin.value_or(0.0), mix.value_or(0.0), lm.swing_llr);
            check(pin.has_value() and *pin < 0.0, "the believed (wrong) swing alone reads the open door SHUT (the try87 offer of `open`)");
            check(lm.swing_llr < 0.0f, "the swing log-odds moves to the other side");
            check(mix.has_value() and *mix > 0.0, "the swing-marginal P(passable) > 1/2: cross offered, open (P(shut)) not");
        }
        // (c) a SHUT leaf is the same under every hypothesis: the swing and hinge log-odds do not move
        {
            const auto c = bump(0.0f, 0.1f, 20.f);
            const auto lm = rc::door::phi_leaf_marginal(c, c, c, c, 1.5f, -0.5f, 0.0f, 0.2f);
            check(std::abs(lm.hinge_llr - 1.5f) < 1e-4f and std::abs(lm.swing_llr + 0.5f) < 1e-4f,
                  "identical curves (a shut leaf) teach nothing about hinge or swing");
        }
    }

    std::print("(15) Task 21: no anchor outlives its passage; a new cross puts the robot on the NEAR side\n");
    {
        // try89 leg 2 -> leg 3: through the room switch (crossing_through), then the door is asked again (leg 3's approach).
        check(rc::door::passage_over_after_through(true, true, false), "after the switch, this door claimed again: the passage is over");
        check(rc::door::passage_over_after_through(true, false, true), "after the switch, the predicted window spent: the passage is over");
        check(not rc::door::passage_over_after_through(true, false, false), "still crossing on it (claimed nothing, seen/predicted): kept");
        check(not rc::door::passage_over_after_through(false, true, true), "before the switch the cross's own lifecycle rules (retry etc.)");
        // …and the next cross claim creates a fresh one (Task 19): nothing anchored ⇒ the cross creates it.
        check(rc::door::anchor_creation_claim(true, false), "a new cross claim creates a fresh anchor");

        // The near side. Door at the origin, wall_u = +x, across_u = +y. s = the robot's coordinate along the anchor's +y.
        const Eigen::Vector2f c(0.f, 0.f), wu(1.f, 0.f), n(0.f, 1.f);
        const auto s_of = [&](const Eigen::Vector2f& p_robot, float latched)
        {
            const auto ax = rc::door::anchor_opening_axes(wu, n, rc::door::anchor_near_side(latched, n.dot(p_robot - c)));
            const float det = ax.ux.x() * ax.uy.y() - ax.ux.y() * ax.uy.x();
            check(std::abs(det - 1.f) < 1e-5f, "the opening frame stays right-handed");
            return ax.uy.dot(p_robot - c);
        };
        const Eigen::Vector2f leg2_robot(0.1f, -1.9f), leg3_robot(0.2f, 1.8f);   // the two legs start on opposite sides
        check(s_of(leg2_robot, -1.f) < 0.f, "leg 2 (robot on -n, leaving the -n room): s < 0");
        check(s_of(leg3_robot, +1.f) < 0.f, std::format("leg 3 (robot on the OTHER side, +n): s {:+.2f} < 0 — not leg 2's +1.78", s_of(leg3_robot, +1.f)));
        check(s_of(leg3_robot, 0.f) < 0.f and s_of(leg2_robot, 0.f) < 0.f, "unlatched: the robot's own side, s < 0 either way");
        // A retry mid-doorway, 0.1 m past the plane: the latched side (the room it is leaving) keeps the normal pointing on.
        check(s_of(Eigen::Vector2f(0.f, 0.1f), -1.f) > 0.f, "mid-doorway retry: the side the cross LEAVES holds (s > 0, past the plane, not flipped)");
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
