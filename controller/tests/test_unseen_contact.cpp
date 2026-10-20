// Task 5 (local-motion policy): unseen-contact evidence from commanded-but-absent motion.
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_unseen_contact.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tuc && /tmp/tuc
#include "../src/local_policy/unseen_contact.h"

#include <cmath>
#include <random>
#include <span>
#include <vector>
#include <cstdio>

using namespace rc::policy;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

static BaseLimits limits() { BaseLimits b; b.v_max = 0.5f; b.w_max = 1.5f; b.v_min_eff = 0.05f; b.w_min_eff = 0.1f; return b; }

int main()
{
    const auto b = limits();
    auto body = rc::RobotFootprint::shadow(); body.set_safety_margin(0.05f);
    const Pose2 pose{{1.f, 2.f}, 0.f};   // forward = +x
    const float dt = 0.1f, sig = 0.01f;
    const Eigen::Vector2f fwd(1.f, 0.f);
    // ★ D3 (Task 6): L starts at the stated prior logit(0.01); limits() has no acceleration rates (instantaneous ramps),
    //   so the free prediction there is cmd*dt and the original checks keep their meaning.
    const float L0prior = std::log(UnseenContact::kPriorBlocked / (1.f - UnseenContact::kPriorBlocked));

    auto fwd_h = [](const UnseenContact& u) { const auto h = u.hypotheses(); return h.empty() ? std::optional<Return>{} : std::optional<Return>{h[0]}; };
    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto hs = u.hypotheses(); const auto h = fwd_h(u);
      CHECK("commanded_no_motion_grows: one hypothesis, p_exist > 0.9", hs.size() == 1 and h and h->p_exist > 0.9f);
      const float bare = body.support_radius_yaw(0.f, fwd) - body.safety_margin();
      CHECK("  in front of the body, at the bare hull edge + sigma", h and h->p.x() > pose.p.x() + bare - 1e-3f
            and h->p.x() < pose.p.x() + bare + 3.f * sig and std::abs(h->p.y() - pose.p.y()) < 0.05f);
      CHECK("  sigma = sigma_odo", h and std::abs(h->sigma - sig) < 1e-6f);
      u.reset();
      CHECK("reset clears", u.hypotheses().empty()); }

    { UnseenContact u;
      for (int i = 0; i < 40; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.2f * dt, 0.f, dt, pose, b, body, sig);
      const auto h = fwd_h(u);
      CHECK("moving_freely_none: no blocked evidence, no hypothesis (R1)", not h); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      for (int i = 0; i < 30; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.2f * dt, 0.f, dt, pose, b, body, sig);
      CHECK("evidence reverses once it moves again", fwd_h(u)->p_exist < 1e-3f); }

    { UnseenContact u;
      for (int i = 0; i < 50; ++i) u.update(0.02f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      CHECK("below_deadband_no_evidence", u.hypotheses().empty() and u.log_odds(0) == L0prior); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(-0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto h = u.hypotheses();
      CHECK("reverse_blocked_behind", h.size() == 1 and h[0].p_exist > 0.9f and h[0].p.x() < pose.p.x() - 0.1f); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const Eigen::Vector2f before = fwd_h(u)->p;
      Pose2 q = pose;
      for (int i = 0; i < 15; ++i) { q.p.x() -= 0.02f; u.update(-0.2f, 0.f, 0.f, 0.f, -0.2f * dt, 0.f, dt, q, b, body, sig); }
      const auto hs = u.hypotheses();
      CHECK("hypothesis_stays_after_backing_away", hs.size() == 1 and (hs[0].p - before).norm() < 1e-5f
            and hs[0].p.x() > q.p.x() + 0.3f);
      UnseenContact w;
      for (int i = 0; i < 20; ++i) w.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const float L0 = w.log_odds(0);
      for (int i = 0; i < 15; ++i) w.update(-0.2f, 0.f, 0.f, 0.f, -0.2f * dt, 0.f, dt, q, b, body, sig);
      CHECK("backing_leaves_forward_evidence", w.log_odds(0) == L0); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.1f * dt, 0.f, dt, pose, b, body, sig);
      CHECK("partial_motion_break_even", std::abs(u.log_odds(0) - L0prior) < 1e-4f); }

    { UnseenContact u;   // pure rotation commanded, yaw does not change: a hull corner on the leading side
      for (int i = 0; i < 20; ++i) u.update(0.f, 0.8f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto h = u.hypotheses();
      CHECK("rotation_blocked: hypothesis at a corner", h.size() == 1 and h[0].p_exist > 0.9f
            and (h[0].p - pose.p).norm() > body.inscribed_radius());
      UnseenContact v;
      for (int i = 0; i < 20; ++i) v.update(0.f, -0.8f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto g = v.hypotheses();
      CHECK("  opposite sense leads with a different corner", h.size() == 1 and g.size() == 1 and (h[0].p - g[0].p).norm() > 0.1f); }

    { // driving_over_hypothesis_refutes_it (round 6, B2): a forward hypothesis from 20 blocked ticks, then the body
      // drives freely (odometry = command) onto the place: the hypothesis is refuted, the direction back at the prior.
        UnseenContact u;
        for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
        const auto h0 = u.hypotheses();
        Pose2 q = pose; bool gone = false;
        for (int i = 0; i < 20 and not gone; ++i)
        {
            q.p.x() += 0.2f * dt;
            u.update(0.2f, 0.f, 0.2f, 0.f, 0.2f * dt, 0.f, dt, q, b, body, sig);
            gone = u.hypotheses().empty();
        }
        std::printf("  driving over: hypothesis at x=%.3f, refuted with the body at x=%.3f\n", h0.empty() ? 0.f : h0[0].p.x(), q.p.x());
        CHECK("driving_over_hypothesis_refutes_it", h0.size() == 1 and gone and u.log_odds(0) == L0prior);
        // and a body that does NOT move as commanded over it (blocked) keeps it
        UnseenContact v;
        for (int i = 0; i < 20; ++i) v.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
        Pose2 r = pose; r.p.x() = h0[0].p.x();   // (teleported onto it, but odometry says blocked)
        v.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, r, b, body, sig);
        CHECK("  a blocked tick does not refute", not v.hypotheses().empty());
    }
    { // odometry_frame (review fix round 2): stored in the odometry frame, converted with the current room<-odom pair; a
      // room-frame jump moves the hypothesis with the body (it stays in front of it), not into it.
        UnseenContact u;
        const Pose2 odom{{0.f, 0.f}, 0.f};
        for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, 0.f, 0.f, dt, odom, b, body, sig);
        const auto h_odom = u.hypotheses();
        const Pose2 room_a{{1.f, 2.f}, 0.f}, room_b{{1.3f, 2.1f}, 0.2f};   // the localiser jumps 0.3 m / 0.2 rad
        const auto ha = u.hypotheses(room_a, odom), hb = u.hypotheses(room_b, odom);
        const Eigen::Vector2f fwd_b(std::cos(0.2f), std::sin(0.2f));
        const float ahead_a = (ha[0].p - room_a.p).dot(Eigen::Vector2f(1.f, 0.f)), ahead_b = (hb[0].p - room_b.p).dot(fwd_b);
        std::printf("  odometry frame: ahead of the body %.3f m before the jump, %.3f m after\n", ahead_a, ahead_b);
        CHECK("odometry_frame: the hypothesis follows the body through a localisation jump",
              h_odom.size() == 1 and std::abs(ahead_a - ahead_b) < 1e-4f and std::abs(ahead_a - h_odom[0].p.x()) < 1e-4f);
    }
    { // rapid_reversal_no_phantom (review fix round 3): w flips +-0.3 every 50 ms tick, each command executed 60 +- 20 ms
      // after it is sent (uniform jitter); the base ramps at a_rot toward the command in force and odometry reports that
      // motion. Given the commands IN FORCE per sub-interval, no direction may rise above 0.05. (With the latest command
      // alone the same data read as blocked turning: printed for contrast.)
        BaseLimits rb = b; rb.a_lin = 1.f; rb.d_lin = 1.f; rb.a_rot = 2.f; rb.d_rot = 2.f; rb.v_min_eff = 0.03f; rb.w_min_eff = 0.08f;
        std::mt19937 g(9); std::uniform_real_distribution<float> jit(-0.02f, 0.02f);
        struct Sent { float t_apply, w; };
        std::vector<Sent> sent;
        UnseenContact u, naive;
        const float tick = 0.05f, sub = 0.001f;
        float w = 0.f, cmd_in_force = 0.f, last_sent = 0.f;
        Pose2 odom = pose;
        for (int i = 0; i < 200; ++i)
        {
            const float t0 = i * tick;
            const float wc = (i % 2 == 0) ? 0.3f : -0.3f;
            sent.push_back({t0 + 0.06f + jit(g), wc}); last_sent = wc;
            std::vector<UnseenContact::CmdSeg> segs;
            const float w_start = w; float dyaw = 0.f;
            for (float tt = 0.f; tt < tick - 1e-6f; tt += sub)
            {
                for (const auto& sd : sent) if (sd.t_apply <= t0 + tt + 1e-6f) cmd_in_force = sd.w;   // latest applied
                std::erase_if(sent, [&](const Sent& sd) { return sd.t_apply <= t0 + tt + 1e-6f and sd.w != cmd_in_force; });
                dyaw += UnseenContact::ramp(w, cmd_in_force, rb.a_rot, rb.d_rot, sub, &w);
                if (not segs.empty() and segs.back().w == cmd_in_force) segs.back().dt += sub; else segs.push_back({sub, 0.f, cmd_in_force});
            }
            odom.yaw += dyaw;
            u.update(std::span<const UnseenContact::CmdSeg>(segs), 0.f, w_start, 0.f, dyaw, odom, rb, body, 0.005f);
            naive.update(0.f, last_sent, 0.f, w_start, 0.f, dyaw, tick, odom, rb, body, 0.005f);
        }
        float worst = 0.f, worst_naive = 0.f;
        for (const auto& h : u.hypotheses()) worst = std::max(worst, h.p_exist);
        for (const auto& h : naive.hypotheses()) worst_naive = std::max(worst_naive, h.p_exist);
        std::printf("  rapid reversal: max p_exist %.2e with the commands in force, %.3f with the latest command alone\n", worst, worst_naive);
        CHECK("rapid_reversal_no_phantom", worst <= 0.05f);
    }
    { // startup_ramp_no_hypothesis (D3): a real base (a_lin 1 m/s^2, a_rot 2 rad/s^2, 60 ms of nothing first) asked for
      // 0.5 m/s and 1.5 rad/s from rest; odometry reports exactly the ramp it executes. The free prediction follows the
      // ramp, so no direction may rise above 0.05 (was p ~ 0.45 on the hull edge with cmd*dt: Stuck in open space).
        BaseLimits rb = b; rb.a_lin = 1.f; rb.d_lin = 1.f; rb.a_rot = 2.f; rb.d_rot = 2.f; rb.v_min_eff = 0.03f; rb.w_min_eff = 0.08f;
        UnseenContact u; float v = 0.f, w = 0.f; const float tick = 0.05f;
        for (int i = 0; i < 30; ++i)
        {
            // the base: one tick of latency, then the ramp (the driver passes the command in effect at the wheels)
            const float vc = i == 0 ? 0.f : 0.5f, wc = i == 0 ? 0.f : 1.5f;
            const float ds = UnseenContact::free_travel(v, vc, rb.a_lin, rb.d_lin, tick), dy = UnseenContact::free_travel(w, wc, rb.a_rot, rb.d_rot, tick);
            u.update(0.5f, 1.5f, v, w, ds, dy, tick, pose, rb, body, 0.005f);
            v = std::min(0.5f, v + (i == 0 ? 0.f : rb.a_lin * tick)); w = std::min(1.5f, w + (i == 0 ? 0.f : rb.a_rot * tick));
        }
        float worst = 0.f; for (const auto& h : u.hypotheses()) worst = std::max(worst, h.p_exist);
        std::printf("  start-up ramp: %zu directions with evidence, max p_exist %.2e\n", u.hypotheses().size(), worst);
        CHECK("startup_ramp_no_hypothesis_above_0.05", worst <= 0.05f);
        // and the same base, blocked from rest (measured velocity 0, nothing moves): the evidence still accrues
        UnseenContact x;
        for (int i = 0; i < 200; ++i) x.update(0.3f, 0.f, 0.f, 0.f, 0.f, 0.f, tick, pose, rb, body, 0.005f);
        std::printf("  blocked from rest with a real ramp: p_exist %.3f after 10 s\n", x.hypotheses().empty() ? 0.f : x.hypotheses()[0].p_exist);
    }
    return fails ? 1 : 0;
}
