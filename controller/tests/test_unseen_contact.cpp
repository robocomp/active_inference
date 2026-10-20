// Task 5 (local-motion policy): unseen-contact evidence from commanded-but-absent motion.
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_unseen_contact.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tuc && /tmp/tuc
#include "../src/local_policy/unseen_contact.h"

#include <cmath>
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

    auto fwd_h = [](const UnseenContact& u) { const auto h = u.hypotheses(); return h.empty() ? std::optional<Return>{} : std::optional<Return>{h[0]}; };
    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto hs = u.hypotheses(); const auto h = fwd_h(u);
      CHECK("commanded_no_motion_grows: one hypothesis, p_exist > 0.9", hs.size() == 1 and h and h->p_exist > 0.9f);
      const float bare = body.support_radius_yaw(0.f, fwd) - body.safety_margin();
      CHECK("  in front of the body, at the bare hull edge + sigma", h and h->p.x() > pose.p.x() + bare - 1e-3f
            and h->p.x() < pose.p.x() + bare + 3.f * sig and std::abs(h->p.y() - pose.p.y()) < 0.05f);
      CHECK("  sigma = sigma_odo", h and std::abs(h->sigma - sig) < 1e-6f);
      u.reset();
      CHECK("reset clears", u.hypotheses().empty()); }

    { UnseenContact u;
      for (int i = 0; i < 40; ++i) u.update(0.2f, 0.f, 0.2f * dt, 0.f, dt, pose, b, body, sig);
      const auto h = fwd_h(u);
      CHECK("moving_freely_none: weak hypothesis, no switch", h and h->p_exist < 1e-6f); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      for (int i = 0; i < 30; ++i) u.update(0.2f, 0.f, 0.2f * dt, 0.f, dt, pose, b, body, sig);
      CHECK("evidence reverses once it moves again", fwd_h(u)->p_exist < 1e-3f); }

    { UnseenContact u;
      for (int i = 0; i < 50; ++i) u.update(0.02f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      CHECK("below_deadband_no_evidence", u.hypotheses().empty() and u.log_odds(0) == 0.f); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(-0.2f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto h = u.hypotheses();
      CHECK("reverse_blocked_behind", h.size() == 1 and h[0].p_exist > 0.9f and h[0].p.x() < pose.p.x() - 0.1f); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const Eigen::Vector2f before = fwd_h(u)->p;
      Pose2 q = pose;
      for (int i = 0; i < 15; ++i) { q.p.x() -= 0.02f; u.update(-0.2f, 0.f, -0.2f * dt, 0.f, dt, q, b, body, sig); }
      const auto hs = u.hypotheses();
      CHECK("hypothesis_stays_after_backing_away", hs.size() == 2 and (hs[0].p - before).norm() < 1e-5f
            and hs[0].p.x() > q.p.x() + 0.3f);
      UnseenContact w;
      for (int i = 0; i < 20; ++i) w.update(0.2f, 0.f, 0.f, 0.f, dt, pose, b, body, sig);
      const float L0 = w.log_odds(0);
      for (int i = 0; i < 15; ++i) w.update(-0.2f, 0.f, -0.2f * dt, 0.f, dt, q, b, body, sig);
      CHECK("backing_leaves_forward_evidence", w.log_odds(0) == L0); }

    { UnseenContact u;
      for (int i = 0; i < 20; ++i) u.update(0.2f, 0.f, 0.1f * dt, 0.f, dt, pose, b, body, sig);
      CHECK("partial_motion_break_even", std::abs(u.log_odds(0)) < 1e-4f); }

    { UnseenContact u;   // pure rotation commanded, yaw does not change: a hull corner on the leading side
      for (int i = 0; i < 20; ++i) u.update(0.f, 0.8f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto h = u.hypotheses();
      CHECK("rotation_blocked: hypothesis at a corner", h.size() == 1 and h[0].p_exist > 0.9f
            and (h[0].p - pose.p).norm() > body.inscribed_radius());
      UnseenContact v;
      for (int i = 0; i < 20; ++i) v.update(0.f, -0.8f, 0.f, 0.f, dt, pose, b, body, sig);
      const auto g = v.hypotheses();
      CHECK("  opposite sense leads with a different corner", h.size() == 1 and g.size() == 1 and (h[0].p - g[0].p).norm() > 0.1f); }
    return fails ? 1 : 0;
}
