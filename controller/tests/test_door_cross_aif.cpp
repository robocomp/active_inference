// door_cross_aif (src/door_cross_aif.h) vs anchor_cross_command (the 10-18 law) on a kinematic UNICYCLE (Shadow is
// differential): a 1.0 m opening in a wall, Shadow's footprint, anchor noise at the logged sigma. Spec §5 test 1/2.
// Build+run from controller/ (relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I../common -Isrc $(pkg-config --cflags Qt6Core) -fPIC
//     tests/test_door_cross_aif.cpp -o /tmp/tdca -lfastdds -lfastcdr $(pkg-config --libs Qt6Core) && /tmp/tdca
#include "../src/door_cross_aif.h"

#include <cmath>
#include <numbers>
#include <print>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace
{
constexpr float kHalfW = 0.30f, kHalfL = 0.33f;
float kDoorW = 1.0f, kNoiseXY = 0.02f, kNoiseTh = 0.0175f;
constexpr float kVmax = 0.35f, kWmax = 1.0f, kDt = 0.05f, kStandoff = 2.0f, kTmax = 60.f;
const float kRBody = std::hypot(kHalfW, kHalfL);

struct Outcome { bool through = false, contact = false, reversed = false; float t = 0.f, min_clear = 1e9f; };

// world: door centre (0,0), wall along x, the robot starts at y < 0 and goes to +y. Robot pose (x, y, phi); forward =
// (-sin phi, cos phi) (robot +y), side = (cos phi, sin phi) (robot +x).
template <class Law>
Outcome run(float x0, float y0, float phi0, Law law, std::mt19937& rng)
{
    std::normal_distribution<float> nxy(0.f, kNoiseXY), nth(0.f, kNoiseTh);
    float x = x0, y = y0, phi = phi0;
    Outcome o;
    for (float t = 0.f; t < kTmax; t += kDt)
    {
        // the anchor in the robot frame (with noise): A = R(-phi)(c - p), psi = -phi
        const float c = std::cos(phi), s = std::sin(phi);
        const Eigen::Vector2f d(-x, -y);
        rc::interaction::RelPose r;
        r.x = Eigen::Vector3f(c * d.x() + s * d.y() + nxy(rng), -s * d.x() + c * d.y() + nxy(rng), -phi + nth(rng));
        r.P = Eigen::Vector3f(kNoiseXY * kNoiseXY, kNoiseXY * kNoiseXY, kNoiseTh * kNoiseTh).asDiagonal();
        const rc::AnchorCommand cmd = law(r);
        if (cmd.adv < -1e-4f) o.reversed = true;
        x += cmd.adv * -std::sin(phi) * kDt;
        y += cmd.adv * std::cos(phi) * kDt;
        phi += cmd.rot * kDt;
        // contact: any footprint edge crossing the wall line outside the opening; clearance at the line otherwise
        const float cc = std::cos(phi), ss = std::sin(phi);
        const auto corner = [&](float bx, float by) { return Eigen::Vector2f(x + cc * bx - ss * by, y + ss * bx + cc * by); };
        const Eigen::Vector2f k[4] = {corner(-kHalfW, -kHalfL), corner(kHalfW, -kHalfL), corner(kHalfW, kHalfL), corner(-kHalfW, kHalfL)};
        for (int i = 0; i < 4; ++i)
        {
            const auto &p = k[i], &q = k[(i + 1) % 4];
            if ((p.y() > 0.f) == (q.y() > 0.f)) continue;
            const float xi = p.x() + (q.x() - p.x()) * (0.f - p.y()) / (q.y() - p.y());
            o.min_clear = std::min(o.min_clear, 0.5f * kDoorW - std::abs(xi));
            if (std::abs(xi) > 0.5f * kDoorW) o.contact = true;
        }
        if (y > kRBody) { o.through = true; o.t = t; return o; }   // THROUGH (drive_door_route's test, true pose)
    }
    o.t = kTmax;
    return o;
}

template <class Law>
void sweep(const char* name, Law law)
{
    std::mt19937 rng(7);
    int n = 0, ok = 0, contact = 0, stalled = 0, reversed = 0;
    float t_sum = 0.f, clear_min = 1e9f, t_max = 0.f;
    std::vector<std::string> bad;
    for (const float y0 : {-0.8f, -1.4f, -2.0f})
        for (const float x0 : {-0.5f, -0.25f, 0.f, 0.25f, 0.5f})
            for (const float deg : {-45.f, -20.f, 0.f, 20.f, 45.f})
            {
                const auto o = run(x0, y0, deg * std::numbers::pi_v<float> / 180.f, law, rng);
                ++n;
                if (o.reversed) ++reversed;
                if (o.contact) { ++contact; bad.push_back(std::format("contact x0 {:+.2f} y0 {:+.1f} {:+.0f}deg", x0, y0, deg)); }
                else if (not o.through) { ++stalled; bad.push_back(std::format("stall x0 {:+.2f} y0 {:+.1f} {:+.0f}deg", x0, y0, deg)); }
                else { ++ok; t_sum += o.t; t_max = std::max(t_max, o.t); clear_min = std::min(clear_min, o.min_clear); }
            }
    std::println("{:28s} {}/{} through clean | contact {} stall {} reversed {} | t mean {:.1f} s max {:.1f} s | min clearance {:.3f} m",
                 name, ok, n, contact, stalled, reversed, ok ? t_sum / ok : 0.f, t_max, clear_min);
    for (std::size_t i = 0; i < std::min<std::size_t>(bad.size(), 8); ++i) std::println("    {}", bad[i]);
}
}  // namespace

int main()
{
  for (const auto [w, nxy, nth] : {std::tuple{1.0f, 0.02f, 0.0175f}, std::tuple{0.9f, 0.05f, 0.052f}})
  {
    kDoorW = w; kNoiseXY = nxy; kNoiseTh = nth;
    std::println("-- door {:.1f} m, anchor noise {:.0f} cm / {:.0f} deg", w, nxy * 100, nth * 57.3);
    sweep("old law (anchor_cross_command)", [](const rc::interaction::RelPose& r)
          { return rc::anchor_cross_command(r, kDoorW, kHalfW, kStandoff, kVmax, kWmax, false, {}, kHalfL); });
    sweep("AIF point-ahead (door_cross_aif)", [](const rc::interaction::RelPose& r)
          { return rc::door_cross_aif(r, kDoorW, {}, rc::DoorCrossAifBody{kHalfW, kHalfL}, kStandoff, kVmax, kWmax, kDt).cmd; });
  }
    // ── warm24 (2026-11-01): 1.27 m in front of DOOR_1 (clear 0.88 m), turned 52 deg off the normal, e ~ 0: the law sent adv 0, rot 0
    //    for minutes — P(fit) 0 because of the heading, and nothing in the objective asked for the heading. Must cross, no contact.
    int fails = 0;
    {
        kDoorW = 0.88f;
        std::mt19937 rng(11);
        for (const float deg : {52.f, -52.f, 70.f, -70.f})
        {
            const auto o = run(0.f, -1.27f, deg * std::numbers::pi_v<float> / 180.f, [](const rc::interaction::RelPose& r)
                { return rc::door_cross_aif(r, kDoorW, {}, rc::DoorCrossAifBody{kHalfW, kHalfL}, kStandoff, kVmax, kWmax, kDt).cmd; }, rng);
            const bool ok = o.through and not o.contact;
            std::println("  warm24 start {:+.0f} deg: through {} contact {} t {:.1f} s -> {}", deg, o.through, o.contact, o.t, ok ? "ok" : "FAIL");
            if (not ok) ++fails;
        }
        kDoorW = 1.0f;
    }
    // ── warm24 leg 2: THROUGH (body wholly past the opening), then the drive-in stalled ~0.9 m past the door — the anchor seen
    //    from inside re-initialised, heading_err grew to -59 deg, and P(fit AT ENTRY) (0.003-0.09) still scaled the progress.
    //    The entry has happened: P(fit) no longer gates the drive-in.
    {
        const float x = 0.f, y = 0.9f, phi = -59.f * std::numbers::pi_v<float> / 180.f;   // centre 0.9 m past, turned 59 deg
        const float c = std::cos(phi), sn = std::sin(phi);
        const Eigen::Vector2f d(-x, -y);
        rc::interaction::RelPose r;
        r.x = Eigen::Vector3f(c * d.x() + sn * d.y(), -sn * d.x() + c * d.y(), -phi);
        r.P = Eigen::Vector3f(4e-4f, 4e-4f, 3e-4f).asDiagonal();
        // DOOR_2 swings into the room entered: after the crossing its leaf stands on the robot's side, beside it (live: P(MAP) 1)
        rc::interaction::AnchorLeaf leaf;
        leaf.hinge = Eigen::Vector2f(0.44f, 0.02f);
        leaf.tip = Eigen::Vector2f(0.44f - 0.29f, 0.02f + 0.86f);
        leaf.phi = 1.90f; leaf.sigma_phi = 0.17f; leaf.p = 1.f;
        const std::vector<rc::interaction::AnchorLeaf> leaves{leaf};
        const auto st = rc::door_cross_aif(r, 0.88f, leaves, rc::DoorCrossAifBody{kHalfW, kHalfL}, kStandoff, kVmax, kWmax, kDt);
        std::println("  drive-in 0.9 m past, -59 deg, leaf beside: p_fit {:.3f} adv {:+.3f} rot {:+.3f}", st.p_fit, st.cmd.adv, st.cmd.rot);
        if (not (st.p_fit > 0.99f and (st.cmd.adv > 0.05f or std::abs(st.cmd.rot) > 0.1f)))
        { ++fails; std::println("  FAIL drive-in held back by the entry fit"); }
    }
    std::println("{}", fails == 0 ? "warm24 cases: ALL PASS" : std::format("warm24 cases: {} FAILED", fails));
    return fails == 0 ? 0 : 1;
}
