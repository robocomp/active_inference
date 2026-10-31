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
    return 0;
}
