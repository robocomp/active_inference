// Task C: the robot body in the aperture is evidence the leaf does not block it (src/door_body_evidence.h), run through the
// leaf channel DoorFitter::estimate_phi runs. Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_body_evidence.cpp src/door_belief.cpp -o /tmp/tdbe && /tmp/tdbe
#include <cmath>
#include <format>
#include <print>
#include <string>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_body_evidence.h"
#include "../src/door_leaf_channel.h"
#include "../src/door_passability.h"

using namespace rc::door;
using V2 = Eigen::Vector2f;
static int g_fail = 0;
static void check(bool ok, const std::string& what) { std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what); if (not ok) ++g_fail; }

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f, W = 1.0f, BODY = 0.60f;
constexpr int NSTEP = 25;

static rc::DoorBelief belief()
{
    rc::DoorBeliefParams p;
    p.thickness = 0.05f; p.floor_z = 0.0f;
    p.wall_u = V2(1, 0);
    p.wall_O = V2(-0.5f * W, 0);
    p.leaf = {1.5708f, HingeSide::Near, +1.0f};
    return rc::DoorBelief({0.0f, W, 2.1f}, p);
}

// Run the leaf channel for `frames` frames from a prior of P(open) ~ 0.7 (mean 60 deg, sigma 30 deg); the body curve is
// `curve` (empty = none). Returns P(passable) after.
static float run(const std::vector<float>& curve, int frames, V2 sensor, float sweep_y = 0.f)
{
    LeafChannel ch;
    auto b = belief();
    float hl = 0.f, sl = 0.f, pe = 1.05f, ps = 0.52f;
    PhiPosteriorStep last;
    for (int f = 0; f < frames; ++f)
    {
        LeafChannelInput in;
        if (not curve.empty())
            for (int k = 0; k < NSTEP; ++k)
            {   // the body blocks (or not) whichever jamb and swing: the SAME curve for all four (hinge x swing) hypotheses
                const float phi = PHI_MIN + (PHI_MAX - PHI_MIN) * float(k) / float(NSTEP - 1);
                in.bb.emplace_back(phi, curve[std::size_t(k)]); in.ob.emplace_back(phi, curve[std::size_t(k)]);
                in.bs.emplace_back(phi, curve[std::size_t(k)]); in.os.emplace_back(phi, curve[std::size_t(k)]);
            }
        in.have_sensor = true; in.sensor = sensor + V2(0.f, sweep_y * float(f) / float(std::max(1, frames - 1)));
        in.dt = f == 0 ? std::numeric_limits<float>::infinity() : 0.05f;
        in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
        last = leaf_channel_step(ch, b, hl, sl, pe, ps, in).post;
    }
    return passable_prob(last.curve, W, BODY).value_or(-1.f);
}

int main()
{
    const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * 0.0004f;   // sigma 2 cm
    std::print("(a) units\n");
    check(body_in_aperture_prob(0.f, 0.f, cov, 0.5f, 0.025f) > 0.7f, "centre mid-aperture, on the wall line: p_in high");
    check(body_in_aperture_prob(0.f, -0.5f, cov, 0.5f, 0.025f) < 1e-6f, "0.5 m in front of the door: p_in ~ 0");
    check(body_in_aperture_prob(0.9f, 0.f, cov, 0.5f, 0.025f) < 1e-6f, "beside the door (0.9 m along the wall): p_in ~ 0");
    check(body_phi_loglik(0.f, W, BODY, PHI_MIN, PHI_MAX, NSTEP).empty(), "p_in = 0: no curve (exactly no change)");
    const auto ll = body_phi_loglik(1.f, W, BODY, PHI_MIN, PHI_MAX, NSTEP);
    check(ll.size() == NSTEP and ll.front() < -5.f and ll.back() == 0.f, "shut leaf (phi 0) ruled out, wide open (120 deg) free");

    std::print("(b) the body passes through the aperture: P(open) rises\n");
    const float base = run({}, 1, V2(0, -3.f));
    const auto in_ap = body_phi_loglik(body_in_aperture_prob(0.f, 0.f, cov, 0.5f, 0.025f), W, BODY, PHI_MIN, PHI_MAX, NSTEP);
    // The body drives through: ~6 frames while its centre is inside the wall band, the sensor moving across it (new viewpoints).
    const float thru = run(in_ap, 6, V2(0, -0.08f), 0.16f);
    std::print("  P(passable): prior {:.3f} -> {:.3f} after 6 frames with the body in the aperture\n", base, thru);
    check(thru > 0.95f and thru > base, "P(passable) rises to ~1");

    std::print("(c) the body beside the door: no change\n");
    const auto beside = body_phi_loglik(body_in_aperture_prob(1.2f, -0.4f, cov, 0.5f, 0.025f), W, BODY, PHI_MIN, PHI_MAX, NSTEP);
    const float same = run(beside, 8, V2(1.2f, -0.4f));
    check(beside.empty() and std::abs(same - run({}, 8, V2(1.2f, -0.4f))) < 1e-6f, "P(passable) identical to the no-evidence run");

    std::print("(d) a pose off by more than the opening cannot claim passage (honest sigma)\n");
    const Eigen::Matrix2f wide = Eigen::Matrix2f::Identity() * 1.0f;   // sigma 1 m
    const float p_wide = body_in_aperture_prob(0.f, 0.f, wide, 0.5f, 0.025f);
    check(p_wide < 0.2f, std::format("sigma 1 m: p_in {:.3f} small", p_wide));
    std::print("{}\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
