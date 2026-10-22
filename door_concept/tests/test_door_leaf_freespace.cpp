// Task L: the leaf's FREE-SPACE evidence — a ray that flew THROUGH the region a leaf at angle phi would occupy is evidence
// AGAINST that phi (the beam model's P(return beyond | surface there) = eps_beyond), a return on the leaf is evidence FOR it —
// and the clear span the planner reads comes from the whole phi posterior, not from its best angle.
// Synthetic world, the agent's own code: DoorBelief::phi_field_loglik (the beam model) -> leaf_channel_step (the filter) ->
// clear_width_expected / passable_logit (what `cross` reads). Rays are simulated against the TRUE leaf (an OBB from leaf_pose),
// the wall with its hole, and a far wall 3.5 m behind it.
//   (a) leaf at 45 deg, sensor facing the door, rays through the opening: phi near 45 deg, clear span <= 0.7 m, within 12 frames
//   (b) fully open (90 deg): span ~1.0 m
//   (c) a SHUT door seen obliquely (60 deg off the normal) and sparsely (<= 20 rays/frame): never read open
//   (d) a door at 45 deg seen obliquely and sparsely: span <= 0.7 m
// Build: g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_leaf_freespace.cpp src/door_belief.cpp -o /tmp/t_fs && /tmp/t_fs
#include <algorithm>
#include <clocale>
#include <cmath>
#include <format>
#include <numbers>
#include <print>
#include <random>
#include <string>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_leaf_channel.h"
#include "../src/door_passability.h"
#include "../src/door_phi_posterior.h"

using V2 = Eigen::Vector2f;
using V3 = Eigen::Vector3f;
static int g_fail = 0;
static void check(bool ok, const std::string& what) { std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what); if (not ok) ++g_fail; }

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f;
constexpr int NSTEP = 25;
constexpr float W = 1.0f, H = 2.1f, NEEDED = 0.70f;

static rc::DoorBelief make_belief()
{
    rc::DoorBeliefParams p;
    p.thickness = 0.05f; p.floor_z = 0.0f;
    p.wall_u = V2(1, 0);
    p.wall_O = V2(-0.5f * W, 0.0f);
    p.leaf = {1.5708f, rc::door::HingeSide::Near, +1.0f};
    return rc::DoorBelief({0.0f, W, H}, p);
}

// First-hit range of a ray against: the wall (plane y=0 outside the opening), the true leaf, the far wall y=-3.5.
static float cast(const rc::DoorBelief& truth, float phi, const V3& o, const V3& u)
{
    float best = 1e9f;
    const auto ap = truth.aperture();
    // wall with the hole
    if (std::abs(u.y()) > 1e-6f)
    {
        const float t = -o.y() / u.y();
        if (t > 0.0f)
        {
            const V3 P = o + t * u;
            const bool hole = P.x() >= -0.5f * W and P.x() <= 0.5f * W and P.z() >= 0.0f and P.z() <= H;
            if (not hole) best = std::min(best, t);
        }
        const float tf = (-3.5f - o.y()) / u.y();
        if (tf > 0.0f) best = std::min(best, tf);
    }
    // leaf: slab test in the leaf's own axes
    rc::door::LeafState lf = truth.params().leaf;
    lf.phi = phi;
    const auto L = rc::door::leaf_pose(ap, lf);
    const float e[3] = {L.ex.dot(o.head<2>() - L.centre_xy), L.ey.dot(o.head<2>() - L.centre_xy), o.z() - L.centre_z};
    const float f[3] = {L.ex.dot(u.head<2>()), L.ey.dot(u.head<2>()), u.z()};
    const float half[3] = {L.half_w, L.half_t, L.half_h};
    float tmin = 0.0f, tmax = 1e9f;
    bool ok = true;
    for (int a = 0; a < 3 and ok; ++a)
    {
        if (std::abs(f[a]) > 1e-6f)
        {
            float t1 = (-e[a] - half[a]) / f[a], t2 = (-e[a] + half[a]) / f[a];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1); tmax = std::min(tmax, t2);
            if (tmin > tmax) ok = false;
        }
        else if (std::abs(e[a]) > half[a]) ok = false;
    }
    if (ok and tmin > 0.0f) best = std::min(best, tmin);
    return best;
}

struct Frame { rc::ai::LidarRays rays; };
static Frame scan(const rc::DoorBelief& truth, float phi, V2 sensor, int n_rays, std::mt19937& g)
{
    Frame fr;
    fr.rays.origin = V3(sensor.x(), sensor.y(), 1.0f);
    fr.rays.max_range_m = 8.0f; fr.rays.robust_c_m = 0.05f; fr.rays.precision = 400.f;
    // rays aimed at points spread over the doorway column and ±0.4 m beyond it (the selection's slab)
    std::normal_distribution<float> noise(0.0f, 0.015f);
    std::uniform_real_distribution<float> aim(-0.5f * W - 0.3f, 0.5f * W + 0.3f), elev(0.2f, 1.9f);
    for (int i = 0; i < n_rays; ++i)
    {
        const V3 target(aim(g), 0.0f, elev(g));
        const V3 d = target - fr.rays.origin;
        const V3 u = d.normalized();
        const float t = cast(truth, phi, fr.rays.origin, u);
        if (t > fr.rays.max_range_m) continue;
        fr.rays.endpoints.push_back(fr.rays.origin + (t + noise(g)) * u);
    }
    return fr;
}

struct Run { float phi_mean, phi_sigma, span_exp, p_pass; int frames_to_span; };
static Run run(float phi_true, float swing_true, V2 sensor, int rays_per_frame, int frames, unsigned seed)
{
    std::mt19937 g(seed);
    auto belief = make_belief();                       // believed: Near hinge, swing +1 (toward the sensor side)
    auto truth = make_belief();
    {
        auto p = truth.params(); p.leaf.swing = swing_true; truth = rc::DoorBelief({0.0f, W, H}, p);
    }
    rc::door::LeafChannel ch;
    float hl = 0.f, sl = 0.f, pe = std::numeric_limits<float>::quiet_NaN(), ps = std::numeric_limits<float>::quiet_NaN();
    Run out{};
    out.frames_to_span = -1;
    for (int f = 0; f < frames; ++f)
    {
        // small viewpoint jitter: a moving robot
        const V2 at = sensor + V2(0.04f * std::sin(0.31f * f), 0.03f * std::cos(0.17f * f));
        auto fr = scan(truth, phi_true, at, rays_per_frame, g);
        auto nz = rc::door::phi_field_nuisance(belief.aperture());
        nz.other_hinge = true; nz.other_swing = true;
        nz.lat_sigma = 0.05f;
        const auto fl = belief.phi_field_loglik(fr.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        rc::door::LeafChannelInput in;
        in.bb = fl.loglik; in.ob = fl.loglik_other; in.bs = fl.loglik_swing; in.os = fl.loglik_both;
        in.have_sensor = true; in.sensor = at;
        in.dt = f == 0 ? std::numeric_limits<float>::infinity() : 0.1f;
        in.phi_min = PHI_MIN; in.phi_max = PHI_MAX; in.nstep = NSTEP; in.view_rad = PHI_STEP;
        const auto r = rc::door::leaf_channel_step(ch, belief, hl, sl, pe, ps, in);
        out.phi_mean = r.post.mean; out.phi_sigma = r.post.sigma;
        std::vector<std::pair<float, double>> lc;
        for (const auto& [p, l] : r.post.log_curve) lc.emplace_back(p, l);
        out.span_exp = rc::door::clear_width_expected(lc, W);
        out.p_pass = float(rc::door::sigmoid(rc::door::passable_logit(lc, W, NEEDED).value_or(0.0)));
        if (out.frames_to_span < 0 and out.span_exp <= 0.7f and std::abs(out.phi_mean - phi_true) < 0.35f) out.frames_to_span = f + 1;
    }
    return out;
}

int main()
{
    std::setlocale(LC_ALL, "");
    constexpr float D = std::numbers::pi_v<float> / 180.0f;
    auto show = [](const char* what, const Run& r) {
        std::print("  {}: phi {:.1f} deg +- {:.1f}, E[clear span] {:.2f} m, P(passable) {:.2f}, in-band after {} frames\n", what,
                   r.phi_mean / D, r.phi_sigma / D, r.span_exp, r.p_pass, r.frames_to_span); };
    std::print("(a) leaf at 45 deg, sensor 2.5 m in front, 100 rays/frame, 12 frames\n");
    for (float sw : {+1.0f, -1.0f})
    {
        const auto r = run(45.0f * D, sw, V2(0.1f, 2.5f), 100, 12, 7);
        show(sw > 0 ? "swung toward the sensor" : "swung away from the sensor", r);
        check(std::abs(r.phi_mean - 45.0f * D) < 15.0f * D, "phi within 15 deg of 45");
        check(r.span_exp <= 0.7f and r.frames_to_span > 0 and r.frames_to_span <= 12, "E[clear span] <= 0.7 m within 12 frames");
    }
    std::print("(b) fully open, 90 deg\n");
    for (float sw : {+1.0f, -1.0f})
    {
        const auto r = run(90.0f * D, sw, V2(0.1f, 2.5f), 100, 12, 8);
        show(sw > 0 ? "toward" : "away", r);
        check(r.span_exp > 0.85f and r.p_pass > 0.8f, "span ~1.0 m and passable");
    }
    std::print("(c) SHUT door, oblique (sensor 60 deg off the normal), sparse (15 rays/frame), 40 frames\n");
    for (unsigned seed : {1u, 2u, 3u})
    {
        const auto r = run(0.0f, +1.0f, V2(2.6f, 1.5f), 15, 40, seed);
        show("shut oblique", r);
        check(r.span_exp < 0.35f and r.p_pass < 0.25f, "never read open: E[span] < 0.35 m, P(passable) < 0.25");
    }
    std::print("(d) leaf at 45 deg, oblique and sparse, 60 frames\n");
    for (unsigned seed : {1u, 2u})
    {
        const auto r = run(45.0f * D, +1.0f, V2(2.6f, 1.5f), 15, 60, seed);
        show("45 oblique", r);
        check(r.span_exp <= 0.7f, "E[clear span] <= 0.7 m");
    }
    std::print("(e) the expected span is over the posterior, not at its mean angle\n");
    {
        // half the mass on a shut leaf, half on a fully open one: the mean angle is 45 deg (span 0.29 m), the expected span 0.5 m
        std::vector<std::pair<float, double>> bi;
        for (int k = 0; k < NSTEP; ++k) bi.emplace_back(PHI_MIN + (PHI_MAX - PHI_MIN) * float(k) / float(NSTEP - 1), (k == 0 or k == 18) ? 0.0 : -50.0);
        const float e = rc::door::clear_width_expected(bi, W);
        std::print("  bimodal {{0, {:.0f} deg}}: E[span] {:.3f} m vs span at the mean angle {:.3f} m\n", bi[18].first / D, e, rc::door::clear_width(W, 0.5f * bi[18].first));
        check(std::abs(e - 0.5f * rc::door::clear_width(W, bi[18].first)) < 1e-3f, "E[span] = the mixture's mean span");
        check(rc::door::clear_width_expected({}, W) == 0.0f, "an empty curve has span 0, never NaN");
    }
    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", g_fail));
    return g_fail == 0 ? 0 : 1;
}
