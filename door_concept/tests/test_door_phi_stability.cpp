// Offline harness for the leaf-angle (phi) channel: DoorBelief::phi_field_loglik + rc::door::phi_posterior_step
// (door_phi_posterior.h) — the SAME code DoorFitter::estimate_phi runs — on a synthetic ray-cast scene with
// the real geometry of the live failure: a 0.11 m wall, a 1.0 m doorway, a 2.1 m leaf hinged on the FAR face
// (the Webots Door sits at room_1's face of the wall), a bpearl-like downward dome at 0.67 m.
//
//   (0) equivalence: phi_field_loglik with beta forced to 1 == log(phi_field_likelihood) on the old nuisance
//   (a) the try36-40 failure: the crossed door carried into room_2, leaf ~95 deg OUT of room_2 (beside the
//       doorway, behind the wall), seen from 1.4-2.1 m while the robot wanders, with 3-100 % of the rays
//       kept per frame, and the model aperture misplaced 0 / -0.35 / +0.35 m along the wall (live: 0.23-0.40)
//       → P(passable) stays high and stable. The OLD pipeline is run beside it and its flips are printed.
//   (b) a SHUT door seen frontally → P(passable) low, from an unknown prior AND from a wrong carried "open"
//   (c) a door opened INTO the robot's room, leaf visible → high, from an unknown prior
//   (d) a door CLOSING during observation (95 deg → 0 at 0.8 rad/s) → P(passable) follows within a few frames
//   (e) a COMMANDED swing (the phi_cmd prior moving 0 → 90 deg at 0.5 rad/s with the leaf) → follows
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tests/test_door_phi_stability.cpp src/door_belief.cpp -o /tmp/test_door_phi_stability && /tmp/test_door_phi_stability
// PHI_TRACE=1 prints section (c) frame by frame.
#include <algorithm>
#include <chrono>
#include <format>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <print>
#include <random>
#include <string>
#include <vector>

#include "../src/door_belief.h"
#include "../src/door_phi_posterior.h"
#include "../src/door_passability.h"   // passable_prob — the SAME marginalisation the agent publishes

using V2 = Eigen::Vector2f;
using V3 = Eigen::Vector3f;

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

// ─── the scene ───────────────────────────────────────────────────────────────────────────────────
// Room_2 frame: the wall's room_2 face is y = 0, the wall fills y in [-T, 0], room_2 is y > 0 (the robot),
// room_1 is y < -T. The real doorway is x in [0, W].
struct Box { V2 c; V2 ex; float hx, hy, z0, z1; };
static Box aabb(float x0, float x1, float y0, float y1, float z0, float z1)
{ return {V2(0.5f * (x0 + x1), 0.5f * (y0 + y1)), V2(1, 0), 0.5f * (x1 - x0), 0.5f * (y1 - y0), z0, z1}; }
static float ray_box(const V3& o, const V3& u, const Box& b)
{
    const V2 ey(-b.ex.y(), b.ex.x());
    const V3 c(b.c.x(), b.c.y(), 0.5f * (b.z0 + b.z1));
    const V3 ax[3] = {V3(b.ex.x(), b.ex.y(), 0), V3(ey.x(), ey.y(), 0), V3(0, 0, 1)};
    const float h[3] = {b.hx, b.hy, 0.5f * (b.z1 - b.z0)};
    const V3 d = c - o;
    float tmin = 0, tmax = 1e9f;
    for (int i = 0; i < 3; ++i)
    {
        const float e = ax[i].dot(d), f = ax[i].dot(u);
        if (std::abs(f) > 1e-7f)
        {
            float t1 = (e - h[i]) / f, t2 = (e + h[i]) / f;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1); tmax = std::min(tmax, t2);
            if (tmin > tmax) return -1;
        }
        else if (-e - h[i] > 0 or -e + h[i] < 0) return -1;
    }
    return tmin > 0 ? tmin : -1;
}

struct Scene
{
    float T = 0.11f, W = 1.0f, H = 2.1f;
    float phi = 1.66f;        // TRUE leaf angle
    float swing = -1.0f;      // -1: into room_1 (OUT of the robot's room), hinged on the far face; +1: into room_2
    bool  hinge_far = false;  // false: hinged on the x = 0 jamb; true: on the x = W jamb (Task 11: DOOR_1 from room_2)
    std::vector<Box> boxes() const
    {
        std::vector<Box> b;
        b.push_back(aabb(-6, 0, -T, 0, 0, 3));              // wall left of the doorway
        b.push_back(aabb(W, 6, -T, 0, 0, 3));               // wall right of it
        b.push_back(aabb(0, W, -T, 0, H, 3));               // lintel
        b.push_back(aabb(-6, 6, -4.2f, -4.0f, 0, 3));       // room_1: far wall + side walls
        b.push_back(aabb(-3.2f, -3.0f, -4.2f, -T, 0, 3));
        b.push_back(aabb(4.0f, 4.2f, -4.2f, -T, 0, 3));
        // The leaf: hinged on the x = 0 jamb at the face of the wall it opens toward (the Webots Door sits
        // at room_1's face), 4 cm thick, centred on its swing line.
        const float y_h = swing < 0 ? -T : 0.0f;
        const V2 dir(hinge_far ? -std::cos(phi) : std::cos(phi), swing * std::sin(phi));
        const float lw = W - 0.01f;
        b.push_back({V2(hinge_far ? W : 0.0f, y_h) + 0.5f * lw * dir, dir, 0.5f * lw, 0.02f, 0.01f, H - 0.01f});
        return b;
    }
};

// A bpearl-like dome: 720 azimuths x 32 rings from horizontal to straight down, 1 cm range noise.
static std::vector<V3> scan(const Scene& s, const V3& o, std::mt19937& rng, float keep = 1.0f)
{
    std::normal_distribution<float> n(0, 0.01f);
    std::uniform_real_distribution<float> U(0, 1);
    const auto bx = s.boxes();
    std::vector<V3> pts;
    for (int a = 0; a < 720; ++a)
        for (int e = 0; e < 32; ++e)
        {
            if (keep < 1.0f and U(rng) > keep) continue;
            const float az = a * 0.5f * std::numbers::pi_v<float> / 180.f;
            const float el = -e * (90.f / 31.f) * std::numbers::pi_v<float> / 180.f;
            const V3 u(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            float t = 50;
            if (u.z() < -1e-6f) t = std::min(t, -o.z() / u.z());
            for (const auto& b : bx) { const float tb = ray_box(o, u, b); if (tb > 0) t = std::min(t, tb); }
            if (t > 8.0f) continue;
            pts.push_back(o + (t + n(rng)) * u);
        }
    return pts;
}

// ─── the model door ──────────────────────────────────────────────────────────────────────────────
static rc::DoorBelief make_belief(float model_slide, float swing,
                                  rc::door::HingeSide hinge = rc::door::HingeSide::Near)
{
    rc::DoorBeliefParams p;
    p.wall_O = {-2, 0}; p.wall_u = {1, 0}; p.thickness = 0.05f; p.floor_z = 0;
    p.leaf = {0, hinge, swing};
    return rc::DoorBelief({2.0f + model_slide, 1.0f, 2.1f}, p);
}

constexpr float PHI_MIN = 0.0f, PHI_MAX = 2.0944f, PHI_STEP = 0.0873f, DT = 0.1f, NEEDED = 0.70f;
constexpr int   NSTEP = 25;

struct Tracker   // the carried belief, exactly as DoorInstance carries it
{
    float hinge_llr = 0.0f;   // log P(believed hinge) / P(other) — DoorInstance::hinge_llr
    float phi_est = 0.0f, phi_sigma = PHI_MAX;
    bool  first = true;
    float p = 0.0f;
    float ninf = 0.0f, beta = 1.0f, lat = 0.0f;
};

// One cycle of the NEW channel: select → phi_field_loglik → posterior → rate limit → P(passable).
static void step_new(Tracker& t, rc::DoorBelief& b, const V3& o, const std::vector<V3>& pts,
                     float prior_mean_override = std::numeric_limits<float>::quiet_NaN(), float swing_rate = 2.0f)
{
    auto nz  = rc::door::phi_field_nuisance(b.aperture());
    nz.lat_sigma = std::sqrt(std::max(1e-8f, b.covariance()(0, 0)));   // as DoorFitter::estimate_phi does
    const auto sel = rc::door::select_doorway_rays(b.aperture(), o, pts, std::max(-nz.lat_lo, nz.lat_hi), 1.0f);
    const float dt = t.first ? std::numeric_limits<float>::infinity() : DT;
    t.first = false;
    const float sp = rc::door::phi_prior_sigma(t.phi_sigma, swing_rate, dt, PHI_STEP, PHI_MAX);
    const float prior = std::isnan(prior_mean_override) ? t.phi_est : prior_mean_override;
    std::vector<std::pair<float, float>> ll;
    nz.other_hinge = true;   // as DoorFitter::estimate_phi: the hinge is marginalised
    if (not sel.rays.endpoints.empty())
    {
        auto fr = b.phi_field_loglik(sel.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        t.ninf = fr.n_inf; t.beta = fr.beta; t.lat = fr.lat_map;
        auto hm = rc::door::phi_hinge_marginal(fr.loglik, fr.loglik_other, t.hinge_llr, prior, sp);
        ll = std::move(hm.loglik);
        t.hinge_llr = hm.hinge_llr;
        rc::door::adopt_map_hinge(b, t.hinge_llr);
    }
    const auto post = rc::door::phi_posterior_step(ll, PHI_MIN, PHI_MAX, NSTEP, prior, sp);
    t.phi_sigma = post.sigma;
    t.phi_est   = std::clamp(post.mean, PHI_MIN, PHI_MAX);
    b.set_leaf_phi(t.phi_est);
    t.p = rc::door::passable_prob(post.curve, b.aperture().w, NEEDED).value_or(-1.0f);
}

// The OLD channel, reproduced verbatim for the before/after comparison: ±6 cm slide, summed likelihood,
// linear-float posterior whose all-zero case reset sigma to the full range.
static void step_old(Tracker& t, rc::DoorBelief& b, const V3& o, const std::vector<V3>& pts)
{
    const auto sel = rc::door::select_doorway_rays(b.aperture(), o, pts, 0.0f, 1.0f);
    const float dt = t.first ? std::numeric_limits<float>::infinity() : DT;
    t.first = false;
    const float sp = rc::door::phi_prior_sigma(t.phi_sigma, 2.0f, dt, PHI_STEP, PHI_MAX);
    std::vector<float> lidar(NSTEP, 0.0f);
    if (not sel.rays.endpoints.empty())
    {
        rc::DoorBelief::FieldNuisance nz;
        const auto c = b.phi_field_likelihood(sel.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        for (int k = 0; k < NSTEP; ++k) lidar[k] = c[k].second;
    }
    const double lsum = std::accumulate(lidar.begin(), lidar.end(), 0.0);
    double cv = 0.0;
    if (lsum > 1e-12)
    {
        const double mu = lsum / NSTEP; double v = 0;
        for (float x : lidar) v += (x - mu) * (x - mu);
        cv = std::sqrt(v / NSTEP) / mu;
    }
    const bool measured = cv > 0.10;
    std::vector<std::pair<float, float>> curve;
    double num = 0, den = 0;
    for (int k = 0; k < NSTEP; ++k)
    {
        const float phi = PHI_MIN + (PHI_MAX - PHI_MIN) * k / (NSTEP - 1);
        const float d = (phi - t.phi_est) / sp;
        const double prior = std::exp(-0.5 * d * d);
        const double s_n = lsum > 1e-12 ? lidar[k] / lsum : 0.0;
        const float w = static_cast<float>((measured ? s_n : 1.0) * prior);
        curve.emplace_back(phi, w); num += phi * w; den += w;
    }
    if (den > 1e-12)
    {
        const float mean = num / den; double var = 0;
        for (auto& [phi, w] : curve) var += w * (phi - mean) * (phi - mean);
        t.phi_sigma = std::sqrt(var / den);
        t.phi_est += std::clamp(mean - t.phi_est, -2.0f * (std::isfinite(dt) ? dt : 0.1f), 2.0f * (std::isfinite(dt) ? dt : 0.1f));
    }
    else
        t.phi_sigma = PHI_MAX - PHI_MIN;
    t.phi_est = std::clamp(t.phi_est, PHI_MIN, PHI_MAX);
    b.set_leaf_phi(t.phi_est);
    t.p = rc::door::passable_prob(curve, b.aperture().w, NEEDED).value_or(-1.0f);
}

static int flips(const std::vector<float>& p)
{
    int n = 0;
    for (std::size_t i = 1; i < p.size(); ++i)
        if ((p[i - 1] >= 0.5f) != (p[i] >= 0.5f)) ++n;
    return n;
}

int main()
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: reproduce the agent's locale (nothing here parses text, but stay honest)
    std::mt19937 rng(7);

    // ── (0) equivalence with the field likelihood the agent used before ──────────────────────────────
    std::print("(0) phi_field_loglik == log(phi_field_likelihood) when untempered, on the old nuisance\n");
    {
        Scene sc; sc.phi = 0.6f; sc.swing = +1.0f;
        auto b = make_belief(0.0f, +1.0f);
        float worst = 0.0f;
        for (const V3& o : {V3(0.5f, 1.8f, 0.67f), V3(-0.3f, 1.4f, 0.67f), V3(1.2f, 2.2f, 0.67f)})
        {
            const auto pts = scan(sc, o, rng);
            const auto sel = rc::door::select_doorway_rays(b.aperture(), o, pts, 0.0f, 1.0f);
            rc::DoorBelief::FieldNuisance nz; nz.ray_icc = 0.0f;   // beta = 1
            const auto a = b.phi_field_likelihood(sel.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
            const auto c = b.phi_field_loglik(sel.rays, PHI_MIN, PHI_MAX, NSTEP, nz);
            for (int k = 0; k < NSTEP; ++k)
                if (a[k].second > 1e-6f)   // compare where the old float curve has not underflowed
                    worst = std::max(worst, std::abs(std::log(a[k].second) - c.loglik[k].second));
        }
        check(worst < 1e-2f, std::format("max |log L_old - L_new| = {:.2e} nats", worst));
    }

    // ── (a) the carried open-out leaf, seen from room_2 ──────────────────────────────────────────────
    std::print("\n(a) carried leaf at 95 deg OUT of room_2, robot wandering 1.4-2.1 m in front, 3-100% of rays kept\n");
    for (const float slide : {0.0f, -0.35f, +0.35f})
    {
        Scene sc; sc.phi = 1.66f; sc.swing = -1.0f;
        auto bn = make_belief(slide, -1.0f), bo = make_belief(slide, -1.0f);
        Tracker tn, to;
        tn.phi_est = to.phi_est = 1.66f; tn.phi_sigma = to.phi_sigma = 0.035f;   // remembered from the crossing
        tn.first = to.first = false;
        std::vector<float> pn, po;
        std::uniform_real_distribution<float> U(0, 1);
        float tmax_ms = 0;
        for (int f = 0; f < 60; ++f)
        {
            const float lat  = 0.5f + 0.9f * std::sin(0.21f * f) ;             // across the doorway and beyond
            const float dist = 1.75f + 0.35f * std::cos(0.13f * f);
            const float keep = f % 7 == 3 ? 0.03f : 0.1f + 0.9f * U(rng);       // some near-empty frames
            const V3 o(lat, dist, 0.67f);
            const auto pts = scan(sc, o, rng, keep);
            const auto t0 = std::chrono::steady_clock::now();
            step_new(tn, bn, o, pts);
            tmax_ms = std::max(tmax_ms, std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count());
            step_old(to, bo, o, pts);
            pn.push_back(tn.p); po.push_back(to.p);
        }
        const float pmin = *std::ranges::min_element(pn);
        std::print("  slide {:+.2f} m: NEW min P {:.3f}, flips {}, final phi {:.0f} deg sigma {:.0f} deg, n_inf {:.0f} beta {:.3f} lat {:+.2f} | OLD min P {:.3f}, flips {} | worst {:.1f} ms/frame\n",
                   slide, pmin, flips(pn), tn.phi_est * 57.3f, tn.phi_sigma * 57.3f, tn.ninf, tn.beta, tn.lat,
                   *std::ranges::min_element(po), flips(po), tmax_ms);
        check(pmin > 0.8f and flips(pn) == 0,
              std::format("slide {:+.2f}: P(passable) stays high and never flips (min {:.3f})", slide, pmin));
    }

    // ── (b) a SHUT door, frontally ───────────────────────────────────────────────────────────────────
    std::print("\n(b) SHUT door seen frontally from 1.8 m\n");
    for (const float start : {-1.0f, 1.66f})   // -1 = no prior (first look); 1.66 = a wrong carried "open"
    {
        Scene sc; sc.phi = 0.0f; sc.swing = -1.0f;
        auto b = make_belief(0.0f, -1.0f);
        Tracker t;
        if (start >= 0) { t.phi_est = start; t.phi_sigma = 0.035f; t.first = false; }
        int first_low = -1;
        for (int f = 0; f < 20; ++f)
        {
            const V3 o(0.5f + 0.2f * std::sin(0.3f * f), 1.8f, 0.67f);
            step_new(t, b, o, scan(sc, o, rng, 0.5f));
            if (first_low < 0 and t.p < 0.05f) first_low = f;
        }
        std::print("  prior {}: P(passable) {:.4f} after 20 frames, first < 0.05 at frame {}, phi {:.0f} deg\n",
                   start < 0 ? "none" : "open 95deg", t.p, first_low, t.phi_est * 57.3f);
        check(t.p < 0.05f and first_low >= 0 and first_low <= 8,
              std::format("shut door reads shut (P={:.4f}, within {} frames)", t.p, first_low));
    }

    // ── (c) door opened INTO the robot's room, leaf visible ───────────────────────────────────────────
    std::print("\n(c) door open 90 deg INTO room_2 (leaf in view), first look\n");
    {
        Scene sc; sc.phi = 1.57f; sc.swing = +1.0f;
        auto b = make_belief(0.0f, +1.0f);
        Tracker t;
        int first_high = -1;
        std::vector<float> p;
        for (int f = 0; f < 20; ++f)
        {
            const V3 o(1.4f + 0.3f * std::sin(0.3f * f), 1.9f, 0.67f);
            step_new(t, b, o, scan(sc, o, rng, 0.5f));
            p.push_back(t.p);
            if (std::getenv("PHI_TRACE")) std::print("    f{} P {:.3f} phi {:.0f} sigma {:.0f} n_inf {:.1f} beta {:.3f} lat {:+.2f} llr {:+.2f} hinge {}\n", f, t.p, t.phi_est * 57.3f, t.phi_sigma * 57.3f, t.ninf, t.beta, t.lat, t.hinge_llr, b.params().leaf.hinge == rc::door::HingeSide::Far ? "far" : "near");
            if (first_high < 0 and t.p > 0.9f) first_high = f;
        }
        std::print("  P(passable) {:.3f}, first > 0.9 at frame {}, phi {:.0f} deg, flips after it {}\n",
                   t.p, first_high, t.phi_est * 57.3f,
                   first_high >= 0 ? flips(std::vector<float>(p.begin() + first_high, p.end())) : -1);
        const int fl = first_high >= 0 ? flips(std::vector<float>(p.begin() + first_high, p.end())) : -1;
        check(t.p > 0.9f and first_high >= 0 and first_high <= 8 and fl == 0, "open-into-room door reads open, and stays open");
    }

    // ── (d) the door CLOSES while the robot watches ──────────────────────────────────────────────────
    std::print("\n(d) door closing 95 deg -> 0 at 0.8 rad/s (truth passable while phi > 72.5 deg)\n");
    for (const float swing : {-1.0f, +1.0f})
    {
        Scene sc; sc.swing = swing;
        auto b = make_belief(0.0f, swing);
        Tracker t; t.phi_est = 1.66f; t.phi_sigma = 0.035f; t.first = false;
        int truth_shut_at = -1, model_shut_at = -1;
        const float phi_pass = std::acos(1.0f - NEEDED);   // clear span = w (1 - cos phi) = needed
        for (int f = 0; f < 50; ++f)
        {
            sc.phi = std::max(0.0f, 1.66f - 0.8f * DT * std::max(0, f - 5));
            const V3 o(0.6f, 1.8f, 0.67f);
            step_new(t, b, o, scan(sc, o, rng, 0.5f));
            if (truth_shut_at < 0 and sc.phi < phi_pass) truth_shut_at = f;
            if (model_shut_at < 0 and t.p < 0.5f) model_shut_at = f;
        }
        std::print("  swing {:+.0f}: truth not passable from frame {}, P<0.5 from frame {} (lag {} frames), final P {:.4f} phi {:.0f} deg\n",
                   swing, truth_shut_at, model_shut_at, model_shut_at - truth_shut_at, t.p, t.phi_est * 57.3f);
        check(model_shut_at >= 0 and model_shut_at - truth_shut_at <= 10 and t.p < 0.05f,
              std::format("closing door followed (swing {:+.0f}, lag {} frames)", swing, model_shut_at - truth_shut_at));
    }

    // ── (e) a COMMANDED swing: the phi_cmd prior moves with the leaf ─────────────────────────────────
    std::print("\n(e) commanded open 0 -> 90 deg at 0.5 rad/s, door opening INTO room_2 in step with the command\n");
    {
        Scene sc; sc.swing = +1.0f; sc.phi = 0.0f;
        auto b = make_belief(0.0f, +1.0f);
        Tracker t; t.phi_est = 0.0f; t.phi_sigma = 0.035f; t.first = false;
        int pass_at = -1, truth_at = -1;
        const float phi_pass = std::acos(1.0f - NEEDED);
        for (int f = 0; f < 50; ++f)
        {
            const float cmd = std::min(1.5708f, 0.5f * DT * f);
            sc.phi = cmd;
            const V3 o(0.5f, 1.8f, 0.67f);
            step_new(t, b, o, scan(sc, o, rng, 0.5f), cmd, 0.5f);
            if (truth_at < 0 and sc.phi >= phi_pass) truth_at = f;
            if (pass_at < 0 and t.p > 0.5f) pass_at = f;
        }
        std::print("  truth passable from frame {}, P>0.5 from frame {}, final P {:.3f} phi {:.0f} deg\n",
                   truth_at, pass_at, t.p, t.phi_est * 57.3f);
        check(pass_at >= 0 and std::abs(pass_at - truth_at) <= 6 and t.p > 0.9f, "commanded swing followed");
    }

    // ── (f) phi_reads_open_from_far_side (Task 11, live try69 leg 2) ─────────────────────────────────
    // DOOR_1 seen from room_2: the leaf stands open 90 deg AWAY from the robot (into room_1), hinged on the jamb the
    // model does NOT believe in (room_1's config pin, Near, was the wrong jamb, and the mirror inherited it as the
    // geometry it was). The robot walks from in front of the believed hinge across the doorway to the far side, as
    // the controller's wait-turn did. Truth: phi = 90 deg, passable throughout.
    std::print("\n(f) phi_reads_open_from_far_side: leaf 90 deg AWAY, hinged on the jamb the model does not believe\n");
    // swing +1 is the same door seen from room_1 (leg 1): the leaf swung TOWARD the robot, its own returns dropped by
    // select_doorway_rays (short of the plane), so the hinge must be read from the rays it does or does not block.
    for (const float swing : {-1.0f, +1.0f})
    for (const bool believed_right : {true, false})
    {
        Scene sc; sc.phi = 1.5708f; sc.swing = swing; sc.hinge_far = true;
        std::print("  -- leaf {} the robot\n", swing < 0 ? "AWAY from" : "TOWARD");
        auto b = make_belief(0.0f, swing, believed_right ? rc::door::HingeSide::Far : rc::door::HingeSide::Near);
        Tracker t; t.phi_est = 1.66f; t.phi_sigma = 0.035f; t.first = false;   // remembered from the crossing
        std::vector<float> p;
        for (int f = 0; f < 40; ++f)
        {
            const V3 o(-0.4f + 1.8f * f / 39.0f, 1.6f, 0.67f);   // from beside the believed hinge to past the far jamb
            step_new(t, b, o, scan(sc, o, rng, 0.5f));
            p.push_back(t.p);
            if (std::getenv("PHI_TRACE")) std::print("    f{} x {:+.2f} P {:.3f} phi {:.0f} sigma {:.0f} lat {:+.2f} hinge {}\n", f, o.x(), t.p, t.phi_est * 57.3f, t.phi_sigma * 57.3f, t.lat, b.params().leaf.hinge == rc::door::HingeSide::Far ? "far" : "near");
        }
        const float pmin = *std::ranges::min_element(p);
        std::print("  believed hinge {}: min P {:.3f}, final P {:.3f}, phi {:.0f} deg, hinge now {}\n",
                   believed_right ? "right" : "WRONG", pmin, t.p, t.phi_est * 57.3f,
                   b.params().leaf.hinge == rc::door::HingeSide::Far ? "far (true)" : "near");
        check(b.params().leaf.hinge == rc::door::HingeSide::Far, "the hinge ends on the true jamb");
        check(std::abs(t.phi_est - 1.5708f) < 0.1745f and pmin > 0.8f,
              std::format("believed hinge {}: phi {:.0f} deg within 90 +- 10, P(passable) min {:.3f} > 0.8",
                          believed_right ? "right" : "wrong", t.phi_est * 57.3f, pmin));
    }

    std::print("\n{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
