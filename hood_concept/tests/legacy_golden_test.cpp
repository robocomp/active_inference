// legacy_golden_test — the hood belief's LEGACY trajectory (the per-frame recursive-Laplace update,
// common/ai_belief/recursive_laplace.h), recorded BEFORE the shared-engine port (Task 10 of the shared-engine plan) so the
// port can prove that Model.Engine = "legacy" reproduces it bit for bit.
//
// The sequence drives one HoodBelief with the default parameters (volatility inference on) through:
//   - 12 updates from the front (camera points with per-point R, the common-mode chain variances, the robust top, the wall
//     behind it, and LiDAR rays on the front face through the legacy first-hit factor; the box hangs, so the underside is seen too);
//   - two predict-only cycles and an inflate_for_age;
//   - 10 updates from the side (a new view; the wall still staged), the last 4 with a bpearl ray-set in lidar_extra;
//   - 6 updates with the box moved 5 cm, so the filter has somewhere to go.
// After each step it records θ, the full Σ, the log-volatility ω and the returned free energy, as doubles converted from
// the floats the belief holds. Compare: EXACT equality of every value (the legacy path must be untouched, not merely close).
//
// Run: ./legacy_golden_test            (compare against golden/legacy_golden.bin)
//      ./legacy_golden_test --record   (rewrite the golden file; only for a change that is MEANT to move the legacy path)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "hood_belief.h"

using namespace rc;

namespace
{
struct Lcg
{
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float uniform(float a)   // in [−a, a]
    {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const double u = double(s >> 11) * (1.0 / 9007199254740992.0);
        return float((2.0 * u - 1.0) * a);
    }
};

struct Box { float cx, cy, H, w, h, yaw; float extent; };   // the hood HANGS: z ∈ [H − extent, H]

Eigen::Vector3f world(const Box& b, float lx, float ly, float z)
{
    const float c = std::cos(b.yaw), s = std::sin(b.yaw);
    return {b.cx + c * lx - s * ly, b.cy + s * lx + c * ly, z};
}

// Points on the faces a camera at `cam` sees (outward normal towards it) and on the top, with noise.
std::vector<Eigen::Vector3f> visible_points(const Box& b, const Eigen::Vector2f& cam, int per, Lcg& rng)
{
    std::vector<Eigen::Vector3f> pts;
    const float hw = 0.5f * b.w, hh = 0.5f * b.h;
    const float c = std::cos(b.yaw), s = std::sin(b.yaw);
    const Eigen::Vector2f d = cam - Eigen::Vector2f(b.cx, b.cy);
    const float lx = c * d.x() + s * d.y(), ly = -s * d.x() + c * d.y();   // camera in the box frame
    for (int i = 0; i < per; ++i)
        for (int j = 0; j < per; ++j)
        {
            const float a = (i + 0.5f) / per, t = (j + 0.5f) / per, z = b.H - b.extent + t * b.extent;
            const float n = rng.uniform(0.01f);
            if (ly < -hh) pts.push_back(world(b, -hw + a * b.w, -hh + n, z));
            if (ly >  hh) pts.push_back(world(b, -hw + a * b.w,  hh + n, z));
            if (lx < -hw) pts.push_back(world(b, -hw + n, -hh + a * b.h, z));
            if (lx >  hw) pts.push_back(world(b,  hw + n, -hh + a * b.h, z));
            pts.push_back(world(b, -hw + a * b.w, -hh + t * b.h, b.H - b.extent + n));   // the underside, seen from below
        }
    return pts;
}

// LiDAR rays from `origin` to points on the face(s) the origin sees, at mid height band.
rc::ai::LidarRays rays_to(const Box& b, const Eigen::Vector3f& origin, int n, Lcg& rng, float precision)
{
    rc::ai::LidarRays r;
    r.origin = origin;
    r.precision = precision;
    const auto pts = visible_points(b, origin.head<2>(), n, rng);
    for (std::size_t k = 0; k < pts.size(); k += 3) r.endpoints.push_back(pts[k]);
    return r;
}

std::vector<double> record_of(const HoodBelief& b, float fe)
{
    std::vector<double> v;
    const auto th = b.state().vec();
    for (int i = 0; i < 6; ++i) v.push_back(double(th(i)));
    const auto& S = b.covariance();
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) v.push_back(double(S(i, j)));
    const auto& om = b.log_volatility();
    for (int i = 0; i < 6; ++i) v.push_back(double(om(i)));
    v.push_back(double(fe));
    return v;
}
}  // namespace

int main(int argc, char** argv)
{
    const bool record = argc > 1 and std::string(argv[1]) == "--record";
    Lcg rng;
    HoodBeliefParams p;
    p.volatility_infer = true;
    p.wall_flush_prior = 1.0f;   // a hood is definitionally wall-mounted (its config's value)
    Box truth{1.20f, -0.60f, 2.05f, 0.90f, 0.50f, 0.0f, p.vertical_extent_m};
    WallRef wall;
    wall.ok = true;
    wall.p = Eigen::Vector2f(0.0f, truth.cy + 0.5f * truth.h);   // the back face (+y) against the wall
    wall.n = Eigen::Vector2f(0.0f, -1.0f);                        // into the room
    wall.length_m = 4.0f;

    HoodBelief b(HoodBeliefState{1.15f, -0.68f, 2.00f, 0.85f, 0.50f, 0.05f}, p);
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));

    std::vector<std::vector<double>> rows;
    const auto frame_from = [&](const Eigen::Vector2f& cam, int per, bool lidar, bool bpearl)
    {
        HoodFrame f;
        f.points = visible_points(truth, cam, per, rng);
        for (std::size_t k = 0; k < f.points.size(); ++k) f.R.push_back(0.0009f * (1.0f + 0.5f * std::abs(rng.uniform(1.0f))));
        f.chain_cov_xx = 4e-4f; f.chain_cov_yy = 3e-4f; f.chain_cov_yaw = 2e-4f; f.chain_cov_size = 1e-4f;
        f.z_top_obs = truth.H - 0.01f;
        f.wall = wall;
        if (lidar) f.lidar = rays_to(truth, Eigen::Vector3f(cam.x(), cam.y(), 1.2f), 6, rng, 2500.0f);
        if (bpearl) f.lidar_extra.push_back(rays_to(truth, Eigen::Vector3f(cam.x(), cam.y(), 0.3f), 4, rng, 1500.0f));
        return f;
    };
    const Eigen::Vector2f front(1.25f, -2.6f), side(3.0f, -0.7f);
    for (int k = 0; k < 12; ++k) rows.push_back(record_of(b, b.update(frame_from(front, 6, k % 3 != 2, false))));
    b.predict(); rows.push_back(record_of(b, 0.0f));
    b.predict(); rows.push_back(record_of(b, 0.0f));
    b.inflate_for_age(2.5f, 0.1f); rows.push_back(record_of(b, 0.0f));
    for (int k = 0; k < 10; ++k) rows.push_back(record_of(b, b.update(frame_from(side, 6, true, k >= 6))));
    truth.cx += 0.05f;
    for (int k = 0; k < 6; ++k) rows.push_back(record_of(b, b.update(frame_from(front, 6, true, false))));

    if (record)
    {
        std::ofstream out(GOLDEN_FILE, std::ios::binary);
        const std::uint64_t n = rows.size(), m = rows.front().size();
        out.write(reinterpret_cast<const char*>(&n), sizeof n);
        out.write(reinterpret_cast<const char*>(&m), sizeof m);
        for (const auto& r : rows) out.write(reinterpret_cast<const char*>(r.data()), std::streamsize(m * sizeof(double)));
        std::printf("recorded %llu rows x %llu values to %s\n", (unsigned long long)n, (unsigned long long)m, GOLDEN_FILE);
        return out.good() ? 0 : 1;
    }
    std::ifstream in(GOLDEN_FILE, std::ios::binary);
    std::uint64_t n = 0, m = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof n);
    in.read(reinterpret_cast<char*>(&m), sizeof m);
    if (not in or n != rows.size() or m != rows.front().size())
    {
        std::printf("FAIL: golden file missing or shape differs (%llu x %llu vs %zu x %zu)\n", (unsigned long long)n,
                    (unsigned long long)m, rows.size(), rows.front().size());
        return 1;
    }
    int bad = 0;
    for (std::size_t r = 0; r < n; ++r)
    {
        std::vector<double> g(m);
        in.read(reinterpret_cast<char*>(g.data()), std::streamsize(m * sizeof(double)));
        for (std::size_t c = 0; c < m; ++c)
            if (std::memcmp(&g[c], &rows[r][c], sizeof(double)) != 0)
            {
                if (bad < 10) std::printf("FAIL row %zu value %zu: golden %.17g now %.17g\n", r, c, g[c], rows[r][c]);
                ++bad;
            }
    }
    std::printf("%s: %llu rows x %llu values, %d differ (exact compare)\n", bad == 0 ? "PASS" : "FAIL",
                (unsigned long long)n, (unsigned long long)m, bad);
    return bad == 0 ? 0 : 1;
}
