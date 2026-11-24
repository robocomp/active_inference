// hood_engine_test — the hood belief on the shared persistent engine (shared-engine Task 10a).
//
// (a) Consistency on a scene where every channel acts. A synthetic HANGING hood (the body z ∈ [H − extent, H]) AWAY from the
//     class mean (H 2.15 vs 2.05, w 0.80 vs 0.90, h 0.55 vs 0.50), its back face against a staged wall, seen from a robot
//     walking a 60° arc at 2.5 m:
//       - camera: toy_model.h's surface_points of the body (lifted) plus its underside, filtered to the faces the camera sees, 1 cm iid noise, and ONE constant per-run
//         bias drawn from the camera latent's marginal (position σ = AI2CommonModePosStd, yaw σ = AI2CommonModeYawStd): every
//         point of every frame is shifted by (b_x, b_y) and turned by b_ψ about the box centre — the error the latent b_cam
//         models, which no number of frames averages away;
//       - the robust top z_top_obs, the wall (so the wall prior, the no-cross constraint and the flush terms act);
//       - LiDAR: a constructed beam ray set (exact first hits on the faces the Helios sees, label m = 0.9, ε_px = 0.02), with
//         a BeamConfig built by BeamConfig::make, so the L channel measures;
//       - latents as the fitter stages them: b_cam / b_L marginals = the constant common-mode stds, ρ from the path since the
//         last update, a small robot pose covariance, the manifest's layer spread Δ.
//     The belief is seeded far off. The manifest's Δ_w = 0.58 m makes w nearly the class prior's: the shape latent caps each
//     channel at Δ/√12 = 0.167 m, two channels at 0.118 m, so with the prior's 0.08 m the stated σ_w ≈ 0.066 m and a seed
//     ≥ 6σ off needs |e0_w| ≥ 0.40 m: e0_w = −0.45 m. Likewise Δ_h = 0.23 m caps each channel at 0.066 m, two at 0.047 m, with
//     the prior's 0.08 m σ_h ≈ 0.041 m, so e0_h = +0.25 m (≥ 6σ). The other seeds 0.20 m on cx, cy, H and 0.25 rad on yaw.
//     Assertions, stated before running:
//       (a1) the seed error is at least 6σ on every DOF (σ = √P_ii after the 60 updates) — the scene is not trivial;
//       (a2) every DOF ends within 3σ of the truth;
//       (a3) the error shrank by k = 2: |err| < |seed err|/2. Derivation: with (a1) |e0| ≥ 6σ and (a2) |err| ≤ 3σ,
//            |err| ≤ |e0|/2. It is the check a belief stuck at its seed fails by itself (|err| = |e0|), even where an
//            inflated σ would let (a2) pass;
//       (a4) the LiDAR channel measured (staged rays > 0) and the wall prior entered.
// (b) The beam part (the LiDAR channel's convex box, hanging: cz = H − extent/2): its J = ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ against central differences.
// (c) The switch: params.persistent = false runs the legacy update (the engine never seeds its joint).
// (d) Mode rotation on the persistent path (k = 1, w ↔ h): θ, P and the latent mean permute together — θ's rows/columns and
//     every 6-sized latent block (b_cam, b_L, u_cam, u_L, s_cam, s_L) by the same w ↔ h swap, e untouched — and the
//     belief's float Σ mirror stays P's θθ block. Exact equality (a permutation moves numbers, it computes none).
//
// Run: ./hood_engine_test
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <vector>

#include "hood_belief.h"
#include "ai_belief/tests/toy_model.h"

using namespace rc;

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace
{
struct Lcg
{
    std::uint64_t s = 0x2545F4914F6CDD1Dull;
    double u01()
    {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (double(s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
    float uniform(float a) { return float((2.0 * u01() - 1.0) * a); }
    float normal() { const double u = u01(), v = u01(); return float(std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * std::numbers::pi * v)); }
};
constexpr float kPosStd = 0.03f, kSizeStd = 0.02f, kYawStd = 0.03f;   // the hood config's AI2CommonMode*Std
constexpr double kEpsPx = 0.02, kMOn = 0.9;                            // the ray labels (ε_px, the label on the object)
using Engine = HoodBelief::Engine;

// The faces of the hanging body a viewer at `eye` sees (outward normal towards it): toy::surface_points of a box of height
// `extent` lifted so its top is at t.H, plus its underside.
std::vector<Eigen::Vector3f> visible(const toy::ToyState& t, int per, const Eigen::Vector3f& eye, float extent)
{
    std::vector<Eigen::Vector3f> out;
    const float c = std::cos(t.yaw), s = std::sin(t.yaw), lift = t.H - extent;
    toy::ToyState body = t;
    body.H = extent;
    std::vector<Eigen::Vector3f> pts = toy::surface_points(body, per);
    for (auto& q : pts) q.z() += lift;
    for (int i = 0; i < per; ++i)
        for (int j = 0; j < per; ++j)
        {
            const float lx = (-0.5f + (i + 0.5f) / per) * t.w, ly = (-0.5f + (j + 0.5f) / per) * t.h;
            pts.emplace_back(t.cx + c * lx - s * ly, t.cy + s * lx + c * ly, lift);
        }
    for (const auto& p : pts)
    {
        const float dx = p.x() - t.cx, dy = p.y() - t.cy, lx = c * dx + s * dy, ly = -s * dx + c * dy;
        Eigen::Vector3f n;
        if (std::abs(p.z() - t.H) < 1e-5f) n = Eigen::Vector3f::UnitZ();
        else if (std::abs(p.z() - lift) < 1e-5f) n = -Eigen::Vector3f::UnitZ();
        else if (std::abs(std::abs(ly) - 0.5f * t.h) < 1e-4f) n = Eigen::Vector3f(-s, c, 0.0f) * (ly > 0 ? 1.0f : -1.0f);
        else n = Eigen::Vector3f(c, s, 0.0f) * (lx > 0 ? 1.0f : -1.0f);
        if (n.dot(eye - p) > 0.0f) out.push_back(p);
    }
    return out;
}

// Exact first hits from `origin` on the side faces it sees (a convex box: a point on a face facing the origin is a first hit).
rc::ai::beam::BeamRays rays_on(const toy::ToyState& t, const Eigen::Vector3f& origin, float extent)
{
    rc::ai::beam::BeamRays r;
    r.origin = origin.cast<double>();
    for (const auto& p : visible(t, 5, origin, extent))
        if (p.z() < t.H - 1e-4f)                                   // the sides and the underside: the rings do not reach the top
        {
            r.endpoints.push_back(p.cast<double>());
            r.m.push_back(kMOn);
            r.q_sum += (kMOn - kEpsPx) / (1.0 - 2.0 * kEpsPx);
        }
    return r;
}
}  // namespace

int main()
{
    const toy::ToyState truth{1.20f, -0.60f, 2.15f, 0.80f, 0.55f, 0.10f};
    const auto class_params = []
    {   // the hood config's class means (etc/config.toml AI2PriorFootprintM / AI2PriorDepthM / AI2PriorHeightM, WallFlushPrior)
        HoodBeliefParams q;
        q.prior_footprint_m = 0.90f; q.prior_depth_m = 0.50f; q.prior_height_m = 2.05f; q.wall_flush_prior = 1.0f;
        return q;
    };
    HoodBeliefParams p = class_params();
    const float extent = p.vertical_extent_m;
    p.persistent = true;
    p.beam = rc::ai::beam::BeamConfig::make(*rc::ai::beam::StudentT::make(2500.0, 0.05));   // LidarPrecision 2500, c 5 cm
    p.layer_spread_m = {0.11f, 0.11f, 0.0f, 0.58f, 0.23f, 0.0f};                           // hood.concept.toml [shape]
    const Eigen::Matrix<float, 6, 1> e0 = (Eigen::Matrix<float, 6, 1>() << 0.20f, -0.20f, -0.20f, -0.45f, 0.25f, 0.25f).finished();
    const Eigen::Matrix<float, 6, 1> tv = truth.vec();
    HoodBelief b(HoodBeliefState::from_vec(tv + e0), p);
    const Eigen::Vector2f c(truth.cx, truth.cy), ydir(-std::sin(truth.yaw), std::cos(truth.yaw));
    b.set_room_interior(c - 2.5f * ydir);                                      // the room is in front: local +y is the back
    WallRef wall;
    wall.ok = true;
    wall.p = c + 0.5f * truth.h * ydir;                                        // through the back face
    wall.n = -ydir;                                                            // into the room
    wall.sigma_m = 0.02f;
    wall.length_m = 4.0f;

    // (a) 60 persistent updates
    {
        Lcg rng;
        const float bx = kPosStd * rng.normal(), by = kPosStd * rng.normal(), bpsi = kYawStd * rng.normal();
        std::printf("(a) camera bias drawn: (%+.4f, %+.4f) m, %+.4f rad\n", bx, by, bpsi);
        const float cb = std::cos(bpsi), sb = std::sin(bpsi);
        const auto biased = [&](const Eigen::Vector3f& q)
        {
            const Eigen::Vector2f d = q.head<2>() - c;
            return Eigen::Vector3f(c.x() + cb * d.x() - sb * d.y() + bx, c.y() + sb * d.x() + cb * d.y() + by, q.z());
        };
        constexpr float kRange = 2.5f;
        const auto at = [&](float a) { return Eigen::Vector2f(c + kRange * Eigen::Vector2f(std::cos(a), std::sin(a))); };
        const float front = std::atan2(-ydir.y(), -ydir.x());                 // bearing of the room side
        const float a0 = front - std::numbers::pi_v<float> / 6.0f, da = (std::numbers::pi_v<float> / 3.0f) / 60.0f;
        rc::ai::view::ViewpointPath path;
        Eigen::Vector2f prev = at(a0);
        for (int k = 0; k < 60; ++k)
        {
            const Eigen::Vector2f robot = at(a0 + k * da);
            if (k > 0) rc::ai::view::accumulate_path(path, prev, robot - prev, c);
            const Eigen::Vector3f cam(robot.x(), robot.y(), 0.95f), helios(robot.x(), robot.y(), 1.20f);
            HoodFrame f;
            for (const auto& q : visible(truth, 6, cam, extent))
                f.points.push_back(biased(q) + Eigen::Vector3f(rng.uniform(0.01f), rng.uniform(0.01f), rng.uniform(0.01f)));
            f.z_top_obs = truth.H;
            f.wall = wall;
            auto& in = f.persist;
            const float p2 = kPosStd * kPosStd, s2 = kSizeStd * kSizeStd, y2 = kYawStd * kYawStd;
            in.sig2_cam = {p2, p2, s2, s2, s2, y2};
            in.sig2_L = in.sig2_cam;
            const float rho = rc::ai::view::rho_camera(path, kRange, truth.w);
            in.rho_cam.fill(rho);
            in.rho_L.fill(rho);                                                // the camera's path stands in for the Helios'
            in.rho_chain = 1.0f;
            in.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 1e-5).asDiagonal();
            in.chain_dr = robot - prev;
            in.robot_xy = robot;
            in.camera_xy = robot;
            in.beam = rays_on(truth, helios, extent);
            b.update(f);
            path = {};
            prev = robot;
        }
        const auto th = b.state().vec();
        const auto& S = b.covariance();
        const char* names[6] = {"cx", "cy", "H", "w", "h", "yaw"};
        for (int i = 0; i < 6; ++i)
        {
            float d = th(i) - tv(i);
            if (i == 5) d = std::remainder(d, 2.0f * std::numbers::pi_v<float>);
            const float sd = std::sqrt(S(i, i));
            std::printf("(a) %-3s truth %+.4f est %+.4f  err %+.4f  sigma %.4f (%.2f sigma)  seed err %+.3f (%.1f sigma)\n",
                        names[i], tv(i), th(i), d, sd, std::abs(d) / sd, e0(i), std::abs(e0(i)) / sd);
            CHECK(std::isfinite(sd) and sd > 0.0f);
            CHECK(std::abs(e0(i)) >= 6.0f * sd);           // (a1)
            CHECK(std::abs(d) <= 3.0f * sd);               // (a2)
            CHECK(std::abs(d) < std::abs(e0(i)) / 2.0f);   // (a3), k = 2
        }
        std::printf("(a) beam rays staged %d, gap rays %d, wall prior entered %d\n", b.last_beam_staged(), b.last_gap_rays(),
                    int(b.wall_prior_added()));
        CHECK(b.last_beam_staged() > 0);                   // (a4)
        CHECK(b.wall_prior_added());
        CHECK(b.latent_std().allFinite());
    }

    // (d) mode rotation on the persistent path (the belief of (a), its joint seeded)
    {
        const auto P0 = b.joint_covariance();
        const auto m0 = b.latent_mean();
        const auto s0 = b.state();
        b.apply_mode_rotation_for_test(1);
        constexpr int N = 6, K = Engine::K;
        std::array<int, N + K> full{};
        for (int i = 0; i < N + K; ++i) full[i] = i;
        for (const int off : {0, N + Engine::kCam, N + Engine::kL, N + Engine::kFc, N + Engine::kFL, N + Engine::kSc, N + Engine::kSL})
            std::swap(full[off + 3], full[off + 4]);
        const auto& P1 = b.joint_covariance();
        const auto& m1 = b.latent_mean();
        int bad_p = 0, bad_m = 0;
        for (int r = 0; r < N + K; ++r)
            for (int col = 0; col < N + K; ++col) bad_p += P1(r, col) != P0(full[r], full[col]);
        for (int j = 0; j < K; ++j) bad_m += m1(j) != m0(full[N + j] - N);
        std::printf("(d) rotation k = 1: P entries off %d, latent means off %d\n", bad_p, bad_m);
        CHECK(bad_p == 0);
        CHECK(bad_m == 0);
        CHECK(b.state().w == s0.h and b.state().h == s0.w);
        CHECK(std::abs(std::remainder(b.state().yaw - s0.yaw - 0.5f * std::numbers::pi_v<float>, 2.0f * std::numbers::pi_v<float>)) < 1e-6f);
        const Eigen::Matrix<float, 6, 6> mirror = P1.topLeftCorner<6, 6>().cast<float>();
        CHECK(b.covariance() == mirror);
        CHECK(m0.segment<3>(Engine::kChain) == m1.segment<3>(Engine::kChain));   // e is room-frame: untouched
    }

    // (b) the beam part's J against central differences of beam_parts
    {
        HoodBelief bb(HoodBeliefState{}, class_params());
        const HoodBeliefState s{truth.cx, truth.cy, truth.H, truth.w, truth.h, truth.yaw};
        const auto pack = [&](const HoodBeliefState& st)
        {
            const auto P = bb.beam_parts(st).front();
            Eigen::Matrix<double, 7, 1> v;
            v << P.c.x(), P.c.y(), P.c.z(), P.yaw, P.half.x(), P.half.y(), P.half.z();
            return v;
        };
        const auto J = bb.beam_parts(s).front().J;
        double worst = 0.0;
        for (int k = 0; k < 6; ++k)
        {
            Eigen::Matrix<float, 6, 1> vp = s.vec(), vm = s.vec();
            vp(k) += 1e-3f; vm(k) -= 1e-3f;
            const Eigen::Matrix<double, 7, 1> fd = (pack(HoodBeliefState::from_vec(vp)) - pack(HoodBeliefState::from_vec(vm))) / 2e-3;
            worst = std::max(worst, (fd - J.col(k)).cwiseAbs().maxCoeff());
        }
        std::printf("(b) beam part J vs central differences: worst |diff| %.2e\n", worst);
        CHECK(worst < 1e-3);   // the part is linear in θ: only the float state's rounding (~1e-7·2/2e-3 = 1e-4) separates them
    }

    // (c) the legacy switch: params.persistent = false never touches the engine
    {
        HoodBelief bl(HoodBeliefState{truth.cx, truth.cy, truth.H, truth.w, truth.h, truth.yaw}, class_params());
        HoodFrame f;
        f.points = visible(truth, 4, Eigen::Vector3f(truth.cx, truth.cy - 2.0f, 1.0f), extent);
        bl.update(f);
        CHECK(not bl.first_update_done());
        CHECK(bl.joint_covariance().isZero(0.0));
    }

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
