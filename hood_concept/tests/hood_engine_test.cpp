// hood_engine_test — the hood belief on the shared persistent engine (shared-engine Task 10, Step 1).
//
// (a) Consistency: a synthetic HANGING hood (toy_model.h's scene builder, surface_points of the box body lifted to z ∈
//     [H − extent, H], plus its underside, with 1 cm iid noise) seen by the camera
//     from a robot that walks a 60° arc round it at 2.5 m, 60 persistent updates with the latents staged as the fitter
//     stages them (b_cam marginal = the constant common-mode stds, ρ_cam from the odometry path since the last update, a
//     small robot pose covariance with no localiser correction). The belief is seeded 5 cm / 3° off the truth. After the 60
//     updates every component of θ is within 3σ of the truth (σ = √P_ii, the published Σθθ).
// (b) The beam part (the LiDAR channel's convex box, hanging: cz = H − extent/2): its J = ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ against central differences of
//     beam_parts itself.
// (c) The switch: params.persistent = false runs the legacy update (the engine never seeds its joint).
//
// Run: ./hood_engine_test
#include <cmath>
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
    float uniform(float a)
    {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const double u = double(s >> 11) * (1.0 / 9007199254740992.0);
        return float((2.0 * u - 1.0) * a);
    }
};
constexpr float kPosStd = 0.03f, kSizeStd = 0.02f, kYawStd = 0.03f;   // the hood config's AI2CommonMode*Std
}  // namespace

int main()
{
    const toy::ToyState truth{1.20f, -0.60f, 2.08f, 0.86f, 0.52f, 0.10f};
    const auto class_params = []
    {   // the hood config's class means (etc/config.toml AI2PriorFootprintM / AI2PriorDepthM / AI2PriorHeightM)
        HoodBeliefParams p;
        p.prior_footprint_m = 0.90f; p.prior_depth_m = 0.50f; p.prior_height_m = 2.05f; p.wall_flush_prior = 1.0f;
        return p;
    };
    // The body is a box of height `extent` lifted so its top is at H; the underside is seen from below.
    const auto hood_points = [&](int per, float extent)
    {
        toy::ToyState body = truth;
        body.H = extent;
        std::vector<Eigen::Vector3f> pts = toy::surface_points(body, per);
        const float lift = truth.H - extent;
        for (auto& q : pts) q.z() += lift;
        const float c = std::cos(truth.yaw), sn = std::sin(truth.yaw);
        for (int i = 0; i < per; ++i)
            for (int j = 0; j < per; ++j)
            {
                const float lx = (-0.5f + (i + 0.5f) / per) * truth.w, ly = (-0.5f + (j + 0.5f) / per) * truth.h;
                pts.emplace_back(truth.cx + c * lx - sn * ly, truth.cy + sn * lx + c * ly, lift);
            }
        return pts;
    };

    // (a) consistency after 60 persistent updates
    {
        HoodBeliefParams p = class_params();
        p.persistent = true;
        p.layer_spread_m = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        HoodBelief b(HoodBeliefState{truth.cx + 0.05f, truth.cy - 0.04f, truth.H - 0.05f, truth.w - 0.04f,
                                                     truth.h + 0.03f, truth.yaw + 0.05f}, p);
        Lcg rng;
        const Eigen::Vector2f c(truth.cx, truth.cy);
        constexpr float kRange = 2.5f;
        const auto at = [&](float a) { return Eigen::Vector2f(c + kRange * Eigen::Vector2f(std::cos(a), std::sin(a))); };
        const float a0 = -0.5f * std::numbers::pi_v<float> - std::numbers::pi_v<float> / 6.0f;
        const float da = (std::numbers::pi_v<float> / 3.0f) / 60.0f;
        rc::ai::view::ViewpointPath path;
        Eigen::Vector2f prev = at(a0);
        for (int k = 0; k < 60; ++k)
        {
            const Eigen::Vector2f robot = at(a0 + k * da);
            if (k > 0) rc::ai::view::accumulate_path(path, prev, robot - prev, c);
            HoodFrame f;
            for (auto q : hood_points(6, p.vertical_extent_m))
            {
                q += Eigen::Vector3f(rng.uniform(0.01f), rng.uniform(0.01f), rng.uniform(0.01f));
                f.points.push_back(q);
            }
            auto& in = f.persist;
            const float p2 = kPosStd * kPosStd, s2 = kSizeStd * kSizeStd, y2 = kYawStd * kYawStd;
            in.sig2_cam = {p2, p2, s2, s2, s2, y2};
            in.sig2_L = in.sig2_cam;
            in.rho_cam.fill(rc::ai::view::rho_camera(path, kRange, truth.w));
            in.rho_chain = 1.0f;
            in.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 1e-5).asDiagonal();
            in.chain_dr = robot - prev;
            in.robot_xy = robot;
            in.camera_xy = robot;
            b.update(f);
            path = {};
            prev = robot;
        }
        const auto th = b.state().vec();
        const Eigen::Matrix<float, 6, 1> tv = Eigen::Matrix<float, 6, 1>(truth.vec());
        const auto& S = b.covariance();
        const char* names[6] = {"cx", "cy", "H", "w", "h", "yaw"};
        for (int i = 0; i < 6; ++i)
        {
            float d = th(i) - tv(i);
            if (i == 5) d = std::remainder(d, 2.0f * std::numbers::pi_v<float>);
            const float sd = std::sqrt(S(i, i));
            std::printf("(a) %-3s truth %+.4f est %+.4f  err %+.4f  sigma %.4f  (%.2f sigma)\n", names[i], tv(i), th(i), d, sd,
                        std::abs(d) / sd);
            CHECK(std::isfinite(sd) and sd > 0.0f);
            CHECK(std::abs(d) <= 3.0f * sd);
        }
        CHECK(b.first_update_done());
        CHECK(b.latent_std().allFinite());
    }

    // (b) the beam part's J against central differences of beam_parts
    {
        HoodBelief b(HoodBeliefState{}, class_params());
        const HoodBeliefState s{truth.cx, truth.cy, truth.H, truth.w, truth.h, truth.yaw};
        const auto pack = [&](const HoodBeliefState& st)
        {
            const auto P = b.beam_parts(st).front();
            Eigen::Matrix<double, 7, 1> v;
            v << P.c.x(), P.c.y(), P.c.z(), P.yaw, P.half.x(), P.half.y(), P.half.z();
            return v;
        };
        const auto J = b.beam_parts(s).front().J;
        double worst = 0.0;
        for (int k = 0; k < 6; ++k)
        {
            Eigen::Matrix<float, 6, 1> vp = s.vec(), vm = s.vec();
            vp(k) += 1e-3f; vm(k) -= 1e-3f;
            const Eigen::Matrix<double, 7, 1> fd = (pack(HoodBeliefState::from_vec(vp)) - pack(HoodBeliefState::from_vec(vm))) / 2e-3;
            worst = std::max(worst, (fd - J.col(k)).cwiseAbs().maxCoeff());
        }
        std::printf("(b) beam part J vs central differences: worst |diff| %.2e\n", worst);
        CHECK(worst < 1e-3);   // float state, h = 1e-3: the differences' own rounding is ~1e-4
    }

    // (c) the legacy switch: params.persistent = false never touches the engine
    {
        HoodBeliefParams p = class_params();
        HoodBelief b(HoodBeliefState{truth.cx, truth.cy, truth.H, truth.w, truth.h, truth.yaw}, p);
        HoodFrame f;
        f.points = hood_points(4, p.vertical_extent_m);
        b.update(f);
        CHECK(not b.first_update_done());
        CHECK(b.joint_covariance().isZero(0.0));
    }

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
