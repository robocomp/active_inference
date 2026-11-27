// engine_offer_f1_test — the offer promises the ρ the update then applies (review F1-I2; task-F1).
//
// With F1 the countable fitter stages each frame's viewpoints and the latents' anchors (FrameLatents::view), and
// PersistentEngine::predicted_gain reads the anchors the engine committed to measure a look's ρ. Nothing else checks that the
// promise and the delivery use the same ρ (an index slip — a_lat read for height, the camera advanced from the LiDAR's position —
// would make them disagree silently, and first surface in grading (b) as an "estimator" defect).
//
// Toy box with kHeight and a Helios. Five updates at standpoints stepping 0.02 m sideways at a fixed heading (pure translation,
// so the robot's displacement IS each sensor's), each staged exactly as CountableFitter::stage_persistent / consume_paths do
// (view_stage with the robot displacement, ρ from anchor_separation, view_commit after the update). Then three looks:
//   parked   — at the last standpoint: every separation 0, every ρ = 1;
//   outward  — one more step: separations > 0, ρ < 1;
//   inside   — back at the second standpoint, inside what was paid: separations 0, ρ = 1 (★ the feasibility bound).
// For each, the offer's kernel arguments (dbg_offer_rho) must equal the fitter-rule separations of a frame staged there, per
// latent (camera, LiDAR lateral, LiDAR height), and the offer's lateral ρ must equal the kernel of those separations. The
// outward look is then delivered as an update with the fitter-staged frame, and its staged lateral ρ equals the promise.
// Run: ./engine_offer_f1_test
#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

#include "toy_model.h"
#include "ai_belief/beam_ray.h"
#include "ai_belief/offer_view.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace {
struct ToyH : toy::ToyBox { static constexpr int kHeight = 2; };
using Engine = rc::ai::engine::PersistentEngine<ToyH>;
constexpr int N = Engine::N;
namespace of = rc::ai::offer;
namespace vw = rc::ai::view;
const toy::ToyState kTruth{0.0f, 0.0f, 0.6f, 1.6f, 0.6f, 0.0f};

of::OfferLidar test_lidar(const rc::ai::beam::StudentT& st)
{   // as offer_return_view_test: Shadow's ring span, 0.2° azimuth step, a select box about the toy's footprint
    of::OfferLidar L;
    L.offset = {0.0f, -0.08f, 1.075f};
    for (int k = 0; k < 32; ++k) L.ring_elev_rad.push_back((-15.5 + k * 70.0 / 31.0) * std::numbers::pi / 180.0);
    L.azimuth_step_rad = 0.2 * std::numbers::pi / 180.0;
    L.eps_px = 0.05;
    L.room.floor_sigma = 0.005;
    L.st = st;
    constexpr float margin = 0.1f;
    L.box = {0.5f * std::hypot(1.6f, 0.6f) + margin, -margin, 0.6f + margin};
    return L;
}
bool same(const vw::ViewpointPath& a, const vw::ViewpointPath& b)
{   // the offer advances in 10 chords of exact increments, the fitter in one: equal to float rounding
    return std::abs(a.dphi_rad - b.dphi_rad) <= 2e-6f and std::abs(a.dz_m - b.dz_m) <= 2e-6f;
}
}  // namespace

int main()
{
    rc::nbv::Sensor sensor;
    sensor.hfov_rad = 110.0f * std::numbers::pi_v<float> / 180.0f;
    sensor.vfov_rad = 70.0f * std::numbers::pi_v<float> / 180.0f;
    sensor.height_m = 0.945f;
    const std::vector<rc::nbv::Obstacle> no_obstacles;
    ToyH m;
    m.s = kTruth;
    m.prior_mean << 0.f, 0.f, kTruth.H, kTruth.w, kTruth.h, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    m.sigma_m = 0.05f;
    m.beam_cfg = rc::ai::beam::BeamConfig::make(*rc::ai::beam::StudentT::make(2500.0, 0.05));
    const of::OfferLidar L = test_lidar(m.beam_cfg.st);
    const of::OfferSensors sns{sensor, no_obstacles, L};
    const float g = of::max_ring_gap(L.ring_elev_rad);
    // The object centre the fitter and the offer measure viewpoints about: the belief's current θ (it moves with the updates).
    const auto centre = [&] { return Eigen::Vector2f(m.state().vec()(0), m.state().vec()(1)); };
    const float yaw = std::numbers::pi_v<float> / 2;                         // looking +y at the box from below it
    const auto at = [&](int k) { return of::Standpoint{Eigen::Vector2f(-0.05f + 0.02f * float(k), -2.3f), yaw}; };
    const auto cam_xy = [&](const of::Standpoint& s) { return of::camera_at(s, sns).head<2>().eval(); };
    const auto L_xy = [&](const of::Standpoint& s) { return Eigen::Vector2f(of::lidar_origin(s, L).head<2>().cast<float>()); };

    // The fitter's rule (stage_persistent): the frame's view from the committed state and the robot displacement since.
    struct Staged { vw::ViewState v; vw::ViewpointPath cam, lat, H; float rho_lat = 1.f, rho_H = 1.f, rho_cam = 1.f; };
    const auto stage = [&](const vw::ViewState& committed, const of::Standpoint& s, const Eigen::Vector2f& dr)
    {
        Staged st;
        const Eigen::Vector2f c = centre();
        st.v = vw::view_stage(committed, cam_xy(s), L_xy(s), dr, c);
        st.cam = vw::anchor_separation(st.v.a_cam, st.v.cam);
        st.lat = vw::anchor_separation(st.v.a_lat, st.v.L);
        st.H   = vw::anchor_separation(st.v.a_H, st.v.L);
        const float z = (cam_xy(s) - c).norm(), zl = (L_xy(s) - c).norm();
        st.rho_cam = vw::rho_camera(st.cam, z, 0.5f * kTruth.w);
        st.rho_lat = vw::rho_lidar_lateral(st.lat, g);
        st.rho_H   = vw::rho_lidar_height(st.H, zl, std::atan2(kTruth.H - 1.075f, zl), g);
        return st;
    };
    Engine e;
    vw::ViewState committed;
    of::Standpoint prev = at(0);
    const auto deliver = [&](const of::Standpoint& s, const Staged& st)
    {
        const auto o = e.offer_observation(m, s, sns);
        toy::ToyFrame f = toy::make_frame(o.points, o.rays);
        f.R.resize(o.points.size());
        for (std::size_t k = 0; k < o.points.size(); ++k) f.R[k] = o.R / o.weight[k];
        rc::ai::engine::FrameLatents<N> fl;
        fl.sig2_cam = {9e-4f, 9e-4f, 4e-4f, 4e-4f, 4e-4f, 9e-4f};
        fl.sig2_L = fl.sig2_cam;
        fl.rho_cam.fill(st.rho_cam);
        fl.rho_L = {st.rho_lat, st.rho_lat, st.rho_H, st.rho_lat, st.rho_lat, st.rho_lat};
        fl.rho_chain = 1.0f;
        fl.robot_xy = s.xy;
        fl.camera_xy = cam_xy(s);
        fl.view = st.v;
        fl.camera_measured = not o.points.empty();
        fl.lidar_measured = o.lidar;
        CHECK(fl.camera_measured and fl.lidar_measured);
        e.update(m, f, fl);
        committed = vw::view_commit(st.v, true, true, st.rho_cam, st.rho_lat, st.rho_H);   // consume_paths
        prev = s;
    };
    for (int k = 0; k < 5; ++k) deliver(at(k), stage(committed, at(k), at(k).xy - prev.xy));

    struct Probe { const char* name; of::Standpoint s; bool moved; };
    for (const Probe& p : {Probe{"parked", at(4), false}, Probe{"outward", at(5), true}, Probe{"inside", at(1), false}})
    {
        (void)e.predicted_gain(m, p.s, sns);
        const auto& d = e.dbg_offer_rho();
        const Staged st = stage(committed, p.s, p.s.xy - prev.xy);
        const Eigen::Vector2f c = centre();
        // ρ tolerance from the separation's: the offer's 10 chords and the fitter's one step agree to 2e-6 rad (same()), and
        // dρ = ρ·dΔ/g, so 2e-6/g ≈ 5e-5.
        const float tol_rho = 2e-6f / g;
        std::printf("  %-8s offer: cam (%.5f, %.5f) lat (%.5f, %.5f) H (%.5f, %.5f) ρ_lat %.6f ρ_H %.6f | fitter: cam (%.5f, %.5f) "
                    "lat (%.5f, %.5f) H (%.5f, %.5f) ρ_lat %.6f\n", p.name, d.cam.dphi_rad, d.cam.dz_m, d.lat.dphi_rad, d.lat.dz_m,
                    d.H.dphi_rad, d.H.dz_m, d.rho_L[0], d.rho_L[ToyH::kHeight], st.cam.dphi_rad, st.cam.dz_m, st.lat.dphi_rad,
                    st.lat.dz_m, st.H.dphi_rad, st.H.dz_m, st.rho_lat);
        CHECK(d.f1_cam and d.f1_L);                                  // the anchors were read, not the straight path
        CHECK(same(d.cam, st.cam) and same(d.lat, st.lat) and same(d.H, st.H));
        CHECK(std::abs(d.rho_L[0] - st.rho_lat) <= tol_rho);
        CHECK(std::abs(d.rho_L[ToyH::kHeight] - vw::rho_lidar_height(d.H, (L_xy(p.s) - c).norm(),
                                                    std::atan2(kTruth.H - 1.075f, (L_xy(p.s) - c).norm()), g)) <= tol_rho);
        if (p.moved) { CHECK(d.lat.dphi_rad > 0.f and d.rho_L[0] < 1.f and d.rho_cam[0] < 1.f); }
        else         { CHECK(d.rho_cam[0] >= 1.f - tol_rho and d.rho_L[0] >= 1.f - tol_rho and d.rho_L[ToyH::kHeight] >= 1.f - tol_rho); }
        // (parked / inside: 1 up to float — the frontier's leading edge a + m is re-referenced in float and can sit ~1e-7 rad
        //  inside the last committed view, which the kernel turns into ρ = 1 − 3e-6.)
        if (p.moved)
        {   // deliver it: the update's staged ρ is the one promised
            const float promised = d.rho_L[0];
            deliver(p.s, st);
            CHECK(std::abs(st.rho_lat - promised) <= tol_rho);
        }
    }
    std::printf(bad == 0 ? "engine_offer_f1_test: PASS\n" : "engine_offer_f1_test: %d FAILED\n", bad);
    return bad == 0 ? 0 : 1;
}
