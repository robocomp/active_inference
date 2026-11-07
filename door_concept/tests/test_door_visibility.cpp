// test_door_visibility.cpp — Task 2 of the door-existence model: door_view(), "should this door be visible in this image, and
// how big does it project?", for the ZED pinhole AND the ricoh 360. The visibility rules are the silhouette's, moved
// (src/core/door_visibility.{h,cpp}); compute_silhouette_existence now walks its samples through the same ViewRig.
//
// The scene is test_door_core_existence.cpp's: a 10 x 10 m room, a door in the top wall (y = 5) at x = 1, a 1280x720 ZED
// (f = 700) 1.2 m high looking +y from (1, 0.5).
//   zed_behind_vis_zero          camera looking -y: nothing visible
//   zed_around_wall_vis_zero     a partition between camera and door: vis 0 (a wall hides it)
//   zed_overflow_fill            0.6 m face-on: fill_v > 1 (the door overflows the image)
//   ricoh_outside_window_vis_zero  the door's bearing outside every looked window: vis 0
//   ricoh_unknown_windows_no_miss  windows unknown (producer without the field): vis 0, no miss evidence
//   ricoh_face_on_1p4m           equirect 1920x960, the door at 1.4 m face-on: vis ≈ 1, fill = angular spans / (2π/3, π)
//   open_leaf_vis_is_marginal    a phi posterior with mass at 0 and 80°: vis is the mass-weighted mean of the single-angle values
//   silhouette_unchanged         compute_silhouette_existence on this scene returns the numbers it returned BEFORE the move
//                                (captured from a316097f; -DSIL_ONLY builds the golden-capture part alone, against that tree)
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_visibility.cpp \
//       src/core/door_core.cpp src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_visibility.cpp src/core/door_visibility.cpp \
//       src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp \
//       ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
//       ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc \
//       -o /tmp/test_door_visibility && /tmp/test_door_visibility
#include <chrono>
#include <cmath>
#include <numbers>
#include <print>
#include <vector>
#include "../src/core/door_core.h"
#include "../src/core/door_existence.h"
#include "../src/door_belief.h"
#ifndef SIL_ONLY
#include "../src/core/door_visibility.h"
#endif

using namespace rc::door::core;

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

constexpr float kDoorX = 1.0f, kRobotY = 0.5f;
constexpr int kW = 1280, kH = 720;
constexpr float kF = 700.f;

static Mat4d pose(double x, double y, double z, bool turned_away = false)
{
    Mat4d T = Mat4d::Identity();
    if (turned_away) { T(0, 0) = -1.0; T(1, 1) = -1.0; }
    T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
    return T;
}

struct Scene
{
    DoorCoreConfig cfg;
    DoorCore core{cfg};
    std::vector<Eigen::Vector2f> polygon{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};
    rc::DoorFitState inst;
    CameraIntrinsics zed;
    MaskPacket pkt;
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> doors;

    explicit Scene(float width = 1.0f, float height = 2.1f)
    {
        rc::DoorState st0;
        st0.w = width; st0.h = height; st0.thickness = 0.1f;
        core.fit().init_instance(inst, 4242, "door_1", st0);
        // The belief, set up by hand: the top wall y = 5 seen from inside the room (u runs −x, so across_u = −y points INTO
        // the room, as DoorFit's wall association orients it), the aperture centred at x = kDoorX.
        rc::DoorBeliefParams p;
        p.wall_O = Eigen::Vector2f(5.0f, 5.0f); p.wall_u = Eigen::Vector2f(-1.0f, 0.0f); p.wall_len = 10.0f; p.thickness = 0.1f;
        rc::DoorBeliefState s0;
        s0.w = width; s0.h = height; s0.s = (5.0f - kDoorX) - 0.5f * width;
        inst.ai2_belief = rc::DoorBelief(s0, p);
        inst.ai2_initialized = true;
        core.fit().refresh_geometry(inst);
        zed.present = true; zed.width = kW; zed.height = kH; zed.focal_x = kF; zed.focal_y = kF;
        doors = {{4242, &inst}};
        set_mask(false, kRobotY, kDoorX);
    }
    // A mask frame: a small unrelated "plant" in the corner, plus (lit) the door's projected pixels from (robot_x, robot_y).
    void set_mask(bool lit, float robot_y, float robot_x)
    {
        pkt = MaskPacket{};
        pkt.valid = true; pkt.frame_id = 1; pkt.timestamp_ms = 1001;
        MaskSlice pl; pl.label = "plant"; pl.class_id = 5.f; pl.confidence = 0.8f; pl.source = 0;
        for (int r = 0; r < 30; r += 3) for (int c = 0; c < 30; c += 3) pkt.mask_pixels.emplace_back(c, r);
        pl.pixel_begin = 0; pl.pixel_end = pkt.mask_pixels.size();
        pkt.slices.push_back(pl);
        if (not lit) return;
        const float Y = 4.99f - robot_y;
        const float c0 = kW / 2 + kF * (kDoorX - 0.5f - robot_x) / Y, c1 = kW / 2 + kF * (kDoorX + 0.5f - robot_x) / Y;
        const float r0 = kH / 2 - kF * (2.1f - 1.2f) / Y, r1 = kH / 2 - kF * (0.0f - 1.2f) / Y;
        const auto b = pkt.mask_pixels.size();
        for (float r = r0; r < r1; r += 3.0f)
            for (float c = c0; c < c1; c += 3.0f)
                pkt.mask_pixels.emplace_back(c, r);
        MaskSlice d; d.label = "door"; d.class_id = 1000.f; d.confidence = 0.9f; d.source = 0;
        d.pixel_begin = b; d.pixel_end = pkt.mask_pixels.size();
        pkt.slices.push_back(d);
    }
    ExistenceScene scene(const std::optional<Mat4d>& T, const CameraIntrinsics& cam)
    {
        return ExistenceScene{cfg.door, T, cam, pkt, polygon, core.fit().room_band(), doors};
    }
    void open_to(float phi_rad)   // the single-angle rendering: no posterior, the leaf at phi
    {
        inst.leaf.phi = phi_rad;
        inst.leaf_pose = rc::door::leaf_pose(inst.aperture, inst.leaf);
        inst.phi_curve.clear();
    }
};

static void print_sil(const char* tag, const DoorSilhouette& s)
{
    std::print("SIL {} e_occ={:.6f} e_free={:.6f} n_det={} n_tot={} n_occl={} n_central={} det_w={:.6f} n_cells={} marg={} ndm={:.6f}\n",
               tag, s.e_occ, s.e_free, s.n_detectable, s.n_total, s.n_occluded, s.n_central, s.det_w, s.n_cells,
               s.phi_marginalised, s.n_det_marg);
}

static void check_sil(const char* tag, const DoorSilhouette& s, float occ, float free_, int det, int occl, int central, float det_w,
                      int cells, float ndm)
{
    print_sil(tag, s);
    CHECK(std::abs(s.e_occ - occ) < 1e-3f);
    CHECK(std::abs(s.e_free - free_) < 1e-3f);
    CHECK(s.n_detectable == det and s.n_total == 420 and s.n_occluded == occl and s.n_central == central);
    CHECK(std::abs(s.det_w - det_w) < 1e-3f and s.n_cells == cells);
    CHECK(std::abs(s.n_det_marg - ndm) < 1e-3f);
}

int main()
{
    // ── silhouette_unchanged: the numbers captured from a316097f (before the move). ──────────────────────────────────
    {
        Scene sc;
        const auto T = std::optional<Mat4d>(pose(1.0, kRobotY, 1.2));
        sc.set_mask(true, kRobotY, 1.0f);
        const auto lit = compute_silhouette_existence(sc.scene(T, sc.zed), sc.inst, nullptr);
        sc.set_mask(false, kRobotY, 1.0f);
        const auto absent = compute_silhouette_existence(sc.scene(T, sc.zed), sc.inst, nullptr);
        // marginal over a phi posterior: mass at 0 and 80°
        sc.set_mask(true, kRobotY, 1.0f);
        sc.inst.phi_curve = {{0.0f, 0.5f}, {80.0f * std::numbers::pi_v<float> / 180.0f, 0.5f}};
        const auto marg = compute_silhouette_existence(sc.scene(T, sc.zed), sc.inst, nullptr);
        sc.set_mask(true, 4.0f, 1.0f);   // oblique/near
        sc.inst.phi_curve.clear();
        const auto near = compute_silhouette_existence(sc.scene(std::optional<Mat4d>(pose(1.4, 4.0, 1.2)), sc.zed), sc.inst, nullptr);
        // behind a partition 2 cm from the jamb, and from its open side
        Scene sp;
        sp.polygon = {{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {0.48f, 5.f}, {0.48f, 1.5f}, {0.38f, 1.5f}, {0.38f, 5.f}, {-5.f, 5.f}};
        const auto wall = compute_silhouette_existence(sp.scene(std::optional<Mat4d>(pose(-1.2, kRobotY, 1.2)), sp.zed), sp.inst, nullptr);
        const auto openside = compute_silhouette_existence(sp.scene(std::optional<Mat4d>(pose(1.6, kRobotY, 1.2)), sp.zed), sp.inst, nullptr);
        print_sil("wall", wall); print_sil("openside", openside);
        print_sil("lit", lit); print_sil("absent", absent); print_sil("marg", marg); print_sil("near", near);
#ifndef SIL_ONLY
        // captured from a316097f (the tree before the move) with -DSIL_ONLY
        check_sil("wall",     wall,     0.f,       0.f,       0,   420, 0,   0.f,   0,   0.f);
        check_sil("openside", openside, 0.f,       420.f,     420, 0,   389, 420.f, 420, 420.f);
        check_sil("lit",      lit,      287.52f,   132.48f,   420, 0,   389, 420.f, 397, 420.f);
        check_sil("absent",   absent,   0.f,       420.f,     420, 0,   389, 420.f, 397, 420.f);
        check_sil("marg",     marg,     402.5f,    17.5f,     420, 0,   389, 420.f, 397, 420.f);
        check_sil("near",     near,     96.f,      11.f,      107, 0,   48,  107.f, 107, -1.f);
#endif
    }
#ifdef SIL_ONLY
    return 0;
#else
    // ── door_view ───────────────────────────────────────────────────────────────────────────────────────────────────
    {   // zed_behind_vis_zero
        Scene sc; sc.open_to(0.0f);   // a SHUT leaf in the aperture
        const CameraSpec cam = camera_spec(sc.zed);
        const auto T = pose(1.0, kRobotY, 1.2, /*turned_away=*/true);
        const auto g = door_view({cam, T, {}, false}, sc.scene(T, sc.zed), sc.inst);
        std::print("zed_behind: vis={} projectable={}\n", g.vis, g.projectable);
        CHECK(g.vis == 0.0f);
    }
    {   // zed_around_wall_vis_zero: a partition 2 cm from the jamb, the camera west of it
        Scene sc; sc.open_to(0.0f);
        sc.polygon = {{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {0.48f, 5.f}, {0.48f, 1.5f}, {0.38f, 1.5f}, {0.38f, 5.f}, {-5.f, 5.f}};
        const CameraSpec cam = camera_spec(sc.zed);
        const auto Tb = pose(-1.2, kRobotY, 1.2), To = pose(1.6, kRobotY, 1.2);
        const auto gb = door_view({cam, Tb, {}, false}, sc.scene(Tb, sc.zed), sc.inst);
        const auto go = door_view({cam, To, {}, false}, sc.scene(To, sc.zed), sc.inst);
        std::print("zed_around_wall: behind the partition vis={} ; open side vis={}\n", gb.vis, go.vis);
        CHECK(gb.vis == 0.0f);
        CHECK(go.vis > 0.5f);
    }
    {   // zed_overflow_fill
        Scene sc; sc.open_to(0.0f);
        const CameraSpec cam = camera_spec(sc.zed);
        const auto T = pose(1.0, 4.4, 1.2);
        const auto g = door_view({cam, T, {}, false}, sc.scene(T, sc.zed), sc.inst);
        std::print("zed_overflow: fill_h={:.3f} fill_v={:.3f} vis={:.3f} range={:.3f} aspect={:.3f}\n", g.fill_h, g.fill_v, g.vis,
                   g.range_m, g.aspect);
        CHECK(g.fill_v > 1.0f);
    }
    // ── the ricoh: equirect 1920 x 960, 1.2 m high, 3.6 m south of the door's face... 1.4 m from it, face-on ──
    CameraIntrinsics rcam;
    rcam.present = true; rcam.projection_model = 1; rcam.width = 1920; rcam.height = 960;
    rcam.equirect_az_sign = 1.f; rcam.equirect_az_offset = 0.f;
    const CameraSpec rspec = camera_spec(rcam);
    {   // ricoh_face_on_1p4m
        Scene sc(0.85f, 2.05f); sc.open_to(0.0f);
        const auto& L = sc.inst.leaf_pose;
        const float leaf_y = L.centre_xy.y();
        const auto T = pose(kDoorX, leaf_y - 1.4, 1.2);
        const float bearing = std::numbers::pi_v<float> / 2;
        const std::vector<Vec2f> win{{bearing - 0.5f, bearing + 0.5f}};
        const auto g = door_view({rspec, T, win, true}, sc.scene(T, rcam), sc.inst);
        // expected: the four corners' azimuth / elevation spans over (2π/3, π)
        double az_lo = 1e9, az_hi = -1e9, el_lo = 1e9, el_hi = -1e9;
        for (const float lx : {-L.half_w, L.half_w})
            for (const float lz : {L.centre_z - L.half_h, L.centre_z + L.half_h})
            {
                const Eigen::Vector3f P = rc::door::leaf_point(L, lx, 0.0f, lz);
                const double dx = P.x() - T(0, 3), dy = P.y() - T(1, 3), dz = P.z() - T(2, 3);
                const double az = std::atan2(dx, dy), el = std::asin(dz / std::sqrt(dx * dx + dy * dy + dz * dz));
                az_lo = std::min(az_lo, az); az_hi = std::max(az_hi, az); el_lo = std::min(el_lo, el); el_hi = std::max(el_hi, el);
            }
        const double want_h = (az_hi - az_lo) / (2.0 * std::numbers::pi / 3.0), want_v = (el_hi - el_lo) / std::numbers::pi;
        std::print("ricoh_face_on_1p4m: door {:.3f} x {:.3f} m  vis={:.4f} fill_h={:.5f} (want {:.5f}) fill_v={:.5f} (want {:.5f}) "
                   "range={:.3f} aspect={:.3f}\n", 2 * L.half_w, 2 * L.half_h, g.vis, g.fill_h, want_h, g.fill_v, want_v, g.range_m,
                   g.aspect);
        CHECK(std::abs(2 * L.half_w - 0.85f) < 1e-3f and std::abs(2 * L.half_h - 2.05f) < 1e-3f);
        CHECK(g.projectable and g.vis > 0.95f);
        CHECK(std::abs(g.fill_h - want_h) < 2e-3 and std::abs(g.fill_v - want_v) < 2e-3);   // 1 px of 1920 = 5e-4 of the circle
        CHECK(g.aspect > 0.99f);
        // ricoh_outside_window_vis_zero: the window is on the far side of the room
        const std::vector<Vec2f> far{{-bearing - 0.5f, -bearing + 0.5f}};
        const auto go = door_view({rspec, T, far, true}, sc.scene(T, rcam), sc.inst);
        std::print("ricoh_outside_window: vis={}\n", go.vis);
        CHECK(go.vis == 0.0f);
        // ricoh_unknown_windows_no_miss: unknown is not looked
        const auto gu = door_view({rspec, T, {}, false}, sc.scene(T, rcam), sc.inst);
        std::print("ricoh_unknown_windows: vis={}\n", gu.vis);
        CHECK(gu.vis == 0.0f);
    }
    {   // open_leaf_vis_is_marginal
        Scene sc;
        const CameraSpec cam = camera_spec(sc.zed);
        const auto T = pose(1.0, 3.0, 1.2);
        const float phi80 = 80.0f * std::numbers::pi_v<float> / 180.0f;
        sc.open_to(0.0f);
        const float v0 = door_view({cam, T, {}, false}, sc.scene(T, sc.zed), sc.inst).vis;
        sc.open_to(phi80);
        const float v80 = door_view({cam, T, {}, false}, sc.scene(T, sc.zed), sc.inst).vis;
        sc.open_to(0.0f);
        sc.inst.phi_curve = {{0.0f, 0.7f}, {phi80, 0.3f}};
        const float vm = door_view({cam, T, {}, false}, sc.scene(T, sc.zed), sc.inst).vis;
        std::print("open_leaf: vis(0)={:.4f} vis(80)={:.4f} marginal(.7/.3)={:.4f} want {:.4f}\n", v0, v80, vm, 0.7f * v0 + 0.3f * v80);
        CHECK(std::abs(v0 - v80) > 0.05f);
        CHECK(vm > std::min(v0, v80) and vm < std::max(v0, v80));
        CHECK(std::abs(vm - (0.7f * v0 + 0.3f * v80)) < 1e-4f);
    }
    std::print("{}\n", fails == 0 ? "test_door_visibility: ALL PASS" : "test_door_visibility: FAILURES");
    return fails == 0 ? 0 : 1;
#endif
}
