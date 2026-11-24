// engine_neighbour_test — a neighbour's published belief as a foreign density (spec 2026-11-23 shared-engine G3, "Answers to
// the builder check" 5; plan Task 6, Review Focus 5). A YOLO-seg mask can take in the object next to it: live14fixA's
// nightstand mask swallowed the bed (a 2.46 m cloud for a 0.51 m nightstand). Here the toy nightstand's mask cloud is its own
// visible faces (front, both sides, top) merged with a slab of the bed beside it (the bed's top and its front face, as in
// countable_concept's birth_footprint_test case 1). The toy model has no clutter component: without a foreign explanation
// every bed point is nightstand evidence.
//
//   (1) With the bed's belief given as a ForeignBox (its own std), after 50 updates the nightstand's h is within 3σ of 0.51
//       and its centre within 3σ of the truth on each axis (σ = the engine's posterior σ of that component).
//   (2) Contrast, stated so (1) is the neighbour's doing: the same run with no neighbour leaves h or the centre outside 3σ.
//   (3) The same scene with a ForeignBox whose std is all zero: no NaN, and the result is bit-identical to the run without
//       that neighbour (it is skipped, never a zero-width density — Review Focus 5).
//   (4) The LiDAR side (beam_room.h): a ray whose first surface is the neighbour gets its foreign mass there (p_f at the
//       neighbour's face ≫ the room-only p_f), and an empty neighbour list stages p_f bit-identically to the room only.
//
// The 3σ bounds are the brief's; nothing here is fitted. ε = 0.094 is nightstand_concept's Model.MaskPixelErrorRate (the
// owner-decided value). Run: ./engine_neighbour_test
#include <cmath>
#include <cstdio>
#include <vector>

#include "ai_belief/beam_room.h"
#include "toy_model.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace {

using Engine = rc::ai::engine::PersistentEngine<toy::ToyBox>;
constexpr int N = Engine::N;
constexpr double kEps = 0.094;                      // nightstand_concept Model.MaskPixelErrorRate

// The nightstand (truth) and the bed beside it, 2 cm away along +y.
const toy::ToyState kNs{0.0f, 0.0f, 0.55f, 0.51f, 0.51f, 0.0f};
const toy::ToyState kBed{0.0f, 0.255f + 0.02f + 1.0f, 0.50f, 1.60f, 2.00f, 0.0f};

// The merged mask cloud: the nightstand's front (−y), both sides and top, plus the bed's top and its front (−y) face
// within the nightstand's x span. The nightstand's +y face is hidden behind it from a camera at −y.
std::vector<Eigen::Vector3f> merged_cloud()
{
    std::vector<Eigen::Vector3f> pts;
    constexpr int n = 8;
    const float hw = 0.5f * kNs.w, hh = 0.5f * kNs.h;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const float a = (i + 0.5f) / n, b = (j + 0.5f) / n;
            pts.emplace_back(-hw + a * kNs.w, -hh, b * kNs.H);           // front
            pts.emplace_back( hw, -hh + a * kNs.h, b * kNs.H);           // sides
            pts.emplace_back(-hw, -hh + a * kNs.h, b * kNs.H);
            pts.emplace_back(-hw + a * kNs.w, -hh + b * kNs.h, kNs.H);   // top
        }
    const float bed_y0 = kBed.cy - 0.5f * kBed.h;                        // the bed's front face
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 40; ++j)
        {
            const float x = -hw + (i + 0.5f) / 12.0f * kNs.w;
            pts.emplace_back(x, bed_y0 + 0.02f + 1.58f * (j + 0.5f) / 40.0f, kBed.H);   // bed top, 1.6 m deep
        }
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 6; ++j)
            pts.emplace_back(-hw + (i + 0.5f) / 12.0f * kNs.w, bed_y0, (j + 0.5f) / 6.0f * kBed.H);   // bed front
    return pts;
}

rc::ai::beam::ForeignBox bed_box(float s)
{
    rc::ai::beam::ForeignBox b;
    b.c = {kBed.cx, kBed.cy};
    b.H = kBed.H; b.w = kBed.w; b.h = kBed.h; b.yaw = kBed.yaw;
    b.std << s, s, s, s, s, s;
    return b;
}

struct Run { toy::ToyState s; Eigen::Matrix<double, N, 1> sigma; bool finite = true; Engine::MatJd P; };

// 50 camera updates from one standpoint (parked: ρ = 1), the nightstand seeded off its truth at the class mean sizes.
Run run(const std::vector<rc::ai::beam::ForeignBox>& nb)
{
    toy::ToyBox m;
    m.sigma_m = 0.01f;
    m.s = toy::ToyState{0.03f, 0.05f, 0.55f, 0.51f, 0.51f, 0.0f};
    m.prior_mean << 0.f, 0.f, 0.55f, 0.51f, 0.51f, 0.f;
    m.Sigma = m.prior_diag.asDiagonal();
    Engine eng;
    const auto cloud = merged_cloud();
    for (int t = 0; t < 50; ++t)
    {
        rc::ai::engine::FrameLatents<N> fl;
        const float p2 = 0.03f * 0.03f, s2 = 0.02f * 0.02f, y2 = 0.03f * 0.03f;
        fl.sig2_cam = {p2, p2, s2, s2, s2, y2};
        fl.sig2_L = fl.sig2_cam;
        fl.rho_cam.fill(1.0f); fl.rho_L.fill(1.0f);
        fl.rho_chain = 1.0f;
        fl.robot_xy = {0.0f, -2.5f};
        fl.camera_measured = true;
        fl.lidar_measured = false;
        fl.foreign_boxes = nb;
        fl.mask_error_rate = kEps;
        toy::ToyBox::Frame f;
        f.points = cloud;
        eng.update(m, f, fl);
    }
    Run r;
    r.s = m.s;
    r.P = eng.P();
    for (int i = 0; i < N; ++i) r.sigma(i) = std::sqrt(std::max(0.0, eng.P()(i, i)));
    r.finite = m.s.vec().allFinite() and eng.P().allFinite();
    return r;
}

bool within(const Run& r, bool print, const char* tag)
{
    const double dh = std::abs(double(r.s.h) - double(kNs.h)), dx = std::abs(double(r.s.cx) - kNs.cx),
                 dy = std::abs(double(r.s.cy) - kNs.cy);
    if (print)
        std::printf("  %-14s h %.4f (σ %.4f, |Δ|/σ %.2f)  cx %.4f (σ %.4f, %.2f)  cy %.4f (σ %.4f, %.2f)  H %.3f w %.3f\n", tag,
                    r.s.h, r.sigma(4), dh / r.sigma(4), r.s.cx, r.sigma(0), dx / r.sigma(0), r.s.cy, r.sigma(1), dy / r.sigma(1),
                    r.s.H, r.s.w);
    return dh <= 3.0 * r.sigma(4) and dx <= 3.0 * r.sigma(0) and dy <= 3.0 * r.sigma(1);
}

}  // namespace

int main()
{
    std::printf("engine_neighbour_test (shared-engine G3, Review Focus 5)\n");

    // (1) the bed's belief as a ForeignBox: the nightstand keeps its own size and place
    const Run with_bed = run({bed_box(0.03f)});
    CHECK(with_bed.finite);
    CHECK(within(with_bed, true, "with bed:"));

    // (2) contrast: no neighbour, the bed slab drags the nightstand
    const Run none = run({});
    CHECK(none.finite);
    CHECK(not within(none, true, "no neighbour:"));

    // (3) a neighbour with an all-zero std is skipped: bit-identical to no neighbour, no NaN
    const Run zero = run({bed_box(0.0f)});
    CHECK(zero.finite);
    const bool same = zero.s.vec() == none.s.vec() and zero.P == none.P;
    std::printf("  zero std:      %s the run without it\n", same ? "bit-identical to" : "DIFFERS from");
    CHECK(same);
    {   // and the other missing-belief forms are refused by usable()
        auto nan = bed_box(0.03f); nan.std(2) = std::nanf("");
        auto neg = bed_box(0.03f); neg.std(0) = -0.01f;
        CHECK(rc::ai::beam::usable(bed_box(0.03f)));
        CHECK(not rc::ai::beam::usable(bed_box(0.0f)) and not rc::ai::beam::usable(nan) and not rc::ai::beam::usable(neg));
    }

    // (4) the rays: the neighbour is a known surface of p_f
    {
        const auto st = *rc::ai::beam::StudentT::make(5.0 / (0.05 * 0.05), 0.05);   // any valid Student-t (ν = 4)
        rc::ai::beam::RoomSurfaces room;
        room.floor = true; room.floor_z = 0.0; room.floor_sigma = 0.005;
        rc::ai::beam::BeamRays rays;
        rays.rho_max = 6.0;
        rays.origin = {0.0, -1.0, 0.3};
        const double y_face = kBed.cy - 0.5 * kBed.h;                     // the bed's front face, the ray's first surface
        rays.endpoints = {Eigen::Vector3d(0.0, y_face, 0.3), Eigen::Vector3d(0.0, y_face + 0.5, 0.3)};
        rays.m = {0.5, 0.5};
        rc::ai::beam::stage_room_foreign_density(rays, room, st);
        const std::vector<double> room_only = rays.p_f;
        room.neighbours = {};                                              // empty list: bit-identical
        rc::ai::beam::stage_room_foreign_density(rays, room, st);
        CHECK(rays.p_f == room_only);
        room.neighbours = {bed_box(0.03f)};
        rc::ai::beam::stage_room_foreign_density(rays, room, st);
        std::printf("  rays: p_f at the bed face %.4g (room only %.4g); 0.5 m behind it %.4g (room only %.4g)\n", rays.p_f[0],
                    room_only[0], rays.p_f[1], room_only[1]);
        CHECK(rays.p_f[0] > 10.0 * room_only[0]);                         // the return on the bed is foreign-explained
        CHECK(rays.p_f[1] < room_only[1]);                                 // the mass went to the bed, not beyond it
        room.neighbours = {bed_box(0.0f)};                                 // a zero-std neighbour is skipped here too
        rc::ai::beam::stage_room_foreign_density(rays, room, st);
        CHECK(rays.p_f == room_only);
    }

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
