/*
 * door_geometry.h  —  SINGLE SOURCE OF TRUTH for door panel geometry (M0 of the openable-door work).
 *
 * A door is an APERTURE plus a LEAF:
 *
 *   APERTURE — the static rectangular hole in a wall. This is what the robot navigates THROUGH and what
 *              every downstream consumer treats as a landmark: it is rigidly attached to its wall and never
 *              moves when the door swings. The DSR node's RT edge, its wall association, and its
 *              width_m/depth_m/height_m all describe the APERTURE.
 *   LEAF     — the rigid panel hinged on one vertical edge of the aperture, opening by phi. At phi == 0 the
 *              leaf sits flush inside the aperture and the two coincide EXACTLY.
 *
 * Every SDF, silhouette sample, mesh vertex, ROI corner, footprint and planner face MUST come from
 * leaf_pose() / leaf_pose_from_box() below — never from a locally reconstructed cos/sin(yaw) rectangle.
 *
 * WHY THIS FILE EXISTS. The panel rectangle used to be re-derived independently in ~10 places, and two
 * separate SDFs disagreed about which one was authoritative. Every one of those sites silently assumed the
 * panel lies in the WALL PLANE. Two of them delete the door outright the moment a leaf swings:
 *   · DoorFitter::compute_silhouette_existence sampled the face at a hardcoded ly = 0 in the wall plane, so
 *     an opened leaf's predicted silhouette goes unlit, e_free spikes and the existence channel removes the
 *     door BECAUSE it opened;
 *   · the room-containment prior keyed on the panel centre, which an outward swing moves by w/2 (0.45 m for
 *     the fitted w = 0.90 observed live) against Existence.RoomMarginM = 0.40 — and that branch bypasses the
 *     sensor channel entirely, so nothing can rescue it.
 * Collapsing the geometry into one function is what stops M1 (phi as a fitted DOF) and M2 (discrete
 * hinge/swing hypotheses) from reintroducing that class of bug.
 *
 * NOTE ON EXACTNESS: there is deliberately NO `if (phi == 0)` fast path. The phi == 0 result is produced by
 * the SAME code as phi != 0, and reduces term-for-term to the old wall-plane expressions (cos(0) is exactly
 * 1.0f, sin(0) is exactly +0.0f, and multiplying by 1.0f / adding +/-0.0f are exact in IEEE-754). self_test()
 * asserts this with operator==, not a tolerance.
 *
 * Header-only: pure Eigen, no DSR / config / torch — unit-testable in isolation.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

namespace rc::door
{

// Which vertical edge of the aperture carries the hinge. Near = the `s` edge, Far = the `s + w` edge.
// Unknown until M2 resolves it; M0 pins it to Near, which at phi == 0 is geometrically indistinguishable.
enum class HingeSide : std::uint8_t { Near = 0, Far = 1 };

// Exact box SDF from per-axis face distances (|local| − half_extent). 0 on the surface, >0 outside, <0 in.
// This is the ONLY copy — it was duplicated in door_belief.cpp and door_model.cpp.
inline float box_sdf(float dx, float dy, float dz)
{
    const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f), oz = std::max(dz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
    return outside + inside;
}

// ─── The APERTURE: the static hole in the wall ──────────────────────────────────────────────────
// Wall frame: near corner O, along-wall unit u, across-wall unit n = (−u.y, u.x). [s, w, h] is exactly the
// belief's theta. Nothing here moves when the leaf swings — that invariant is what lets the DSR RT edge,
// resolve_wall, merge, ghost identity and the room-containment prior all key on the aperture.
struct Aperture
{
    Eigen::Vector2f wall_O{0.0f, 0.0f};   // near corner of the wall (room frame, m)
    Eigen::Vector2f wall_u{1.0f, 0.0f};   // along-wall unit direction (room frame; MUST be normalised)
    float s = 0.0f;                        // along-wall offset of the near edge from O (m)
    float w = 0.70f;                       // aperture width  (m, along the wall)
    float h = 2.00f;                       // aperture height (m, vertical; base pinned to the floor)
    float floor_z   = 0.0f;                // room-frame floor datum: the base is PINNED here
    float thickness = 0.05f;               // across-wall extent (m, fixed — not a DOF)

    Eigen::Vector2f across_u() const { return {-wall_u.y(), wall_u.x()}; }
    Eigen::Vector2f centre_xy() const { return wall_O + (s + 0.5f * w) * wall_u; }
    float           centre_z()  const { return floor_z + 0.5f * h; }
    float           yaw()       const { return std::atan2(wall_u.y(), wall_u.x()); }
};

// ─── The LEAF's articulation state ──────────────────────────────────────────────────────────────
// NOT part of theta: the belief stays 3-DOF [s, w, h] and the shared inference engine is untouched. phi
// becomes a fitted DOF in M1; hinge/swing become discrete hypotheses in M2. In M0 all three are pinned.
struct LeafState
{
    float     phi   = 0.0f;               // opening angle (rad); 0 = flush inside the aperture
    HingeSide hinge = HingeSide::Near;    // which vertical aperture edge carries the hinge
    float     swing = +1.0f;              // +1 / −1: which side of the wall the leaf opens toward
};

// ─── The LEAF's rigid pose in the ROOM frame ────────────────────────────────────────────────────
// `ex` always runs hinge → free edge; `ey` is the leaf's face normal (across its thickness).
struct LeafPose
{
    Eigen::Vector2f centre_xy{0.0f, 0.0f};
    Eigen::Vector2f ex{1.0f, 0.0f};        // room-frame unit along the leaf WIDTH  (leaf local +x)
    Eigen::Vector2f ey{0.0f, 1.0f};        // room-frame unit across the leaf FACE  (leaf local +y)
    Eigen::Vector2f hinge_xy{0.0f, 0.0f};  // room-frame XY of the vertical hinge axis
    float centre_z = 0.0f;
    float half_w = 0.35f, half_t = 0.025f, half_h = 1.0f;

    float yaw() const { return std::atan2(ex.y(), ex.x()); }
};

// ★ THE mapping. Everything else in the agent is a consumer of this one function.
//
// The trick that makes phi == 0 exact rather than merely close: rotate in WALL-FRAME SCALARS (along,
// across), then map to the room with the same `wall_O + along·u + across·n` expression the old code used.
// At phi == 0 this yields ax == 1.0f and an == +0.0f, so every term collapses to the old one identically.
inline LeafPose leaf_pose(const Aperture& a, const LeafState& l)
{
    // The hinge is one vertical aperture edge; the free edge is w along the wall from it.
    const float sgn         = (l.hinge == HingeSide::Near) ? +1.0f : -1.0f;
    const float hinge_along = (l.hinge == HingeSide::Near) ? a.s : (a.s + a.w);

    // The leaf's width axis in wall-frame scalars. Opening rotates it out of the wall plane toward `swing`.
    const float ax = sgn * std::cos(l.phi);      // along-wall component of the leaf's +x
    const float an = l.swing * std::sin(l.phi);  // across-wall component of the leaf's +x

    // The leaf centre sits half a width out from the hinge, along +x.
    const float along_c  = hinge_along + 0.5f * a.w * ax;
    const float across_c =               0.5f * a.w * an;

    const Eigen::Vector2f u = a.wall_u, n = a.across_u();
    LeafPose L;
    L.hinge_xy  = a.wall_O + hinge_along * u;
    L.centre_xy = a.wall_O + along_c * u + across_c * n;
    L.centre_z  = a.floor_z + 0.5f * a.h;
    L.ex        = ax * u + an * n;
    L.ey        = Eigen::Vector2f(-L.ex.y(), L.ex.x());
    L.half_w    = 0.5f * a.w;
    L.half_t    = 0.5f * a.thickness;
    L.half_h    = 0.5f * a.h;
    return L;
}

// Entry point for code that only holds the published room-frame box (rc::DoorState): mesh, ROI, merge
// footprint, and the pre-belief candidate/residual split. Reproduces the old cos/sin(yaw) construction
// exactly. `cz` is the BASE datum (the box spans [cz, cz + h]), matching DoorState.
inline LeafPose leaf_pose_from_box(float cx, float cy, float cz, float yaw,
                                   float w, float h, float thickness)
{
    const float c = std::cos(yaw), sn = std::sin(yaw);
    LeafPose L;
    L.ex        = Eigen::Vector2f(c, sn);
    L.ey        = Eigen::Vector2f(-sn, c);
    L.centre_xy = Eigen::Vector2f(cx, cy);
    L.centre_z  = cz + 0.5f * h;
    L.half_w    = 0.5f * w;
    L.half_t    = 0.5f * thickness;
    L.half_h    = 0.5f * h;
    L.hinge_xy  = L.centre_xy - L.half_w * L.ex;   // Near-hinge convention (M0 pins hinge = Near)
    return L;
}

// Room-frame point at leaf-local (lx along the width, ly across the face) and ABSOLUTE height z.
inline Eigen::Vector3f leaf_point(const LeafPose& L, float lx, float ly, float z_abs)
{
    const Eigen::Vector2f xy = L.centre_xy + lx * L.ex + ly * L.ey;
    return {xy.x(), xy.y(), z_abs};
}

inline float leaf_sdf(const LeafPose& L, const Eigen::Vector3f& p)
{
    const Eigen::Vector2f d(p.x() - L.centre_xy.x(), p.y() - L.centre_xy.y());
    return box_sdf(std::abs(d.dot(L.ex)) - L.half_w,
                   std::abs(d.dot(L.ey)) - L.half_t,
                   std::abs(p.z() - L.centre_z) - L.half_h);
}

// The 8 box corners (room frame) — the projected-ROI bbox and the display mesh both need these.
inline std::array<Eigen::Vector3f, 8> leaf_corners(const LeafPose& L)
{
    std::array<Eigen::Vector3f, 8> out{};
    int k = 0;
    for (const float sx : {-1.0f, 1.0f})
        for (const float sy : {-1.0f, 1.0f})
            for (const float sz : {-1.0f, 1.0f})
                out[k++] = leaf_point(L, sx * L.half_w, sy * L.half_t, L.centre_z + sz * L.half_h);
    return out;
}

// Oriented footprint rectangles (room-plane XY), for overlap / containment tests.
inline std::array<Eigen::Vector2f, 4> footprint(const LeafPose& L)
{
    return {L.centre_xy + L.half_w * L.ex + L.half_t * L.ey,
            L.centre_xy + L.half_w * L.ex - L.half_t * L.ey,
            L.centre_xy - L.half_w * L.ex - L.half_t * L.ey,
            L.centre_xy - L.half_w * L.ex + L.half_t * L.ey};
}

inline std::array<Eigen::Vector2f, 4> footprint(const Aperture& a)
{
    const Eigen::Vector2f c = a.centre_xy(), u = a.wall_u, n = a.across_u();
    const float hw = 0.5f * a.w, ht = 0.5f * a.thickness;
    return {c + hw * u + ht * n, c + hw * u - ht * n, c - hw * u - ht * n, c - hw * u + ht * n};
}

// ─── self_test ──────────────────────────────────────────────────────────────────────────────────
// Called as a block from DoorBelief::self_test() so the agent keeps exactly one test entry point.
// The core claim is EXACTNESS at phi == 0 (operator==, not a tolerance) plus the rigid-body invariants
// across a full 0 → 90 deg sweep, which together are the regression net for M1/M2.
inline bool self_test()
{
    bool ok = true;
    const auto check = [&ok](bool c, const char* m)
    { if (not c) { ok = false; std::printf("  FAIL: door_geometry %s\n", m); } };

    // The legacy wall-plane SDF, verbatim from the pre-M0 DoorBelief::sdf_panel, as the reference.
    const auto legacy_sdf = [](const Aperture& a, const Eigen::Vector3f& p)
    {
        const Eigen::Vector2f& u = a.wall_u;
        const Eigen::Vector2f  nrm(-u.y(), u.x());
        const Eigen::Vector2f  centre_xy = a.wall_O + (a.s + 0.5f * a.w) * u;
        const float centre_z = a.floor_z + 0.5f * a.h;
        const Eigen::Vector2f d_xy(p.x() - centre_xy.x(), p.y() - centre_xy.y());
        return box_sdf(std::abs(d_xy.dot(u))   - 0.5f * a.w,
                       std::abs(d_xy.dot(nrm)) - 0.5f * a.thickness,
                       std::abs(p.z() - centre_z) - 0.5f * a.h);
    };

    // Deliberately NON-axis-aligned wall yaws: an axis-aligned wall can pass by accident (sin/cos hit 0/1).
    for (const float wall_yaw : {0.0f, 0.6458f, -1.9722f, 2.3f})
    {
        Aperture a;
        a.wall_O = {1.3f, -0.7f};
        a.wall_u = {std::cos(wall_yaw), std::sin(wall_yaw)};
        a.s = 1.20f; a.w = 0.72f; a.h = 2.05f; a.floor_z = 0.0f; a.thickness = 0.05f;

        const Eigen::Vector2f ap_c   = a.centre_xy();
        const float           ap_yaw = a.yaw();
        const LeafPose        L0     = leaf_pose(a, {});   // phi = 0, Near, +1

        // (a) phi == 0 reproduces the old wall-plane expressions BIT-EXACTLY.
        const Eigen::Vector2f n(-a.wall_u.y(), a.wall_u.x());
        check(L0.centre_xy == (a.wall_O + (a.s + 0.5f * a.w) * a.wall_u), "phi=0 centre not bit-exact");
        check(L0.ex == a.wall_u,                                  "phi=0 ex != wall_u");
        check(L0.ey == n,                                         "phi=0 ey != wall normal");
        check(L0.centre_z == a.floor_z + 0.5f * a.h,              "phi=0 centre_z");
        check(L0.yaw() == ap_yaw,                                 "phi=0 yaw != wall tangent");

        // (b) phi == 0 SDF is bit-identical to the legacy formula over a cloud spanning the panel + clutter.
        std::vector<Eigen::Vector3f> cloud;
        for (int i = -6; i <= 6; ++i)
            for (int j = -3; j <= 3; ++j)
                for (int k = 0; k <= 6; ++k)
                    cloud.emplace_back(ap_c.x() + 0.11f * static_cast<float>(i),
                                       ap_c.y() + 0.09f * static_cast<float>(j),
                                       0.05f + 0.33f * static_cast<float>(k));
        float dmax = 0.0f;
        for (const auto& p : cloud) dmax = std::max(dmax, std::abs(leaf_sdf(L0, p) - legacy_sdf(a, p)));
        check(dmax == 0.0f, "phi=0 SDF is not bit-identical to the wall-plane formula");

        // (c) phi = 90 deg PINS the hinge: the axis does not move and the leaf's −x edge sits on it.
        const float kHalfPi = 1.57079632679f;
        const LeafPose L9 = leaf_pose(a, {kHalfPi, HingeSide::Near, +1.0f});
        check(L9.hinge_xy == L0.hinge_xy, "hinge moved when the leaf opened");
        check((leaf_point(L9, -L9.half_w, 0.0f, 1.0f)
               - Eigen::Vector3f(L9.hinge_xy.x(), L9.hinge_xy.y(), 1.0f)).norm() < 1e-5f,
              "hinge edge is off the hinge axis");

        // (d) phi = 90 deg SWINGS the free edge: w from the hinge, perpendicular to the wall, mirrored by swing.
        const auto free_of = [](const LeafPose& L) { return Eigen::Vector2f(L.centre_xy + L.half_w * L.ex); };
        check(std::abs((free_of(L9) - L9.hinge_xy).norm() - a.w) < 1e-5f, "free edge stretched");
        check(std::abs((free_of(L9) - L9.hinge_xy).normalized().dot(a.wall_u)) < 1e-5f,
              "phi=90 free edge is not perpendicular to the wall");
        const LeafPose Lm = leaf_pose(a, {kHalfPi, HingeSide::Near, -1.0f});
        check(((free_of(Lm) - Lm.hinge_xy) + (free_of(L9) - L9.hinge_xy)).norm() < 1e-5f,
              "swing sign is not mirrored");

        // (e) M2 pre-condition: flipping the hinge at phi = 0 must NOT move the panel. Its yaw flips by pi,
        //     which is exactly why the DSR pose must come from the APERTURE and not the leaf.
        const LeafPose Lf = leaf_pose(a, {0.0f, HingeSide::Far, +1.0f});
        check(Lf.centre_xy == L0.centre_xy, "far-hinge phi=0 moved the panel");
        float fmax = 0.0f;
        for (const auto& p : cloud) fmax = std::max(fmax, std::abs(leaf_sdf(Lf, p) - leaf_sdf(L0, p)));
        check(fmax == 0.0f, "far-hinge phi=0 SDF differs");

        // (f) Rigid-body invariants over the whole sweep + the APERTURE never moves on phi (so the RT edge
        //     can never be dragged by the leaf). This is the regression net for M1/M2.
        for (int d = 0; d <= 90; ++d)
        {
            const LeafPose L = leaf_pose(a, {static_cast<float>(d) * kHalfPi / 90.0f, HingeSide::Near, +1.0f});
            check(std::isfinite(L.centre_xy.x()) and std::isfinite(L.centre_xy.y()), "non-finite leaf pose");
            check(std::abs(L.ex.norm() - 1.0f) < 1e-5f and std::abs(L.ex.dot(L.ey)) < 1e-5f,
                  "leaf axes are not orthonormal");
            check(std::abs((L.centre_xy - L.hinge_xy).norm() - 0.5f * a.w) < 1e-5f,
                  "leaf stretched during the swing");
            check(a.centre_xy() == ap_c and a.yaw() == ap_yaw, "the aperture moved when the leaf opened");
        }

        // (g) The DoorState entry point agrees with the aperture entry point.
        const LeafPose Lb = leaf_pose_from_box(L0.centre_xy.x(), L0.centre_xy.y(), a.floor_z, ap_yaw,
                                               a.w, a.h, a.thickness);
        float bmax = 0.0f;
        for (const auto& p : cloud) bmax = std::max(bmax, std::abs(leaf_sdf(Lb, p) - leaf_sdf(L0, p)));
        check(bmax < 1e-6f, "leaf_pose_from_box disagrees with leaf_pose");
    }
    return ok;
}

}  // namespace rc::door
