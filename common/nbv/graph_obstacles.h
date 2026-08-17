/*
 * common/nbv/graph_obstacles.h  —  the DSR-side companion to viewpoint_score.h: read the other objects in the
 * graph as viewpoint obstacles. Header-only.
 *
 * Split from viewpoint_score.h on purpose: that header is pure geometry (Eigen only) and stays testable in a
 * standalone harness with no graph, no Qt and no DDS. This one is the thin adapter that knows where the
 * footprints live. Include it only from an agent.
 *
 * Every concept agent needs the identical query — "which other objects could I stand inside, or have to look
 * through?" — and each had (or was about to grow) its own copy. Two mistakes were already live in the two
 * copies that existed, and both were SILENT:
 *
 *   1. They read the DEPRECATED int attributes obj_width / obj_depth. Nothing in this tree writes those; the
 *      agents publish width_m / depth_m (float, metres). Guarded by has_value(), the loop skipped every node
 *      and the function always returned an EMPTY list — the obstacle and line-of-sight logic never ran at all.
 *   2. They pre-inflated each footprint by the robot radius before returning it. That is right for "can I
 *      stand here" and wrong for "can I see through there": a box inflated by 0.6 m casts a shadow 0.6 m
 *      wider than the object does, marking clear sightlines blocked. rc::nbv inflates internally, for the
 *      stand-inside test only, from the robot_radius_m argument — so TRUE extents belong here.
 */

#pragma once

#include <cstdint>
#include <string>
#include <cstdio>
#include <vector>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "viewpoint_score.h"

namespace rc::nbv
{

// Every fitted object in the graph except `self_id`, as TRUE oriented footprints in the room frame.
//
// Nodes without a fitted footprint yet (no width_m/depth_m, or zero extent) are skipped: an object we cannot
// size is one we cannot reason about geometrically, and inventing a size for it would fabricate occlusion.
// ts==0 on get_transformation_matrix ⇒ MAIN-THREAD ONLY (the InnerEigenAPI ts==0 cache is unlocked; see
// CLAUDE.md). Every current caller is on the compute/main thread.
// The same footprints WITH the identity of the node they came from. Obstacle stays pure geometry (it is the
// input to the standalone-testable viewpoint scorer), so identity rides alongside rather than inside it.
// rc::exclusion needs it: "somebody else already claims this space" is not a useful thing to log without
// saying WHO, and a claim has to be recognisable across cycles.
//
// ★AND THE VERTICAL BAND, added 2026-08-16 — the z was always in the RT matrix and was being thrown away.
// A footprint alone cannot answer "is this space already taken": a kitchen has a hood over a worktop, a wall
// unit over a base unit and a bottle on a table, and in plan view each of those pairs reads as one object
// fully inside another. Measured on the live run: hood_1 (z 1.99..2.28) sits 100.0% inside the footprint of
// cabinet_w13_base (z 0.02..0.76), so the explained-away rule was discarding ~0.97 m of the very wall run
// the cabinet was fitting, as "already explained" by something 1.2 m above it.
//
// The convention is uniform and deliberate across the fleet (see the identical comment in every
// <concept>_scene_graph.cpp): NODE ORIGIN = BASE, top = origin.z + height_m. So the band is
// [M(2,3), M(2,3) + height_m] and needs no per-concept adapter.
//
// z1 <= z0 means UNKNOWN vertical extent (no height_m published). Callers must read that as "spans every
// height" so an unsized node keeps its old, purely-2-D behaviour rather than silently ceasing to claim.
struct IdentifiedObstacle
{
    Obstacle      fp{};
    std::string   name;
    std::uint64_t id = 0;
    float         z0 = 0.0f;   // base, room frame (m)
    float         z1 = 0.0f;   // top; <= z0 ⇒ height unknown ⇒ treat as unbounded
};

inline std::vector<IdentifiedObstacle> collect_graph_obstacles_identified(DSR::DSRGraph& G,
                                                                          DSR::InnerEigenAPI* inner_eigen,
                                                                          std::uint64_t self_id)
{
    std::vector<IdentifiedObstacle> obs;
    if (inner_eigen == nullptr)
        return obs;
    for (const char* type : {"object", "box"})
        for (const auto& n : G.get_nodes_by_type(type))
        {
            if (n.id() == self_id)
                continue;
            const auto w = G.get_attrib_by_name<width_m_att>(n);
            const auto d = G.get_attrib_by_name<depth_m_att>(n);
            if (not (w.has_value() and d.has_value()))
                continue;
            if (not (w.value() > 0.0f and d.value() > 0.0f))
                continue;
            const auto tr = inner_eigen->get_transformation_matrix("room", n.name(), 0);
            if (not tr.has_value())   // ALWAYS check: a missing node/edge in the RT chain returns nullopt
                continue;
            const auto& M = tr.value();
            // Vertical band: origin is the BASE by fleet convention, top = base + height_m. A node with no
            // height_m leaves z1 == z0, which every consumer reads as "unknown ⇒ spans all heights".
            const float z0 = static_cast<float>(M(2, 3));
            const auto  hm = G.get_attrib_by_name<height_m_att>(n);
            const float z1 = (hm.has_value() and hm.value() > 0.0f) ? z0 + hm.value() : z0;
            obs.push_back({{static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)),
                            w.value(), d.value(),
                            std::atan2(static_cast<float>(M(1, 0)), static_cast<float>(M(0, 0)))},
                           n.name(), n.id(), z0, z1});
        }
    return obs;
}

// The geometry-only view every existing caller uses. ONE walk of the graph, two projections of it — adding a
// second loop here is exactly the duplication this file was written to end.
//
// ★`with_height` DEFAULTS OFF, AND THAT DEFAULT IS THE WHOLE POINT. Leaving the band unset makes every
// obstacle infinitely tall, which is what the sight test did before it could reason about z. That
// over-occludes ⇒ visible_fraction too low ⇒ p_detect too low ⇒ absence charged too weakly ⇒ objects are
// HELD. Held is the recoverable error; the honest 3-D answer raises p_detect and therefore REMOVES more,
// and it would do so on top of a detector envelope that is still flagged UNCALIBRATED. So the geometry
// lands correct and inert: pass true only where the result is being measured against detect_probe.
inline std::vector<Obstacle> collect_graph_obstacles(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner_eigen,
                                                    std::uint64_t self_id, bool with_height = false)
{
    std::vector<Obstacle> obs;
    for (const auto& o : collect_graph_obstacles_identified(G, inner_eigen, self_id))
    {
        obs.push_back(o.fp);
        if (with_height) { obs.back().z0 = o.z0; obs.back().z1 = o.z1; }
    }
    return obs;
}

// ─── the camera model, read from the graph ────────────────────────────────────────────────────────────────
//
// ★MUST BE CALLED PER CYCLE, NOT ONCE AT STARTUP. The zed node's intrinsics are published by robot_concept
// when RGB frames start arriving (cam_rgb_focalx/focaly/width/height), so an agent that reads them in
// initialize() is racing the producer. Losing that race is SILENT and its failure mode is specific and bad:
// fy/H unavailable ⇒ vfov stays 0 ⇒ has_vertical() false ⇒ the fill model collapses to HORIZONTAL-ONLY, which
// is precisely the bug rc::nbv exists to fix. Measured on the live 0.64 x 0.64 x 1.92 m refrigerator belief:
// fully bound → 3.43 m stand-off from the face; vfov missing → 0.64 m. Same code, same object, one missing
// attribute, and the second one drives the robot nose-to-nose with the box.
//
// So: call this every cycle and pass the result in. It is two graph reads and one inner_eigen lookup — far
// cheaper than being wrong, and it self-heals the moment the producer comes up. `env` is the agent's detector
// envelope (config-driven); the returned Sensor carries it so the caller has one complete object.
//
// A missing zed node / camera API / room→zed transform leaves the corresponding field at its DEFAULT, which
// for vfov means "no vertical channel". Callers that want to notice should check has_vertical().
inline Sensor sensor_from_graph(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner_eigen,
                                const rc::detect::DetectorEnvelope& env = {})
{
    Sensor s;
    s.env = env;
    const auto zed = G.get_node("zed");
    if (not zed.has_value())
        return s;
    if (auto cam = G.get_camera_api(zed.value()); cam)
    {
        const float fx = cam->get_focal_x(), fy = cam->get_focal_y();
        const float W  = static_cast<float>(cam->get_width()), H = static_cast<float>(cam->get_height());
        if (fx > 0.0f and W > 0.0f) s.hfov_rad = 2.0f * std::atan(0.5f * W / fx);
        if (fy > 0.0f and H > 0.0f) s.vfov_rad = 2.0f * std::atan(0.5f * H / fy);
    }
    // Mount height in ROOM frame — NOT the proto's body-relative z: the room floor datum is offset from the
    // body origin (~3 cm) and the object z-spans are room-frame, so mixing them biases the binding axis.
    // ts==0 ⇒ MAIN THREAD ONLY (the InnerEigenAPI ts==0 cache is unlocked; CLAUDE.md).
    if (inner_eigen != nullptr)
        if (const auto rtz = inner_eigen->get_transformation_matrix("room", "zed", 0); rtz.has_value())
            s.height_m = static_cast<float>(rtz.value()(2, 3));

    // Say ONCE what is missing. Silence is what made this class of failure expensive: an incomplete model
    // still produces a confident answer, so the only visible symptom was a robot parked against a box. The
    // height_m case is the long one — room->zed needs the ROOM node, so it can stay 0 for many seconds after
    // the intrinsics have arrived. Callers refuse via Sensor::complete(); this just makes the wait legible.
    static bool announced = false;
    if (not announced and not s.complete())
    {
        announced = true;
        std::fprintf(stderr, "[nbv] camera model INCOMPLETE — no viewpoint will be proposed until it fills in "
                             "(hfov=%.1f deg vfov=%.1f deg height=%.2f m).%s%s\n",
                     s.hfov_rad * 57.29578f, s.vfov_rad * 57.29578f, s.height_m,
                     s.has_vertical() ? "" : "  MISSING: vfov (zed cam_rgb_focaly/height).",
                     s.height_m > 1e-3f ? "" : "  MISSING: mount height (room->zed; needs the ROOM node).");
    }
    return s;
}

}  // namespace rc::nbv
