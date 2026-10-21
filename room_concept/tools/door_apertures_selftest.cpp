/*
 *  door_apertures_selftest.cpp — DoorApertures::refresh on a REAL DSR graph (private DDS domain).
 *
 *  Task R2 side finding (live rtD1i): room_concept seated door_5 (= DOOR_0, a horizontal opening at ~(0.67, 2.54) in
 *  room_1) as a VERTICAL aperture at x ≈ 0, y 2.16–3.20, on every recall. Where it came from: door_concept's own state for
 *  the restored door — its fitter sat at (−0.006, 2.67) yaw 1.547 right after each restore (ai2_log.csv) although the
 *  restore seeded (0.89, 2.66) yaw −3.20, and in visit 1 its fit converged to (0.11, 2.02) yaw −1.56 after the leaf was
 *  read open — written as the door node's RT edge. room_concept reads that edge verbatim, so the 90° error is the
 *  producer's (door_concept, not touched here). What room_concept DID add: it walked each door's chain ONCE per node and
 *  reused the segment for the node's whole life ("walked once, reused from here"), so a door the producer later re-fitted
 *  stayed wherever room_concept first read it. This test pins the fix: the cache follows the door's own RT edge.
 *
 *  Expect: (1) the first read is the edge as written; (2) door_concept rewrites the edge (the correct horizontal
 *  aperture) ⇒ the next refresh returns the NEW segment (before: the first one, for ever); (3) an unchanged edge is not
 *  re-walked (the chain-walk counter does not move).
 *
 *  Build + run:  make -C build door_apertures_selftest && ../bin/door_apertures_selftest   (exit code = failures)
 */
#include "door_apertures.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QCoreApplication>

#include <clocale>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <string>

namespace
{
    int failures = 0;
    void check(bool ok, const char* what)
    {
        std::printf("  %-78s %s\n", what, ok ? "PASS" : "FAIL");
        failures += ok ? 0 : 1;
    }
    bool near_seg(const rc::DoorAperture& d, const Eigen::Vector2f& a, const Eigen::Vector2f& b, float tol = 0.02f)
    {
        return ((d.a - a).norm() < tol and (d.b - b).norm() < tol) or ((d.a - b).norm() < tol and (d.b - a).norm() < tol);
    }
}   // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);   // the InnerEigenAPI's cache invalidation is a queued slot: it needs an event loop
    std::setlocale(LC_ALL, "");          // behave like the Qt agent (es_ES) — CLAUDE.md
    auto G = std::make_shared<DSR::DSRGraph>("door_apertures_selftest", 991, std::string(SHADOW_JSON), true,
                                             static_cast<int8_t>(73), DSR::Queue);
    auto rt = G->get_rt_api();
    auto inner = G->get_inner_eigen_api();
    auto root = G->get_node_root();
    if (not root.has_value()) { std::printf("no root node in %s\n", SHADOW_JSON); return 1; }

    DSR::Node room = DSR::Node::create<room_node_type>("room_t");
    const auto room_id = G->insert_node(room);
    if (not room_id.has_value()) { std::printf("room insert failed\n"); return 1; }
    rt->insert_or_assign_edge_RT(root.value(), room_id.value(), std::vector<float>{0.f, 0.f, 0.f}, std::vector<float>{0.f, 0.f, 0.f});

    DSR::Node door = DSR::Node::create<object_node_type>("door_t");
    G->add_or_modify_attrib_local<object_subtype_att>(door, std::string("door"));
    G->add_or_modify_attrib_local<width_m_att>(door, 1.0f);
    G->add_or_modify_attrib_local<door_open_prob_att>(door, 0.0f);
    const auto door_id = G->insert_node(door);
    if (not door_id.has_value()) { std::printf("door insert failed\n"); return 1; }
    auto write_door = [&](float x, float y, float yaw)
    {
        auto rn = G->get_node(room_id.value());
        rt->insert_or_assign_edge_RT(rn.value(), door_id.value(), std::vector<float>{x, y, 0.f}, std::vector<float>{0.f, 0.f, yaw});
        for (int k = 0; k < 5; ++k) QCoreApplication::processEvents();
    };

    rc::DoorApertures doors;
    // (1) door_concept's restored state, as live: vertical at x ≈ 0 (yaw π/2 about the aperture centre).
    write_door(-0.006f, 2.67f, 0.5f * std::numbers::pi_v<float>);
    auto out = doors.refresh(*G, *inner, "room_t");
    check(out.size() == 1 and near_seg(out[0], {-0.006f, 2.17f}, {-0.006f, 3.17f}),
          "(1) the first read is the edge as door_concept wrote it (vertical)");
    const long walks1 = doors.resolves();

    // (2) door_concept re-fits it to the true horizontal opening at (0.67, 2.54), yaw π.
    write_door(0.67f, 2.54f, std::numbers::pi_v<float>);
    // (Measured while writing this test: re-walking the chain through the SHARED InnerEigenAPI at ts == 0 after the
    //  rewrite still returned the first placement, (−0.01, 2.67), where a fresh instance returned (0.67, 2.54) — its
    //  cache entry for room<-door was not invalidated by the local edge rewrite. So refresh() reads the door's own link
    //  through RT_API and uses the cache only for the links above it, which a re-fit does not touch.)
    out = doors.refresh(*G, *inner, "room_t");
    check(out.size() == 1 and near_seg(out[0], {0.17f, 2.54f}, {1.17f, 2.54f}),
          "(2) a rewritten RT edge is followed (pre-R2: the first read was kept for the node's life)");

    // (3) nothing written since: no new chain walk.
    const long walks2 = doors.resolves();
    out = doors.refresh(*G, *inner, "room_t");
    check(walks2 == walks1 + 1 and doors.resolves() == walks2 and out.size() == 1,
          "(3) an unchanged edge is not re-walked (one walk per placement)");

    std::printf("door_apertures_selftest: %d failure(s)\n", failures);
    G.reset();
    return failures;
}
