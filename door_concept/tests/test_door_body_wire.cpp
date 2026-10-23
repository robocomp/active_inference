// Task C fix round 1: the body-in-aperture evidence at the WIRE level — DoorFit::estimate_phi itself, not only the LeafChannel —
// and the touching-a-shut-leaf case (centre ~0.25 m in front of the leaf: no evidence).
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_body_wire.cpp src/core/door_core.cpp src/core/door_fit.cpp \
//     src/core/door_existence.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp src/epistemic_planner.cpp \
//     src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
//     ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc -o /tmp/tdbw && /tmp/tdbw
#include <cmath>
#include <print>
#include "../src/core/door_fit.h"
#include "../src/door_body_evidence.h"
#include "../src/door_passability.h"

using namespace rc::door;
static int g_fail = 0;
#define CHECK(c) do { const bool ok_ = (c); std::print("  [{}] {}\n", ok_ ? "PASS" : "FAIL", #c); if (not ok_) ++g_fail; } while (0)

static float p_pass(const std::optional<Eigen::Vector2f>& body, const Eigen::Matrix2f& cov, int frames)
{
    rc::DoorConfig cfg;
    rc::door::core::DoorFit fit(cfg);
    rc::DoorFitState inst;
    rc::DoorState s; s.w = 1.0f; s.h = 2.1f; s.thickness = 0.05f;
    fit.init_instance(inst, 1, "door_1", s);
    rc::DoorBeliefParams bp;   // a door at the origin, wall along +x, leaf hung from the near jamb
    bp.thickness = 0.05f; bp.floor_z = 0.0f; bp.wall_u = Eigen::Vector2f(1, 0); bp.wall_O = Eigen::Vector2f(-0.5f, 0);
    bp.leaf = {1.5708f, HingeSide::Near, +1.0f};
    inst.ai2_belief = rc::DoorBelief({0.0f, 1.0f, 2.1f}, bp);
    inst.ai2_initialized = true;
    fit.refresh_geometry(inst);   // seats inst.aperture from the belief
    fit.set_robot_body(body, cov);
    for (int f = 0; f < frames; ++f) fit.estimate_phi(inst);
    return passable_prob(inst.phi_curve, inst.aperture.w, cfg.pragmatic.robot_passage_width_m).value_or(-1.f);
}

int main()
{
    const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * 0.0004f;
    {   // the unit: touching a shut leaf, centre 0.25 m in front, realistic belief spread (leaf plane sd 0.08 m)
        Eigen::Matrix2f C = Eigen::Matrix2f::Identity() * 0.0016f; C(1, 1) += 0.0064f; C(0, 0) += 0.0015f;
        const float p_in = body_in_aperture_prob(0.f, -0.25f, C, 0.5f, 0.025f);
        const auto ll = body_phi_loglik(p_in, 1.0f, 0.6f, 0.f, 2.0944f, 25);
        float worst = 0.f; for (float v : ll) worst = std::max(worst, std::abs(v));
        std::print("  touching a shut leaf: p_in {:.4f}, max |log L| {:.4f}\n", p_in, worst);
        CHECK(p_in < 0.01f and worst < 0.02f);
    }
    const float base = p_pass(std::nullopt, cov, 3);
    std::print("  wire: DoorFit initialised = {}\n", base != -2.f);
    CHECK(base != -2.f);
    if (base != -2.f)
    {
        // centre ~ aperture centre of a door at the origin (init puts it at (0,0) yaw 0): body at the leaf plane
        const float thru = p_pass(Eigen::Vector2f(0.f, 0.f), cov, 3);
        const float beside = p_pass(Eigen::Vector2f(3.f, -0.3f), cov, 3);
        std::print("  P(passable): no body {:.3f}, body in the aperture {:.3f}, body beside {:.3f}\n", base, thru, beside);
        CHECK(thru > base);
        CHECK(std::abs(beside - base) < 1e-6f);
    }
    std::print("{}\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
