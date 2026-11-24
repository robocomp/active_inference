// gain_query_test — the engine's offer gain at a GIVEN standpoint (EpistemicPlanner::gain_at; shared-engine plan rev 2, P1
// follow-up) and the request-line parser (countable_gain_query.h).
//   1. At the planner's own winning standpoint, gain_at == the proposal's promised_nats, bit for bit: one computation, not two.
//   2. After parked camera updates from the front, a look from that same front standpoint promises less than a look from the
//      side (ρ = 1 along a zero path: the shared camera error is already known; the side view decorrelates it).
//   3. Request lines parse with std::from_chars under the agents' es_ES locale (comma decimal separator), malformed ones drop.
// Fixture: the golden nightstand belief of persistent_belief_test (persistent path). Run: ./gain_query_test
#include <bit>
#include <clocale>
#include <cmath>
#include <cstdio>
#include "countable_belief.h"
#include "countable_gain_query.h"
#include "epistemic_planner.h"
#include "beam_fixture.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr float tx = 0.30f, ty = -0.255f, tH = 0.62f, tw = 0.51f;
static const Eigen::Vector2f kRobot(tx, ty - 1.5f);

static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 0.90f;
    p.room_axis_std    = 3.0f * 0.017453293f;
    p.volatility_infer = true;
    p.move_rate_per_day = 0.011f; p.move_scale_m = 0.3f; p.move_scale_rad = 0.5f;
    p.persistent       = true;
    p.beam             = rc::test::test_beam_config();
    return p;
}
// 200 noise-free points on the front and left faces of the truth box, the latents at the config floors, the robot parked at kRobot.
static CountableFrame camera_frame()
{
    CountableFrame f;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        f.points.emplace_back(tx - 0.5f * tw + tw * (i + 0.5f) / 12.0f, ty - 0.5f * tw, tH * (j + 0.5f) / 10.0f);
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 10; ++j)
        f.points.emplace_back(tx - 0.5f * tw, ty - 0.5f * tw + tw * (i + 0.5f) / 8.0f, tH * (j + 0.5f) / 10.0f);
    const CountableBeliefParams p = params();
    f.R.assign(f.points.size(), p.sigma_base_m * p.sigma_base_m);
    f.z_top_obs = tH;
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 0.0).asDiagonal();
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
    f.robot_xy = kRobot;
    return f;
}

int main()
{
    std::setlocale(LC_ALL, "es_ES.UTF-8");   // the agents run under it (CLAUDE.md): from_chars must not care

    CountableBelief b(CountableBeliefState{tx + 0.05f, ty - 0.04f, tH + 0.03f, tw, tw, 0.0f}, params());
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    const CountableFrame f = camera_frame();
    for (int k = 0; k < 20; ++k) b.update(f);

    rc::nbv::Sensor cam;                     // the ZED as the agent models it: real FoVs and the mount height
    cam.hfov_rad = 1.92f; cam.vfov_rad = 1.18f; cam.height_m = 0.98f;
    EpistemicPlanner planner;
    const EpistemicProposal prop = planner.compute(b, 0.0f, params().sigma_base_m, cam);
    CHECK(prop.valid);
    if (prop.valid)
    {
        const float g = planner.gain_at(b, {prop.epistemic_target_x_m, prop.epistemic_target_y_m}, prop.epistemic_target_yaw_rad, cam);
        std::printf("1. offer at (%.4f, %.4f, %.4f): promised %.9g, gain_at %.9g\n", prop.epistemic_target_x_m,
                    prop.epistemic_target_y_m, prop.epistemic_target_yaw_rad, prop.promised_nats, g);
        CHECK(std::bit_cast<std::uint32_t>(g) == std::bit_cast<std::uint32_t>(prop.promised_nats));
    }

    // 2. seen (the parked front standpoint) against unseen (the left side, 1.5 m off the left face), both facing the centre
    const Eigen::Vector2f c(tx, ty), side(tx - 0.5f * tw - 1.5f, ty);
    const float g_seen   = planner.gain_at(b, kRobot, std::atan2(c.y() - kRobot.y(), c.x() - kRobot.x()), cam);
    const float g_unseen = planner.gain_at(b, side, std::atan2(c.y() - side.y(), c.x() - side.x()), cam);
    std::printf("2. seen front %.6f nats, unseen side %.6f nats\n", g_seen, g_unseen);
    CHECK(std::isfinite(g_seen) and std::isfinite(g_unseen));
    CHECK(g_seen >= 0.0f);
    CHECK(g_unseen > g_seen);

    // 3. the request lines
    const auto q = parse_gain_query("1795512057339 nightstand_2 -1.25 3.5e-1 1.5707963");
    CHECK(q.has_value());
    if (q) { CHECK(q->id == 1795512057339L); CHECK(q->node == "nightstand_2"); CHECK(q->x == -1.25f); CHECK(q->y == 0.35f);
             CHECK(q->yaw == 1.5707963f); }
    CHECK(not parse_gain_query("7 bed_1 1,5 2 0").has_value());        // a comma decimal: not a number here
    CHECK(not parse_gain_query("0 * 1 2 3").has_value());              // ids are positive
    CHECK(not parse_gain_query("8 * 1 2").has_value());                // a field missing
    CHECK(not parse_gain_query("9 * 1 2 3 4").has_value());            // one too many
    const auto all = new_gain_queries("3 * 0 0 0\nbad line\n5 bed_1 1 2 3\r\n4 * 1 1 1\n", 3);
    CHECK(all.size() == 2);
    if (all.size() == 2) { CHECK(all[0].id == 5 and all[0].yaw == 3.0f); CHECK(all[1].id == 4); }

    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}
