// furniture_prior_test — the FURNITURE PRIORS (owner 2026-11-14) on a synthetic front-only nightstand.
// A real nightstand 0.51 x 0.51 x 0.62, back on a north wall (y = 0, interior −y), front face at y = −0.51, yaw 0.
// The belief starts on the CORRECT side of the seen face (where birth_on_priors puts it — a single face fits a box on
// either side, and the GN never crosses: see the note in main), but 20° off the room axis and 0.20 m off the wall.
// Only the FRONT face is ever seen.
#include <cmath>
#include <cstdio>
#include <random>
#include "countable_belief.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static CountableBeliefState run(bool priors, float& yaw_deg, float& back_gap, float& yaw_full)
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.51f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.62f; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;   // etc/config.toml, not the fridge's 1.20
    p.wall_flush_prior = priors ? 0.90f : 0.0f;
    p.room_axis_std    = priors ? 3.0f * 0.017453293f : 0.0f;
    CountableBeliefState s0{0.10f, -0.455f, 0.62f, 0.51f, 0.51f, 0.3491f};  // 20° off, back 0.20 m off the wall
    CountableBelief b(s0, p);
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    std::mt19937 rng(7); std::normal_distribution<float> nz(0.0f, 0.005f); std::uniform_real_distribution<float> U(0, 1);
    for (int it = 0; it < 60; ++it)
    {
        CountableFrame f;
        for (int i = 0; i < 400; ++i)                                   // the FRONT face only: y = −0.51
            f.points.emplace_back(-0.255f + 0.51f * U(rng), -0.51f + nz(rng), 0.62f * U(rng));
        f.z_top_obs = 0.62f;
        f.wall.ok = true; f.wall.p = Eigen::Vector2f(0, 0); f.wall.n = Eigen::Vector2f(0, -1);
        f.wall.sigma_m = 0.02f; f.wall.length_m = 4.0f;
        f.room_axis_ok = priors; f.room_axis = 0.0f;
        b.update(f);
    }
    const auto s = b.state();
    float y = std::fmod(s.yaw * 57.29578f, 90.0f); if (y > 45) y -= 90; if (y < -45) y += 90;
    yaw_deg = y; yaw_full = std::remainder(s.yaw, 6.2831853f) * 57.29578f;   // 0 = front to the room back_gap = -(b.back_centre(s).y());               // distance of the back face from the wall (+ = in front of it)
    std::printf("%s: cx=%.3f cy=%.3f yaw=%.1f° (mod 90 → %.1f°) w=%.3f h=%.3f H=%.3f back gap=%.3f m\n",
                priors ? "WITH priors   " : "WITHOUT priors", s.cx, s.cy, s.yaw * 57.29578f, y, s.w, s.h, s.H, back_gap);
    return s;
}

int main()
{
    float y0, g0, y1, g1, f0, f1;
    const auto s0 = run(false, y0, g0, f0);
    const auto s1 = run(true, y1, g1, f1);
    (void) s0;
    CHECK(std::abs(y1) < 2.0f);          // aligned with the room axis
    CHECK(std::abs(g1) < 0.05f);         // back on the wall
    CHECK(std::abs(f1) < 5.0f);          // FRONT faces the room (yaw 0), not the wall (yaw 180)
    CHECK(std::abs(std::abs(f0) - 180.0f) < 5.0f);   // without the priors this case ends front-to-the-wall
    CHECK(std::abs(s1.H - 0.62f) < 0.05f and std::abs(s1.cy + 0.255f) < 0.03f);
    std::printf("furniture_prior_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
