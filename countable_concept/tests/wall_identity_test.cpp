// wall_identity_test — the wall priors enter ONCE per physical wall (F1 round 6.3; review I2). The identity is geometric (same
// room, same facing, within 3σ in offset and angle on the lines' own σ), not room_concept's edge index: (1) the outline is
// renumbered between updates (edge 0 → 3 → 7 → 0) and the staged line jitters by 5 mm — the priors entered on the first update
// only; (2) the nearest wall alternates A → B → A → B (a box in a corner) — the priors entered on the first A and the first B
// only, never again (CW1 fix round 2: on the first update of each that converged, since an entry is committed only when absorbed).
// Run: ./wall_identity_test
#include <cstdio>
#include <vector>
#include "countable_belief.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.5f; p.prior_depth_m = 0.5f; p.prior_footprint_std = 0.06f; p.prior_height_m = 0.6f; p.prior_height_std = 0.1f;
    p.wall_flush_prior = 1.0f; p.room_axis_std = 3.0f * 0.017453293f; p.persistent = true;
    p.wall_standoff_m = 0.03f; p.wall_standoff_std_m = 0.02f;
    return p;
}
static WallRef wall(float px, float py, float nx, float ny, int edge)
{
    WallRef w; w.ok = true; w.p = {px, py}; w.n = {nx, ny}; w.sigma_m = 0.02f; w.length_m = 3.0f; w.room = 1; w.edge = edge;
    return w;
}
static CountableFrame frame(const WallRef& w)
{
    CountableFrame f;
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 8; ++j)      // the front face of a 0.5 box centred at (0.7, −0.28)
        f.points.emplace_back(0.45f + 0.5f * (i + 0.5f) / 10.0f, -0.53f, 0.6f * (j + 0.5f) / 8.0f);
    f.R.assign(f.points.size(), 0.03f * 0.03f);
    f.wall = w;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    const float c2 = 0.0289f * 0.0289f;
    f.sig2_cam = {c2, c2, c2, c2, c2, 9e-4f}; f.sig2_L = f.sig2_cam;
    f.sigma_pose = Eigen::Vector3d(1e-8, 1e-8, 0.0).asDiagonal();
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
    f.robot_xy = {0.7f, -2.5f};
    return f;
}

int main()
{
    std::printf("wall_identity_test (F1 round 6.3)\n");
    const WallRef A0 = wall(0.0f, 0.0f, 0.0f, -1.0f, 0);   // the wall behind the box (y = 0, room at y < 0)
    {   // (1) renumbering + 5 mm jitter
        CountableBelief b(CountableBeliefState{0.7f, -0.28f, 0.6f, 0.5f, 0.5f, 0.0f}, params());
        b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
        std::vector<int> entered;
        int k = 0;
        for (const int edge : {0, 3, 7, 0, 5})
        {
            WallRef w = A0; w.edge = edge; w.p.y() += (k % 2 ? 0.005f : -0.005f);
            b.update(frame(w));
            entered.push_back(b.last_wall_entered() ? 1 : 0);
            ++k;
        }
        std::printf("  renumbered 0,3,7,0,5 (+-5 mm): entered %d %d %d %d %d\n", entered[0], entered[1], entered[2], entered[3], entered[4]);
        CHECK(entered == (std::vector<int>{1, 0, 0, 0, 0}));
    }
    {   // (2) alternation A → B → A → B (B: the side wall x = 1.0, room at x < 1)
        const WallRef B = wall(1.0f, 0.0f, -1.0f, 0.0f, 1);
        CountableBelief b(CountableBeliefState{0.7f, -0.28f, 0.6f, 0.5f, 0.5f, 0.0f}, params());
        b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
        std::vector<int> entered, conv;
        for (int k = 0; k < 6; ++k)
        {
            b.update(frame(k % 2 ? B : A0));
            entered.push_back(b.last_wall_entered() ? 1 : 0);
            conv.push_back(b.last_gn_converged() ? 1 : 0);
        }
        std::printf("  alternating A,B,A,B,A,B: entered %d %d %d %d %d %d  (GN converged %d %d %d %d %d %d)\n", entered[0], entered[1], entered[2],
                    entered[3], entered[4], entered[5], conv[0], conv[1], conv[2], conv[3], conv[4], conv[5]);
        // CW1 fix round 2 (N2): a wall's priors are recorded as entered only on an update that ABSORBED them (converged; an unconverged
        // update keeps P = P̄). So each physical wall enters exactly once, on its first converged update: A on the first A update that
        // converged, B on the first converged B update, and never again. (Here the B updates are side-on to B, where the parallel term
        // sits at its maximum — review M5, parked — and the first B updates cap.)
        std::vector<int> expect(6, 0);
        for (const int parity : {0, 1})
            for (int k = parity; k < 6; k += 2)
                if (conv[std::size_t(k)]) { expect[std::size_t(k)] = 1; break; }
        CHECK(entered == expect);
    }
    std::printf(bad ? "wall_identity_test: %d FAILED\n" : "wall_identity_test: OK\n", bad);
    return bad != 0;
}
