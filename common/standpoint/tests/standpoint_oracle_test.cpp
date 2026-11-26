// standpoint_oracle_test.cpp — the shared standpoint oracle (common/standpoint/standpoint_oracle.h) against synthetic grids.
//   1. a free room: P(stand and turn) = 1;
//   2. a wall approaching the turn circle: P = 1 while the wall is beyond the body's reach, 0 once it is inside, never rising as
//      it comes closer; between the two there is a distance where the body STANDS at one heading but cannot TURN;
//   3. partial occupancy: soft cells multiply P by exp(−nats) each, counted once through the turn; hard ⇒ 0; collapsed ⇒ nothing;
//      a live-hard cell vetoes only when the caller does not ask for persistent causes;
//   4. continuity: P is a continuous, monotone function of the residual belief behind a cell (no step anywhere in the evidence);
//   5. no world ⇒ 1 (no belief, no veto); a residual grid with no field ⇒ every cell hard (the controller's convention).
#include "../standpoint_oracle.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace
{
int failures = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  [%s] %s\n", ok_ ? "ok" : "FAIL", msg); if (not ok_) ++failures; } while (0)

using V2 = Eigen::Vector2f;
using rc::standpoint::Oracle;
constexpr float kInf = std::numeric_limits<float>::infinity();
const rc::RobotFootprint kBody = rc::RobotFootprint::shadow();

std::vector<V2> rect(float x0, float y0, float x1, float y1) { return {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}; }

rc::OccupiedCells soft(std::vector<V2> c, std::vector<float> r) { rc::OccupiedCells o; o.cell_size_m = 0.05f; o.centres = std::move(c); o.risk_nats = std::move(r); return o; }
}   // namespace

int main()
{
    const float cell = 0.05f;
    const std::vector<std::vector<V2>> none;

    std::printf("1. a free room\n");
    {
        Oracle o(kBody, cell, 0.f);
        const auto room = rect(0, 0, 4, 4);
        o.set_world(room, none, none, {});
        CHECK(o.p_stand_and_turn({2.f, 2.f}) == 1.0 and o.p_stand({2.f, 2.f}, 0.4f) == 1.0, "P(stand and turn) = 1 in open floor");
        CHECK(o.p_stand_and_turn({0.02f, 2.f}) == 0.0, "...and 0 with the centre on the wall");
        CHECK(o.p_stand_and_turn({9.f, 9.f}) == 0.0, "...and 0 off the lattice");
    }

    std::printf("2. a wall at the edge of the turn circle\n");
    {
        Oracle o(kBody, cell, 0.f);
        const float rin = kBody.inscribed_radius(), rout = kBody.circumscribed_radius();
        double last = 1.0;
        bool monotone = true, stand_not_turn = false;
        float d_first_zero = -1.f;
        for (float d = 0.60f; d >= 0.10f; d -= 0.005f)
        {
            const auto room = rect(0, 0, 4, 4);   // the wall x = 0; the standpoint d from it
            o.set_world(room, none, none, {});
            const double p = o.p_stand_and_turn({d, 2.f});
            if (p > last) monotone = false;
            last = p;
            if (p == 0.0 and d_first_zero < 0.f) d_first_zero = d;
            bool some_heading = false;
            for (int h = 0; h < 8; ++h) some_heading = some_heading or o.p_stand({d, 2.f}, static_cast<float>(h) * 0.785398f) > 0.0;
            if (p == 0.0 and some_heading) stand_not_turn = true;
        }
        o.set_world(rect(0, 0, 4, 4), none, none, {});
        CHECK(o.p_stand_and_turn({rout + 2.f * cell, 2.f}) == 1.0, "wall beyond the circumscribed radius + a cell: P = 1");
        CHECK(o.p_stand_and_turn({rin - cell, 2.f}) == 0.0, "wall inside the inscribed radius: P = 0 at every heading");
        CHECK(monotone, "P never rises as the wall comes closer");
        std::printf("      first P = 0 at d = %.3f m (inscribed %.3f, circumscribed %.3f)\n", d_first_zero, rin, rout);
        CHECK(d_first_zero > rin and d_first_zero <= rout + cell, "the turn is lost between the inscribed and circumscribed radii");
        CHECK(stand_not_turn, "there is a distance where the body stands at one heading but cannot turn (a rigid, non-round body)");
    }

    std::printf("3. partial occupancy\n");
    {
        const auto room = rect(0, 0, 4, 4);
        const V2 p{2.f, 2.f};
        Oracle o(kBody, cell, 0.f);
        o.set_world(room, none, none, soft({{2.025f, 2.025f}}, {1.5f}));
        CHECK(std::abs(o.p_stand_and_turn(p) - std::exp(-1.5)) < 1e-9, "one soft cell under the body: P = exp(-nats), counted ONCE through the turn");
        o.set_world(room, none, none, soft({{2.025f, 2.025f}, {2.125f, 1.975f}, {1.925f, 2.075f}}, {1.0f, 1.0f, 1.0f}));
        CHECK(std::abs(o.p_stand_and_turn(p) - std::exp(-3.0)) < 1e-6, "three 1-nat cells: P = exp(-3) (independent cells, nats add)");
        o.set_world(room, none, none, soft({{3.025f, 2.025f}}, {5.0f}));
        CHECK(o.p_stand_and_turn(p) == 1.0, "a soft cell 1 m away: P = 1");
        o.set_world(room, none, none, soft({{2.025f, 2.025f}}, {0.0f}));
        CHECK(o.p_stand_and_turn(p) == 1.0, "a collapsed cell (explained by a modelled object, 0 nats): P = 1");
        o.set_world(room, none, none, soft({{2.025f, 2.025f}}, {kInf}));
        CHECK(o.p_stand_and_turn(p) == 0.0, "a hard cell (no belief behind it): P = 0");
        rc::OccupiedCells live = soft({{2.025f, 2.025f}}, {kInf});
        live.live_hard = {1};
        o.set_world(room, none, none, live);
        CHECK(o.p_stand_and_turn(p, /*persistent_only=*/true) == 1.0 and o.p_stand_and_turn(p, false) == 0.0,
              "a live-hard cell (a return this cycle) vetoes only when transient causes count");
        const std::vector<std::vector<V2>> box{rect(2.2f, 1.9f, 2.4f, 2.1f)};
        o.set_world(room, none, box, {});
        CHECK(o.p_stand_and_turn(p) == 0.0 and o.p_stand_and_turn({3.2f, 3.2f}) == 1.0, "an object polygon in the turn: 0; away from it: 1");
        const std::vector<std::vector<V2>> passage{rect(3.9f, 1.6f, 5.0f, 2.4f)};
        o.set_world(room, passage, none, {});
        CHECK(o.p_stand_and_turn({4.4f, 2.0f}) > 0.0, "a passage polygon outside the room is floor (the controller's crossing corridor)");
    }

    std::printf("4. continuity in the belief\n");
    {
        const auto room = rect(0, 0, 4, 4);
        Oracle o(kBody, cell, 0.f);
        // one residual cell under the body, its published mean P rising from barely-seen to confirmed at fixed concentration
        double last = 1.0, max_step = 0.0;
        bool monotone = true, all_positive = true;
        const std::vector<float> xyz{2.025f, 2.025f, 0.3f};
        for (int k = 0; k <= 400; ++k)
        {
            const float P = 0.30f + 0.69f * static_cast<float>(k) / 400.f;
            const float kappa = 20.f;
            rc::residual_risk::BeliefField f;
            f.xmin = 0.f; f.ymin = 0.f; f.cell = 0.05f; f.width = 80; f.height = 80;
            f.prob.assign(80 * 80, 0.f);
            f.var.assign(80 * 80, 0.f);
            const int i = 40 * 80 + 40;
            f.prob[i] = P;
            f.var[i] = P * (1.f - P) / (kappa + 1.f);
            o.set_world(room, none, none, rc::standpoint::cells_from_xyz(xyz, 0.05f, f));
            const double p = o.p_stand_and_turn({2.f, 2.f});
            if (p > last + 1e-12) monotone = false;
            if (k > 0) max_step = std::max(max_step, std::abs(p - last));
            if (not (p > 0.0)) all_positive = false;
            last = p;
        }
        std::printf("      P over 401 evidence steps: largest step %.4f, final %.3g\n", max_step, last);
        CHECK(monotone, "P falls monotonically as the evidence for the cell grows");
        CHECK(max_step < 0.02, "no jump: every 0.17 % step in the cell's mean moves P by < 0.02 (continuous, no threshold)");
        CHECK(all_positive, "a soft cell never vetoes outright: P > 0 at every evidence level");
    }

    std::printf("5. no belief, no veto; no field, every cell hard\n");
    {
        Oracle o(kBody, cell, 0.f);
        CHECK(o.p_stand_and_turn({2.f, 2.f}) == 1.0, "no world yet: P = 1 (no verdict)");
        const auto room = rect(0, 0, 4, 4);
        const std::vector<float> xyz{2.025f, 2.025f, 0.3f};
        o.set_world(room, none, none, rc::standpoint::cells_from_xyz(xyz, 0.05f, rc::residual_risk::BeliefField{}));
        CHECK(o.p_stand_and_turn({2.f, 2.f}) == 0.0, "residual cells with no field: hard, as in the controller");
    }

    std::printf(failures == 0 ? "\nALL PASS (0 failures)\n" : "\nFAILURES: %d\n", failures);
    return failures == 0 ? 0 : 1;
}
