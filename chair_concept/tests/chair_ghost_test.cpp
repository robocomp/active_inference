// The ghost at a remembered chair's old spot (changed-object, ghost_p_vis) must be a chair of the PRIOR's size.
// Live run moved_chair_2 (2026-11-16): it was built on a default ChairModel, whose all-zero bounds clamp every
// dimension to 0, so its projected box had fill 0, P(detect) 0, and the old spot never gathered evidence.
// Run: ./chair_ghost_test
#include "chair_ghost.h"
#include "chair_prior.h"
#include <clocale>
#include <cmath>
#include <print>
#include <string>

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const auto prior = rc::load_chair_prior(std::string(AGENT_DIR) + "/../common/concept_manifest/chair.concept.toml");
    CHECK(prior.has_value());
    if (not prior) return 1;
    rc::ChairModelParams mp;
    mp.chair = *prior;

    const auto m = rc::ghost_model(-2.57f, -0.61f, mp);
    const auto& s = m.state();
    CHECK(std::abs(s.cx + 2.57f) < 1e-6f and std::abs(s.cy + 0.61f) < 1e-6f);
    CHECK(s.yaw == 0.f);
    CHECK(std::abs(s.seat_w - prior->seat_w) < 1e-6f);
    CHECK(std::abs(s.seat_d - prior->seat_d) < 1e-6f);
    CHECK(std::abs(s.seat_h - prior->seat_h) < 1e-6f);
    CHECK(std::abs(s.back_h - prior->back_h) < 1e-6f);
    CHECK(s.seat_w > 0.f and s.back_h > 0.f);
    if (failures == 0) std::println("chair_ghost_test: all passed");
    return failures == 0 ? 0 : 1;
}
