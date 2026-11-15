// class_factor_test — the class factor of existence (spec 2026-11-14 countable class existence, §3 and §9): a pure step
// function from the shape evidence Λ to L_cls, the worker-side apply_class_step that holds on an unmeasured cycle, and
// the product p_exists() = occupancy x class. Run: ./class_factor_test
#include <cmath>
#include <cstdio>
#include <vector>
#include "countable_class_evidence.h"

using namespace rc::countable;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static rc::exist::RemovalPolicy policy()
{
    rc::exist::RemovalPolicy p;
    p.logodds_max = 4.0f; p.removal_prob = 0.12f; p.remove_frames = 15.0f;
    return p;
}

int main()
{
    const float pi_cls = 0.8f, inh = 1.0f;

    // a shape the class argues against: L_cls falls below the removal boundary and the step says so
    {
        const auto s = class_factor_step({-40.0f}, {0.9f}, pi_cls, inh);
        CHECK(s.size() == 1 and s[0].L_cls < logit(0.05f) and s[0].argued_against);
    }
    // a good shape holds
    {
        const auto s = class_factor_step({5.0f}, {0.9f}, pi_cls, inh);
        CHECK(class_probability(s[0].L_cls) > 0.99f and not s[0].argued_against);
    }
    // singleton: the weaker of two is inhibited by the stronger one's product p; the stronger is untouched
    {
        const auto alone = class_factor_step({2.0f}, {0.9f}, pi_cls, inh);
        const auto two   = class_factor_step({5.0f, 2.0f}, {0.95f, 0.9f}, pi_cls, inh);
        const auto first = class_factor_step({5.0f}, {0.95f}, pi_cls, inh);
        CHECK(std::abs((alone[0].L_cls - two[1].L_cls) - 0.95f) < 1e-4f);
        CHECK(std::abs(first[0].L_cls - two[0].L_cls) < 1e-6f);
    }
    // an unmeasured cycle changes nothing: not L_cls, not the debounce, not the logged Λ
    {
        rc::exist::ExistenceBelief cls(0.7f, 4.0f);
        rc::exist::RemovalDebounce deb; deb.streak = 3.0f;
        float lam = 1.25f;
        const bool removed = apply_class_step(cls, deb, lam, /*measured=*/false, /*warm=*/true, -40.0f,
                                              ClassStep{-9.0f, true}, policy());
        CHECK(not removed and cls.logodds() == 0.7f and deb.streak == 3.0f and lam == 1.25f);
    }
    // a measured cycle sets L_cls and the logged Λ; a warm, argued-against class eventually removes, a cold one never
    {
        rc::exist::ExistenceBelief cls(logit(pi_cls), 4.0f);
        rc::exist::RemovalDebounce deb; float lam = 0.0f;
        bool removed = false;
        for (int i = 0; i < 40 and not removed; ++i)
            removed = apply_class_step(cls, deb, lam, true, true, -40.0f, ClassStep{-9.0f, true}, policy());
        CHECK(removed and lam == -40.0f and cls.logodds() == -4.0f);
        rc::exist::ExistenceBelief cold(logit(pi_cls), 4.0f);
        rc::exist::RemovalDebounce deb2; float lam2 = 0.0f;
        for (int i = 0; i < 40; ++i)
            CHECK(not apply_class_step(cold, deb2, lam2, true, /*warm=*/false, -40.0f, ClassStep{-9.0f, true}, policy()));
        CHECK(deb2.streak == 0.0f);
    }
    // the product: occupancy 0.98 x class 0.02
    {
        // CountableInstance::p_exists() is exactly this (the instance header needs Qt/DSR, so it is tested here)
        rc::exist::ExistenceBelief occ(0.0f, 8.0f), cls(0.0f, 8.0f);
        occ.set(logit(0.98f)); cls.set(logit(0.02f));
        CHECK(std::abs(existence_product(occ, cls) - 0.0196f) < 1e-4f);
    }
    std::printf(bad ? "class_factor_test: %d FAILED\n" : "class_factor_test: OK\n", bad);
    return bad != 0;
}
