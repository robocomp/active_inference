// back_panel_test — the back panel (headboard) comes from the class manifest, not from config (Task 3).
// Run: LANG=es_ES.UTF-8 ./back_panel_test   (setlocale(LC_ALL,"") below: the harness must see the agent's locale)
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "back_panel.h"
#include "countable_belief.h"

#ifndef MANIFEST_DIR
#error "MANIFEST_DIR must name common/concept_manifest"
#endif

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static std::string mf(const char* cls) { return std::string(MANIFEST_DIR) + "/" + cls + ".concept.toml"; }

// The params the agent builds (countable_fitter.cpp) for the headboard fields, from the manifest.
static rc::CountableBeliefParams params_for(const char* cls)
{
    rc::CountableBeliefParams p;
    if (const auto bp = rc::manifest::load_back_panel(mf(cls)))
    { p.headboard_h_m = bp->height_m; p.headboard_t_m = bp->thickness_m; }
    return p;
}
static bool bits_equal(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

int main()
{
    std::setlocale(LC_ALL, "");
    const auto bed = rc::manifest::load_back_panel(mf("bed"));
    CHECK(bed.has_value());
    if (bed) { CHECK(bed->height_m == 1.25f); CHECK(bed->thickness_m == 0.10f); }
    CHECK(not rc::manifest::load_back_panel(mf("nightstand")).has_value());
    CHECK(not rc::manifest::load_back_panel(mf("toilet")).has_value());
    CHECK(not rc::manifest::load_back_panel(mf("no_such_class")).has_value());

    const rc::CountableBeliefState s{0.0f, 0.0f, 0.6f, 0.5f, 0.5f, 0.3f};
    std::mt19937 rng(20261114); std::uniform_real_distribution<float> U(-1.5f, 1.5f), Uz(-1.0f, 1.0f);
    for (const char* cls : {"nightstand", "toilet", "bed"})
    {
        const rc::CountableBelief b(s, params_for(cls));
        int diff = 0;
        for (int i = 0; i < 200; ++i)
        {
            const Eigen::Vector3f p(U(rng), U(rng), Uz(rng) + 1.0f);   // 3x3x2 m cube around the state
            const float prim = b.sdf_prim(p, s, 0);
            const float want = std::strcmp(cls, "bed") == 0 ? std::min(b.sdf_box(p, s), b.sdf_headboard(p, s))
                                                            : b.sdf_box(p, s);
            if (not bits_equal(prim, want)) ++diff;
        }
        std::printf("%s: %d / 200 points differ\n", cls, diff);
        CHECK(diff == 0);
    }
    std::printf("back_panel_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
