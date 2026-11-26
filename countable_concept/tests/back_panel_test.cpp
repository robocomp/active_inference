// back_panel_test — the back panel (headboard) comes from the class manifest, not from config (Task 3).
// Run: LANG=es_ES.UTF-8 ./back_panel_test   (setlocale(LC_ALL,"") below: the harness must see the agent's locale)
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include "back_panel.h"
#include "class_prior.h"
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
    bool opened = false;
    CHECK(not rc::manifest::load_back_panel(mf("no_such_class"), &opened).has_value());
    CHECK(not opened);                                   // an open failure ...
    CHECK(not rc::manifest::load_back_panel(mf("nightstand"), &opened).has_value());
    CHECK(opened);                                       // ... is not the same as "no block"

    for (const char* cls : {"bed", "nightstand", "toilet"})
    {
        const auto cp = rc::manifest::load_class_prior(mf(cls));
        CHECK(cp.has_value());
        if (cp) CHECK(cp->p_label_true == 0.8f);
    }
    {
        struct E { const char* c; float H, w, h; };
        for (const E& e : {E{"toilet", 0.05f, 0.26f, 0.15f}, E{"bed", 0.10f, 0.05f, 0.05f}, E{"nightstand", 0.03f, 0.04f, 0.04f}})
        {
            const auto cp = rc::manifest::load_class_prior(mf(e.c));
            CHECK(cp.has_value());
            if (cp) CHECK(cp->fit_H_m == e.H and cp->fit_w_m == e.w and cp->fit_h_m == e.h);
        }
        const std::string tmp = "back_panel_test_nofit.toml";               // [prior.class] without the fit keys
        { std::ofstream o(tmp); o << "[prior.class]\nkind = \"label_reliability\"\np_label_true = 0.8\n"; }
        const auto none = rc::manifest::load_class_prior(tmp);
        std::remove(tmp.c_str());
        CHECK(none.has_value());
        if (none) CHECK(none->p_label_true == 0.8f and none->fit_H_m == 0.0f and none->fit_w_m == 0.0f and none->fit_h_m == 0.0f);
    }
    CHECK(not rc::manifest::load_class_prior(mf("cabinet")).has_value());   // a manifest without the section (was the table, which declares one since K2c)
    bool cp_opened = true;
    CHECK(not rc::manifest::load_class_prior(mf("no_such_class"), &cp_opened).has_value());
    CHECK(not cp_opened);

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
