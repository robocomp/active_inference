// config_load_test — K1: every countable class's config loads through the agent's OWN loader (rc::load_countable_config, the
// real ConfigLoader.cpp, from the class folder as the agent's cwd, under es_ES — CLAUDE.md), and the class facts arrive:
//   refrigerator (etc/countable.toml): floor-anchored (vertical_extent_m ≤ 0), its own fitted envelope, eps_px 0.066,
//                 permanence + stand-off from its manifest;
//   hood (etc/countable.toml): HANGS — vertical_extent_m = the manifest's extent_m 0.50, π_flush 1, its own depth std;
//   bed / nightstand / toilet (etc/config.toml): still floor-anchored (unchanged by K1).
// A refused start (CountableConfigError: an inherited manifest block, a malformed [prior.*]) fails the test.
// Run: LANG=es_ES.UTF-8 ./config_load_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "countable_config.h"
#include <genericworker.h>   // the test stub: the real ConfigLoader (manifest_stub/)

#ifndef REPO_DIR
#error "REPO_DIR must name the repository root"
#endif
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

static std::optional<rc::CountableConfig> load(const std::string& cls, const std::string& rel)
{
    std::filesystem::current_path(std::filesystem::path(REPO_DIR) / (cls + "_concept"));
    ConfigLoader cl;
    cl.load(rel);
    try { return rc::load_countable_config(cl); }
    catch (const std::exception& e) { std::printf("%s: REFUSED: %s\n", cls.c_str(), e.what()); return std::nullopt; }
}

int main()
{
    std::setlocale(LC_ALL, "");
    for (const char* cls : {"bed", "nightstand", "toilet"})
    {
        const auto c = load(cls, "etc/config.toml");
        CHECK(c.has_value());
        if (c) CHECK(c->concept_class == cls and c->vertical_extent_m <= 0.0f and c->ai2_prior_depth_std <= 0.0f);
    }
    if (const auto c = load("refrigerator", "etc/countable.toml"); c)
    {
        CHECK(c->concept_class == "refrigerator" and c->agent_name == "refrigerator_concept");
        CHECK(c->vertical_extent_m <= 0.0f);
        CHECK(near(c->ai2_prior_footprint_m, 0.60f) and near(c->ai2_prior_height_m, 1.90f));
        CHECK(near(c->detect_min_fill, 0.0f) and near(c->detect_max_fill, 1.32f) and near(c->detect_soft, 0.26f));
        CHECK(near(c->mask_pixel_error_rate, 0.066f));
        CHECK(near(c->move_rate_per_day, 0.00021f) and near(c->wall_standoff_m, 0.05f) and near(c->wall_standoff_std_m, 0.02f));
        CHECK(near(c->ai2_wall_flush_prior, 0.0f) and c->front_detect_enabled);
        std::printf("refrigerator: LOADED, floor-anchored, H %.2f, stand-off %.3f ± %.3f\n", c->ai2_prior_height_m,
                    c->wall_standoff_m, c->wall_standoff_std_m);
    }
    else CHECK(false);
    if (const auto c = load("hood", "etc/countable.toml"); c)
    {
        CHECK(c->concept_class == "hood" and c->agent_name == "hood_concept");
        CHECK(near(c->vertical_extent_m, 0.50f));
        CHECK(near(c->ai2_prior_footprint_m, 0.90f) and near(c->ai2_prior_footprint_std, 0.30f));
        CHECK(near(c->ai2_prior_depth_m, 0.50f) and near(c->ai2_prior_depth_std, 0.15f) and near(c->ai2_prior_height_m, 2.05f));
        CHECK(near(c->ai2_wall_flush_prior, 1.0f) and not c->front_detect_enabled);
        CHECK(near(c->existence_detection_prob, 0.55f));
        CHECK(near(c->move_rate_per_day, 0.000196f) and near(c->wall_standoff_m, 0.0f) and near(c->wall_standoff_std_m, 0.01f));
        std::printf("hood: LOADED, hangs [%.2f, %.2f]\n", c->ai2_prior_height_m - c->vertical_extent_m, c->ai2_prior_height_m);
    }
    else CHECK(false);
    std::printf(bad ? "config_load_test FAILED (%d)\n" : "config_load_test OK\n", bad);
    return bad ? 1 : 0;
}
