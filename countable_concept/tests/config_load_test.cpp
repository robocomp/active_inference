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
        // K2b: every other class keeps the pre-K2b channels (the defaults)
        if (c) CHECK(c->look_attrs == "table" and not c->appearance_enabled and not c->publish_exist_prob and not c->thumbnails_enabled
                     and not c->approach_enabled and c->epistemic_offer_enabled and not c->changed_object_enabled and c->mesh_path.empty());
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
    // K2b: the chair, floor-anchored, the parts ARE its solid (seat, backrest, four legs) on one proportion, the seat top
    if (const auto c = load("chair", "etc/countable.toml"); c)
    {
        CHECK(c->concept_class == "chair" and c->agent_name == "chair_concept");
        CHECK(c->vertical_extent_m <= 0.0f);
        CHECK(c->solid_parts and c->shape_parts.size() == 6 and c->shape_proportions.size() == 1);
        if (c->shape_proportions.size() == 1)
            CHECK(c->shape_proportions[0].name == "seat_top" and near(c->shape_proportions[0].lo, 0.377f) and near(c->shape_proportions[0].hi, 0.667f));
        CHECK(near(c->ai2_prior_footprint_m, 0.47f) and near(c->ai2_prior_footprint_std, 0.046f));
        CHECK(near(c->ai2_prior_depth_m, 0.45f) and near(c->ai2_prior_depth_std, 0.035f));
        CHECK(near(c->ai2_prior_height_m, 0.905f) and near(c->ai2_prior_height_std, 0.089f));
        CHECK(near(c->ai2_wall_flush_prior, 0.5f) and c->ai2_room_axis_std_deg <= 0.0f and not c->front_detect_enabled);
        CHECK(near(c->move_rate_per_day, 3.0f) and near(c->wall_standoff_m, 0.05f) and near(c->wall_standoff_std_m, 0.05f));
        CHECK(near(c->layer_spread_m[3], 0.055f) and near(c->layer_spread_m[4], 0.045f) and c->layer_spread_m[2] == 0.0f);
        // the consumers' channels the legacy chair published, all on (K2b spike)
        CHECK(c->look_attrs == "chair" and c->appearance_enabled and c->publish_exist_prob and c->thumbnails_enabled);
        CHECK(c->approach_enabled and c->epistemic_offer_enabled and c->changed_object_enabled);
        CHECK(c->mesh_path == "chair_concept/meshes/chair_back_plus_y.obj");
        CHECK(near(c->link_pi_move, 0.10f) and near(c->link_pi_gone, 0.10f) and near(c->link_lambda_new, 0.10f) and near(c->ghost_evidence_gain, 0.15f));
        CHECK(near(c->recall_pi_move, 0.5f) and near(c->tracker_birth_min_sep_m, 0.70f));
        std::printf("chair: LOADED, floor-anchored, %zu parts on '%s', H %.3f\n", c->shape_parts.size(),
                    c->shape_proportions.empty() ? "-" : c->shape_proportions[0].name.c_str(), c->ai2_prior_height_m);
    }
    else CHECK(false);
    std::printf(bad ? "config_load_test FAILED (%d)\n" : "config_load_test OK\n", bad);
    return bad ? 1 : 0;
}
