// refrigerator_card_test: make_card(config as etc/config.toml sets it) carries the agent's real envelope, channels and
// geometry, and the card survives a JSON round trip. Run under the agent locale:  LANG=es_ES.UTF-8 ./refrigerator_card_test
#include "refrigerator_card.h"
#include "perception_card/tests/toml_peek.h"   // via -I common
#include <clocale>
#include <cmath>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }
using namespace rc;

int main(int argc, char**)
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const std::string toml = std::string(AGENT_DIR) + "/etc/config.toml";
    RefrigeratorConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.detect_min_fill = toml_peek(toml, "DetectMinFill", cfg.detect_min_fill);
    cfg.detect_max_fill = toml_peek(toml, "DetectMaxFill", cfg.detect_max_fill);
    cfg.detect_soft = toml_peek(toml, "DetectSoft", cfg.detect_soft);
    cfg.existence_detection_prob = toml_peek(toml, "ExistenceDetectionProb", cfg.existence_detection_prob);
    cfg.ai2_prior_footprint_m = toml_peek(toml, "AI2PriorFootprintM", cfg.ai2_prior_footprint_m);
    cfg.ai2_prior_height_m = toml_peek(toml, "AI2PriorHeightM", cfg.ai2_prior_height_m);
    // Expected, literally: the fridge tour fit 0.0 / 1.32 / 0.26 (etc/config.toml, calibrated 2026-08-08).
    const card::Card c = make_card(cfg);
    if (argc > 1) std::puts(card::to_json(c).c_str());   // any argument: print the card JSON (for the task report)
    CHECK(c.concept_ == "refrigerator" and c.level == card::Level::Countable);
    CHECK(c.channels.size() == 3 and c.channels[0].sensor == "zed" and c.channels[0].role == card::Role::Identity);
    CHECK(c.channels[1].sensor == "helios" and c.channels[1].role == card::Role::Occupancy);
    CHECK(c.channels[2].sensor == "ricoh" and not c.channels[2].modelled);
    const auto& p = c.subtypes.at("");
    CHECK(near(p.env.min_fill, 0.0f) and near(p.env.max_fill, 1.32f) and near(p.env.soft, 0.26f));
    CHECK(near(p.q_return, 0.85f));
    CHECK(p.terms.size() == 2 and p.terms[0].name == "envelope" and p.terms[1].name == "central");
    CHECK(near(c.geometry.w, 0.60f) and near(c.geometry.d, 0.60f) and near(c.geometry.h, 1.90f));
    CHECK(near(c.geometry.z0, 0.0f) and near(c.geometry.z1, 1.90f));
    CHECK(c.calibration == "fitted" and c.calibration_date == "2026-08-08");
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));
    std::printf(bad ? "refrigerator_card_test: %d FAILED\n" : "refrigerator_card_test: OK\n", bad);
    return bad != 0;
}
