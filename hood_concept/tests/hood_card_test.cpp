// hood_card_test: make_card(config as etc/config.toml sets it) carries the agent's real envelope, channels and
// geometry, and the card survives a JSON round trip. Run under the agent locale:  LANG=es_ES.UTF-8 ./hood_card_test
#include "hood_card.h"
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
    HoodConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.detect_min_fill = toml_peek(toml, "DetectMinFill", cfg.detect_min_fill);
    cfg.detect_max_fill = toml_peek(toml, "DetectMaxFill", cfg.detect_max_fill);
    cfg.detect_soft = toml_peek(toml, "DetectSoft", cfg.detect_soft);
    cfg.existence_detection_prob = toml_peek(toml, "ExistenceDetectionProb", cfg.existence_detection_prob);
    cfg.ai2_prior_footprint_m = toml_peek(toml, "AI2PriorFootprintM", cfg.ai2_prior_footprint_m);
    cfg.ai2_prior_depth_m = toml_peek(toml, "AI2PriorDepthM", cfg.ai2_prior_depth_m);
    cfg.ai2_prior_height_m = toml_peek(toml, "AI2PriorHeightM", cfg.ai2_prior_height_m);
    // Expected, literally: the hood runs the fleet prior (etc/config.toml carries no Detect* keys).
    const card::Card c = make_card(cfg);
    if (argc > 1) std::puts(card::to_json(c).c_str());   // any argument: print the card JSON (for the task report)
    CHECK(c.concept_ == "hood" and c.level == card::Level::Countable);
    CHECK(c.channels.size() == 3 and c.channels[0].sensor == "zed" and c.channels[0].role == card::Role::Identity);
    CHECK(c.channels[1].sensor == "helios" and c.channels[1].role == card::Role::Occupancy);
    CHECK(c.channels[2].sensor == "ricoh" and not c.channels[2].modelled);
    const auto& p = c.subtypes.at("");
    CHECK(near(p.env.min_fill, 0.10f) and near(p.env.max_fill, 0.60f) and near(p.env.soft, 0.06f));
    CHECK(near(p.q_return, 0.55f));   // etc/config.toml ExistenceDetectionProb (the manifest's measured 0.355 is not what runs)
    CHECK(p.terms.size() == 2 and p.terms[0].name == "envelope" and p.terms[1].name == "central");
    CHECK(near(c.geometry.w, 0.90f) and near(c.geometry.d, 0.50f) and near(c.geometry.h, 0.50f));
    CHECK(near(c.geometry.z0, 1.55f) and near(c.geometry.z1, 2.05f));
    CHECK(c.calibration == "prior");
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));
    std::printf(bad ? "hood_card_test: %d FAILED\n" : "hood_card_test: OK\n", bad);
    return bad != 0;
}
