// table_card_test: make_card(config as etc/config.toml sets it) carries the agent's real envelope, channels and
// geometry, and the card survives a JSON round trip. Run under the agent locale:  LANG=es_ES.UTF-8 ./table_card_test
#include "table_card.h"
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
    TableConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.detect_min_fill = toml_peek(toml, "DetectMinFill", cfg.detect_min_fill);
    cfg.detect_max_fill = toml_peek(toml, "DetectMaxFill", cfg.detect_max_fill);
    cfg.detect_soft = toml_peek(toml, "DetectSoft", cfg.detect_soft);
    cfg.existence_detection_prob = toml_peek(toml, "ExistenceDetectionProb", cfg.existence_detection_prob);
    // Expected, literally: table runs the fleet prior 0.10 / 0.60 / 0.06 (etc/config.toml sets none of them).
    const card::Card c = make_card(cfg);
    if (argc > 1) std::puts(card::to_json(c).c_str());   // any argument: print the card JSON (for the task report)
    CHECK(c.concept_ == "table" and c.graph_type == "object" and c.level == card::Level::Countable);
    CHECK(c.channels.size() == 3);
    CHECK(c.channels[0].sensor == "zed" and c.channels[0].role == card::Role::Identity and c.channels[0].modelled);
    CHECK(c.channels[1].sensor == "helios" and c.channels[1].role == card::Role::Occupancy and c.channels[1].modelled);
    CHECK(c.channels[2].sensor == "ricoh" and not c.channels[2].modelled);
    CHECK(c.subtypes.contains("round") and c.subtypes.contains("square"));
    for (const char* s : {"round", "square"})
    {
        const auto& p = c.subtypes.at(s);
        CHECK(near(p.env.min_fill, 0.10f) and near(p.env.max_fill, 0.60f) and near(p.env.soft, 0.06f));
        CHECK(near(p.q_return, 0.85f));
        CHECK(p.terms.size() == 2 and p.terms[0].name == "envelope" and p.terms[1].name == "central");
        CHECK(near(p.terms[0].params.at("max_fill"), 0.60f));
    }
    CHECK(c.calibration == "prior" and c.blind_spot_map_path.empty());
    CHECK(near(c.geometry.z1, 0.75f) and near(c.geometry.z0, 0.0f));
    // A loaded blind-spot map switches calibration and carries its path.
    cfg.existence_blind_spot_map = true;
    const card::Card m = make_card(cfg, 120);
    CHECK(m.calibration == "blind_spot_map" and m.blind_spot_map_path == cfg.existence_blind_spot_file);
    // JSON round trip (both cards).
    for (const card::Card* k : {&c, &m})
    {
        const auto back = card::card_from_json(card::to_json(*k));
        CHECK(back.has_value() and card::to_json(*back) == card::to_json(*k));
    }
    std::printf(bad ? "table_card_test: %d FAILED\n" : "table_card_test: OK\n", bad);
    return bad != 0;
}
