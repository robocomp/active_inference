// chair_card_test: make_card(config as etc/config.toml sets it) carries the agent's real envelope, channels, terms and
// geometry, and the card survives a JSON round trip. Run under the agent locale:  LANG=es_ES.UTF-8 ./chair_card_test
#include "chair_card.h"
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
    ChairConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.detect_min_fill = toml_peek(toml, "DetectMinFill", cfg.detect_min_fill);
    cfg.detect_max_fill = toml_peek(toml, "DetectMaxFill", cfg.detect_max_fill);
    cfg.detect_soft = toml_peek(toml, "DetectSoft", cfg.detect_soft);
    cfg.tracker_birth_seat_w = toml_peek(toml, "BirthSeatW", cfg.tracker_birth_seat_w);
    cfg.tracker_birth_seat_d = toml_peek(toml, "BirthSeatD", cfg.tracker_birth_seat_d);
    cfg.tracker_birth_seat_h = toml_peek(toml, "BirthSeatH", cfg.tracker_birth_seat_h);
    cfg.tracker_birth_back_h = toml_peek(toml, "BirthBackH", cfg.tracker_birth_back_h);
    const card::Card c = make_card(cfg);
    if (argc > 1) std::puts(card::to_json(c).c_str());
    CHECK(c.concept_ == "chair" and c.graph_type == "object" and c.level == card::Level::Countable);
    CHECK(c.channels.size() == 2);
    CHECK(c.channels[0].sensor == "zed" and c.channels[0].role == card::Role::Identity and c.channels[0].modelled);
    CHECK(c.channels[1].sensor == "ricoh" and c.channels[1].role == card::Role::Identity and not c.channels[1].modelled);
    const auto& p = c.subtypes.at("");
    CHECK(near(p.env.min_fill, 0.10f) and near(p.env.max_fill, 0.60f) and near(p.env.soft, 0.06f));   // etc sets none: fleet prior
    const std::vector<std::string> want{"envelope", "in_fov", "central"};
    CHECK(p.terms.size() == want.size());
    for (std::size_t i = 0; i < want.size() and i < p.terms.size(); ++i) CHECK(p.terms[i].name == want[i]);
    CHECK(near(p.terms[0].params.at("max_fill"), 0.60f) and near(p.terms[0].params.at("axes"), 1.0f));
    CHECK(near(c.geometry.w, 0.60f) and near(c.geometry.d, 0.52f) and near(c.geometry.h, 1.25f));   // Tracker.BirthSeat* template
    CHECK(near(c.geometry.z0, 0.0f) and near(c.geometry.z1, 1.25f));
    CHECK(c.calibration == "prior" and c.instance_terms.empty());
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));
    std::printf(bad ? "chair_card_test: %d FAILED\n" : "chair_card_test: OK\n", bad);
    return bad != 0;
}
