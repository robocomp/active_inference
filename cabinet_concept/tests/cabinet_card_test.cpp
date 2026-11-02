// cabinet_card_test: make_card(config as etc/config.toml sets it) carries the agent's real terms, channels and
// geometry, and the card survives a JSON round trip. Run under the agent locale:  LANG=es_ES.UTF-8 ./cabinet_card_test
#include "cabinet_card.h"
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
    CabinetConfig cfg;
    cfg.existence_detection_prob = toml_peek(toml, "ExistenceDetectionProb", cfg.existence_detection_prob);
    cfg.existence_absence_range_ref_m = toml_peek(toml, "ExistenceAbsenceRangeRefM", cfg.existence_absence_range_ref_m);
    cfg.existence_absence_range_power = toml_peek(toml, "ExistenceAbsenceRangePower", cfg.existence_absence_range_power);
    const card::Card c = make_card(cfg);
    if (argc > 1) std::puts(card::to_json(c).c_str());
    CHECK(c.concept_ == "cabinet" and c.graph_type == "box" and c.level == card::Level::Countable);
    CHECK(c.channels.size() == 2);
    CHECK(c.channels[0].sensor == "zed" and c.channels[0].role == card::Role::Identity and c.channels[0].modelled);
    CHECK(c.channels[1].sensor == "helios" and c.channels[1].role == card::Role::Occupancy and c.channels[1].modelled);
    const auto& p = c.subtypes.at("");
    CHECK(near(p.q_return, 0.85f));
    CHECK(p.terms.size() == 3 and p.terms[0].name == "range_conf" and p.terms[1].name == "in_fov" and p.terms[2].name == "central");
    CHECK(near(p.terms[0].params.at("ref_m"), 2.5f) and near(p.terms[0].params.at("power"), 2.0f));
    CHECK(near(c.geometry.w, 1.0f) and near(c.geometry.d, 0.6f) and near(c.geometry.h, 0.9f));
    CHECK(near(c.geometry.z0, 0.0f) and near(c.geometry.z1, 0.9f));
    CHECK(c.instance_terms.empty() and c.calibration == "prior" and c.blind_spot_map_path.empty());
    CHECK(c.walls and near(c.walls->own_wall_skip_m, kCabinetOwnWallSkipM) and near(kCabinetOwnWallSkipM, 0.70f));
    CHECK(c.walls and c.walls->own_wall_plane_m < 0.0f and near(c.walls->wall_thickness_m, 0.0f));
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));
    std::printf(bad ? "cabinet_card_test: %d FAILED\n" : "cabinet_card_test: OK\n", bad);
    return bad != 0;
}
