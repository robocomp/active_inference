// table_card_test: make_card(config as etc/config.toml sets it) carries the agent's real envelope, channels and
// geometry, and the card survives a JSON round trip. Run under the agent locale:  LANG=es_ES.UTF-8 ./table_card_test
#include "table_card.h"
#include "perception_card/tests/toml_peek.h"   // via -I common
#include <clocale>
#include <cmath>
#include <cstdio>
#include <algorithm>

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
        CHECK(near(p.terms[0].params.at("max_fill"), 0.60f) and near(p.terms[0].params.at("axes"), 1.0f));
    }
    CHECK(c.instance_terms == (std::vector<std::string>{"p_clear", "blind_spot_map"}) and near(c.instance_prior.at("p_clear"), 1.0f));
    CHECK(near(c.geometry.w, 1.0f) and near(c.geometry.d, 0.6f) and near(c.geometry.h, 0.75f));   // TableState defaults
    CHECK(c.calibration == "prior" and c.blind_spot_map_path.empty());
    CHECK(c.walls and near(c.walls->own_wall_skip_m, rc::occlusion::kOwnWallSkipM) and near(c.walls->own_wall_skip_m, 0.15f));
    CHECK(c.walls and c.walls->own_wall_plane_m < 0.0f and near(c.walls->wall_thickness_m, 0.0f));
    { // existence_absence_envelope = false: the terms the agent then runs (config ref/power, in_fov, central), no envelope
        TableConfig f = cfg;  f.existence_absence_envelope = false;
        f.existence_absence_range_ref_m = toml_peek(toml, "ExistenceAbsenceRangeRefM", f.existence_absence_range_ref_m);
        f.existence_absence_range_power = toml_peek(toml, "ExistenceAbsenceRangePower", f.existence_absence_range_power);
        const card::Card k = make_card(f);
        const auto& t = k.subtypes.at("round").terms;
        CHECK(t.size() == 3 and t[0].name == "range_conf" and t[1].name == "in_fov" and t[2].name == "central");
        CHECK(near(t[0].params.at("ref_m"), 2.5f) and near(t[0].params.at("power"), 2.0f));
        CHECK(not std::ranges::any_of(t, [](const card::TermSpec& s) { return s.name == "envelope"; }));
        const auto back = card::card_from_json(card::to_json(k));
        CHECK(back.has_value() and card::to_json(*back) == card::to_json(k));
    }
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
