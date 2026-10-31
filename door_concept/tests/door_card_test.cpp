// door_card_test: make_card(config as etc/config.toml sets it) carries the agent's real terms, channels and
// geometry, and the card survives a JSON round trip. Also: an unmeasured door is marginalised over the phi prior.
// Run under the agent locale:  LANG=es_ES.UTF-8 ./door_card_test
#include "door_card.h"
#include "perception_card/tests/toml_peek.h"   // via -I common
#include "perception_card/evaluator.h"
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
    DoorConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.detect_min_fill = toml_peek(toml, "DetectMinFill", cfg.detect_min_fill);
    cfg.detect_max_fill = toml_peek(toml, "DetectMaxFill", cfg.detect_max_fill);
    cfg.detect_soft = toml_peek(toml, "DetectSoft", cfg.detect_soft);
    cfg.exist_detection_prob = toml_peek(toml, "DetectionProb", cfg.exist_detection_prob);   // [Existence]
    cfg.openable_phi_max_rad = toml_peek(toml, "PhiMaxRad", cfg.openable_phi_max_rad);       // [Openable] (absent: default)
    const card::Card c = make_card(cfg);
    if (argc > 1) std::puts(card::to_json(c).c_str());
    CHECK(c.concept_ == "door" and c.graph_type == "door" and c.level == card::Level::Articulated);
    CHECK(c.channels.size() == 2);
    CHECK(c.channels[0].sensor == "zed" and c.channels[0].role == card::Role::Identity and c.channels[0].modelled);
    CHECK(c.channels[1].sensor == "bpearl" and c.channels[1].role == card::Role::Occupancy and c.channels[1].modelled);
    const auto& p = c.subtypes.at("");
    CHECK(near(p.env.min_fill, 0.10f) and near(p.env.max_fill, 0.60f) and near(p.env.soft, 0.06f));
    CHECK(near(p.q_return, 0.85f));
    const std::vector<std::string> want{"envelope", "leaf_obliquity", "resolvability", "in_fov", "central", "kappa"};
    CHECK(p.terms.size() == want.size());
    for (std::size_t i = 0; i < want.size() and i < p.terms.size(); ++i) CHECK(p.terms[i].name == want[i]);
    CHECK(near(p.terms[0].params.at("max_fill"), 0.60f) and near(p.terms[0].params.at("axes"), 1.0f));
    CHECK(near(p.terms[1].params.at("phi_max"), 1.5707963f));
    CHECK(near(c.geometry.w, 1.0f) and near(c.geometry.d, 0.05f) and near(c.geometry.h, 2.1f));
    CHECK(near(c.geometry.z0, 0.0f) and near(c.geometry.z1, 2.1f));
    CHECK(c.instance_terms.empty() and c.calibration == "prior" and c.blind_spot_map_path.empty());
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));

    { // evaluator: one door target, phi unset lies strictly between shut and wide open (along the leaf normal)
        card::World w;
        w.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        rc::nbv::Sensor z;   // the ZED at its mount (640x360 at f = 448 px)
        z.hfov_rad = 2.0f * std::atan(640.0f / 448.0f); z.vfov_rad = 2.0f * std::atan(360.0f / 448.0f); z.height_m = 0.945f;
        w.zed = z; w.zed_mount_z = 0.945f;
        card::Hypothesis h; h.concept_ = "door"; h.x = 5.0f; h.y = 2.0f; h.yaw = 0.0f;
        const card::Viewpoint vp{{5.0f, 0.5f}, std::nullopt};
        card::Hypothesis h0 = h, h90 = h;  h0.phi = 0.0f;  h90.phi = 1.5707963f;
        const float pu = card::evaluate_camera(c, w, h, vp).p, p0 = card::evaluate_camera(c, w, h0, vp).p,
                    p90 = card::evaluate_camera(c, w, h90, vp).p;
        std::printf("door phi: p(unset)=%.4f p(0)=%.4f p(pi/2)=%.4f\n", pu, p0, p90);
        CHECK(p90 < pu and pu < p0);
    }
    std::printf(bad ? "door_card_test: %d FAILED\n" : "door_card_test: OK\n", bad);
    return bad != 0;
}
