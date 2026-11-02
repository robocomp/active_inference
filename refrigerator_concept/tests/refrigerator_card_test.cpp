// refrigerator_card_test: make_card(config as etc/config.toml sets it) carries the agent's real envelope, channels and
// geometry, and the card survives a JSON round trip. Run under the agent locale:  LANG=es_ES.UTF-8 ./refrigerator_card_test
#include "refrigerator_card.h"
#include "perception_card/evaluator.h"
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
    CHECK(near(p.terms[0].params.at("axes"), 1.0f));
    CHECK(near(c.geometry.w, 0.60f) and near(c.geometry.d, 0.60f) and near(c.geometry.h, 1.90f));
    CHECK(near(c.geometry.z0, 0.0f) and near(c.geometry.z1, 1.90f));
    CHECK(c.calibration == "fitted" and c.calibration_date == "2026-08-08");
    // Level 1 = refrigerator_projection.cpp's own-wall skip (rc::kRefrigeratorOwnWallSkipM) against LINE walls.
    CHECK(c.walls and near(c.walls->own_wall_skip_m, 0.70f) and near(c.walls->own_wall_skip_m, kRefrigeratorOwnWallSkipM));
    CHECK(c.walls and c.walls->own_wall_plane_m < 0.0f and near(c.walls->wall_thickness_m, 0.0f));
    { // I4 end to end with THIS card: back 0.35 m from the x = 6 wall, 3.15 m stand-off, ZED + helios. The same P with
      // that wall and with it 3 m further away, and nothing reads BehindWall. Then the wall line through the carcass
      // (x = 5.5, the back 0.15 m beyond it): still the same P, because it is the fridge's own wall.
        card::World w;
        w.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        rc::nbv::Sensor z;
        z.hfov_rad = 2.0f * std::atan(640.0f / 448.0f); z.vfov_rad = 2.0f * std::atan(360.0f / 448.0f); z.height_m = 0.945f;
        w.zed = z; w.zed_mount_z = 0.945f;
        card::LidarRig hel;
        hel.model.version = 1; hel.model.rings = 32; hel.model.fov_start_deg = -55.0f; hel.model.fov_end_deg = 15.0f;
        hel.model.azimuth_step_deg = 0.4f; hel.model.range_min_m = 0.1f; hel.model.range_max_m = 80.0f;
        hel.mount = Eigen::Isometry3f::Identity();
        hel.mount.linear() = Eigen::AngleAxisf(std::numbers::pi_v<float>, Eigen::Vector3f::UnitX()).toRotationMatrix();
        hel.mount.translation() = Eigen::Vector3f(0.0f, -0.155f, 1.075f);
        w.helios = hel;
        const float half = 0.5f * c.geometry.d;
        card::Hypothesis h; h.concept_ = "refrigerator"; h.x = 6.0f - 0.35f - half; h.y = 2.0f;
        const card::Viewpoint vp{{h.x - 3.15f, 2.0f}, std::nullopt};
        card::World far = w; far.polygon = {{0, 0}, {9, 0}, {9, 4}, {0, 4}};
        card::World cut = w; cut.polygon = {{0, 0}, {h.x + half - 0.15f, 0}, {h.x + half - 0.15f, 4}, {0, 4}};
        const auto rw = card::evaluate(c, w, h, vp), rf = card::evaluate(c, far, h, vp), rc_ = card::evaluate(c, cut, h, vp);
        std::printf("fridge I4: wall rec=%.4f notice=%.4f limit=%d | far rec=%.4f | cut rec=%.4f limit=%d\n",
                    rw.p_recognise, rw.p_notice, int(rw.limit), rf.p_recognise, rc_.p_recognise, int(rc_.limit));
        CHECK(std::fabs(rw.p_recognise - rf.p_recognise) < 1e-6f and std::fabs(rw.p_notice - rf.p_notice) < 1e-6f);
        CHECK(std::fabs(rc_.p_recognise - rf.p_recognise) < 1e-6f and std::fabs(rc_.p_notice - rf.p_notice) < 1e-6f);
        CHECK(rw.limit != card::Limit::BehindWall and rc_.limit != card::Limit::BehindWall);
        for (const auto& ch : rc_.channels) CHECK(ch.limit != card::Limit::BehindWall);
        CHECK(rw.p_notice >= rw.p_recognise and rw.p_recognise > 0.5f);
    }
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));
    std::printf(bad ? "refrigerator_card_test: %d FAILED\n" : "refrigerator_card_test: OK\n", bad);
    return bad != 0;
}
