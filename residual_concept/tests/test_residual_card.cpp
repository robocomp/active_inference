// test_residual_card: make_card(config as etc/config.toml sets it) carries the agent's real channels, band and
// footprint; the card survives a JSON round trip; and the Mass evaluator path gives COVERAGE only.
// Run under the agent locale:  LANG=es_ES.UTF-8 ./test_residual_card
#include "residual_card.h"
#include "perception_card/tests/toml_peek.h"   // via -I common
#include "perception_card/evaluator.h"
#include <clocale>
#include <cmath>
#include <cstdio>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }
using namespace rc;

static card::World room()
{
    card::World w;
    w.polygon = {{-8, -8}, {8, -8}, {8, 8}, {-8, 8}};
    nbv::Sensor s;
    constexpr float f = 448.0f;
    s.hfov_rad = 2.0f * std::atan(640.0f / f); s.vfov_rad = 2.0f * std::atan(360.0f / f); s.height_m = 0.945f;
    w.zed = s; w.zed_mount_z = 0.945f;
    return w;
}

int main(int argc, char**)
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const std::string toml = std::string(AGENT_DIR) + "/etc/config.toml";
    ResidualConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.cluster.floor_z0 = toml_peek(toml, "FloorZ0", cfg.cluster.floor_z0);
    cfg.cluster.ceil_z = toml_peek(toml, "CeilZ", cfg.cluster.ceil_z);
    // UseBpearl / ZedBoost.* are booleans in etc/config.toml (true); the struct defaults are the same, so they are not peeked.
    const card::Card c = make_card(cfg, 0.05f);
    if (argc > 1) std::puts(card::to_json(c).c_str());
    CHECK(c.concept_ == "residual" and c.level == card::Level::Mass);
    CHECK(c.channels.size() == 3);
    CHECK(c.channels[0].sensor == "helios" and c.channels[0].role == card::Role::Occupancy and c.channels[0].modelled);
    CHECK(c.channels[1].sensor == "bpearl" and c.channels[1].role == card::Role::Occupancy and c.channels[1].modelled);
    CHECK(c.channels[2].sensor == "zed" and c.channels[2].role == card::Role::Occupancy and c.channels[2].modelled);
    const auto& p = c.subtypes.at("");
    CHECK(p.terms.size() == 1 and p.terms[0].name == "coverage");
    CHECK(near(p.q_return, 1.0f));
    CHECK(near(c.geometry.w, 0.05f) and near(c.geometry.d, 0.05f));
    CHECK(near(c.geometry.z0, 0.06f) and near(c.geometry.z1, 1.80f) and near(c.geometry.h, 1.74f));   // Clusterer.FloorZ0 / CeilZ
    CHECK(c.calibration == "prior" and c.instance_terms.empty());
    ResidualConfig off = cfg; off.use_bpearl = false;
    CHECK(not make_card(off).channels[1].modelled);
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));

    // Mass path: a 1x1 m floor region 2 m ahead of the ZED. No obstacle: coverage ~ 1. A box between: it drops.
    card::Hypothesis h; h.concept_ = "residual"; h.x = 2.0f; h.y = 0.0f;
    h.size = card::GeometryPrior{1.0f, 1.0f, 0.05f, 0.0f, 0.05f};
    const card::Viewpoint vp{{0.0f, 0.0f}, 0.0f};
    const auto clear = card::evaluate_camera(c, room(), h, vp);
    std::printf("clear: p=%.4f\n", clear.p);
    CHECK(clear.p > 0.9f);
    card::World w = room();
    w.obstacles.push_back({"box_1", nbv::Obstacle{1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.2f}});
    const auto blocked = card::evaluate_camera(c, w, h, vp);
    std::printf("blocked: p=%.4f limit=%d\n", blocked.p, int(blocked.limit));
    CHECK(blocked.p < clear.p and blocked.p < 0.5f);
    std::printf(bad ? "test_residual_card: %d FAILED\n" : "test_residual_card: OK\n", bad);
    return bad != 0;
}
