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

// Mean of N(mean, sigma) truncated to [lo, hi], by midpoint integration (same reference as common/perception_card/tests).
static double truncated_mean(double mean, double sigma, double lo, double hi)
{
    double num = 0, den = 0;
    const int N = 200000;
    for (int i = 0; i < N; ++i)
    {
        const double x = lo + (hi - lo) * (i + 0.5) / N, p = std::exp(-0.5 * (x - mean) * (x - mean) / (sigma * sigma));
        num += x * p; den += p;
    }
    return num / den;
}

int main(int argc, char**)
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const std::string toml = std::string(AGENT_DIR) + "/etc/config.toml";
    ChairConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.detect_min_fill = toml_peek(toml, "DetectMinFill", cfg.detect_min_fill);
    cfg.detect_max_fill = toml_peek(toml, "DetectMaxFill", cfg.detect_max_fill);
    cfg.detect_soft = toml_peek(toml, "DetectSoft", cfg.detect_soft);
    const auto prior = load_chair_prior(std::string(AGENT_DIR) + "/../common/concept_manifest/chair.concept.toml");
    CHECK(prior.has_value());
    if (prior) cfg.prior = *prior;
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
    CHECK(near(c.geometry.w, 0.60f) and near(c.geometry.d, 0.52f) and near(c.geometry.h, 1.25f));   // manifest [prior.template]
    CHECK(near(c.geometry.z0, 0.0f) and near(c.geometry.z1, 1.25f));
    CHECK(c.calibration == "prior" and c.instance_terms.empty());
    CHECK(c.walls and near(c.walls->own_wall_skip_m, rc::occlusion::kOwnWallSkipM) and near(c.walls->own_wall_skip_m, 0.15f));
    CHECK(c.walls and c.walls->own_wall_plane_m < 0.0f and near(c.walls->wall_thickness_m, 0.0f));
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));
    // the generative vars carry the four per-variable sigmas, and with them the truncated mean stays at the template
    CHECK(c.generative and c.generative->vars.size() == 4);
    if (not c.generative) return 1;
    const float want_sigma[4] = {0.05f, 0.04f, 0.03f, 0.06f}, want_mean[4] = {0.60f, 0.52f, 0.595f, 0.655f};
    const float want_lo[4] = {0.30f, 0.30f, 0.38f, 0.30f}, want_hi[4] = {0.65f, 0.65f, 0.70f, 0.80f};
    for (std::size_t i = 0; i < c.generative->vars.size() and i < 4; ++i)
    {
        const auto& v = c.generative->vars[i];
        CHECK(near(v.sigma, want_sigma[i]) and near(v.mean, want_mean[i]) and near(v.lo, want_lo[i]) and near(v.hi, want_hi[i]));
        const double tm = truncated_mean(v.mean, v.sigma, v.lo, v.hi);
        std::printf("truncated mean %s = %.4f (template %.4f)\n", v.var.c_str(), tm, v.mean);
        CHECK(std::fabs(tm - v.mean) < 0.02);
    }
    std::printf(bad ? "chair_card_test: %d FAILED\n" : "chair_card_test: OK\n", bad);
    return bad != 0;
}
