// chair_prior_test: the concept manifest is the single source of the chair's size prior. load_chair_prior reads
// [prior.template]; the perception card's geometry and the birth ChairBeliefParams template both equal it.
// Run under the agent locale:  LANG=es_ES.UTF-8 ./chair_prior_test
#include "chair_card.h"
#include "chair_belief.h"
#include "chair_model.h"
#include "chair_prior.h"
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }
using namespace rc;

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const std::string manifest = std::string(AGENT_DIR) + "/../common/concept_manifest/chair.concept.toml";
    const auto pr = load_chair_prior(manifest);
    CHECK(pr.has_value());
    if (not pr) { std::printf("%s\n", pr.error().c_str()); return 1; }
    CHECK(near(pr->seat_w, 0.60f) and near(pr->seat_d, 0.52f) and near(pr->seat_h, 0.595f)
          and near(pr->back_h, 0.655f) and near(pr->seat_thick, 0.075f) and near(pr->size_std, 0.15f));

    // a manifest missing seat_w_m is an error naming the key
    const std::string bad_path = std::string(AGENT_DIR) + "/tests/chair_prior_missing.tmp.toml";
    {
        std::ifstream in(manifest);
        std::ofstream out(bad_path);
        std::string line;
        while (std::getline(in, line))
            if (line.find("seat_w_m") == std::string::npos) out << line << '\n';
    }
    const auto miss = load_chair_prior(bad_path);
    std::remove(bad_path.c_str());
    CHECK(not miss.has_value() and miss.error().find("seat_w_m") != std::string::npos);
    CHECK(not load_chair_prior("/nonexistent/chair.toml").has_value());

    ChairConfig cfg;
    cfg.prior = *pr;
    const card::Card c = make_card(cfg);
    CHECK(near(c.geometry.w, 0.60f) and near(c.geometry.d, 0.52f) and near(c.geometry.z1, 1.25f));

    ChairBeliefParams p;
    p.tpl_seat_w = p.tpl_seat_d = p.tpl_seat_h = p.tpl_back_h = p.seat_thickness = 0.0f;
    p.seed_template(*pr);
    CHECK(near(p.tpl_seat_w, pr->seat_w) and near(p.tpl_seat_d, pr->seat_d) and near(p.tpl_seat_h, pr->seat_h)
          and near(p.tpl_back_h, pr->back_h) and near(p.seat_thickness, pr->seat_thick));

    ChairState s;
    CHECK(s.seat_w == 0.0f and s.back_h == 0.0f);
    s.seed_size(*pr);
    CHECK(near(s.seat_w, 0.60f) and near(s.seat_d, 0.52f) and near(s.seat_h, 0.595f) and near(s.back_h, 0.655f));
    std::printf(bad ? "chair_prior_test: %d FAILED\n" : "chair_prior_test: OK\n", bad);
    return bad != 0;
}
