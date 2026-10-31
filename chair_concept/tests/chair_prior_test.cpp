// chair_prior_test: the concept manifest is the single source of the chair's size prior. load_chair_prior reads
// [prior.template]; the perception card's geometry and the birth ChairBeliefParams template both equal it.
// Run under the agent locale:  LANG=es_ES.UTF-8 ./chair_prior_test
#include "chair_card.h"
#include "chair_belief.h"
#include "chair_mesh.h"
#include "chair_model.h"
#include "chair_prior.h"
#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>

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
          and near(pr->back_h, 0.655f) and near(pr->seat_thick, 0.075f) and near(pr->seat_w_std, 0.05f)
          and near(pr->leg_half, 0.0375f) and near(pr->seat_d_std, 0.04f) and near(pr->seat_h_std, 0.03f) and near(pr->back_h_std, 0.06f));
    CHECK(near(pr->p_free, 0.5f) and near(pr->p_against_wall, 0.5f)
          and near(pr->wall_gap_mean, 0.05f) and near(pr->wall_gap_sigma, 0.05f));   // [prior.placement]

    // [prior.bounds] loads and CONTAINS the template
    CHECK(near(pr->seat_w_b.lo, 0.30f) and near(pr->seat_w_b.hi, 0.65f) and near(pr->seat_d_b.lo, 0.30f) and near(pr->seat_d_b.hi, 0.65f));
    CHECK(near(pr->seat_h_b.lo, 0.38f) and near(pr->seat_h_b.hi, 0.70f) and near(pr->back_h_b.lo, 0.30f) and near(pr->back_h_b.hi, 0.80f));

    // a manifest missing seat_w_m is an error naming the key
    const std::string bad_path = (std::filesystem::temp_directory_path() / "chair_prior_missing.tmp.toml").string();
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
    // each of the four sigmas is required: dropping its line refuses to start, naming the key
    for (const char* key : {"seat_w_std_m", "seat_d_std_m", "seat_h_std_m", "back_h_std_m"})
    {
        std::ifstream in(manifest);
        std::ofstream out(bad_path);
        std::string line;
        while (std::getline(in, line))
            if (line.find(key) == std::string::npos) out << line << '\n';
        out.close();
        const auto r = load_chair_prior(bad_path);
        std::remove(bad_path.c_str());
        CHECK(not r.has_value() and r.error().find(key) != std::string::npos);
    }

    // strictness: each variant of the real manifest must be REJECTED (or, for the string block, ignored)
    const auto variant = [&](const std::string& from, const std::string& to) -> std::expected<ChairPrior, std::string>
    {
        std::ifstream in(manifest);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const auto at = text.find(from);
        CHECK(at != std::string::npos);
        if (at != std::string::npos) text.replace(at, from.size(), to);
        const std::string path = (std::filesystem::temp_directory_path() / "chair_prior_variant.tmp.toml").string();
        { std::ofstream out(path); out << text; }
        auto r = load_chair_prior(path);
        std::remove(path.c_str());
        return r;
    };
    CHECK(not variant("seat_w_m     = 0.60 ", "seat_w_m     = 0.60abc ").has_value());   // trailing garbage
    CHECK(variant("seat_w_m     = 0.60                # x", "seat_w_m     = 0.60 # x").has_value());   // comment is fine
    CHECK(not variant("wall_gap_mean_m  = 0.05 ", "wall_gap_mean_m  = 0.05x ").has_value());   // placement is as strict
    CHECK(not variant("p_free           = 0.5 ", "p_free           = 0 ").has_value());
    CHECK(not variant("leg_half_m   = 0.0375", "leg_half_m   = 0.0").has_value());       // leg_half is as strict
    CHECK(not variant("seat_w_m     = 0.60", "seat_w_m     = nan").has_value());
    CHECK(not variant("seat_d_std_m = 0.04", "seat_d_std_m = 0.0").has_value());          // zero sigma
    CHECK(not variant("seat_w_m     = 0.60", "seat_w_m     = 0.0").has_value());          // zero size
    CHECK(not variant("seat_w_m     = 0.60", "seat_w_m     = 0.60\nseat_w_m = 0.61").has_value());   // duplicate
    CHECK(not variant("seat_h_max_m = 0.70", "seat_h_max_m = 0.55").has_value());          // template 0.595 outside its bounds
    CHECK(not variant("back_h_max_m = 0.80", "back_h_max_m = 0.65").has_value());          // template 0.655 outside its bounds
    CHECK(not variant("seat_w_min_m = 0.30", "seat_w_min_m = 0.70").has_value());          // lo >= hi
    CHECK(not variant("seat_d_max_m = 0.65", "seat_d_max_m = 0.30").has_value());          // lo == hi
    CHECK(not variant("seat_w_max_m = 0.65", "seat_w_max_m = 0.65x").has_value());         // as strict as the template
    {   // a ChairModel built from the prior keeps the template through set_state (no longer clamped)
        ChairModelParams mp; mp.chair = *pr;
        ChairState s; s.seed_size(*pr);
        ChairModel m(s, mp);
        m.set_state(s);
        CHECK(near(m.state().seat_h, 0.595f) and near(m.state().back_h, 0.655f)
              and near(m.state().seat_w, 0.60f) and near(m.state().seat_d, 0.52f));
    }
    {   // a """ block holding `seat_w_m = 9` (and a section header) must not be read
        const auto r = variant("why          = \"\"\"", "why          = \"\"\"\nseat_w_m = 9\n[other]");
        CHECK(r.has_value() and near(r->seat_w, 0.60f));
    }

    // self_test takes the prior now. It FAILS identically at HEAD~1 ("starved far frames rotated a chair"), a
    // pre-existing defect unrelated to the prior, so it runs here as a smoke test of the seeded path, not as a gate.
    (void)ChairBelief::self_test(*pr);

    ChairConfig cfg;
    cfg.prior = *pr;
    const card::Card c = make_card(cfg);
    CHECK(near(c.geometry.w, 0.60f) and near(c.geometry.d, 0.52f) and near(c.geometry.z1, 1.25f));

    ChairBeliefParams p;   // defaults are 0: nothing typed in the struct
    CHECK(p.tpl_seat_w == 0.0f and p.tpl_back_h == 0.0f and p.seat_thickness == 0.0f);
    p.seed_template(*pr);
    CHECK(near(p.tpl_seat_w, pr->seat_w) and near(p.tpl_seat_d, pr->seat_d) and near(p.tpl_seat_h, pr->seat_h)
          and near(p.tpl_back_h, pr->back_h) and near(p.seat_thickness, pr->seat_thick));

    ChairState s;
    CHECK(s.seat_w == 0.0f and s.back_h == 0.0f);
    s.seed_size(*pr);
    CHECK(near(s.seat_w, 0.60f) and near(s.seat_d, 0.52f) and near(s.seat_h, 0.595f) and near(s.back_h, 0.655f));

    // The intended LIVE change, pinned: the published height is the prior's top (1.25), the mesh's seat slab is the manifest
    // thickness (0.075), the legs are the manifest leg_half in the belief, the SDF and the mesh, and ChairModel does not
    // clamp the template (seat_h 0.595 and back_h 0.655 used to be clamped to 0.55 / 0.65).
    {
        CHECK(near(pr->top(), 1.25f));
        ChairState tpl; tpl.seed_size(*pr);
        const auto mesh = make_chair_mesh(tpl, *pr);
        CHECK(mesh.size() == 6u * 108u);
        float zlo = 1e9f, zhi = -1e9f;
        for (std::size_t i = 0; i < 108; ++i) if (i % 3 == 2) { zlo = std::min(zlo, mesh[i]); zhi = std::max(zhi, mesh[i]); }
        CHECK(near(zhi - zlo, 0.075f));                                   // box 0 = the seat slab
        float xlo = 1e9f, xhi = -1e9f;
        for (std::size_t i = 4 * 108; i < 5 * 108; ++i) if (i % 3 == 0) { xlo = std::min(xlo, mesh[i]); xhi = std::max(xhi, mesh[i]); }
        CHECK(near(xhi - xlo, 2.0f * pr->leg_half));                      // box 4 = a leg (yaw 0)
        ChairModelParams mp; mp.chair = *pr;
        ChairModel m(tpl, mp);
        m.set_state(tpl);
        CHECK(near(m.state().seat_h, 0.595f) and near(m.state().back_h, 0.655f));   // not clamped
        ChairBeliefParams bp; bp.seed_template(*pr);
        CHECK(near(bp.leg_half, 0.0375f));
        // the SDF's legs are leg_half wide: a point 0.03 inside the corner leg is inside, one just beyond 0.0375 is outside
        const float ox = 0.5f * pr->seat_w - pr->leg_half, oy = 0.5f * pr->seat_d - pr->leg_half;
        CHECK(m.sdf_point(Eigen::Vector3f(ox + 0.03f, oy, 0.1f)) < 0.0f);
        CHECK(m.sdf_point(Eigen::Vector3f(ox + 0.05f, oy, 0.1f)) > 0.0f);
    }
    std::printf(bad ? "chair_prior_test: %d FAILED\n" : "chair_prior_test: OK\n", bad);
    return bad != 0;
}
