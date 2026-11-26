// birth_seed_rule_test — which classes the birth seeds on a wall (F1 round 9 a, rc::birth_on_wall).
// The rule: only a class that declares a back-face attachment (Model.WallFlushPrior > 0 or a [prior.attachment] stand-off) and
// is not c2v; every other class keeps its cold-start footprint seed. Checked on every shipped countable config loaded through
// rc::load_countable_config:
//   nightstand, bed, toilet (flush + stand-off), refrigerator (stand-off, flush 0), hood (flush + stand-off), chair (flush 0.5 +
//   stand-off) → on the wall, as before;  table (c2v, nothing declared) → NOT on the wall (k2c_table: the wall seed moved every
//   table 1.0–1.3 m off its detection cloud).
// And the contract: a c2v config that declared a flush prior is refused by rc::symmetric_config_violation AND is still not
// wall-seeded.
// Behaviour (rc::seed_birth, F1 round 9 fix round 1): a table cloud 1 m from a wall keeps its moment centre; a nightstand cloud
// is snapped onto the wall. With BASE's decision (the wall seed whenever room geometry is known) the table case fails.
// F2 (final fix wave): the chair's seed is its placement mixture's MAP — a chair cloud 1.5 m from a wall keeps its cloud seed, one
// at the wall is seeded flush; the p = 1 classes (no [prior.placement]) are bit-identical to birth_on_priors. Fails at BASE (the
// 1.5 m chair was wall-seeded).
// Run: LANG=es_ES.UTF-8 ./birth_seed_rule_test
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <genericworker.h>   // the test stub: the real ConfigLoader (manifest_stub/)
#include "countable_birth_seed.h"
#include "countable_config.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static rc::CountableConfig load(const std::string& dir, const std::string& file)
{
    std::filesystem::current_path(std::filesystem::path(REPO_DIR) / dir);
    ConfigLoader cl;
    cl.load(file);
    return rc::load_countable_config(cl);
}

int main()
{
    std::setlocale(LC_ALL, "");
    std::printf("birth_seed_rule_test\n");
    struct Case { const char* dir; const char* file; bool wall; };
    const Case cases[] = {{"nightstand_concept", "etc/config.toml", true}, {"bed_concept", "etc/config.toml", true},
                          {"toilet_concept", "etc/config.toml", true}, {"refrigerator_concept", "etc/countable.toml", true},
                          {"hood_concept", "etc/countable.toml", true}, {"chair_concept", "etc/countable.toml", true},
                          {"table_concept", "etc/countable.toml", false}};
    for (const auto& k : cases)
    {
        const rc::CountableConfig c = load(k.dir, k.file);
        std::printf("  %-22s flush %.2f standoff_std %.3f c2v %d -> birth on wall %d (expected %d)\n", k.dir, c.ai2_wall_flush_prior,
                    c.wall_standoff_std_m, int(c.shape_symmetric), int(rc::birth_on_wall(c)), int(k.wall));
        CHECK(rc::birth_on_wall(c) == k.wall);
    }
    rc::CountableConfig t = load("table_concept", "etc/countable.toml");
    t.ai2_wall_flush_prior = 0.9f;                                   // a c2v class declaring a flush prior
    CHECK(not rc::symmetric_config_violation(t).empty());
    CHECK(not rc::birth_on_wall(t));
    // Behaviour (rc::seed_birth, the fitter's branch): a wall at y = 0 (normal +y into the room) and a box cloud whose front is
    // 1 m from it. The table keeps its cold-start seed (the cloud's moment centre) exactly; the nightstand is snapped onto the wall
    // (centre half its class depth in from it).
    const rc::NearestWall wall = [](const Eigen::Vector2f&) { rc::WallRef w; w.ok = true; w.p = {0.f, 0.f}; w.n = {0.f, 1.f}; w.length_m = 6.f; return w; };
    const auto cloud = [](float cx, float cy, float w, float h, float H)
    {
        std::vector<Eigen::Vector3f> pts;
        for (int i = 0; i <= 20; ++i) for (int j = 0; j <= 10; ++j)
            pts.emplace_back(cx - 0.5f * w + w * i / 20.f, cy - 0.5f * h + h * j / 10.f, H);   // the top
        for (int i = 0; i <= 20; ++i) for (int k = 0; k <= 10; ++k)
            pts.emplace_back(cx - 0.5f * w + w * i / 20.f, cy - 0.5f * h, H * k / 10.f);         // the front face
        return pts;
    };
    {
        const rc::CountableConfig tc = load("table_concept", "etc/countable.toml");
        rc::CountableBeliefState s0; s0.cx = 0.f; s0.cy = 1.4f; s0.w = 1.8f; s0.h = 0.8f; s0.H = 0.73f; s0.yaw = 0.f;
        const rc::CountableBeliefState before = s0;
        const auto pc = rc::seed_birth(s0, cloud(0.f, 1.4f, 1.8f, 0.8f, 0.73f), 0.f, 1.f, tc, true, wall, "table_test");
        std::printf("  table cloud 1 m off the wall: seed (%.3f, %.3f) -> (%.3f, %.3f), wall-placed %d\n", before.cx, before.cy, s0.cx,
                    s0.cy, int(pc.has_value()));
        CHECK(not pc.has_value());
        CHECK(s0.cx == before.cx and s0.cy == before.cy and s0.w == before.w and s0.h == before.h and s0.yaw == before.yaw);
    }
    {
        const rc::CountableConfig nc = load("nightstand_concept", "etc/config.toml");
        rc::CountableBeliefState s0; s0.cx = 0.f; s0.cy = 1.2f; s0.w = 0.5f; s0.h = 0.4f; s0.H = 0.55f; s0.yaw = 0.f;
        const auto pc = rc::seed_birth(s0, cloud(0.f, 1.2f, 0.5f, 0.4f, 0.55f), 0.f, 1.f, nc, true, wall, "nightstand_test");
        const float class_d = nc.ai2_prior_depth_m > 0.f ? nc.ai2_prior_depth_m : nc.ai2_prior_footprint_m;
        std::printf("  nightstand cloud 1 m off the wall: seed -> (%.3f, %.3f), wall-placed %d; centre to wall %.4f = half the class depth %.4f\n",
                    s0.cx, s0.cy, int(pc.has_value()), s0.cy, 0.5f * class_d);
        CHECK(pc.has_value());
        CHECK(std::abs(s0.cy - 0.5f * class_d) <= 4.f * std::numeric_limits<float>::epsilon() * std::max(1.f, class_d));
    }
    // F2 (final fix wave): the chair declares a placement MIXTURE ([prior.placement] p_free 0.5, p_against_wall 0.5), so its seed
    // is the mixture's MAP, not the wall whenever a stand-off is declared. Its cloud is the seat top + the front face it shows the
    // room (the side away from the wall). At BASE every chair cloud was wall-seeded: the 1.5 m case fails there.
    const auto front_cloud = [](float cx, float cy, float w, float h, float H)
    {
        std::vector<Eigen::Vector3f> pts;
        for (int i = 0; i <= 20; ++i) for (int j = 0; j <= 10; ++j)
            pts.emplace_back(cx - 0.5f * w + w * i / 20.f, cy - 0.5f * h + h * j / 10.f, H);   // the top
        for (int i = 0; i <= 20; ++i) for (int k = 0; k <= 10; ++k)
            pts.emplace_back(cx - 0.5f * w + w * i / 20.f, cy + 0.5f * h, H * k / 10.f);         // the front face, room side
        return pts;
    };
    {
        const rc::CountableConfig cc = load("chair_concept", "etc/countable.toml");
        const float d = cc.ai2_prior_depth_m > 0.f ? cc.ai2_prior_depth_m : cc.ai2_prior_footprint_m;
        std::printf("  chair: p_against_wall %s %.2f, stand-off %.3f ± %.3f, class %.3f x %.3f\n",
                    cc.placement_p_against_wall ? "declared" : "ABSENT", cc.placement_p_against_wall.value_or(-1.f), cc.wall_standoff_m,
                    cc.wall_standoff_std_m, cc.ai2_prior_footprint_m, d);
        CHECK(cc.placement_p_against_wall.has_value());
        {   // free-standing: the cloud's centre 1.5 m from the wall → seeded AT its cloud (s0 unchanged)
            rc::CountableBeliefState s0; s0.cx = 0.3f; s0.cy = 1.5f; s0.w = 0.5f; s0.h = 0.465f; s0.H = 0.45f; s0.yaw = 0.f;
            const rc::CountableBeliefState before = s0;
            const auto pc = rc::seed_birth(s0, front_cloud(0.3f, 1.5f, 0.5f, 0.465f, 0.45f), 0.f, 1.f, cc, true, wall, "chair_free_test");
            std::printf("  chair cloud 1.5 m off the wall: seed (%.3f, %.3f) -> (%.3f, %.3f), wall-placed %d (expected 0)\n", before.cx,
                        before.cy, s0.cx, s0.cy, int(pc.has_value()));
            CHECK(not pc.has_value());
            CHECK(s0.cx == before.cx and s0.cy == before.cy and s0.w == before.w and s0.h == before.h and s0.yaw == before.yaw);
        }
        {   // at the wall: the back at the declared stand-off → seeded flush (centre h̄/2 in from the wall, as birth_on_priors)
            const float cy = cc.wall_standoff_m + 0.5f * d;
            rc::CountableBeliefState s0; s0.cx = 0.3f; s0.cy = cy; s0.w = 0.5f; s0.h = 0.465f; s0.H = 0.45f; s0.yaw = 0.f;
            const auto pc = rc::seed_birth(s0, front_cloud(0.3f, cy, 0.5f, d, 0.45f), 0.f, 1.f, cc, true, wall, "chair_wall_test");
            std::printf("  chair cloud at the wall (back %.3f m off it): seed -> (%.3f, %.3f), wall-placed %d (expected 1)\n",
                        cc.wall_standoff_m, s0.cx, s0.cy, int(pc.has_value()));
            CHECK(pc.has_value());
            CHECK(std::abs(s0.cy - 0.5f * d) <= 4.f * std::numeric_limits<float>::epsilon() * std::max(1.f, d));
        }
    }
    {   // the p = 1 classes are unchanged: seed_birth == birth_on_priors bit for bit (nightstand, also 1.5 m off the wall)
        const rc::CountableConfig nc = load("nightstand_concept", "etc/config.toml");
        CHECK(not nc.placement_p_against_wall.has_value());
        rc::CountableBeliefState a; a.cx = 0.f; a.cy = 1.5f; a.w = 0.5f; a.h = 0.4f; a.H = 0.55f; a.yaw = 0.f;
        rc::CountableBeliefState b = a;
        const auto pts = front_cloud(0.f, 1.5f, 0.5f, 0.4f, 0.55f);
        const auto pa = rc::seed_birth(a, pts, 0.f, 1.f, nc, true, wall, "nightstand_seed");
        const auto pb = rc::birth_on_priors(b, pts, 0.f, 1.f, nc, wall, "nightstand_priors");
        CHECK(pa.has_value() and pb.has_value() and *pa == *pb);
        CHECK(a.cx == b.cx and a.cy == b.cy and a.w == b.w and a.h == b.h and a.yaw == b.yaw);
        std::printf("  nightstand 1.5 m off the wall: seed (%.3f, %.3f) = birth_on_priors (%.3f, %.3f), wall-placed %d\n", a.cx, a.cy,
                    b.cx, b.cy, int(pa.has_value()));
        for (const auto& k : cases)
            if (std::string(k.dir) != "chair_concept")
            {
                const rc::CountableConfig c = load(k.dir, k.file);
                std::printf("  %-22s placement mixture %s\n", k.dir, c.placement_p_against_wall ? "DECLARED" : "absent (p = 1 implied)");
                CHECK(not c.placement_p_against_wall.has_value());
            }
    }
    std::printf(bad ? "birth_seed_rule_test: %d FAILED\n" : "birth_seed_rule_test: OK\n", bad);
    return bad != 0;
}
