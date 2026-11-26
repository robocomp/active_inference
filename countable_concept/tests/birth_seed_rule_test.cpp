// birth_seed_rule_test — which classes the birth seeds on a wall (F1 round 9 a, rc::birth_on_wall).
// The rule: only a class that declares a back-face attachment (Model.WallFlushPrior > 0 or a [prior.attachment] stand-off) and
// is not c2v; every other class keeps its cold-start footprint seed. Checked on every shipped countable config loaded through
// rc::load_countable_config:
//   nightstand, bed, toilet (flush + stand-off), refrigerator (stand-off, flush 0), hood (flush + stand-off), chair (flush 0.5 +
//   stand-off) → on the wall, as before;  table (c2v, nothing declared) → NOT on the wall (k2c_table: the wall seed moved every
//   table 1.0–1.3 m off its detection cloud).
// And the contract: a c2v config that declared a flush prior is refused by rc::symmetric_config_violation AND is still not
// wall-seeded.
// Run: LANG=es_ES.UTF-8 ./birth_seed_rule_test
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <string>

#include <genericworker.h>   // the test stub: the real ConfigLoader (manifest_stub/)
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
    std::printf(bad ? "birth_seed_rule_test: %d FAILED\n" : "birth_seed_rule_test: OK\n", bad);
    return bad != 0;
}
