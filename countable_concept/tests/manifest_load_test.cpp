// manifest_load_test — the class manifest is LOADED by the calls the agent makes, from the agent's working directory.
//
// Live finding (objects_step0, objects_live15birth): every bed / nightstand / toilet run printed "[manifest] ★★<class>
// MANIFEST NOT LOADED (../common/concept_manifest/<class>.concept.toml) — THE DECLARATIVE GEOMETRY IS INERT" and ran its
// span on the agent fallback. The textual readers on the same path (back panel, shape parts, class prior) loaded, so the
// path was right; the whole-file TOML load in load_geometry was what failed.
//
// For each countable class this test does what countable_config.cpp does at startup, in the same conditions:
//   - cwd = <repo>/<class>_concept (the agent runs from its own folder; the manifest path is relative to it);
//   - the path from countable_manifest_path(class);
//   - LANG=es_ES.UTF-8 with std::setlocale(LC_ALL, "") (Qt does this in the agent; CLAUDE.md: a harness without it answers
//     a different question);
//   - rc::manifest::load_geometry and rc::manifest::adopt_span (concept_manifest.h, built against the real ConfigLoader.cpp
//     like the agent), then rc::manifest::layer_spread and load_back_panel (the textual readers).
// It requires the geometry to load (valid, floor_anchored, the manifest's z_top_m) and adopt_span to take it from the
// manifest, not from the fallback. Run: LANG=es_ES.UTF-8 ./manifest_load_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include "concept_manifest.h"
#include "countable_config.h"   // countable_manifest_path

#ifndef REPO_DIR
#error "REPO_DIR must name the repository root"
#endif
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main()
{
    std::setlocale(LC_ALL, "");
    struct Cls { const char* name; float z_top; bool panel; };
    const Cls classes[] = {{"bed", 0.65f, true}, {"nightstand", 0.62f, false}, {"toilet", 0.90f, false}};
    for (const auto& c : classes)
    {
        std::filesystem::current_path(std::filesystem::path(REPO_DIR) / (std::string(c.name) + "_concept"));
        const std::string path = rc::countable_manifest_path(c.name);
        const auto g = rc::manifest::load_geometry(path, c.name);
        CHECK(g.valid);
        CHECK(g.support == rc::manifest::Support::floor_anchored);
        CHECK(std::abs(g.z_top_m - c.z_top) < 1e-6f);   // a locale-truncated read would give 0
        CHECK(g.from == rc::manifest::From::measured);
        const auto span = rc::manifest::adopt_span(path, c.name, rc::manifest::Support::floor_anchored, c.z_top, c.z_top);
        CHECK(span.from_manifest);
        const auto ls = rc::manifest::layer_spread(path, 6);
        CHECK(ls.delta_m.size() == 6 and not ls.source.empty());
        CHECK(rc::manifest::load_back_panel(path).has_value() == c.panel);
        std::printf("%s: geometry %s, z_top %.3f, span from %s, layer spread %zu values\n", c.name,
                    g.valid ? "LOADED" : "NOT loaded", g.z_top_m, span.from_manifest ? "manifest" : "FALLBACK", ls.delta_m.size());
    }
    std::printf(bad ? "manifest_load_test FAILED (%d)\n" : "manifest_load_test OK\n", bad);
    return bad ? 1 : 0;
}
