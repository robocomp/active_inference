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
#include "permanence.h"
#include "wall_standoff.h"
#include "shape_parts.h"
#include "class_prior.h"

#ifndef REPO_DIR
#error "REPO_DIR must name the repository root"
#endif
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main()
{
    std::setlocale(LC_ALL, "");
    // K1: refrigerator (floor-anchored, nominal) and hood (HANGS: span [z_top − extent, z_top], nominal) are countable classes too.
    struct Cls { const char* name; float z_top; bool panel; rc::manifest::Support support; float extent; rc::manifest::From from; };
    using S = rc::manifest::Support; using F = rc::manifest::From;
    const Cls classes[] = {{"bed", 0.65f, true, S::floor_anchored, 0.65f, F::measured},
                           {"nightstand", 0.62f, false, S::floor_anchored, 0.62f, F::measured},
                           {"toilet", 0.90f, false, S::floor_anchored, 0.90f, F::measured},
                           {"refrigerator", 1.90f, false, S::floor_anchored, 1.90f, F::nominal},
                           {"hood", 2.05f, false, S::hangs, 0.50f, F::nominal}};
    for (const auto& c : classes)
    {
        std::filesystem::current_path(std::filesystem::path(REPO_DIR) / (std::string(c.name) + "_concept"));
        const std::string path = rc::countable_manifest_path(c.name);
        const auto g = rc::manifest::load_geometry(path, c.name);
        CHECK(g.valid);
        CHECK(g.support == c.support);
        CHECK(std::abs(g.z_top_m - c.z_top) < 1e-6f);   // a locale-truncated read would give 0
        CHECK(std::abs(g.extent_m - c.extent) < 1e-6f);
        CHECK(g.from == c.from);
        CHECK(rc::manifest::provenance_ok(path, c.name));   // the agent refuses to start on an inherited block
        const auto span = rc::manifest::adopt_span(path, c.name, c.support, c.z_top, c.extent);
        CHECK(span.from_manifest);
        CHECK(std::abs(span.z0 - (c.z_top - c.extent)) < 1e-6f and std::abs(span.z1 - c.z_top) < 1e-6f);
        // The countable loader's other manifest readers (countable_config.cpp): they throw on a malformed block.
        bool threw = false;
        try
        {
            const auto pm = rc::manifest::permanence(path);
            CHECK(pm.has_value() and pm->move_rate_per_day > 0.0f);
            const auto so = rc::manifest::wall_standoff(path);
            CHECK(so.has_value() and so->std_m > 0.0f);
            (void) rc::manifest::load_shape_parts(path);
            (void) rc::manifest::load_class_prior(path);
        }
        catch (const std::exception& e) { threw = true; std::printf("%s: %s\n", c.name, e.what()); }
        CHECK(not threw);
        const auto ls = rc::manifest::layer_spread(path, 6);
        CHECK(ls.delta_m.size() == 6 and not ls.source.empty());
        CHECK(rc::manifest::load_back_panel(path).has_value() == c.panel);
        std::printf("%s: geometry %s, z_top %.3f, span from %s, layer spread %zu values\n", c.name,
                    g.valid ? "LOADED" : "NOT loaded", g.z_top_m, span.from_manifest ? "manifest" : "FALLBACK", ls.delta_m.size());
    }
    std::printf(bad ? "manifest_load_test FAILED (%d)\n" : "manifest_load_test OK\n", bad);
    return bad ? 1 : 0;
}
