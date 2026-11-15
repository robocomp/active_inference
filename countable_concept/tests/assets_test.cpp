// assets_test: the display assets a countable instance publishes, by class-folder convention (countable_assets.h).
// nightstand ships meshes/nightstand_basecolor.jpg and publishes it as mesh_texture_path; bed and toilet ship none and
// publish none. Run: ./assets_test
#include "countable_assets.h"

#include <cstdio>
#include <string>

namespace {
int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++bad; } } while (0)
}

int main()
{
    const std::string repo = REPO_DIR;
    const auto ns = rc::countable_texture_path("nightstand", repo + "/nightstand_concept");
    CHECK(ns == std::optional<std::string>("nightstand_concept/meshes/nightstand_basecolor.jpg"));
    for (const std::string cls : {"bed", "toilet"})
        CHECK(not rc::countable_texture_path(cls, repo + "/" + cls + "_concept").has_value());
    CHECK(not rc::countable_texture_path("nightstand", repo + "/no_such_concept").has_value());
    for (const std::string cls : {"nightstand", "bed", "toilet"})
        CHECK(rc::countable_mesh_path(cls) == cls + "_concept/meshes/" + cls + ".obj");
    std::printf("nightstand texture: %s\n", ns ? ns->c_str() : "(none)");
    std::printf(bad ? "assets_test: %d FAILED\n" : "assets_test: OK\n", bad);
    return bad != 0;
}
