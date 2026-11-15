// shape_parts_test — [[shape.part]] manifest parsing and the countable layout on a fitted box.
// Run: LANG=es_ES.UTF-8 ./shape_parts_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include "countable_shape_parts.h"

#ifndef MANIFEST_DIR
#error "MANIFEST_DIR must name common/concept_manifest"
#endif
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static std::string mf(const char* cls) { return std::string(MANIFEST_DIR) + "/" + cls + ".concept.toml"; }
static bool near(float a, float b, float e = 0.01f) { return std::fabs(a - b) <= e; }

static std::string tmp_manifest(const std::string& body)
{
    const std::string p = "/tmp/shape_parts_test_tmp.toml";
    std::ofstream(p) << body;
    return p;
}
static bool throws(const std::string& body)
{
    try { rc::manifest::load_shape_parts(tmp_manifest(body)); } catch (const std::runtime_error&) { return true; }
    return false;
}

int main()
{
    std::setlocale(LC_ALL, "");
    using namespace rc::manifest;
    // absent table
    CHECK(load_shape_parts(tmp_manifest("[concept]\nname = \"x\"\n")).empty());
    const auto whole = rc::countable::carve_parts_for(0.5f, 0.5f, 0.62f, {}, std::nullopt);
    CHECK(whole.size() == 1);
    if (whole.size() == 1)
    { CHECK(whole[0].cx == 0 and whole[0].cy == 0 and whole[0].sx == 0.5f and whole[0].sy == 0.5f and whole[0].z0 == 0 and whole[0].z1 == 0.62f); }
    // the real manifests
    CHECK(load_shape_parts(mf("nightstand")).size() == 5);
    const auto bed = load_shape_parts(mf("bed"));
    CHECK(bed.size() == 2);
    CHECK(load_shape_parts(mf("toilet")).size() == 3);
    // bed layout
    const auto bp = load_back_panel(mf("bed"));
    CHECK(bp.has_value());
    const auto L = rc::countable::carve_parts_for(1.75f, 2.1f, 0.65f, bed, bp);
    CHECK(L.size() == 3);
    if (L.size() == 3)
    {
        CHECK(near(L[0].z0, 0.25f) and near(L[0].z1, 0.65f));
        CHECK(near(L[1].cy, -1.0f) and near(L[1].z1, 0.83f));
        CHECK(near(L[2].cy, 1.0f, 0.06f) and L[2].z1 == 1.25f);
    }
    // refusals
    const std::string ok = "[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 1\nz0_f = 0\nz1_f = 1\n";
    CHECK(not throws(ok));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 0\nz0_f = 0\nz1_f = 1\n"));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 1\nz0_f = 0.5\nz1_f = 0.5\n"));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 1\nz1_f = 1\n"));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 0,5\nh_f = 1\nz0_f = 0\nz1_f = 1\n"));
    std::printf(bad ? "FAILED (%d)\n" : "shape_parts_test OK\n", bad);
    return bad ? 1 : 0;
}
