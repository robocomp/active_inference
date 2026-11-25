// shape_parts_test — [[shape.part]] manifest parsing and the countable layout on a fitted box; the [shape] layer spread
// (shared-engine G4: layer_spread_m with its mandatory layer_spread_source).
// Run: LANG=es_ES.UTF-8 ./shape_parts_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include "countable_shape_parts.h"
#include "layer_spread.h"

#ifndef MANIFEST_DIR
#error "MANIFEST_DIR must name common/concept_manifest"
#endif
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static std::string mf(const char* cls) { return std::string(MANIFEST_DIR) + "/" + cls + ".concept.toml"; }
static bool near(float a, float b, float e = 0.01f) { return std::fabs(a - b) <= e; }

static std::string tmp_manifest(const std::string& body)
{
    const std::string p = (std::filesystem::temp_directory_path() / std::format("shape_parts_test_{}.toml", ::getpid())).string();
    std::ofstream o(p);
    if (not o) { std::printf("FAIL: cannot write %s\n", p.c_str()); std::exit(2); }
    o << body;
    return p;
}
static bool throws(const std::string& body)
{
    try { rc::manifest::load_shape_parts(tmp_manifest(body)); } catch (const std::runtime_error&) { return true; }
    return false;
}
static bool spread_throws(const std::string& body, int n = 6)
{
    try { rc::manifest::layer_spread(tmp_manifest(body), n); } catch (const std::runtime_error&) { return true; }
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
    CHECK(load_shape_parts(mf("toilet")).size() == 2);   // K2a: bowl + tank (class knowledge), the model's solid
    CHECK(load_solid_is_parts(mf("toilet")) and not load_solid_is_parts(mf("bed")) and not load_solid_is_parts(mf("nightstand")));
    // bed layout
    const auto bp = load_back_panel(mf("bed"));
    CHECK(bp.has_value());
    const auto L = rc::countable::carve_parts_for(1.75f, 2.1f, 0.65f, bed, bp);
    CHECK(L.size() == 3);
    if (L.size() == 3)
    {
        CHECK(near(L[0].z0, 0.25f) and near(L[0].z1, 0.65f));
        CHECK(near(L[1].cy, -1.0f) and near(L[1].z1, 0.83f));
        CHECK(near(L[2].cy, 1.0f, 0.01f) and L[2].z1 == 1.25f);
    }
    const std::string ok = "[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 1\nz0_f = 0\nz1_f = 1\n";
    // nightstand, non-square: legs at +-0.44 of w (0.50) and h (0.60); a w/h swap fails
    {
        const auto ns = load_shape_parts(mf("nightstand"));
        const auto N = rc::countable::carve_parts_for(0.50f, 0.60f, 0.62f, ns, std::nullopt);
        CHECK(N.size() == 5);
        if (N.size() == 5)
        {
            CHECK(near(N[1].cx, -0.22f, 0.001f) and near(N[1].cy, -0.264f, 0.001f));
            CHECK(near(N[1].sx, 0.04f, 0.001f) and near(N[1].sy, 0.048f, 0.001f));
            CHECK(near(N[0].z0, 0.4092f, 0.001f) and near(N[1].z1, 0.4092f, 0.001f));
        }
        const auto ts = load_shape_parts(mf("toilet"));
        const auto T = rc::countable::carve_parts_for(0.58f, 0.733f, 1.02f, ts, std::nullopt);
        CHECK(T.size() == 2);
        if (T.size() == 2)
        {   // bowl: full length, 0.76 of the width, floor to 0.54 H; tank: full width, back 0.29 of the length, 0.54 H to H
            CHECK(near(T[0].cy, 0.0f, 0.001f) and near(T[0].sx, 0.76f * 0.58f, 0.001f) and near(T[0].sy, 0.733f, 0.001f));
            CHECK(T[0].z0 == 0.0f and near(T[0].z1, 0.54f * 1.02f, 0.001f));
            CHECK(near(T[1].cy, 0.355f * 0.733f, 0.001f) and near(T[1].sx, 0.58f, 0.001f) and near(T[1].sy, 0.29f * 0.733f, 0.001f));
            CHECK(near(T[1].z0, 0.54f * 1.02f, 0.001f) and near(T[1].z1, 1.02f, 0.001f));
            CHECK(near(T[1].cy + 0.5f * T[1].sy, 0.5f * 0.733f, 0.001f));   // the tank's back is the box's back
        }
    }
    // block ended by a following table, comment inside, shuffled keys
    {
        const auto v = load_shape_parts(tmp_manifest(
            "[[shape.part]]\nz1_f = 1\n# a comment\nname = \"s\"\nh_f = 0.5\nz0_f = 0.25\nw_f = 0.75\ny_f = 0.1\nx_f = -0.1\n"
            "[other]\nx_f = 99\nz1_f = 1\n"));
        CHECK(v.size() == 1);
        if (v.size() == 1)
        { CHECK(v[0].name == "s" and near(v[0].x_f, -0.1f, 1e-6f) and near(v[0].y_f, 0.1f, 1e-6f) and near(v[0].w_f, 0.75f, 1e-6f)
                and near(v[0].h_f, 0.5f, 1e-6f) and near(v[0].z0_f, 0.25f, 1e-6f) and near(v[0].z1_f, 1.0f, 1e-6f)); }
    }
    CHECK(throws(ok + "x_f = 0.2\n"));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 1\nz0_f = 0\nz1_f = inf\n"));
    // refusals
    CHECK(not throws(ok));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 0\nz0_f = 0\nz1_f = 1\n"));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 1\nz0_f = 0.5\nz1_f = 0.5\n"));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 1\nh_f = 1\nz1_f = 1\n"));
    CHECK(throws("[[shape.part]]\nname = \"p\"\nx_f = 0\ny_f = 0\nw_f = 0,5\nh_f = 1\nz0_f = 0\nz1_f = 1\n"));
    // ── [shape] solid (K2a) ──
    {
        const auto solid_throws = [](const std::string& body)
        { try { load_solid_is_parts(tmp_manifest(body)); } catch (const std::runtime_error&) { return true; } return false; };
        CHECK(not load_solid_is_parts(tmp_manifest("[shape]\nfrom = \"x\"\n" + ok)));          // absent: the box
        CHECK(not load_solid_is_parts(tmp_manifest("[shape]\nsolid = \"box\"\n" + ok)));
        CHECK(load_solid_is_parts(tmp_manifest("[shape]\nsolid = \"parts\"  # c\n" + ok)));
        CHECK(not load_solid_is_parts(tmp_manifest("[other]\nsolid = \"parts\"\n" + ok)));   // outside [shape]: not read
        CHECK(solid_throws("[shape]\nsolid = \"legs\"\n" + ok));                              // unknown value
        CHECK(solid_throws("[shape]\nsolid = parts\n" + ok));                                  // not a string
        CHECK(solid_throws("[shape]\nsolid = \"parts\"\n"));                                  // parts without a row
    }
    // ── [shape] layer spread ──
    {
        CHECK(layer_spread(tmp_manifest("[concept]\nname = \"x\"\n"), 6).delta_m.empty());   // absent: no layering declared
        // the real manifests: six values each, every one with its source, and the parts still read beside them
        const float want[3][6] = {{0.015f, 0.015f, 0.10f, 0.20f, 0.03f, 0.0f},    // bed
                                  {0.025f, 0.025f, 0.0f, 0.04f, 0.05f, 0.0f},     // nightstand
                                  {0.06f, 0.06f, 0.0f, 0.25f, 0.12f, 0.0f}};      // toilet
        const char* cls[3] = {"bed", "nightstand", "toilet"};
        for (int c = 0; c < 3; ++c)
        {
            const auto ls = layer_spread(mf(cls[c]), 6);
            CHECK(ls.delta_m.size() == 6 and not ls.source.empty());
            for (std::size_t k = 0; k < ls.delta_m.size() and k < 6; ++k) CHECK(ls.delta_m[k] == want[c][k]);
        }
        // one line, a trailing comma, a comment after each element over several lines; the block ends at the next table
        const auto one = layer_spread(tmp_manifest("[shape]\nlayer_spread_m = [0.1, 0.2]\nlayer_spread_source = \"a # b\"\n"), 2);
        CHECK(one.delta_m.size() == 2 and one.delta_m[1] == 0.2f and one.source == "a # b");
        const auto multi = layer_spread(tmp_manifest("[shape]\nlayer_spread_m = [\n  0.5,  # x, y\n  0.25, # z\n]\n"
                                                     "layer_spread_source = \"s\"\n[other]\nlayer_spread_m = [9.0]\n"), 2);
        CHECK(multi.delta_m.size() == 2 and multi.delta_m[0] == 0.5f and multi.delta_m[1] == 0.25f);
        // refusals: no source, blank source, source alone, wrong length, negative, comma decimal, not an array, not finite;
        // and a layer_spread_m outside [shape] is not read
        CHECK(spread_throws("[shape]\nlayer_spread_m = [0.1, 0.2]\n", 2));
        CHECK(spread_throws("[shape]\nlayer_spread_m = [0.1, 0.2]\nlayer_spread_source = \" \"\n", 2));
        CHECK(spread_throws("[shape]\nlayer_spread_source = \"s\"\n", 2));
        CHECK(spread_throws("[shape]\nlayer_spread_m = [0.1, 0.2]\nlayer_spread_source = \"s\"\n", 3));
        CHECK(spread_throws("[shape]\nlayer_spread_m = [0.1, -0.2]\nlayer_spread_source = \"s\"\n", 2));
        CHECK(spread_throws("[shape]\nlayer_spread_m = [0,05, 0.2]\nlayer_spread_source = \"s\"\n", 2));   // "0,05": not a TOML number
        CHECK(spread_throws("[shape]\nlayer_spread_m = 0.1\nlayer_spread_source = \"s\"\n", 1));          // not an array
        CHECK(spread_throws("[shape]\nlayer_spread_m = [inf, 0.2]\nlayer_spread_source = \"s\"\n", 2));
        CHECK(layer_spread(tmp_manifest("[other]\nlayer_spread_m = [0.1, 0.2]\n"), 2).delta_m.empty());
    }
    std::printf(bad ? "FAILED (%d)\n" : "shape_parts_test OK\n", bad);
    return bad ? 1 : 0;
}
