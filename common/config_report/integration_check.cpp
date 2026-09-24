/*
 * integration_check.cpp — rc::cfg::Reader against a REAL ConfigLoader and a real .toml.
 *
 * NOT named *_test.cpp on purpose: common/run_tests.sh compiles with a bare g++ and has neither
 * toml++ nor $ROBOCOMP/classes, so this one needs its own line. Build+run:
 *     common/config_report/build_integration_check.sh
 *
 * ★It calls setlocale(LC_ALL, "") because a harness has no Qt and would otherwise stay in the "C"
 * locale, answering a different question than the agent does (CLAUDE.md, locale section).
 *
 * The case it reproduces is commit b6c40b4's: a config that is SILENT about six solver keys, which
 * must come out as `default` and not as `false`.
 */
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <print>

#include "config_read.h"

using namespace rc::cfg;

namespace { int failures = 0; }
static void check(bool ok, std::string_view what)
{
    if (not ok) { std::print("  FAIL: {}\n", what); ++failures; }
}

int main()
{
    std::setlocale(LC_ALL, "");
    std::print("integration_check (locale '{}')\n", std::setlocale(LC_NUMERIC, nullptr));

    const auto dir = std::filesystem::temp_directory_path() / "cfg_report_integration";
    std::filesystem::create_directories(dir);
    const auto toml = (dir / "config.toml").string();
    {
        std::ofstream f(toml);
        f.imbue(std::locale::classic());          // never let the WRITER acquire a comma separator
        f << "[Agent]\nname = \"probe\"\nid = 77\n\n"
             "[RoomConcept]\n"
             "GnMaxIters = 20          # the file DOES state this one\n"
             "StateEps = 0.0375        # a decimal POINT, read back under es_ES\n"
             "RelocLegacyGridSearch = true\n"
             "OrphanKey = 5            # in the file, read by nothing\n"
             "NotAnInt = \"twelve\"      # present, wrong type\n\n"
             "[Masks]\n"
             "legacy_room_frame = false\n";
    }

    ConfigLoader cl;
    cl.load(toml);
    Reader cfg(cl, "probe_agent");

    // Stated in the file.
    check(cfg.i("RoomConcept.GnMaxIters", 10, "GN iteration cap") == 20, "file value wins over the default");
    // ★ The locale trap, end to end: 0.0375 written with a POINT must come back as 0.0375.
    const float eps = cfg.f("RoomConcept.StateEps", 0.04f, "convergence band on the state delta (m)");
    check(eps > 0.0374f and eps < 0.0376f, "a decimal point survives es_ES (toml++ parses, not strtof)");
    check(cfg.b("Masks.legacy_room_frame", false,
                "A/B: masks already in the room frame (pre-2026-07 producer)", experiment) == false,
          "an experiment switch at its default still reads correctly");
    check(cfg.b("RoomConcept.RelocLegacyGridSearch", false,
                "A/B: the pre-09-16 4-stage lattice relocaliser", experiment) == true, "experiment arm on");

    // ★ THE b6c40b4 CASE: six keys the file says nothing about.
    for (const auto* k : {"RoomConcept.HierPrecBoundaryEnabled", "RoomConcept.WindowStrideEnabled",
                          "RoomConcept.BoundaryFejSchur", "RoomConcept.CornerEarlyExitCheck",
                          "RoomConcept.AdaptiveCovEnabled"})
        check(cfg.b(k, false, "solver stage the config never mentions") == false, "silent key reads its default");
    check(cfg.i("RoomConcept.GnMaxItersOld", 10, "the older cap, unstated") == 10, "silent int reads its default");

    check(registry().record("RoomConcept.HierPrecBoundaryEnabled").origin == Origin::Default,
          "★ a key absent from the file is provenance `default`, NOT `false` — this is the whole point");
    check(registry().record("RoomConcept.GnMaxIters").origin == Origin::File, "a stated key is `file`");

    // Present but unusable: looks configured, is not.
    check(cfg.i("RoomConcept.NotAnInt", 12, "a key whose type does not match") == 12, "type error falls back");
    check(registry().record("RoomConcept.NotAnInt").origin == Origin::TypeError, "and is marked TypeError");

    registry().set_agent("probe_agent");
    exempt_generated_prefixes();
    registry().declare_complete("probe_agent");
    const auto p = cfg.publish((dir / "config_effective.csv").string());

    check(p.unread == 1, "exactly one orphan key found (OrphanKey); [Agent] absorbed by the exemption");
    check(p.arm == "Masks.legacy_room_frame=false,RoomConcept.RelocLegacyGridSearch=true",
          "the arm names both experiment switches");
    check(p.sweep_armed, "sweep armed");

    if (failures == 0) std::print("integration_check: all checks passed\n");
    else               std::print("integration_check: {} CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
