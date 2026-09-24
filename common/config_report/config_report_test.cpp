/*
 * config_report_test.cpp — standalone. Built by common/run_tests.sh with a bare g++.
 *
 * The point of these cases is NOT that the table prints. It is that the table cannot lie in the four
 * ways this fleet has actually been lied to:
 *   1. a key absent from the file must read as `default`, not as `false`          (commit b6c40b4)
 *   2. an unarmed sweep must say INCONCLUSIVE, never print an empty unread list
 *   3. an overlay value PARSED and applied by nothing must be named               (CmdNoiseRot)
 *   4. the run fingerprint must move when an ARM moves and stay put when a DIAGNOSTIC moves
 */
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <string>

#include "config_report.h"

using namespace rc::cfg;

namespace
{
int failures = 0;

void check(bool ok, std::string_view what)
{
    if (not ok) { std::print("  FAIL: {}\n", what); ++failures; }
}

Record rec(std::string key, std::string type, std::string def, std::string eff, Origin o,
           Kind k = Kind::Permanent, std::string desc = "d")
{
    Record r;
    r.key = std::move(key); r.type = std::move(type);
    r.code_default = std::move(def); r.effective = std::move(eff);
    r.origin = o; r.kind = k; r.description = std::move(desc);
    return r;
}

std::string dump_path(std::string_view name)
{
    const auto dir = std::filesystem::temp_directory_path() / "cfg_report_test";
    std::filesystem::create_directories(dir);
    return (dir / name).string();
}

std::string slurp(const std::string& p)
{
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
}  // namespace

int main()
{
    std::print("config_report_test\n");

    // ── 1. provenance: absent from the file is `default`, and it is NOT the same as false ────────
    {
        Registry r;
        r.note(rec("A.Gate", "bool", "false", "false", Origin::Default));
        r.note(rec("A.Other", "bool", "false", "true", Origin::File));
        check(r.record("A.Gate").origin == Origin::Default, "absent key keeps Origin::Default");
        check(r.record("A.Gate").effective == r.record("A.Gate").code_default,
              "a default-origin key's effective value IS its code default");
        check(r.record("A.Other").origin == Origin::File, "a present key is Origin::File");
        // Re-reading the same key counts, it does not duplicate: two read sites is worth knowing.
        r.note(rec("A.Gate", "bool", "false", "false", Origin::Default));
        check(r.record("A.Gate").reads == 2, "a second read site bumps reads, does not add a row");
        check(r.size() == 2, "re-note upserts by key");
    }

    // ── 2. the unarmed sweep must be INCONCLUSIVE, never a clean bill of health ──────────────────
    {
        Registry r;
        r.set_agent("unarmed_agent");
        r.note(rec("B.Known", "int", "1", "2", Origin::File));
        const auto p = r.publish({{"B.Known", "2"}, {"B.Stranger", "9"}}, dump_path("unarmed.csv"));
        check(not p.sweep_armed, "sweep is not armed without declare_complete()");
        check(p.unread == 1, "the stranger key is counted even while INCONCLUSIVE");
        const auto csv = slurp(dump_path("unarmed.csv"));
        check(csv.find("registry_complete=0") != std::string::npos, "CSV records the unarmed state");
        check(csv.find("B.Stranger") != std::string::npos, "an unread key is still WRITTEN to the record");
    }
    {
        Registry r;
        r.declare_complete("armed_agent");
        r.note(rec("B.Known", "int", "1", "2", Origin::File));
        const auto p = r.publish({{"B.Known", "2"}, {"B.Stranger", "9"}}, dump_path("armed.csv"));
        check(p.sweep_armed, "declare_complete arms the sweep");
        check(p.unread == 1, "armed sweep names the unread key");
    }

    // ── 3. exemptions absorb generated keys, and ONLY those ──────────────────────────────────────
    {
        Registry r;
        r.declare_complete("exempt_agent");
        exempt_generated_prefixes(r);
        r.note(rec("C.Real", "int", "1", "1", Origin::Default));
        const auto p = r.publish({{"Agent.name", "x"}, {"Ice.MessageSizeMax", "65536"},
                                  {"C.Real", "1"}, {"C.Stray", "7"}}, dump_path("exempt.csv"));
        check(p.unread == 1, "framework prefixes are absorbed; a stray key is not");
    }
    {
        // mark_consumed is stronger than a prefix: it marks what was ACTUALLY forwarded, so a key
        // that fell into the catch(...) still surfaces. That is the [*.ThreadPool] bug.
        Registry r;
        r.declare_complete("ice_agent");
        r.mark_consumed("omnirobot.ThreadPool.Size", "forwarded to the Ice communicator");
        const auto p = r.publish({{"omnirobot.ThreadPool.Size", "4"},
                                  {"omnirobot.ThreadPool.SizeMax", "8"}}, dump_path("ice.csv"));
        check(p.unread == 1, "a ThreadPool key that was never forwarded still surfaces");
    }

    // ── 4. parsed-but-applied-by-nothing: the CmdNoiseRot shape. It was READ, so no sweep sees it ─
    {
        Registry r;
        r.declare_complete("overlay_agent");
        r.note(rec("Platform.P3Bot.CmdNoiseRot", "float", "0", "0.02", Origin::File));
        r.note_parsed("Platform.P3Bot.CmdNoiseRot");
        r.note(rec("Platform.P3Bot.GnMaxIters", "int", "10", "20", Origin::File));
        r.note_parsed("Platform.P3Bot.GnMaxIters");
        r.note_overlay_seen("Platform.P3Bot.GnMaxIters");
        const auto p = r.publish({{"Platform.P3Bot.CmdNoiseRot", "0.02"},
                                  {"Platform.P3Bot.GnMaxIters", "20"}}, dump_path("overlay.csv"));
        check(p.unread == 0, "an overlay key that WAS read is not unread — the sweep cannot see this bug");
        check(r.record("Platform.P3Bot.CmdNoiseRot").parsed
                  and not r.record("Platform.P3Bot.CmdNoiseRot").apply_seen,
              "CmdNoiseRot is parsed and never applied");
        check(r.record("Platform.P3Bot.GnMaxIters").apply_seen, "GnMaxIters reached an apply_*");
    }

    // ── 5. the fingerprint is an ARM identity, not a config hash ─────────────────────────────────
    {
        Registry a, b;
        a.note(rec("Z.Last", "int", "1", "1", Origin::Default));
        a.note(rec("A.First", "int", "1", "1", Origin::Default));
        b.note(rec("A.First", "int", "1", "1", Origin::Default));
        b.note(rec("Z.Last", "int", "1", "1", Origin::Default));
        check(a.fingerprint() == b.fingerprint(), "fingerprint is stable under insertion order");

        Registry c;
        c.note(rec("X.Arm", "bool", "false", "false", Origin::Default, Kind::Experiment));
        c.note(rec("X.Csv", "bool", "false", "false", Origin::Default, Kind::Diagnostic));
        const auto base = c.fingerprint();
        Registry d;
        d.note(rec("X.Arm", "bool", "false", "false", Origin::Default, Kind::Experiment));
        d.note(rec("X.Csv", "bool", "false", "true", Origin::File, Kind::Diagnostic));
        check(d.fingerprint() == base, "turning a DIAGNOSTIC on does not change the arm identity");
        Registry e;
        e.note(rec("X.Arm", "bool", "false", "true", Origin::File, Kind::Experiment));
        e.note(rec("X.Csv", "bool", "false", "false", Origin::Default, Kind::Diagnostic));
        check(e.fingerprint() != base, "flipping an EXPERIMENT does change the arm identity");
        check(e.arm() == "X.Arm=true", "the arm string names the experiment keys");

        // A self_tuner knob must be DECLARED runtime_tuned at its read site; then it is out of the
        // fingerprint from the start and tuning it cannot move the arm identity.
        Registry f;
        auto tuned = rec("Y.Tuned", "float", "1", "1", Origin::Default);
        tuned.mut = Mutability::RuntimeTuned;
        f.note(tuned);
        f.note(rec("Y.Fixed", "int", "1", "1", Origin::Default));
        const auto before = f.fingerprint();
        f.note_runtime("Y.Tuned", "3.5");
        check(f.fingerprint() == before,
              "a declared RuntimeTuned knob leaves the fingerprint alone — its startup value is not what ran");
        check(f.record("Y.Tuned").effective == "3.5", "and the table still records what it became");
    }

    // ── 6. the undocumented counter is the documentation backlog, and it must count honestly ─────
    {
        Registry r;
        r.declare_complete("doc_agent");
        r.note(rec("D.Has", "int", "1", "1", Origin::Default, Kind::Permanent, "explained"));
        r.note(rec("D.Lacks", "int", "1", "1", Origin::Default, Kind::Permanent, ""));
        const auto p = r.publish({}, dump_path("doc.csv"));
        check(p.undocumented == 1, "an empty description is counted, not silently accepted");
    }

    // ── 7. the CSV is the run's own record: it must carry the arm, not require a config file ─────
    {
        Registry r;
        r.declare_complete("record_agent");
        r.note(rec("E.Arm", "bool", "false", "true", Origin::File, Kind::Experiment, "an A/B arm"));
        const auto p = r.publish({{"E.Arm", "true"}}, dump_path("record.csv"));
        const auto csv = slurp(dump_path("record.csv"));
        check(csv.find("arm=E.Arm=true") != std::string::npos, "the CSV header names the arm");
        check(csv.find("fingerprint=" + p.fingerprint) != std::string::npos, "and its fingerprint");
        check(csv.find("\"an A/B arm\"") != std::string::npos, "the description reaches the record");
    }

    if (failures == 0) std::print("config_report_test: all checks passed\n");
    else               std::print("config_report_test: {} CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
