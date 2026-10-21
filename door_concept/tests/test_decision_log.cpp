// Offline harness for decision_log.h: the binary per-cycle DECISION log ("DDEC" v1) that today's door_concept writes so
// the DoorCore replay (docs/superpowers/specs/2026-10-21-door-core-replay-design.md) can be compared bit-for-bit with
// the code it replaces. Checks: N decisions written under es_ES come back identical (doubles bit-exact, strings with
// embedded commas/newlines), the header is "DDEC" + version 1, a truncated tail is dropped not misread, a wrong magic
// or version is refused, and a disabled log writes nothing.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 tests/test_decision_log.cpp -o /tmp/test_decision_log && /tmp/test_decision_log
#include <bit>
#include <clocale>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <print>
#include "../src/decision_log.h"

using namespace rc::door;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

static bool same_bits(double x, double y) { return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y); }

int main()
{
    std::setlocale(LC_ALL, "es_ES.UTF-8");
    std::locale::global(std::locale::classic());
    const std::string dir = "/tmp/test_decision_log_dir";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/door_decisions.bin";

    std::vector<DoorDecision> want;
    for (int i = 0; i < 200; ++i)
        want.push_back({.cycle = std::uint64_t(1000 + i), .t_ms = -5 + 100 * std::int64_t(i),
                        .kind = std::uint16_t(1 + i % int(DoorDecisionKind::kCount)), .door_id = std::uint64_t(i) * 7919u,
                        .name = (i % 3 == 0) ? "door_" + std::to_string(i) : "",
                        .a = 0.1 * i + 1e-9, .b = -9.2e-05 * i, .c = std::numeric_limits<double>::quiet_NaN(),
                        .d = 0.260417 * i, .note = (i % 5 == 0) ? "a,b\nc \"q\"" : "x"});
    {
        DoorDecisionLog log;
        CHECK(not log.enabled());
        log.append(want[0]);                       // disabled: silently nothing
        CHECK(log.open(path));
        CHECK(log.enabled());
        for (const auto& d : want) log.append(d);
    }
    {   // header
        std::ifstream f(path, std::ios::binary);
        char h[8]{}; f.read(h, 8);
        CHECK(std::memcmp(h, "DDEC", 4) == 0);
        std::uint32_t v = 0; std::memcpy(&v, h + 4, 4);
        CHECK(v == 1u);
    }
    auto got = DoorDecisionLog::read_all(path);
    CHECK(got.has_value());
    if (got)
    {
        CHECK(got->size() == want.size());
        for (std::size_t i = 0; i < want.size() and i < got->size(); ++i)
        {
            const auto &w = want[i], &g = (*got)[i];
            CHECK(w.cycle == g.cycle and w.t_ms == g.t_ms and w.kind == g.kind and w.door_id == g.door_id);
            CHECK(w.name == g.name and w.note == g.note);
            CHECK(same_bits(w.a, g.a) and same_bits(w.b, g.b) and same_bits(w.c, g.c) and same_bits(w.d, g.d));
        }
    }
    {   // truncated tail: cut 5 bytes off the end -> last record dropped, the rest intact
        const auto sz = std::filesystem::file_size(path);
        std::filesystem::resize_file(path, sz - 5);
        auto t = DoorDecisionLog::read_all(path);
        CHECK(t.has_value() and t->size() == want.size() - 1);
    }
    {   // wrong magic / version refused
        std::ofstream o(dir + "/bad.bin", std::ios::binary); o.write("XXXX\1\0\0\0", 8); o.close();
        CHECK(not DoorDecisionLog::read_all(dir + "/bad.bin").has_value());
        std::ofstream o2(dir + "/v2.bin", std::ios::binary); o2.write("DDEC\2\0\0\0", 8); o2.close();
        CHECK(not DoorDecisionLog::read_all(dir + "/v2.bin").has_value());
        CHECK(not DoorDecisionLog::read_all(dir + "/missing.bin").has_value());
    }
    {   // rotation: re-opening keeps the previous run
        DoorDecisionLog a; CHECK(a.open(path));
        a.append(want[1]);
        a.close();
        DoorDecisionLog b; CHECK(b.open(path));
        b.close();
        int n = 0;
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().filename().string().starts_with("door_decisions.bin.")) ++n;
        CHECK(n >= 1);
    }
    std::print("{}\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails == 0 ? 0 : 1;
}
