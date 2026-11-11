// test_door_config_retired.cpp — Task 10 of the door-existence model: the config keys the old existence patches used are REFUSED
// at load (door_config_retired.h), with a message that names the plan; etc/config.toml sets none of them; and the plan's other
// keys still pass. The loader's own check is `retired_door_keys_present(cfg.exists)`; this test feeds it the real etc/config.toml
// (parsed as `[Section]` / `Key = value` lines, the shape the ConfigLoader exposes as "Section.Key") and synthetic files.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 tests/test_door_config_retired.cpp -o /tmp/test_door_config_retired && /tmp/test_door_config_retired
#include <algorithm>
#include <clocale>
#include <fstream>
#include <print>
#include <set>
#include <sstream>
#include <string>
#include "../src/door_config_retired.h"

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

static std::set<std::string> keys_of(std::istream& in)
{
    std::set<std::string> out;
    std::string line, section;
    while (std::getline(in, line))
    {
        const auto a = line.find_first_not_of(" \t");
        if (a == std::string::npos or line[a] == '#') continue;
        if (line[a] == '[')
        {
            const auto e = line.find(']', a);
            if (e != std::string::npos) section = line.substr(a + 1, e - a - 1);
            continue;
        }
        const auto eq = line.find('=', a);
        if (eq == std::string::npos) continue;
        auto k = line.substr(a, eq - a);
        while (not k.empty() and (k.back() == ' ' or k.back() == '\t')) k.pop_back();
        out.insert(section + "." + k);
    }
    return out;
}

int main()
{
    std::setlocale(LC_ALL, "");
    // the shipped config sets none of them
    {
        std::ifstream f("etc/config.toml");
        CHECK(static_cast<bool>(f));
        const auto keys = keys_of(f);
        CHECK(keys.contains("Tracker.GateMahalanobis") and keys.contains("Existence.MaxRoomS"));   // the parser sees real keys
        const auto stale = rc::retired_door_keys_present([&](const std::string& k) { return keys.contains(k); });
        CHECK(stale.empty());
        for (const auto& k : stale) std::print("config.toml still sets retired key {}\n", k);
    }
    // each retired key, alone, is found and refused
    for (const auto k : rc::kRetiredDoorKeys)
    {
        std::istringstream in("[Tracker]\nGateMahalanobis = 9.0\n[Existence]\nMaxRoomS = 300\n");
        auto keys = keys_of(in);
        keys.insert(std::string(k));
        const auto stale = rc::retired_door_keys_present([&](const std::string& q) { return keys.contains(q); });
        CHECK(stale.size() == 1 and stale[0] == k);
        const auto msg = rc::retired_door_keys_message(stale);
        CHECK(msg.contains(std::string(k)) and msg.contains("docs/superpowers/plans/2026-11-07-door-existence-model.md"));
    }
    // all eighteen named by the plan are in the table, nothing else is
    for (const char* k : {"Existence.BirthLogodds", "Existence.MaxLogodds", "Existence.RemovalProb", "Existence.RemoveFrames",
                          "Existence.OutOfRoomGain", "Existence.ShortGain", "Existence.MinHeightConf", "Existence.ReacquireRadiusM",
                          "Existence.GhostMax", "Existence.CentralRegionFrac", "Existence.SensorSigmaM", "Existence.RoomPrior",
                          "Existence.RoomMarginM", "Existence.MinHeightPrior", "Existence.MinHeightM", "Existence.DetectionProb", "Existence.ClutterProb",
                          "Tracker.BirthFrames"})
        CHECK(std::ranges::contains(rc::kRetiredDoorKeys, std::string_view(k)));
    CHECK(rc::kRetiredDoorKeys.size() == 18);
    // a file with a retired key among live ones lists exactly the retired one(s), in the table's order
    {
        std::istringstream in("[Existence]\nRemoveFrames = 15\nMaxRoomS = 300\nGhostMax = 8\n");
        const auto keys = keys_of(in);
        const auto stale = rc::retired_door_keys_present([&](const std::string& k) { return keys.contains(k); });
        CHECK(stale.size() == 2 and stale[0] == "Existence.RemoveFrames" and stale[1] == "Existence.GhostMax");
    }
    std::print("{}\n", fails == 0 ? "test_door_config_retired: ALL PASS" : "test_door_config_retired: FAILURES");
    return fails == 0 ? 0 : 1;
}
