// test_door_config_load.cpp — Task 10 fix round 1: the REAL load_door_config (door_config.cpp, a real ConfigLoader on a real TOML
// file) refuses a config that sets a retired key: it prints the message to STDERR, naming the plan, and exits non-zero. And it
// accepts the shipped etc/config.toml. The refusal is a process exit, so the test re-executes itself as the child.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -Itests/stub -I/home/pbustos/robocomp/robocomp_core/classes -I/usr/include/eigen3 -I../common tests/test_door_config_load.cpp \
//       src/door_config.cpp /home/pbustos/robocomp/robocomp_core/classes/ConfigLoader/ConfigLoader.cpp -o /tmp/test_door_config_load && /tmp/test_door_config_load
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <print>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <clocale>
#include <ConfigLoader/ConfigLoader.h>
#include "../src/door_config.h"

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

static int child(const std::string& toml)
{
    ConfigLoader cl;
    cl.load(toml);
    const auto cfg = rc::load_door_config(cl);
    return cfg.tracker_gate_mahalanobis > 0.0f ? 0 : 3;
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc == 3 and std::string(argv[1]) == "--child") return child(argv[2]);
    const auto run = [&](const std::string& toml, std::string& err)
    {
        const std::string errf = "/tmp/test_door_config_load.err";
        const std::string cmd = std::string(argv[0]) + " --child " + toml + " >/dev/null 2>" + errf;
        const int rc = std::system(cmd.c_str());
        std::ifstream f(errf); err.assign(std::istreambuf_iterator<char>(f), {});
        return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
    };
    std::string err;
    // the shipped config loads
    CHECK(run("etc/config.toml", err) == 0);
    // the same file plus one retired key is refused, on stderr, naming the plan
    {
        std::ifstream in("etc/config.toml"); std::string body((std::istreambuf_iterator<char>(in)), {});
        const auto at = body.find("[Existence]");
        CHECK(at != std::string::npos);
        body.insert(at + 11, "\nRemoveFrames = 15\n");
        std::ofstream("/tmp/test_door_config_load.toml") << body;
        CHECK(run("/tmp/test_door_config_load.toml", err) == EXIT_FAILURE);
        CHECK(err.contains("Existence.RemoveFrames") and err.contains("docs/superpowers/plans/2026-11-07-door-existence-model.md"));
    }
    std::print("{}\n", fails == 0 ? "test_door_config_load: ALL PASS" : "test_door_config_load: FAILURES");
    return fails == 0 ? 0 : 1;
}
