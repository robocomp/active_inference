// Starting / stopping fleet agents from the chat (fleet_control.h): which agent the words name, whether the request is
// refused, asked about or carried out, the process lookup by name in a /proc tree, the tour lock, the yes that
// confirms, and the start/stop/restart sequence. Build+run: tests/run_tests.sh
#include "../src/fleet_control.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <print>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace rc::chat::fleet;

namespace
{
/// A fake /proc entry: <root>/<pid>/comm holding the kernel's process name.
void proc(const fs::path& root, int pid, const std::string& comm)
{
    fs::create_directories(root / std::to_string(pid));
    std::ofstream(root / std::to_string(pid) / "comm") << comm << "\n";
}
const std::vector<std::string> kAgents{"room_concept", "retina", "residual_concept", "door_concept", "table_concept",
                                       "chair_concept", "controller"};
}   // namespace

int main()
{
    int checks = 0;

    // ── the action and the agent the words name ──────────────────────────────────────────────────────────────────
    assert(parse_action("start") == Action::start); ++checks;
    assert(parse_action(" Restart ") == Action::restart); ++checks;
    assert(not parse_action("kill").has_value()); ++checks;
    assert(resolve_agent("chair_concept", kAgents) == "chair_concept"); ++checks;
    assert(resolve_agent("chair", kAgents) == "chair_concept"); ++checks;            // the model says "chair"
    assert(resolve_agent("Table Concept", kAgents) == "table_concept"); ++checks;
    assert(resolve_agent("the retina", kAgents) == "retina"); ++checks;
    {
        const auto r = resolve_agent("bottle", kAgents);                               // not in the fleet list
        assert(not r.has_value() and r.error().find("chair_concept") != std::string::npos); ++checks;
    }
    assert(split_names(" a, b ,,c ") == (std::vector<std::string>{"a", "b", "c"})); ++checks;

    // ── the kernel's name is the first 15 characters: residual_concept runs as "residual_concep" ────────────────
    assert(process_name("residual_concept") == "residual_concep"); ++checks;
    assert(process_name("controller") == "controller"); ++checks;
    {
        const fs::path root = fs::temp_directory_path() / ("test_fleet_proc_" + std::to_string(getpid()));
        fs::remove_all(root);
        proc(root, 101, "chair_concept");
        proc(root, 102, "chair_concept_x");   // another program: not ours
        proc(root, 103, "residual_concep");
        proc(root, 104, "bash");
        fs::create_directories(root / "self");   // non-numeric entries are skipped
        assert(find_pids(root, "chair_concept") == std::vector<int>{101}); ++checks;
        assert(find_pids(root, "residual_concept") == std::vector<int>{103}); ++checks;
        assert(find_pids(root, "table_concept").empty()); ++checks;

        // ── the tour lock: held only while the PID written in it is alive ────────────────────────────────────────
        const fs::path lock = root / "ai_fleet_tour.lock";
        assert(not tour_running(lock, root)); ++checks;                 // no file
        std::ofstream(lock) << "104\n";
        assert(tour_running(lock, root)); ++checks;
        std::ofstream(lock) << "999\n";                                  // the script crashed and left it
        assert(not tour_running(lock, root)); ++checks;
        std::ofstream(lock) << "junk\n";
        assert(not tour_running(lock, root)); ++checks;
        fs::remove_all(root);
    }

    // ── refused, asked about, or carried out ─────────────────────────────────────────────────────────────────────
    {
        const Policy p{.agents = kAgents, .confirm = {"controller", "retina"}};
        auto d = decide(Action::stop, "chair_concept", p, false, true, false);
        assert(d.kind == Decision::Kind::go); ++checks;
        d = decide(Action::stop, "chair_concept", p, true, true, false);   // a tour is running
        assert(d.kind == Decision::Kind::refuse and d.text.find("tour") != std::string::npos); ++checks;
        d = decide(Action::stop, "controller", p, false, true, false);
        assert(d.kind == Decision::Kind::ask and d.text.find("controller") != std::string::npos); ++checks;
        d = decide(Action::stop, "controller", p, false, true, true);      // the person said yes
        assert(d.kind == Decision::Kind::go); ++checks;
        d = decide(Action::restart, "retina", p, false, true, false);      // a restart stops it too
        assert(d.kind == Decision::Kind::ask); ++checks;
        d = decide(Action::restart, "retina", p, false, false, false);     // nothing to stop: no yes needed
        assert(d.kind == Decision::Kind::go); ++checks;
        d = decide(Action::start, "retina", p, false, false, false);       // starting needs no yes
        assert(d.kind == Decision::Kind::go); ++checks;
        d = decide(Action::start, "chair_concept", p, false, true, false);
        assert(d.kind == Decision::Kind::refuse and d.text.find("already running") != std::string::npos); ++checks;
        d = decide(Action::stop, "chair_concept", p, false, false, false);
        assert(d.kind == Decision::Kind::refuse and d.text.find("not running") != std::string::npos); ++checks;
        d = decide(Action::restart, "chair_concept", p, false, false, false);   // not running: a restart just starts
        assert(d.kind == Decision::Kind::go); ++checks;
        d = decide(Action::stop, "chat_agent", p, false, true, false);
        assert(d.kind == Decision::Kind::refuse and d.text.find("myself") != std::string::npos); ++checks;
        d = decide(Action::restart, "robot_concept", p, false, true, false);
        assert(d.kind == Decision::Kind::refuse and d.text.find("depends") != std::string::npos); ++checks;
    }

    // ── the yes: the same request again, after a NEW message from the person, within a minute ────────────────────
    {
        Confirmations c;
        assert(not c.confirmed(Action::stop, "controller", 1, 0)); ++checks;         // never asked
        c.ask(Action::stop, "controller", 1, 0);
        assert(not c.confirmed(Action::stop, "controller", 1, 1000)); ++checks;      // same turn: the model alone
        assert(not c.confirmed(Action::stop, "retina", 2, 1000)); ++checks;          // another agent
        assert(c.confirmed(Action::stop, "controller", 2, 1000)); ++checks;
        assert(not c.confirmed(Action::stop, "controller", 3, 2000)); ++checks;      // used once
        c.ask(Action::restart, "retina", 4, 0);
        assert(not c.confirmed(Action::restart, "retina", 5, 61000)); ++checks;      // too late
    }

    // ── a plain yes is carried out without the model (it said "stopped" without calling the tool, eval 2026-10-19) ─
    {
        for (const char* y : {"yes", "Yes!", "yes please", "sí", "si", "vale", "ok", "do it", "hazlo", "go ahead"})
        {
            assert(is_yes(y)); ++checks;
        }
        for (const char* n : {"no", "no, leave it", "yesterday", "is it running?", "", "yes but first tell me why"})
        {
            assert(not is_yes(n)); ++checks;
        }
        Confirmations c;
        assert(not c.pending().has_value()); ++checks;
        c.ask(Action::stop, "controller", 1, 0);
        assert(c.pending().has_value() and c.pending()->agent == "controller" and c.pending()->action == Action::stop);
        ++checks;
        assert(c.confirmed(Action::stop, "controller", 2, 1000)); ++checks;
        assert(not c.pending().has_value()); ++checks;   // used
    }

    // ── the sequence: stop waits for the exit, restart then launches, start waits for the new heartbeat ──────────
    {
        Op op(Action::stop, "chair_concept", 0);
        auto e = op.tick(0, true, false);
        assert(not e.launch and not e.done); ++checks;
        e = op.tick(500, false, false);
        assert(e.done and e.report.find("stopped") != std::string::npos); ++checks;
    }
    {
        Op op(Action::stop, "chair_concept", 0);
        auto e = op.tick(kStopWaitMs + 1, true, false);   // ignored SIGTERM: reported, never SIGKILLed
        assert(e.done and e.report.find("still running") != std::string::npos); ++checks;
    }
    {
        Op op(Action::restart, "chair_concept", 0);
        auto e = op.tick(100, true, true);
        assert(not e.launch and not e.done); ++checks;
        e = op.tick(1000, false, false);
        assert(e.launch and not e.done); ++checks;
        e = op.tick(1500, true, false);
        assert(not e.launch and not e.done); ++checks;
        e = op.tick(4000, true, true);
        assert(e.done and e.report.find("up") != std::string::npos); ++checks;
    }
    {
        Op op(Action::start, "chair_concept", 0);
        auto e = op.tick(0, false, false);
        assert(e.launch); ++checks;
        e = op.tick(kStartWaitMs + 1, false, false);
        assert(e.done and e.report.find("exited") != std::string::npos); ++checks;
        Op op2(Action::start, "chair_concept", 0);
        op2.tick(0, false, false);
        e = op2.tick(kStartWaitMs + 1, true, false);
        assert(e.done and e.report.find("no heartbeat") != std::string::npos); ++checks;
    }

    // ── "up" means a heartbeat from the NEW process: uptime no longer than since the launch ──────────────────────
    {
        rc::chat::AgentFact a;
        a.name = "chair_concept";
        a.heartbeat_age_ms = 100;
        a.uptime_s = 3;
        assert(started_since(a, 5000)); ++checks;
        a.uptime_s = 600;   // the old instance's last heartbeat, not yet stale
        assert(not started_since(a, 5000)); ++checks;
        a.uptime_s = 3;
        a.heartbeat_age_ms = 10000;
        assert(not started_since(a, 5000)); ++checks;
    }

    std::println("test_fleet: {} checks passed", checks);
}
