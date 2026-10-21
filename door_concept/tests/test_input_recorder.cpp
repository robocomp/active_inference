// Offline harness for src/input_recorder.h + src/input_latch.h (Task 3 of the DoorCore record/replay plan).
// Checks:
//   1. SEQUENCE: a synthetic run — snapshots with events latched BETWEEN cycles (foreign deletes, superseded upserts,
//      UI presses, an actuator call from a Qt callback), actuator calls made INSIDE cycles, the startup identity load —
//      written through the recorder and read back: same cycles, same order, events in arrival order with their stamps,
//      between-cycle calls BEFORE in-cycle calls, the identity load on the first record only.
//   2. OWN-NODE FILTER: own ids (created_by this agent) never reach a record — node events are dropped by the latch
//      and object/wall rows by keep_foreign, which input_snapshot.cpp applies to every graph walk.
//   3. CAP: at RecordMaxMB the recorder STOPS (one log line), the file ends on a record boundary, is ≤ the cap, reads
//      back CleanEof with exactly the records written, and later record() calls are refused.
//   4. STOP MID-RECORD: a writer killed half-way through a record leaves a file that reads back every complete record
//      and reports Truncated at the offset of the cut one.
// Runs under the es_ES locale on purpose (CLAUDE.md "Parsing numbers"): a binary tape must be immune to it.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_input_recorder.cpp src/core/door_cycle_io.cpp -o /tmp/test_input_recorder && /tmp/test_input_recorder
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <print>
#include <random>
#include <sstream>
#include <unistd.h>
#include "../src/input_latch.h"
#include "../src/input_recorder.h"

using namespace rc::door;
using namespace rc::door::core;
namespace fs = std::filesystem;

namespace
{
int g_fail = 0;
void check(bool ok, const std::string& what)
{
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}

constexpr std::uint32_t kAgent = 25;                                  // door_concept's [Agent] id
std::uint64_t own_id(std::uint64_t seq)     { return (seq << kDsrAgentIdBits) | kAgent; }
std::uint64_t foreign_id(std::uint64_t seq, std::uint32_t agent = 7) { return (seq << kDsrAgentIdBits) | agent; }

ObjectInfo object(std::uint64_t id, const std::string& name)
{
    ObjectInfo o;
    o.id = id; o.name = name; o.type = "object";
    o.width_m = 0.9f;
    Mat4d T = Mat4d::Identity(); T(0, 3) = double(id % 97) * 0.01;
    o.room_T_node = T;
    return o;
}

/// One synthetic snapshot: what input_snapshot.cpp builds at the top of compute(), with the own rows filtered.
DoorCycleInput snapshot(std::uint64_t cycle, std::mt19937& rng)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = std::int64_t(1'000'000'000) + std::int64_t(cycle) * 100'000'000 + std::int64_t(rng() % 1000);
    in.t_wall_ms = 1'792'000'000'000 + std::int64_t(cycle) * 100;
    in.rooms.graph_current_room = foreign_id(3);
    in.rooms.n_current_edges = 1;
    in.rooms.current_edge_room = foreign_id(3);
    RoomInfo r; r.id = foreign_id(3); r.name = "room_1"; r.polygon = {{0, 0}, {4, 0}, {4, 6}, {0, 6}};
    in.rooms.rooms.push_back(r);
    in.objects = {object(foreign_id(10), "table_1"), object(own_id(11), "door_1"), object(foreign_id(12), "chair_2")};
    WallInfo w; w.id = foreign_id(20); w.width_m = 4.f; in.walls.push_back(w);
    WallInfo w_own; w_own.id = own_id(21); in.walls.push_back(w_own);
    keep_foreign(in.objects, kAgent);    // ← the builder's own-node filter
    keep_foreign(in.walls, kAgent);
    in.robot.present = true; in.robot.name = "Shadow"; in.robot.id = foreign_id(2);
    if (cycle % 3 == 0)
    {
        MaskPacket m;
        m.valid = true; m.frame_id = int(cycle); m.timestamp_ms = std::uint64_t(in.t_ms());
        MaskSlice s; s.label = "door"; s.support_begin = 0; s.support_end = 50;
        m.slices.push_back(s);
        for (int i = 0; i < 50; ++i) m.support_points.emplace_back(float(i) * 0.01f, 1.f, float(rng() % 100) * 0.01f);
        in.masks = std::move(m);
    }
    if (cycle % 7 == 0)
    {
        ImageFrame g; g.format = PixelFormat::Gray8; g.width = 1280; g.height = 720; g.frame_ref = 3.5f;
        ImagePatch p; p.x0 = 100; p.y0 = 50; p.w = 40; p.h = 30; p.bytes.assign(std::size_t(40 * 30), std::uint8_t(cycle));
        g.patches.push_back(p);
        in.rgb_gray = std::move(g);
    }
    return in;
}
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    const auto* lc = std::setlocale(LC_NUMERIC, nullptr);
    std::println("LC_NUMERIC {}, decimal point '{}'", lc ? lc : "?", std::localeconv()->decimal_point);

    const fs::path dir = fs::temp_directory_path() / ("test_input_recorder_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string path = (dir / "door_inputs.bin").string();

    DoorRecordingHeader hdr;
    hdr.agent_id = kAgent; hdr.agent_name = "door_concept"; hdr.config_text = "[DoorConcept]\nRecordInputs = true\n";
    hdr.start_steady_ns = 1'000'000'000; hdr.start_wall_ms = 1'792'000'000'000; hdr.note = "test";

    // ── 1 + 2: a synthetic run through the latch and the recorder ─────────────────────────────────────────────────
    std::mt19937 rng(1234);
    InputLatch latch(kAgent);
    DoorInputRecorder rec;
    check(rec.open(path, hdr, 0), "open (no cap)");

    IdentityLoad il; il.file_present = true; il.csv_text = "room,name,x,y\nroom_1,door_1,1.5,0.0\n";
    il.initial_room = foreign_id(3); il.initial_room_name = "room_1";
    il.registration_present = true; il.registration_csv_text = "room_1,door_1,DOOR_0\n";
    latch.identity_load(il);

    constexpr int kCycles = 60;
    std::vector<DoorCycleInput> expect;
    int own_dropped = 0;
    for (int c = 1; c <= kCycles; ++c)
    {
        // between cycles: events arrive (stamps strictly increasing within the gap)
        const std::int64_t t0 = std::int64_t(1'000'000'000) + std::int64_t(c - 1) * 100'000'000 + 5'000'000;
        std::vector<DoorEvent> want_events;
        if (c % 4 == 0)
        {
            check(latch.node_deleted(foreign_id(100 + c), t0 + 1), "foreign delete kept");
            want_events.push_back({t0 + 1, NodeDeleted{foreign_id(100 + c), false}});
            {
                InputLatch::SelfScope self(latch);                                  // door_concept's own delete call
                if (not latch.node_deleted(own_id(200 + c), t0 + 2)) ++own_dropped;   // self-requested: dropped
            }
            if (c % 8 == 0)   // one of OUR nodes deleted by somebody else (ltsm evicting a room: tourP4) — recorded
            {
                check(latch.node_deleted(own_id(400 + c), t0 + 2), "own node deleted by another agent latched");
                want_events.push_back({t0 + 2, NodeDeleted{own_id(400 + c), true}});
            }
        }
        if (c % 5 == 0)
        {
            DoorNodeUpserted u1{object(foreign_id(300), "door_9"), RtEdge{foreign_id(3), {1, 2, 0}, {0, 0, 0.5f}}};
            DoorNodeUpserted u2 = u1; u2.node.width_m = 1.1f;     // the SAME node, changed again: supersedes u1
            DoorNodeUpserted mine{object(own_id(301), "door_2"), std::nullopt};
            check(latch.door_upserted(u1, t0 + 3), "foreign upsert kept");
            latch.ui_request(UiActuationRequest{true, "DOOR_0"}, t0 + 4);
            check(latch.door_upserted(u2, t0 + 5), "foreign upsert (again) kept");
            if (not latch.door_upserted(mine, t0 + 6)) ++own_dropped;
            want_events.push_back({t0 + 4, UiActuationRequest{true, "DOOR_0"}});
            want_events.push_back({t0 + 5, u2});
        }
        if (c == 1)
        {
            latch.owned_swept(OwnedNodesSwept{"first-Operating sweep", {own_id(5), own_id(6)}}, t0 + 7);
            want_events.push_back({t0 + 7, OwnedNodesSwept{"first-Operating sweep", {own_id(5), own_id(6)}}});
        }
        std::vector<ActuatorCall> want_calls;
        if (c % 6 == 0)   // an actuator call from a Qt callback, between cycles
        {
            ActuatorCall q; q.kind = ActuatorCallKind::Request; q.door_name = "door_9"; q.xy_m = {1.f, 2.f};
            q.pending = ActuatorPending{c, "door_9", true, "Queued", false};
            latch.actuator_call(q);
            want_calls.push_back(q);
        }

        DoorCycleInput in = snapshot(std::uint64_t(c), rng);
        latch.begin_cycle(in);
        check(latch.in_cycle(), "in cycle after begin");
        if (c % 6 == 0 or c % 11 == 0)   // calls made INSIDE step(): after the between-cycle ones
        {
            ActuatorCall p; p.kind = ActuatorCallKind::Poll;
            p.poll_in = ActuatorPending{c, "door_9", true, "Queued", false};
            p.pending = ActuatorPending{c, "door_9", true, "Delivered", true};
            latch.actuator_call(p);
            want_calls.push_back(p);
        }
        latch.end_cycle();
        check(not latch.in_cycle(), "out of cycle after end");

        check(in.events == want_events, std::format("cycle {}: events as latched, in arrival order", c));
        check(in.actuator_calls == want_calls, std::format("cycle {}: actuator transcript in call order", c));
        check(in.identity_load.has_value() == (c == 1), std::format("cycle {}: identity load only on the first", c));
        check(rec.record(in), std::format("record cycle {}", c));
        expect.push_back(std::move(in));
    }
    check(own_dropped == kCycles / 4 + kCycles / 5, std::format("own node events dropped ({} of {})", own_dropped,
                                                                 kCycles / 4 + kCycles / 5));
    check(latch.pending_events() == 0 and latch.pending_calls() == 0 and not latch.identity_load_pending(),
          "latch drained");
    const std::uint64_t full_bytes = rec.bytes();
    rec.close();
    check(fs::file_size(path) == full_bytes, "bytes() == file size");

    const auto back = read_input_file(path);
    check(back.has_value(), "read back");
    if (back)
    {
        check(back->header == hdr, "header round-trips");
        check(back->stop == StopReason::CleanEof, "clean EOF");
        check(back->cycles.size() == expect.size(), std::format("{} records", back->cycles.size()));
        bool same = back->cycles.size() == expect.size();
        for (std::size_t i = 0; same and i < expect.size(); ++i)
            same = back->cycles[i] == expect[i];
        check(same, "every record equals what was written, in order");
        bool own_absent = true;
        for (const auto& in : back->cycles)
        {
            for (const auto& o : in.objects) own_absent = own_absent and is_foreign(o.id, kAgent);
            for (const auto& w : in.walls)   own_absent = own_absent and is_foreign(w.id, kAgent);
            for (const auto& e : in.events)
            {
                if (const auto* d = std::get_if<NodeDeleted>(&e.payload))   // own ids only as by_other_agent deletions
                    own_absent = own_absent and (is_foreign(d->id, kAgent) != d->by_other_agent);
                if (const auto* u = std::get_if<DoorNodeUpserted>(&e.payload)) own_absent = own_absent and is_foreign(u->node.id, kAgent);
            }
        }
        check(own_absent, "no own id in any recorded object/wall/node event");
        std::println("sequence: {} cycles, {} bytes, {} events, {} actuator calls", back->cycles.size(), full_bytes,
                     [&] { std::size_t n = 0; for (auto& c : back->cycles) n += c.events.size(); return n; }(),
                     [&] { std::size_t n = 0; for (auto& c : back->cycles) n += c.actuator_calls.size(); return n; }());
    }
    // the filter on its own
    {
        std::vector<ObjectInfo> v{object(own_id(1), "door_1"), object(foreign_id(1), "table"), object(own_id(2), "door_2"),
                                  object(foreign_id(2, 26), "x")};
        keep_foreign(v, kAgent);
        check(v.size() == 2 and v[0].name == "table" and v[1].name == "x", "keep_foreign drops exactly the own rows");
        check(created_by(own_id(77), kAgent) and not created_by(foreign_id(77), kAgent), "created_by on the low 12 bits");
    }
    // self-requested vs another agent's deletion of OUR node (ruling, Task 3)
    {
        InputLatch l(kAgent);
        {
            InputLatch::SelfScope outer(l);
            InputLatch::SelfScope inner(l);                                    // nests
            check(not l.node_deleted(own_id(900), 10), "own delete inside a self scope: not latched");
        }
        check(not l.in_self(), "scope closed");
        check(l.self_deleted().contains(own_id(900)), "self-requested deletion tracked");
        check(not l.node_deleted(own_id(900), 11), "late echo of a self-requested delete: not latched");
        check(l.node_deleted(own_id(901), 12), "own node deleted by another agent: latched");
        check(l.node_deleted(foreign_id(902), 13), "foreign delete: latched");
        { InputLatch::SelfScope s2(l); check(l.node_deleted(foreign_id(903), 14), "foreign delete inside our scope: still latched"); }
        DoorCycleInput in;
        l.begin_cycle(in); l.end_cycle();
        const std::vector<DoorEvent> want{{12, NodeDeleted{own_id(901), true}}, {13, NodeDeleted{foreign_id(902), false}},
                                          {14, NodeDeleted{foreign_id(903), false}}};
        check(in.events == want, "deletion events: by_other_agent exactly on the own id nobody here asked to delete");
    }

    // ── 3: the cap ─────────────────────────────────────────────────────────────────────────────────────────────────
    {
        const std::string cpath = (dir / "door_inputs_cap.bin").string();
        DoorInputRecorder capped;
        const std::uint64_t cap = full_bytes / 3;   // about a third of the sequence fits
        check(capped.open(cpath, hdr, cap), "open (capped)");
        std::size_t written = 0;
        bool refused_once = false;
        for (const auto& in : expect)
        {
            if (capped.record(in)) ++written;
            else { refused_once = true; break; }
        }
        check(refused_once and capped.stopped(), "cap reached → stopped");
        check(not capped.enabled(), "stopped ⇒ closed");
        check(not capped.record(expect.front()), "record after the stop is refused");
        check(fs::file_size(cpath) <= cap, std::format("file {} B ≤ cap {} B", fs::file_size(cpath), cap));
        const auto cb = read_input_file(cpath);
        check(cb.has_value() and cb->stop == StopReason::CleanEof, "capped file ends on a record boundary (CleanEof)");
        check(cb.has_value() and cb->cycles.size() == written and written > 0 and written < expect.size(),
              std::format("capped file holds exactly the {} records written", written));
        if (cb)
            for (std::size_t i = 0; i < cb->cycles.size(); ++i)
                check(cb->cycles[i] == expect[i], std::format("capped record {} intact", i));
        std::println("cap: {} of {} records within {} B", written, expect.size(), cap);
    }

    // ── 4: a writer killed mid-record ──────────────────────────────────────────────────────────────────────────────
    {
        const std::string kpath = (dir / "door_inputs_killed.bin").string();
        DoorInputRecorder r;
        check(r.open(kpath, hdr, 0), "open (killed)");
        for (int i = 0; i < 10; ++i) check(r.record(expect[std::size_t(i)]), "record before the kill");
        const std::uint64_t boundary = r.bytes();
        r.close();
        std::ostringstream one(std::ios::binary);
        check(write(one, expect[10]), "serialise the record being written at the kill");
        const std::string half = one.str().substr(0, one.str().size() / 2);
        {
            std::ofstream f(kpath, std::ios::binary | std::ios::app);
            f.write(half.data(), std::streamsize(half.size()));
        }
        const auto kb = read_input_file(kpath);
        check(kb.has_value(), "killed file opens");
        check(kb.has_value() and kb->cycles.size() == 10, "every complete record survives the kill");
        check(kb.has_value() and kb->stop == StopReason::Truncated and kb->stop_offset == boundary,
              std::format("Truncated at the cut record's offset ({})", boundary));
    }

    fs::remove_all(dir);
    if (g_fail == 0) std::println("ALL PASS");
    else std::println("{} FAILURE(S)", g_fail);
    return g_fail == 0 ? 0 : 1;
}
