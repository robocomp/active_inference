// The record builder (em_builder.h): controller journal + live facts + structural events -> commitment records.
#include "../em_builder.h"

#include <clocale>
#include <cmath>
#include <print>
#include <string>

namespace
{
int failures = 0, checks = 0;

void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

using namespace rc::em;
using K = JournalEntry::Kind;

struct FakeFactSource : FactSource
{
    std::map<std::string, Fact, std::less<>> facts;
    std::string room = "room_1";
    std::optional<Fact> fact(std::string_view node) const override
    {
        const auto it = facts.find(node);
        if (it == facts.end()) return std::nullopt;
        return it->second;
    }
    std::string current_room() const override { return room; }
};

Fact door(std::string node, double open_p, std::optional<double> cxx = {}, std::optional<double> cyy = {})
{
    Fact f{.node = std::move(node), .type = "door", .ltsm_id = "passage_0", .values = {{"door_open_prob", open_p}}};
    if (cxx) f.values["cov_xx"] = *cxx;
    if (cyy) f.values["cov_yy"] = *cyy;
    return f;
}

// A scripted controller journal: serials count from 1, everything is kept (tests slice it to fake a ring).
struct Script
{
    std::uint64_t epoch = 1000;
    std::vector<JournalEntry> all;
    JournalEntry& push(K k, std::uint64_t t)
    {
        JournalEntry e;
        e.kind = k;
        e.serial = all.size() + 1;
        e.t_ms = t;
        all.push_back(e);
        return all.back();
    }
    WireView wire() const { return {epoch, all}; }
    WireView wire_from(std::uint64_t serial) const
    {
        WireView w{epoch, {}};
        for (const auto& e : all)
            if (e.serial >= serial) w.entries.push_back(e);
        return w;
    }
};

std::uint64_t seq = 0;
std::function<std::string(std::uint64_t)> ids() { return [](std::uint64_t) { return std::format("id{}", ++seq); }; }

std::vector<Record> closed(const std::vector<Record>& rs)
{
    std::vector<Record> out;
    for (const Record& r : rs)
        if (r.t1) out.push_back(r);
    return out;
}
const Record* find(const std::vector<Record>& rs, Level l, std::string_view kind)
{
    for (const Record& r : rs)
        if (r.level == l and r.kind == kind) return &r;
    return nullptr;
}

// Primes the builder (ruling R8: first wire only adopts) and leaves drive mode mission_only, so no self mission.
void prime_mission_only(Builder& b, Script& s, std::uint64_t now)
{
    b.on_journal(s.wire(), now);
    s.push(K::DriveMode, now).op = "mission_only";
}

void push_mission(Script& s, std::uint64_t t, std::string m = "m1")
{
    auto& e = s.push(K::MissionStart, t);
    e.mission = std::move(m); e.source = "chat"; e.text = R"({"steps":[]})";
}
void push_step(Script& s, K k, std::uint64_t t, std::string id, int attempt = 1, int iteration = 0, std::string out = "")
{
    auto& e = s.push(k, t);
    e.mission = "m1"; e.step_id = std::move(id); e.op = "cross"; e.attempt = attempt; e.iteration = iteration;
    e.text = R"({"target":"door_3","grounding":{"door":"door_3"}})"; e.outcome = std::move(out);
}
void push_claim(Script& s, std::uint64_t t, std::optional<double> promised = 0.4)
{
    auto& e = s.push(K::ClaimStart, t);
    e.op = "cross_door";
    e.text = R"({"affordance":"aff_cross_door_3","producer":"door_concept","target":"door_3","target_type":"door","contract":{"goal":"other_side"}})";
    e.promised = promised;
    e.candidates = R"([{"name":"aff_cross_door_3","G":-1.2}])";
}
void push_claim_end(Script& s, std::uint64_t t, std::string out)
{
    auto& e = s.push(K::ClaimEnd, t);
    e.op = "cross_door"; e.outcome = std::move(out);
}

void test_mission_step_claim_nesting()
{
    FakeFactSource fs;
    fs.facts["door_3"] = door("door_3", 0.2, 0.04, 0.09);
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "1");
    push_claim(s, 120);
    CHECK(closed(b.on_journal(s.wire(), 130)).empty());
    CHECK(b.open().size() == 3);

    fs.facts["door_3"] = door("door_3", 0.9, 0.01, 0.01);
    fs.room = "room_2";
    push_claim_end(s, 200, "success");
    push_step(s, K::StepEnd, 210, "1", 1, 0, "done");
    auto& me = s.push(K::MissionEnd, 220); me.mission = "m1"; me.outcome = "succeeded";
    const auto c = closed(b.on_journal(s.wire(), 230));
    CHECK(c.size() == 3);
    CHECK(b.open().empty());
    const Record* m = find(c, Level::Mission, "mission");
    const Record* st = find(c, Level::Step, "cross");
    const Record* cl = find(c, Level::Claim, "cross_door");
    CHECK(m and st and cl);
    if (not (m and st and cl)) return;
    CHECK(m->parent.empty() and st->parent == m->id and cl->parent == st->id);
    CHECK(m->author == Author::Human and st->author == Author::Human and cl->author == Author::Human);
    CHECK(m->source == "chat" and m->action_json == R"({"steps":[]})" and m->room == "room_1");
    CHECK(m->pre.empty() and m->post.empty());
    CHECK(m->t0 == 100 and m->t1 == 220u and m->outcome == "succeeded");
    CHECK(st->target == "door_3" and st->attempt == 1 and st->outcome == "done");
    CHECK(cl->target == "door_3" and cl->target_ltsm == "passage_0" and cl->outcome == "success");
    CHECK(cl->pre.size() == 1 and cl->pre[0].values.at("door_open_prob") == 0.2);
    CHECK(cl->post.size() == 1 and cl->post[0].values.at("door_open_prob") == 0.9);
    CHECK(cl->promised == 0.4 and cl->alternatives_json == R"([{"name":"aff_cross_door_3","G":-1.2}])");
    CHECK(cl->action_json == R"({"goal":"other_side"})");
    CHECK(cl->delivered.has_value() and std::abs(*cl->delivered - 0.5 * std::log(0.04 * 0.09 / (0.01 * 0.01))) < 1e-12);
    CHECK(cl->t0 == 120 and cl->t1 == 200u);
}

void test_attempts_are_records()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "2", 1);
    push_step(s, K::StepEnd, 120, "2", 1, 0, "failed");
    push_step(s, K::StepStart, 130, "2", 2);
    push_step(s, K::StepEnd, 140, "2", 2, 0, "done");
    const auto c = closed(b.on_journal(s.wire(), 150));
    int steps = 0;
    for (const Record& r : c)
        if (r.level == Level::Step)
        {
            ++steps;
            CHECK((r.attempt == 1 and r.outcome == "failed") or (r.attempt == 2 and r.outcome == "done"));
        }
    CHECK(steps == 2);
    CHECK(b.open().size() == 1);   // the mission
}

void test_composite_nesting()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    auto& rep = s.push(K::StepStart, 110); rep.mission = "m1"; rep.step_id = "2"; rep.op = "repeat"; rep.text = "{}";
    push_step(s, K::StepStart, 120, "2.1", 1, 1);
    const auto rs = b.on_journal(s.wire(), 130);
    const Record* r2 = find(rs, Level::Step, "repeat");
    const Record* r21 = find(rs, Level::Step, "cross");
    CHECK(r2 and r21);
    if (r2 and r21) CHECK(r21->parent == r2->id and r21->iteration == 1);
    // "21" is not a child of "2": a strict DOTTED prefix only
    auto& o = s.push(K::StepStart, 140); o.mission = "m1"; o.step_id = "21"; o.op = "turn"; o.text = "{}";
    const Record* r_21 = find(b.on_journal(s.wire(), 150), Level::Step, "turn");
    const auto open = b.open();                      // keep the copy alive: b.open() returns by value
    const Record* m = nullptr;
    for (const Record& x : open) if (x.level == Level::Mission) m = &x;
    CHECK(r_21 and m);
    if (r_21 and m) CHECK(r_21->parent == m->id);
}

void test_preempted_claim()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "1");
    push_claim(s, 120);
    push_claim_end(s, 130, "preempted");
    const auto c = closed(b.on_journal(s.wire(), 140));
    CHECK(c.size() == 1 and c[0].level == Level::Claim and c[0].outcome == "preempted");
}

void test_self_mission_split_on_room()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    b.on_journal(s.wire(), 10);
    s.push(K::DriveMode, 20).op = "affordances_only";
    const auto r1 = b.on_journal(s.wire(), 30);
    CHECK(r1.size() == 1 and r1[0].author == Author::Self and r1[0].source == "self" and r1[0].target.empty());
    if (r1.size() != 1) return;
    const std::string id1 = r1[0].id;
    fs.room = "room_2";       // no claim open: the split is immediate
    const auto r2 = b.on_events(std::vector<Event>{{.dt_ms = 0, .kind = "current_moved", .node = "room_2"}}, 50);
    const auto c = closed(r2);
    CHECK(c.size() == 1 and c[0].id == id1 and c[0].outcome == "room_changed" and c[0].t1 == 50u
          and c[0].events.size() == 1);
    CHECK(b.open().size() == 1 and b.open()[0].author == Author::Self and b.open()[0].room == "room_2"
          and b.open()[0].id != id1);
}

// (a) a door-crossing claim changes room mid-claim: the split waits for its ClaimEnd, which keeps its real outcome.
void test_self_split_deferred_while_claim_open()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    b.on_journal(s.wire(), 10);
    s.push(K::DriveMode, 20).op = "affordances_only";
    push_claim(s, 30);
    const auto r1 = b.on_journal(s.wire(), 40);
    const Record* self1 = find(r1, Level::Mission, "mission");
    const Record* cl = find(r1, Level::Claim, "cross_door");
    CHECK(self1 and cl);
    if (not (self1 and cl)) return;
    CHECK(cl->parent == self1->id and cl->author == Author::Self);
    const std::string id1 = self1->id, cid = cl->id;

    fs.room = "room_2";
    CHECK(b.on_events(std::vector<Event>{{.dt_ms = 0, .kind = "current_moved", .node = "room_2"}}, 50).empty());
    CHECK(b.on_journal(s.wire(), 60).empty());
    CHECK(b.open().size() == 2);                     // self mission + claim, unsplit

    push_claim_end(s, 70, "success");
    const auto r3 = b.on_journal(s.wire(), 80);
    const auto c = closed(r3);
    CHECK(c.size() == 2 and c[0].id == cid and c[0].outcome == "success" and c[0].parent == id1);
    CHECK(c.size() == 2 and c[1].id == id1 and c[1].outcome == "room_changed" and c[1].t1 == 80u);
    CHECK(b.open().size() == 1 and b.open()[0].author == Author::Self and b.open()[0].room == "room_2"
          and b.open()[0].t0 == 80);
}

// (b) a human mission supersedes the self mission at once; its open claim stays open and closes by its ClaimEnd.
void test_supersede_leaves_self_claim_open()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    b.on_journal(s.wire(), 10);
    s.push(K::DriveMode, 20).op = "affordances_only";
    push_claim(s, 30);
    b.on_journal(s.wire(), 40);
    push_mission(s, 50);
    const auto c = closed(b.on_journal(s.wire(), 60));
    CHECK(c.size() == 1 and c[0].author == Author::Self and c[0].level == Level::Mission
          and c[0].outcome == "superseded" and c[0].t1 == 50u);
    if (c.size() != 1) return;
    const std::string self_id = c[0].id;
    CHECK(b.open().size() == 2);                     // the human mission + the orphaned self claim
    push_claim_end(s, 70, "preempted");
    const auto c2 = closed(b.on_journal(s.wire(), 80));
    CHECK(c2.size() == 1 and c2[0].level == Level::Claim and c2[0].outcome == "preempted"
          and c2[0].parent == self_id and c2[0].author == Author::Self);
    CHECK(b.open().size() == 1 and b.open()[0].author == Author::Human);
}

void test_self_mission_superseded()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    b.on_journal(s.wire(), 10);
    s.push(K::DriveMode, 20).op = "affordances_only";
    CHECK(b.on_journal(s.wire(), 30).size() == 1);     // the self mission opened
    push_mission(s, 40);
    const auto rs = b.on_journal(s.wire(), 50);
    const auto c = closed(rs);
    CHECK(c.size() == 1 and c[0].author == Author::Self and c[0].outcome == "superseded" and c[0].t1 == 40u);
    CHECK(b.open().size() == 1 and b.open()[0].author == Author::Human);
    // the human mission ends while still affordances_only: a fresh self mission opens
    auto& me = s.push(K::MissionEnd, 60); me.mission = "m1"; me.outcome = "succeeded";
    b.on_journal(s.wire(), 70);
    CHECK(b.open().size() == 1 and b.open()[0].author == Author::Self);
    s.push(K::DriveMode, 80).op = "mission_only";
    const auto c2 = closed(b.on_journal(s.wire(), 90));
    CHECK(c2.size() == 1 and c2[0].outcome == "superseded" and b.open().empty());
}

void test_controller_epoch_change()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    b.on_journal(s.wire(), 110);
    CHECK(b.open().size() == 1);
    Script s2;                       // the controller restarted: new epoch, serials from 1 again
    s2.epoch = 2000;
    s2.push(K::DriveMode, 200).op = "mission_only";
    push_mission(s2, 210, "m1");
    const auto rs = b.on_journal(s2.wire(), 220);
    const auto c = closed(rs);
    CHECK(c.size() == 1 and c[0].outcome == "lost" and c[0].t1 == 220u);
    CHECK(b.open().size() == 1 and b.open()[0].t0 == 210);   // ALL the new epoch's entries were processed
}

void test_journal_gap()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    b.on_journal(s.wire(), 110);
    auto& me = s.push(K::MissionEnd, 120); me.mission = "m1"; me.outcome = "succeeded";   // falls off the ring
    push_mission(s, 130, "m2");
    const auto c = closed(b.on_journal(s.wire_from(s.all.size()), 140));
    CHECK(c.size() == 1 and c[0].outcome == "lost");
    CHECK(b.open().size() == 1 and b.open()[0].t0 == 130);
}

void test_first_wire_only_adopts()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "1");
    CHECK(b.on_journal(s.wire(), 120).empty());
    CHECK(b.open().empty());
    push_step(s, K::StepEnd, 130, "1", 1, 0, "done");   // no open match: ignored (R10)
    s.push(K::DriveMode, 135).op = "mission_only";
    CHECK(b.on_journal(s.wire(), 140).empty());
}

// R13: priming adopts the latest DriveMode op (state only) — history says mission_only, so no self mission.
void test_priming_adopts_drive_mode()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    s.push(K::DriveMode, 10).op = "affordances_only";
    s.push(K::DriveMode, 20).op = "mission_only";
    CHECK(b.on_journal(s.wire(), 30).empty());
    CHECK(b.on_events(std::vector<Event>{{.dt_ms = 0, .kind = "born", .node = "chair_1"}}, 40).empty());
    CHECK(b.on_journal(s.wire(), 50).empty());
    CHECK(b.open().empty());
}

// No self mission before the first journal poll; an unknown room ("") never splits a self mission.
void test_unprimed_and_unknown_room()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    CHECK(b.on_events(std::vector<Event>{{.dt_ms = 0, .kind = "born", .node = "chair_1"}}, 10).empty());
    CHECK(b.open().empty());
    Script s;
    b.on_journal(s.wire(), 20);
    CHECK(b.on_journal(s.wire(), 30).size() == 1);      // R9: affordances_only, a self mission in room_1
    fs.room = "";
    CHECK(b.on_events(std::vector<Event>{{.dt_ms = 0, .kind = "current_moved", .node = ""}}, 40).empty());
    CHECK(b.open().size() == 1 and b.open()[0].room == "room_1");
}

// R13: a MissionEnd never stamps its outcome onto an open child; the child's own *End closes it.
void test_mission_end_leaves_children_open()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_claim(s, 110);
    b.on_journal(s.wire(), 120);
    auto& me = s.push(K::MissionEnd, 130); me.mission = "m1"; me.outcome = "succeeded";
    const auto c = closed(b.on_journal(s.wire(), 140));
    CHECK(c.size() == 1 and c[0].level == Level::Mission and c[0].outcome == "succeeded");
    CHECK(b.open().size() == 1 and b.open()[0].level == Level::Claim);
    push_claim_end(s, 150, "failure");
    const auto c2 = closed(b.on_journal(s.wire(), 160));
    CHECK(c2.size() == 1 and c2[0].level == Level::Claim and c2[0].outcome == "failure" and c2[0].t1 == 150u);
    CHECK(b.open().empty());
}

void test_idempotent_repoll()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "1");
    CHECK(b.on_journal(s.wire(), 120).size() == 2);
    CHECK(b.on_journal(s.wire(), 130).empty());
}

void test_delivered_absent_without_cov()
{
    FakeFactSource fs;
    fs.facts["door_3"] = door("door_3", 0.2, 0.04);    // cov_yy missing
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_claim(s, 110, std::nullopt);
    b.on_journal(s.wire(), 120);
    fs.facts["door_3"] = door("door_3", 0.9, 0.01, 0.01);
    push_claim_end(s, 130, "success");
    const auto c = closed(b.on_journal(s.wire(), 140));
    CHECK(c.size() == 1 and not c[0].delivered.has_value() and not c[0].promised.has_value());
    CHECK(c.size() == 1 and b.open().size() == 1 and c[0].parent == b.open()[0].id);   // no step: the human mission (R10)
}

void test_shutdown_closes_all()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "1");
    push_claim(s, 120);
    b.on_journal(s.wire(), 130);
    b.on_events(std::vector<Event>{{.dt_ms = 999, .kind = "born", .node = "chair_1"}}, 150);
    const auto c = b.shutdown(200);
    CHECK(c.size() == 3 and b.open().empty());
    for (const Record& r : c)
    {
        CHECK(r.outcome == "interrupted_by_em_shutdown" and r.t1 == 200u);
        CHECK(r.events.size() == 1 and r.events[0].dt_ms == std::int64_t(150 - r.t0) and r.events[0].node == "chair_1");
    }
    CHECK(c.size() == 3 and c[0].level == Level::Claim and c[2].level == Level::Mission);   // innermost first
}

void test_owner_lost_closes_all_lost()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "1");
    b.on_journal(s.wire(), 120);
    const auto c = b.owner_lost(200);
    CHECK(c.size() == 2 and b.open().empty());
    for (const Record& r : c) CHECK(r.outcome == "lost" and r.t1 == 200u);
}

// Finding 1: Command mode is affordance-driven too (controller_mission.cpp affordance_driven). After a `cross door_N`
// the controller stays in Command: the claims that follow belong to ONE self mission, not one zero-length self
// mission each.
void test_command_mode_is_self_driving()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    b.on_journal(s.wire(), 10);
    s.push(K::DriveMode, 20).op = "command";
    push_mission(s, 30, "c1");
    auto& st = s.push(K::StepStart, 31); st.mission = "c1"; st.step_id = "1"; st.op = "cross"; st.text = R"({"target":"door_3"})";
    auto& se = s.push(K::StepEnd, 40); se.mission = "c1"; se.step_id = "1"; se.op = "cross"; se.outcome = "satisfied";
    auto& me = s.push(K::MissionEnd, 41); me.mission = "c1"; me.outcome = "completed";
    b.on_journal(s.wire(), 50);
    CHECK(b.open().size() == 1 and b.open()[0].author == Author::Self);
    if (b.open().size() != 1) return;
    const std::string self_id = b.open()[0].id;
    push_claim(s, 60);
    push_claim_end(s, 70, "success");
    push_claim(s, 80);
    push_claim_end(s, 90, "success");
    const auto c = closed(b.on_journal(s.wire(), 100));
    CHECK(c.size() == 2);
    for (const Record& r : c) CHECK(r.level == Level::Claim and r.parent == self_id);
    CHECK(b.open().size() == 1 and b.open()[0].id == self_id);
}

// Finding 6: every record names the controller's mission and the journal serials that opened / closed it.
void test_mission_key_and_serials()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);                                   // serial 2
    push_step(s, K::StepStart, 110, "1");                   // serial 3
    auto& cs = s.push(K::ClaimStart, 120);                  // serial 4
    cs.mission = "m1"; cs.op = "cross_door"; cs.text = R"({"target":"door_3"})";
    auto& ce = s.push(K::ClaimEnd, 130);                    // serial 5
    ce.mission = "m1"; ce.op = "cross_door"; ce.outcome = "success";
    push_step(s, K::StepEnd, 140, "1", 1, 0, "done");       // serial 6
    const auto c = closed(b.on_journal(s.wire(), 150));
    const Record* st = find(c, Level::Step, "cross");
    const Record* cl = find(c, Level::Claim, "cross_door");
    CHECK(st and cl);
    if (not (st and cl)) return;
    CHECK(st->mission_key == "m1" and st->serial_start == 3u and st->serial_end == 6u and st->step_id == "1");
    CHECK(cl->mission_key == "m1" and cl->serial_start == 4u and cl->serial_end == 5u);
    CHECK(b.open().size() == 1);
    if (b.open().size() != 1) return;
    CHECK(b.open()[0].mission_key == "m1" and b.open()[0].serial_start == 2u and not b.open()[0].serial_end);
    const auto lost = b.owner_lost(200);                    // EM-closed: no journal serial ends it
    CHECK(lost.size() == 1 and lost[0].mission_key == "m1" and lost[0].serial_start == 2u and not lost[0].serial_end);
}

// Finding 7: affordance offered / withdrawn go to the innermost open record only; structure events go to all.
void test_affordance_events_innermost_only()
{
    FakeFactSource fs;
    Builder b(fs, ids());
    Script s;
    prime_mission_only(b, s, 10);
    push_mission(s, 100);
    push_step(s, K::StepStart, 110, "1");
    push_claim(s, 120);
    b.on_journal(s.wire(), 130);
    b.on_events(std::vector<Event>{{.dt_ms = 0, .kind = "offered", .node = "aff_1"},
                                   {.dt_ms = 0, .kind = "withdrawn", .node = "aff_2"},
                                   {.dt_ms = 0, .kind = "born", .node = "chair_1"}}, 140);
    const auto open = b.open();
    CHECK(open.size() == 3);
    for (const Record& r : open)
    {
        if (r.level == Level::Claim) CHECK(r.events.size() == 3);
        else CHECK(r.events.size() == 1 and r.events[0].kind == "born");
    }
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "es_ES.UTF-8") or std::setlocale(LC_ALL, "");
    test_mission_step_claim_nesting();
    test_attempts_are_records();
    test_composite_nesting();
    test_preempted_claim();
    test_self_mission_split_on_room();
    test_self_split_deferred_while_claim_open();
    test_supersede_leaves_self_claim_open();
    test_self_mission_superseded();
    test_controller_epoch_change();
    test_journal_gap();
    test_first_wire_only_adopts();
    test_priming_adopts_drive_mode();
    test_unprimed_and_unknown_room();
    test_mission_end_leaves_children_open();
    test_idempotent_repoll();
    test_delivered_absent_without_cov();
    test_shutdown_closes_all();
    test_owner_lost_closes_all_lost();
    test_command_mode_is_self_driving();
    test_mission_key_and_serials();
    test_affordance_events_innermost_only();
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
