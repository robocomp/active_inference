// Offline test of the identity link in the repertory (spec 2026-11-15-changed-object, owner's choice B): a live object
// believed to be a remembered one that moved; the owner's "no" cuts it and marks the memory gone from that room.
#include "../src/human.h"
#include "../src/recall_select.h"
#include "../src/repertory.h"
#include "../src/repertory_io.h"

#include <algorithm>
#include <clocale>
#include <filesystem>
#include <fstream>
#include <print>
#include <unistd.h>

using namespace ltom;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

constexpr std::uint64_t kMem = (1ull << 40) | 50;    // chair_50, remembered in room 2
constexpr std::uint64_t kLive = (1ull << 40) | 57;   // chair_57, born new this run
static const std::vector<std::string> kChairs{"chair"};

static Keyframe kf(std::int64_t t, std::int64_t room, double x, double y)
{
    Keyframe k; k.t_ms = t; k.room_key = room; k.pose.x = x; k.pose.y = y; k.pose.vx = k.pose.vy = 0.01; k.pose.vyaw = 0.1;
    return k;
}
// chair_50 died at t=100 in room 2 at (-2.57, -0.61), magenta.
static Repertory with_memory()
{
    Repertory r(1000);
    Entry m; m.uid = kMem; m.cls = "chair"; m.name = "chair_50"; m.color_name = "magenta"; m.color_prob = 0.74f;
    m.lives.push_back(Life{.born_ms = 10, .first_seen_ms = 10, .last_seen_ms = 100, .end_ms = 100, .end = End::death});
    m.keyframes = {kf(50, 2, -2.57, -0.61)};
    r.insert_loaded(m);
    return r;
}
static Sighting live(float same_prob, float left_prob)
{
    Sighting s; s.uid = kLive; s.cls = "chair"; s.name = "chair_57"; s.room_key = 2; s.born_ms = 1000;
    s.pose.x = 1.2; s.pose.y = 1.5; s.pose.vx = s.pose.vy = 0.01; s.pose.vyaw = 0.1;
    s.link = LinkObs{kMem, same_prob, left_prob};
    return s;
}
static void see(Repertory& r, const Sighting& s, std::int64_t now)
{
    const std::vector<Sighting> v{s};
    r.observe(v, {1}, {s.uid}, now);
}

static void record_link_keeps_old_keyframe()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.2f), 1000);
    const Entry& m = r.entries().at(kMem);
    CHECK(m.continued_by == kLive);
    CHECK(r.entries().at(kLive).continues == kMem);
    CHECK(m.moves.size() == 1);
    CHECK(m.moves.back().from.pose.x == -2.57 and m.moves.back().from.room_key == 2);
    CHECK(m.moves.back().to.pose.x == 1.2);
    CHECK(not m.moves.back().confirmed);
    CHECK(m.keyframes.size() == 1);   // the remembered pose stays in the history
    see(r, live(0.95f, 0.8f), 1100);
    CHECK(r.entries().at(kMem).moves.size() == 1);   // one link, one move: not re-recorded each sighting
}
static void below_map_no_link()
{
    auto r = with_memory();
    see(r, live(0.4f, 0.2f), 1000);
    CHECK(not r.entries().at(kMem).continued_by.has_value());
}
static void link_dropped_below_map_cuts_without_gone()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.2f), 1000);
    see(r, live(0.3f, 0.2f), 1100);
    const Entry& m = r.entries().at(kMem);
    CHECK(not m.continued_by.has_value());
    CHECK(not r.entries().at(kLive).continues.has_value());
    CHECK(m.moves.empty());
    CHECK(m.gone.empty());
}
static void confirmed_link_survives_a_low_sighting()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.2f), 1000);
    r.confirm_link(kMem);
    CHECK(r.entries().at(kMem).moves.back().confirmed);
    see(r, live(0.3f, 0.2f), 1100);
    CHECK(r.entries().at(kMem).continued_by == kLive);   // the owner said yes: evidence no longer undoes it
}
static void gone_from_room_not_recalled()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    r.cut_link(kMem, 1200, "no, I didn't");
    const Entry& m = r.entries().at(kMem);
    CHECK(not m.continued_by.has_value());
    CHECK(m.gone.size() == 1 and m.gone.back().room_key == 2 and m.gone.back().words == "no, I didn't");
    CHECK(m.moves.empty());
    CHECK(select_recalls(r, 2, {kLive}, kChairs).empty());
    CHECK(select_recalls(r, 2, {}, kChairs).empty());
}
static void gone_memory_seen_again_is_recalled_there()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    r.cut_link(kMem, 1200, "no");
    Sighting back; back.uid = kMem; back.cls = "chair"; back.name = "chair_50"; back.room_key = 2; back.born_ms = 1300;
    back.pose.x = -2.5; back.pose.y = -0.6; back.pose.vx = back.pose.vy = 0.01; back.pose.vyaw = 0.1;
    see(r, back, 1300);
    Repertory& rr = r;
    const std::vector<Sighting> none;
    rr.observe(none, {1}, {}, 1400);
    rr.observe(none, {1}, {}, 3000);   // past the grace: dead again
    CHECK(select_recalls(rr, 2, {}, kChairs).size() == 1);   // a later sighting there outdates the "gone" record
}
static void continued_offered_only_while_live_present()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    auto v = select_recalls(r, 2, {kLive}, kChairs);
    CHECK(v.size() == 1 and v[0].uid == kMem and v[0].same_as_uid == kLive);
    CHECK(v[0].same_as_prob > 0.85f and v[0].same_as_prob < 0.95f);
    CHECK(v[0].color_name == "magenta");
    r.confirm_link(kMem);
    CHECK(select_recalls(r, 2, {kLive}, kChairs).at(0).same_as_prob == 1.f);
    CHECK(select_recalls(r, 2, {}, kChairs).empty());
}
static void repertory_io_roundtrips_links_and_questions()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    r.confirm_link(kMem);
    Entry other; other.uid = (1ull << 40) | 44; other.cls = "chair"; other.name = "chair_44";
    other.lives.push_back(Life{.end = End::death});
    other.keyframes = {kf(40, 2, -2.5, -0.6)};
    other.gone.push_back(GoneFromRoom{900, 2, "no"});
    r.insert_loaded(other);
    const fs::path f = fs::temp_directory_path() / ("ltom_links_" + std::to_string(::getpid()) + ".json");
    CHECK(save_repertory(f, r));
    Repertory got(1000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    const Entry& m = got.entries().at(kMem);
    CHECK(m.continued_by == kLive);
    CHECK(got.entries().at(kLive).continues == kMem);
    CHECK(m.moves.size() == 1 and m.moves[0].confirmed and m.moves[0].live_uid == kLive);
    CHECK(m.moves[0].from.pose.x == -2.57 and m.moves[0].to.pose.x == 1.2 and m.moves[0].t_ms == 1000);
    const Entry& o = got.entries().at((1ull << 40) | 44);
    CHECK(o.gone.size() == 1 and o.gone[0].t_ms == 900 and o.gone[0].room_key == 2 and o.gone[0].words == "no");
    fs::remove(f);
}
static void older_file_loads_without_links()
{
    const fs::path f = fs::temp_directory_path() / ("ltom_links_old_" + std::to_string(::getpid()) + ".json");
    {
        std::ofstream o(f);
        o << R"({"version": 1, "entries": [{"uid": "1099511627826", "cls": "chair", "name": "chair_50", "w": 0.5, "d": 0.5, "h": 1.0,
              "n_sightings": 3, "lives": [{"born_ms": 1, "first_seen_ms": 1, "last_seen_ms": 2, "end_ms": 2, "end": "death"}],
              "keyframes": [{"t_ms": 1, "room_key": 2, "x": 0.5, "y": 0.5, "yaw": 0, "vx": 0.01, "vy": 0.01, "vyaw": 0.1}]}]})";
    }
    Repertory got(1000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    const Entry& e = got.entries().at(1099511627826ull);
    CHECK(not e.continued_by.has_value() and not e.continues.has_value() and e.moves.empty() and e.gone.empty());
    fs::remove(f);
}

// ── Task 4: questions, the owner's answers, beliefs ──────────────────────────────────────────────────────────
static Claim moved(const char* v, std::uint64_t q, std::uint64_t stamp, const char* words = "yes, I moved it")
{
    return Claim{ClaimKind::moved, std::string(v) + ":" + std::to_string(q), kMem, words, stamp};
}
static const Belief* belief(const HumanBook& b, std::string_view key)
{
    for (const auto& x : b.human().beliefs) if (x.key == key) return &x;
    return nullptr;
}
static int count(const Belief* b, bool yes)
{
    if (b == nullptr) return 0;
    return static_cast<int>(std::ranges::count_if(b->records, [&](const BeliefRecord& r) { return r.yes == yes; }));
}
static const std::string kGets50 = "gets_moved:" + std::to_string(kMem);

static void no_question_before_old_spot_seen_empty()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.3f), 1000);
    CHECK(r.questions().empty());
    see(r, live(0.9f, 0.8f), 1100);
    const auto q = r.questions();
    CHECK(q.size() == 1);
    if (q.size() == 1)
    {
        const Question& x = q.begin()->second;
        CHECK(x.memory_uid == kMem and x.live_uid == kLive and x.room_key == 2);
        CHECK(x.old_kf.pose.x == -2.57 and x.new_kf.pose.x == 1.2 and not x.again);
        CHECK(x.stamp == 1100);
    }
    see(r, live(0.9f, 0.9f), 1200);
    CHECK(r.questions().size() == 1);   // one question per change
}
static void question_names_link_winner()
{
    auto r = with_memory();
    Entry dup; dup.uid = (1ull << 40) | 44; dup.cls = "chair"; dup.name = "chair_44";
    dup.lives.push_back(Life{.end = End::death}); dup.keyframes = {kf(40, 2, -2.57, -0.61)};
    r.insert_loaded(dup);
    see(r, live(0.9f, 0.8f), 1000);    // the chair agent's MAP named chair_50, not the colourless duplicate
    const auto q = r.questions();
    CHECK(q.size() == 1 and q.begin()->second.memory_uid == kMem);
}
static void question_closes_when_live_dies()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    const std::vector<Sighting> none;
    r.observe(none, {1}, {}, 1100);
    r.observe(none, {1}, {}, 2500);   // past the grace: chair_57 dead
    CHECK(r.questions().empty());
}
static void moved_yes_confirms_and_learns()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    CHECK(b.apply(moved("yes", 1000, 7), r, 1500).text == "recorded");
    CHECK(r.questions().empty());
    CHECK(r.entries().at(kMem).moves.back().confirmed);
    CHECK(count(belief(b, "moves_things"), true) == 1);
    CHECK(count(belief(b, kGets50), true) == 1);
    const Belief* m = belief(b, "moves_things");
    CHECK(m != nullptr and m->records.back().said.words == "yes, I moved it" and m->records.back().said.t_ms == 1500);
    CHECK(m != nullptr and m->records.back().object_uid == kMem);
    see(r, live(0.95f, 0.9f), 1600);
    CHECK(r.questions().empty());   // answered: not asked again
}
static void moved_no_cuts_and_learns_no()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    CHECK(b.apply(moved("no", 1000, 7, "no, I didn't"), r, 1500).text == "recorded");
    CHECK(not r.entries().at(kMem).continued_by.has_value());
    CHECK(r.entries().at(kMem).gone.size() == 1 and r.entries().at(kMem).gone[0].words == "no, I didn't");
    CHECK(count(belief(b, "moves_things"), false) == 1);
    CHECK(belief(b, kGets50) == nullptr);
    CHECK(r.questions().empty());
}
static void moved_unknown_closes_without_learning()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    CHECK(b.apply(moved("unknown", 1000, 7, "no sé"), r, 1500).text == "recorded");
    CHECK(r.questions().empty());
    CHECK(r.entries().at(kMem).continued_by == kLive);
    CHECK(not r.entries().at(kMem).moves.back().confirmed);
    CHECK(b.human().beliefs.empty());
    see(r, live(0.9f, 0.9f), 1600);
    CHECK(r.questions().empty());   // "I don't know" is an answer: the same move is not asked again
}
static void answer_to_closed_question_rejected()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    CHECK(b.apply(moved("yes", 999, 7), r, 1500).text.starts_with("rejected: "));   // stale stamp
    CHECK(b.human().beliefs.empty());
    CHECK(b.apply(moved("yes", 1000, 8), r, 1500).text == "recorded");
    CHECK(b.apply(moved("no", 1000, 9), r, 1600).text.starts_with("rejected: "));   // already answered
    CHECK(count(belief(b, "moves_things"), false) == 0);
    CHECK(b.apply(Claim{ClaimKind::moved, "maybe:1000", kMem, "", 10}, r, 1600).text.starts_with("rejected: "));
}
static void second_move_question_is_again()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    b.apply(moved("yes", 1000, 7), r, 1500);
    // Next visit: chair_57 is the memory now (it continues chair_50), and a new live chair_63 is linked to it.
    const std::vector<Sighting> none;
    r.observe(none, {1}, {}, 2000);
    r.observe(none, {1}, {}, 3500);
    constexpr std::uint64_t k63 = (1ull << 40) | 63;
    Sighting s = live(0.9f, 0.8f); s.uid = k63; s.name = "chair_63"; s.pose.x = 2.0; s.link = LinkObs{kLive, 0.9f, 0.8f};
    see(r, s, 4000);
    const auto q = r.questions();
    CHECK(q.size() == 1 and q.begin()->second.memory_uid == kLive and q.begin()->second.again);
}
static void forget_drops_records_keeps_note()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    b.apply(moved("yes", 1000, 7), r, 1500);
    CHECK(b.apply(Claim{ClaimKind::forget, "moves_things", 0, "forget that", 8}, r, 1700).text == "recorded");
    const Belief* m = belief(b, "moves_things");
    CHECK(m != nullptr and m->records.empty() and m->forgotten.size() == 1 and m->forgotten[0].words == "forget that");
    CHECK(count(belief(b, kGets50), true) == 1);   // only that belief
    CHECK(b.apply(Claim{ClaimKind::forget, "no_such_belief", 0, "forget it", 9}, r, 1800).text.starts_with("rejected: "));
}
static void human_io_roundtrips_beliefs()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    b.apply(moved("yes", 1000, 7), r, 1500);
    b.apply(Claim{ClaimKind::forget, kGets50, 0, "forget that", 8}, r, 1700);
    const fs::path f = fs::temp_directory_path() / ("ltom_beliefs_" + std::to_string(::getpid()) + ".json");
    CHECK(save_repertory(f, r, &b));
    Repertory gr(1000);
    HumanBook gb;
    CHECK(load_repertory(f, gr, &gb) == LoadResult::loaded);
    const Belief* m = belief(gb, "moves_things");
    CHECK(m != nullptr and m->records.size() == 1 and m->records[0].yes and m->records[0].object_uid == kMem);
    CHECK(m != nullptr and m->records[0].said.words == "yes, I moved it" and m->records[0].said.t_ms == 1500);
    const Belief* g = belief(gb, kGets50);
    CHECK(g != nullptr and g->records.empty() and g->forgotten.size() == 1 and g->forgotten[0].t_ms == 1700);
    fs::remove(f);
}
static void open_question_survives_restart()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    const fs::path f = fs::temp_directory_path() / ("ltom_q_" + std::to_string(::getpid()) + ".json");
    CHECK(save_repertory(f, r));
    Repertory got(1000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    CHECK(got.questions().size() == 1 and got.questions().begin()->first == 1000);
    fs::remove(f);
}

// ── Final-review fixes ────────────────────────────────────────────────────────────────────────────────────────
static void link_switch_cuts_previous()   // F2: a live chair whose MAP moves to another memory leaves the first one
{
    auto r = with_memory();
    constexpr std::uint64_t k51 = (1ull << 40) | 51;
    Entry m2; m2.uid = k51; m2.cls = "chair"; m2.name = "chair_51";
    m2.lives.push_back(Life{.end = End::death}); m2.keyframes = {kf(60, 2, 0.5, 0.5)};
    r.insert_loaded(m2);
    see(r, live(0.9f, 0.8f), 1000);                    // linked to chair_50, question open
    Sighting s = live(0.9f, 0.8f); s.link = LinkObs{k51, 0.9f, 0.8f};
    see(r, s, 1100);                                   // now believed to be chair_51
    CHECK(not r.entries().at(kMem).continued_by.has_value());
    CHECK(r.entries().at(kMem).moves.empty());
    CHECK(r.entries().at(k51).continued_by == kLive and r.entries().at(kLive).continues == k51);
    const auto q = r.questions();
    CHECK(q.size() == 1 and q.begin()->second.memory_uid == k51);
}
static void no_relink_after_no()           // F5: the chair node still carries the old link for a cycle after "no"
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    CHECK(b.apply(moved("no", 1000, 7, "no"), r, 1500).text == "recorded");
    see(r, live(0.9f, 0.8f), 1600);                    // stale same_as on the node
    CHECK(not r.entries().at(kMem).continued_by.has_value());
    CHECK(r.questions().empty());
}
static void owner_exit_keeps_question_askable()   // F6: the chair agent restarts; the move is asked when it is back
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    CHECK(r.questions().size() == 1);
    const std::vector<Sighting> none;
    r.observe(none, {}, {}, 1100);                     // chair_concept (owner 1) gone: owner_exit
    CHECK(r.questions().empty());
    see(r, live(0.9f, 0.8f), 2000);                    // back, the same chair_57 with its link
    CHECK(r.questions().size() == 1);
}

// Live run moved_chair_4 (2026-11-16): after "no" for chair_50 the chair agent's MAP went to chair_44, a colourless
// duplicate at the same old spot, and the robot asked again about "the chair in room 2". The owner is the only one
// who moves things (spec), so "no, I didn't move it" says the LIVE chair is new: no memory may continue it.
static void no_settles_live_as_new()
{
    auto r = with_memory();
    constexpr std::uint64_t k44 = (1ull << 40) | 44;
    Entry dup; dup.uid = k44; dup.cls = "chair"; dup.name = "chair_44";
    dup.lives.push_back(Life{.end = End::death}); dup.keyframes = {kf(40, 2, -2.58, -0.62)};
    r.insert_loaded(dup);
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    CHECK(b.apply(moved("no", 1000, 7, "no"), r, 1500).text == "recorded");
    Sighting s = live(0.67f, 0.96f); s.link = LinkObs{k44, 0.67f, 0.96f};
    see(r, s, 1600);
    CHECK(not r.entries().at(k44).continued_by.has_value());
    CHECK(not r.entries().at(kLive).continues.has_value());
    CHECK(r.questions().empty());
}
static void said_new_survives_restart()
{
    auto r = with_memory();
    see(r, live(0.9f, 0.8f), 1000);
    HumanBook b;
    CHECK(b.apply(moved("no", 1000, 7, "no"), r, 1500).text == "recorded");
    const fs::path f = fs::temp_directory_path() / std::format("ltom_said_new_{}.json", ::getpid());
    CHECK(save_repertory(f, r));
    Repertory back(1000);
    CHECK(load_repertory(f, back) == LoadResult::loaded);
    fs::remove(f);
    CHECK(back.entries().at(kLive).said_new);
}

int main()
{
    std::setlocale(LC_ALL, "");   // the agents run under es_ES (CLAUDE.md)
    no_settles_live_as_new();
    said_new_survives_restart();
    record_link_keeps_old_keyframe();
    below_map_no_link();
    link_dropped_below_map_cuts_without_gone();
    confirmed_link_survives_a_low_sighting();
    gone_from_room_not_recalled();
    gone_memory_seen_again_is_recalled_there();
    continued_offered_only_while_live_present();
    repertory_io_roundtrips_links_and_questions();
    older_file_loads_without_links();
    no_question_before_old_spot_seen_empty();
    question_names_link_winner();
    question_closes_when_live_dies();
    moved_yes_confirms_and_learns();
    moved_no_cuts_and_learns_no();
    moved_unknown_closes_without_learning();
    answer_to_closed_question_rejected();
    second_move_question_is_again();
    forget_drops_records_keeps_note();
    human_io_roundtrips_beliefs();
    open_question_survives_restart();
    link_switch_cuts_previous();
    no_relink_after_no();
    owner_exit_keeps_question_askable();
    if (failures == 0) std::println("test_repertory_links: all passed");
    return failures == 0 ? 0 : 1;
}
