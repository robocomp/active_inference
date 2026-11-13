// Offline test of src/human.h (pure model of the human at the chat).
#include "../src/human.h"
#include "../src/repertory.h"

#include <cstdio>
#include <print>

using namespace ltom;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

constexpr std::uint64_t kA = (20ull << 40) | 34;
constexpr std::uint64_t kB = (20ull << 40) | 35;
constexpr std::uint64_t kUnknown = (20ull << 40) | 999;

static Repertory repertory()
{
    Repertory r(3000);
    Entry a; a.uid = kA; a.cls = "chair"; a.name = "chair_34"; r.insert_loaded(a);
    Entry b; b.uid = kB; b.cls = "chair"; b.name = "chair_35"; r.insert_loaded(b);
    return r;
}
static Claim owns(std::uint64_t uid, std::uint64_t stamp) { return Claim{ClaimKind::owns, "", uid, "that's my chair", stamp}; }
static Claim disowns(std::uint64_t uid, std::uint64_t stamp) { return Claim{ClaimKind::disowns, "", uid, "that is not my chair", stamp}; }
static bool rejected(const Answer& a) { return a.text.starts_with("rejected:"); }

static void test_name_is_kept_with_history()
{
    const auto r = repertory();
    HumanBook b;
    CHECK(not b.name().has_value());
    CHECK(b.apply(Claim{ClaimKind::name, "Pablo", 0, "my name is Pablo", 1}, r, 1000).text == "recorded");
    CHECK(b.apply(Claim{ClaimKind::name, "Paul", 0, "call me Paul", 2}, r, 2000).text == "recorded");
    CHECK(b.name() == std::optional<std::string>("Paul"));
    CHECK(b.human().names.size() == 2);
    CHECK(b.human().names[0].value == "Pablo" and b.human().names[0].said.words == "my name is Pablo");
    CHECK(b.human().names[1].said.t_ms == 2000);
    CHECK(b.human().id == "human_1" and b.human().role == "developer");
}
static void test_empty_name_rejected()
{
    const auto r = repertory();
    HumanBook b;
    CHECK(rejected(b.apply(Claim{ClaimKind::name, "", 0, "my name is", 1}, r, 1000)));
    CHECK(b.human().names.empty());
}
static void test_owns_needs_a_memory_entry()
{
    const auto r = repertory();
    HumanBook b;
    const auto a = b.apply(owns(kUnknown, 1), r, 1000);
    CHECK(rejected(a));
    CHECK(a.stamp == 1);
    CHECK(b.owned().empty());
}
static void test_owns_records_once()
{
    const auto r = repertory();
    HumanBook b;
    CHECK(b.apply(owns(kA, 1), r, 1000).text == "recorded");
    CHECK(b.apply(owns(kA, 2), r, 2000).text == "recorded");
    CHECK(b.owned().size() == 1);
    CHECK(b.human().things.size() == 1);
    CHECK(b.human().things[0].cls == "chair" and b.human().things[0].name_when_said == "chair_34");
    CHECK(b.human().things[0].said.words == "that's my chair" and b.human().things[0].said.t_ms == 1000);
}
static void test_disown_keeps_history()
{
    const auto r = repertory();
    HumanBook b;
    b.apply(owns(kA, 1), r, 1000);
    CHECK(b.apply(disowns(kA, 2), r, 2000).text == "recorded");
    CHECK(b.owned().empty());
    CHECK(b.human().things.size() == 1);
    CHECK(b.human().things[0].retracted.has_value());
    CHECK(b.human().things[0].retracted->t_ms == 2000);
}
static void test_disown_of_not_owned_rejected()
{
    const auto r = repertory();
    HumanBook b;
    CHECK(rejected(b.apply(disowns(kA, 1), r, 1000)));
    CHECK(b.human().things.empty());
}
static void test_reclaim_after_disown_is_a_new_entry()
{
    const auto r = repertory();
    HumanBook b;
    b.apply(owns(kA, 1), r, 1000);
    b.apply(disowns(kA, 2), r, 2000);
    b.apply(owns(kA, 3), r, 3000);
    CHECK(b.human().things.size() == 2);
    CHECK(b.owned().size() == 1);
    CHECK(b.owned()[0] == &b.human().things[1]);
}
static void test_same_stamp_is_applied_once()
{
    const auto r = repertory();
    HumanBook b;
    const auto a1 = b.apply(owns(kA, 7), r, 1000);
    CHECK(b.dirty());
    b.clear_dirty();
    const auto a2 = b.apply(owns(kA, 7), r, 1100);
    CHECK(not b.dirty());
    CHECK(a2.text == a1.text and a2.stamp == 7);
    // a rejected claim repeated with its stamp keeps its answer
    const auto a3 = b.apply(owns(kUnknown, 8), r, 1200);
    const auto a4 = b.apply(owns(kUnknown, 8), r, 1300);
    CHECK(rejected(a3) and a4.text == a3.text);
}
static void test_two_owned_chairs()
{
    const auto r = repertory();
    HumanBook b;
    b.apply(owns(kA, 1), r, 1000);
    b.apply(owns(kB, 2), r, 2000);
    CHECK(b.owned().size() == 2);
    CHECK(b.owned()[0]->object_uid == kA and b.owned()[1]->object_uid == kB);
}
static void test_parse_claim_kind()
{
    CHECK(parse_claim_kind("name") == ClaimKind::name);
    CHECK(parse_claim_kind("owns") == ClaimKind::owns);
    CHECK(parse_claim_kind("disowns") == ClaimKind::disowns);
    CHECK(not parse_claim_kind("own").has_value());
}
static void test_set_loads_without_dirty()
{
    HumanBook b;
    Human h; h.names.push_back(NameRecord{"Pablo", Said{5, "my name is Pablo"}});
    b.set(h);
    CHECK(not b.dirty());
    CHECK(b.name() == std::optional<std::string>("Pablo"));
}

static void test_unsaved_is_not_recorded()   // I4: "I will remember" never said when the file was not written
{
    CHECK(answer_after_save(Answer{5, "recorded"}, true).text == "recorded");
    const auto a = answer_after_save(Answer{5, "recorded"}, false);
    CHECK(a.stamp == 5 and rejected(a) and a.text.contains("file"));
    CHECK(answer_after_save(Answer{6, "rejected: it was not yours"}, false).text == "rejected: it was not yours");
}

int main()
{
    test_name_is_kept_with_history();
    test_empty_name_rejected();
    test_owns_needs_a_memory_entry();
    test_owns_records_once();
    test_disown_keeps_history();
    test_disown_of_not_owned_rejected();
    test_reclaim_after_disown_is_a_new_entry();
    test_same_stamp_is_applied_once();
    test_two_owned_chairs();
    test_parse_claim_kind();
    test_set_loads_without_dirty();
    test_unsaved_is_not_recorded();
    if (failures == 0) std::println("test_human: all passed");
    return failures == 0 ? 0 : 1;
}
