// human_refs.h: what "that" and "my chair" refer to (spec 2026-11-11-human-model). Build+run: tests/run_tests.sh
#include "../src/human_refs.h"
#include <cassert>
#include <print>

using namespace rc::chat;

namespace
{
ObjectFact live(std::string name, std::uint64_t uid, double bearing, std::string sub = "chair")
{
    ObjectFact o{.id = uid, .name = std::move(name), .subtype = std::move(sub), .room = "room_1", .in_current_room = true};
    o.dist_m = 2.0; o.bearing_deg = bearing; o.uid = uid;
    return o;
}
RememberedObject remembered(std::string name, std::uint64_t uid, std::string sub = "chair")
{
    RememberedObject r{.name = std::move(name), .subtype = std::move(sub), .room = "room_1"};
    r.uid = uid;
    return r;
}
constexpr double kHalfFov = 55.0;
}   // namespace

int main()
{
    int checks = 0;
    const std::vector<std::string> none;
    {   // a centred chair outranks one at 40 deg
        const auto c = that_candidates({live("chair_1", 1, 40), live("chair_2", 2, 2)}, none, std::nullopt, kHalfFov);
        assert(c.size() == 2 and c[0].name == "chair_2" and c[0].uid == 2); ++checks;
        assert(c[0].weight > c[1].weight); ++checks;
    }
    {   // a chair named last turn outranks a more centred one it is not
        const std::vector<std::string> named{"chair_1"};
        const auto c = that_candidates({live("chair_1", 1, 30), live("chair_2", 2, 5)}, named, std::nullopt, kHalfFov);
        assert(c.size() == 2 and c[0].name == "chair_1"); ++checks;
    }
    {   // behind the robot: not a candidate unless named
        const auto c = that_candidates({live("chair_1", 1, 170), live("chair_2", 2, 10)}, none, std::nullopt, kHalfFov);
        assert(c.size() == 1 and c[0].name == "chair_2"); ++checks;
        const std::vector<std::string> named{"chair_1"};
        const auto d = that_candidates({live("chair_1", 1, 170), live("chair_2", 2, 10)}, named, std::nullopt, kHalfFov);
        assert(d.size() == 2); ++checks;
    }
    {   // the last approach target counts even when out of view
        const auto c = that_candidates({live("chair_1", 1, 170)}, none, std::string("chair_1"), kHalfFov);
        assert(c.size() == 1 and c[0].name == "chair_1"); ++checks;
    }
    {   // the person at the chat is never "that"
        const auto c = that_candidates({live("human_1", 9, 0, "person")}, none, std::nullopt, kHalfFov);
        assert(c.empty()); ++checks;
    }
    {   // my chair
        HumanFacts h{.name = "Pablo", .role = "developer", .owned_uids = {1, 2}};
        const std::vector<ObjectFact> l{live("chair_1", 1, 10), live("table_1", 3, 0, "table")};
        const std::vector<RememberedObject> m{remembered("chair_2", 2), remembered("chair_5", 5)};
        const auto two = my_candidates("chair", h, l, m);
        assert(two.size() == 2 and two[0].name == "chair_1" and two[1].name == "chair_2"); ++checks;   // live first
        assert(my_candidates("chairs", h, l, m).size() == 2); ++checks;                                // class normalised
        HumanFacts one{.role = "developer", .owned_uids = {2}};
        const auto r = my_candidates("chair", one, l, m);
        assert(r.size() == 1 and r[0].name == "chair_2" and r[0].uid == 2); ++checks;                 // only remembered
        HumanFacts nobody{.role = "developer"};
        assert(my_candidates("chair", nobody, l, m).empty()); ++checks;
        assert(my_candidates("table", h, l, m).empty()); ++checks;
        assert(my_candidates("", h, l, m).size() == 2); ++checks;                                      // "mine": any class
    }
    {   // the words that ask for it
        assert(my_class("my chair") == std::optional<std::string>("chair")); ++checks;
        assert(my_class("  My Chairs?") == std::optional<std::string>("chair")); ++checks;
        assert(my_class("mine") == std::optional<std::string>("")); ++checks;
        assert(not my_class("chair_34").has_value()); ++checks;
        assert(not my_class("the magenta chair").has_value()); ++checks;
    }
    {   // "go to my chair" as a whole message: the words after "go to"
        assert(go_to_mine("go to my chair") == std::optional<std::string>("my chair")); ++checks;
        assert(go_to_mine("Go back to my chair.") == std::optional<std::string>("my chair")); ++checks;
        assert(go_to_mine("ve a mi silla") == std::optional<std::string>("my chair")); ++checks;
        assert(not go_to_mine("go to chair_34").has_value()); ++checks;
        assert(not go_to_mine("go to my chair and then to the door").has_value()); ++checks;
    }
    {   // I1: an owned object is followed by its UID, never by a re-issued name
        constexpr std::uint64_t A = 34, B = 99;
        HumanFacts h{.name = "Pablo", .role = "developer", .owned_uids = {A}};
        const std::vector<ObjectFact> l{live("chair_34", B, 5)};                 // a DIFFERENT chair now named chair_34
        const std::vector<RememberedObject> m{remembered("chair_34", A)};
        const auto r = resolve_one_mine("chair", h, l, m);
        assert(r.has_value() and r->uid == A and not r->live); ++checks;
        const auto t = mine_live_target(*r, l);
        assert(not t.has_value() and t.error().contains("another")); ++checks;  // never drives to the live namesake
        assert(not mine_goto_name(*r, l).has_value()); ++checks;               // nor plans to its name
        bool fm = false;
        const auto w = answer_where_mine("chair", h, l, Remembered{.objects = m, .now_ms = 1}, "room_1", &fm);
        assert(w.contains("Remembered") and not w.contains("2.0 m") and fm); ++checks;
        // no namesake: the remembered one is a goto, the live one in this room a target by its live name
        assert(mine_goto_name(*r, {}) == std::optional<std::string>("chair_34")); ++checks;
        HumanFacts hb{.role = "developer", .owned_uids = {B}};
        const auto rb = resolve_one_mine("chair", hb, l, m);
        assert(rb.has_value() and rb->live); ++checks;
        { const auto tb = mine_live_target(*rb, l); assert(tb.has_value() and *tb == "chair_34"); ++checks; }
        // two owned: asks which; none: says so
        HumanFacts two{.role = "developer", .owned_uids = {A, B}};
        const auto r2 = resolve_one_mine("chair", two, l, m);
        assert(not r2.has_value() and r2.error().contains("which")); ++checks;
        assert(not resolve_one_mine("table", two, l, m).has_value()); ++checks;
        // live but in another room: a mission, not "only remembered"
        ObjectFact far = live("chair_7", 7, 0); far.in_current_room = false; far.room = "room_2";
        HumanFacts h7{.role = "developer", .owned_uids = {7}};
        const auto r7 = resolve_one_mine("chair", h7, {far}, {});
        assert(r7.has_value() and not mine_live_target(*r7, {far}).has_value()); ++checks;
    }
    std::println("test_human_refs: {} checks passed", checks);
    return 0;
}
