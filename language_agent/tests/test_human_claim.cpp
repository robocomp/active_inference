// human_claim.h: the claim card's words, which card takes a yes, the answer wait. Build+run: tests/run_tests.sh
#include "../src/human_claim.h"
#include <cassert>
#include <print>

using namespace rc::chat;

int main()
{
    int checks = 0;
    ObjectFact c34{.name = "chair_34", .subtype = "chair", .room = "room_1", .in_current_room = true, .color_name = "magenta", .color_prob = 0.74f};
    c34.dist_m = 2.0; c34.bearing_deg = 4;
    ObjectFact c35{.name = "chair_35", .subtype = "chair", .room = "room_1", .in_current_room = true};
    c35.dist_m = 3.0; c35.bearing_deg = -80;
    assert(object_words(c34) == "chair_34, the magenta one ahead"); ++checks;
    assert(object_words(c35) == "chair_35, to the right"); ++checks;
    RememberedObject r{.name = "chair_36", .subtype = "chair", .room = "room_2", .color_name = "yellow"};
    assert(remembered_words(r) == "chair_36, the yellow one I remember in room_2"); ++checks;

    const auto own = claim_readback(HumanClaim{.kind = "owns", .uid = 34, .words = "that's my chair"}, object_words(c34));
    assert(own.contains("chair_34") and own.contains("magenta") and own.contains("yours?")); ++checks;
    const auto dis = claim_readback(HumanClaim{.kind = "disowns", .uid = 34}, object_words(c34));
    assert(dis.contains("NOT yours?")); ++checks;
    assert(claim_readback(HumanClaim{.kind = "name", .value = "Pablo"}, "") == "Your name is Pablo?"); ++checks;
    assert(which_question({"chair_34, the magenta one ahead", "chair_35, to the right"})
           == "Which one do you mean: chair_34, the magenta one ahead, or chair_35, to the right?"); ++checks;
    assert(which_question({"a", "b", "c"}) == "Which one do you mean: a, b, or c?"); ++checks;

    assert(newer_card_takes_the_yes(3, 5) == YesGoesTo::claim); ++checks;
    assert(newer_card_takes_the_yes(6, 5) == YesGoesTo::mission); ++checks;
    assert(not claim_answer_overdue(1000, 1000 + 4999)); ++checks;
    assert(claim_answer_overdue(1000, 1000 + 5001)); ++checks;
    assert(claim_recorded_words(HumanClaim{.kind = "name", .value = "Pablo"}, "") == "I will remember that your name is Pablo."); ++checks;
    assert(claim_recorded_words(HumanClaim{.kind = "owns", .uid = 34}, "chair_34") == "I will remember that chair_34 is yours."); ++checks;
    assert(claim_recorded_words(HumanClaim{.kind = "disowns", .uid = 34}, "chair_34") == "I will remember that chair_34 is not yours."); ++checks;
    {   // I2 + M7: which object a claim is about
        auto lv = [](std::string n, std::uint64_t uid, double b, std::string col = {}) {
            ObjectFact o{.name = std::move(n), .subtype = "chair", .room = "room_1", .in_current_room = true, .color_name = std::move(col)};
            o.dist_m = 2.0; o.bearing_deg = b; o.uid = uid; return o; };
        const std::vector<std::string> none;
        HumanFacts nobody{.role = "developer"};
        const std::vector<ObjectFact> one{lv("chair_1", 1, 3)};
        auto c = claim_candidates("owns", "my chair", nobody, one, {}, none, std::nullopt, 55);   // first-ever claim
        assert(c.size() == 1 and c[0].uid == 1); ++checks;
        HumanFacts owns1{.role = "developer", .owned_uids = {1}};
        const std::vector<ObjectFact> both{lv("chair_1", 1, 3), lv("chair_2", 2, 10)};
        c = claim_candidates("owns", "my chair", owns1, both, {}, none, std::nullopt, 55);       // a SECOND chair
        assert(c.size() == 1 and c[0].uid == 2); ++checks;
        c = claim_candidates("disowns", "my chair", owns1, both, {}, none, std::nullopt, 55);    // disown: the owned one
        assert(c.size() == 1 and c[0].uid == 1); ++checks;
        const std::vector<ObjectFact> col{lv("chair_1", 1, 3, "magenta"), lv("chair_2", 2, 10, "yellow")};
        c = claim_candidates("owns", "the magenta one", nobody, col, {}, none, std::nullopt, 55);
        assert(c.size() == 1 and c[0].name == "chair_1"); ++checks;
        c = claim_candidates("owns", "that", nobody, col, {}, none, std::nullopt, 55);
        assert(c.size() == 2); ++checks;                                                          // two in view: ask
        // I1 in a claim: disowning "my chair" whose name a live chair now carries describes the REMEMBERED one
        HumanFacts ownsA{.role = "developer", .owned_uids = {34}};
        const std::vector<RememberedObject> mem{RememberedObject{.name = "chair_34", .subtype = "chair", .room = "room_2", .uid = 34}};
        c = claim_candidates("disowns", "my chair", ownsA, {lv("chair_34", 99, 0)}, mem, none, std::nullopt, 55);
        assert(c.size() == 1 and c[0].uid == 34 and c[0].words.contains("remember")); ++checks;
    }
    // I3: one claim per model round
    assert(second_claim_this_turn(std::optional<std::uint64_t>(4), 4)); ++checks;
    assert(not second_claim_this_turn(std::optional<std::uint64_t>(3), 4)); ++checks;
    assert(not second_claim_this_turn(std::nullopt, 4)); ++checks;
    std::println("test_human_claim: {} checks passed", checks);
    return 0;
}
