// g++ -std=c++23 -O2 common/room_resolve/room_resolve_test.cpp -o /tmp/room_resolve_test && /tmp/room_resolve_test
//
// The room-resolution RULE as a pure function (room_resolve_rule.h), pinned without a DSR graph:
//   rule 1  exactly one `current` edge wins, whatever rooms exist;
//   rule 2  otherwise the unique NON-proto room;
//   rule 3  (2026-10-09) otherwise, ONLY when there is no `current` edge at all and NO non-proto room, a
//           unique proto room — room_concept's start-up proto-room, so the controller and residual can work
//           with it before any room is surveyed;
//   else    nothing.
// The door-crossing case is the one rule 3 must never change: an old real room plus a crossing proto ⇒ the
// real room, until ltsm_agent moves `current`.
#include "room_resolve_rule.h"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace
{
    int fails = 0;
    void expect(const char* what, std::optional<std::uint64_t> got, std::optional<std::uint64_t> want)
    {
        const bool ok = got == want;
        fails += ok ? 0 : 1;
        std::printf("%s %-62s got=%s want=%s\n", ok ? "PASS" : "FAIL", what,
                    got ? std::to_string(*got).c_str() : "none", want ? std::to_string(*want).c_str() : "none");
    }
}   // namespace

int main()
{
    using rc::room::RoomFact;
    using rc::room::CurrentEdges;
    using rc::room::resolve_current_room;
    const CurrentEdges none{};
    const CurrentEdges cur_at_7{.count = 1, .room = 7};
    const CurrentEdges two_current{.count = 2, .room = 7};

    // ── existing behaviour, pinned ──────────────────────────────────────────────────────────────────
    expect("rule 1: current edge wins over a non-proto room",
           resolve_current_room(cur_at_7, std::vector<RoomFact>{{5, false}, {7, true}}), 7);
    expect("rule 1: current edge may point at a proto",
           resolve_current_room(cur_at_7, std::vector<RoomFact>{{7, true}}), 7);
    expect("rule 2: unique non-proto room, no current edge",
           resolve_current_room(none, std::vector<RoomFact>{{5, false}}), 5);
    expect("rule 2: non-proto beats a proto (door crossing)",
           resolve_current_room(none, std::vector<RoomFact>{{5, false}, {9, true}}), 5);
    expect("ambiguous: two non-proto rooms, no current",
           resolve_current_room(none, std::vector<RoomFact>{{5, false}, {6, false}}), std::nullopt);
    expect("ambiguous: two non-proto rooms + a proto, no current",
           resolve_current_room(none, std::vector<RoomFact>{{5, false}, {6, false}, {9, true}}), std::nullopt);
    expect("no rooms at all",
           resolve_current_room(none, std::vector<RoomFact>{}), std::nullopt);
    // Two `current` edges: rule 1 abstains; the pre-existing code then fell through to rule 2. Pinned as is.
    expect("two current edges fall through to rule 2 (unchanged)",
           resolve_current_room(two_current, std::vector<RoomFact>{{5, false}, {9, true}}), 5);

    // ── rule 3 ──────────────────────────────────────────────────────────────────────────────────────
    expect("rule 3: unique proto, no current, no non-proto room",
           resolve_current_room(none, std::vector<RoomFact>{{9, true}}), 9);
    expect("rule 3 refuses two protos",
           resolve_current_room(none, std::vector<RoomFact>{{9, true}, {10, true}}), std::nullopt);
    expect("rule 3 never fires with a non-proto room present",
           resolve_current_room(none, std::vector<RoomFact>{{5, false}, {6, false}, {9, true}}), std::nullopt);
    expect("rule 3 never fires with two current edges",
           resolve_current_room(two_current, std::vector<RoomFact>{{9, true}}), std::nullopt);

    std::printf("%s: %d failure(s)\n", fails == 0 ? "ALL PASS" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
