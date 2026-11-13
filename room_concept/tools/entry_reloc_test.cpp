// entry_reloc_test — the ENTRY RELOCALISATION ends on the protocol, not only on geometry (exploration-tour spike, 2026-11-05).
// explore_spike1: after crossing into an unknown room, afford_room was sent to the room's centre; the controller HANDED IT
// BACK ("no route … the body cannot stand there") and drove to repaired standpoints 0.65 m away, while the entry relocalisation
// waited for the robot to come within half its diagonal of the original centre. It never did: the layout explorer stayed locked
// out for 750 s, `finished` never came, and the crossing-born room was never promoted.
// Build + run (from room_concept/):
//   g++ -std=c++23 -O1 tools/entry_reloc_test.cpp -o /tmp/entry_reloc_test && /tmp/entry_reloc_test
#include <cmath>
#include <print>

#include "../src/entry_reloc.h"

int main()
{
    int fails = 0, checks = 0;
    const auto check = [&](bool ok, const char* what) { ++checks; if (not ok) { ++fails; std::println("FAIL: {}", what); } };
    using rc::entry::still_relocalising;
    check(still_relocalising(false, false), "not at the centre and the claim still open: keep relocalising");
    check(not still_relocalising(true, false), "at the centre: the room's planners take over");
    check(not still_relocalising(false, true), "the consumer completed or handed back the claim: the planners take over (explore_spike1)");
    check(not still_relocalising(true, true), "both: done");
    // clean47: the centre is worth the pose information it can still add — zero when the pose is as good as typical
    check(not still_relocalising(false, false, 0.f), "a centre worth 0 nats ends the relocalisation (clean47)");
    {
        rc::entry::PoseReference R;
        check(R.gain(-20.f) == 0.f, "no reference yet: no evidence the pose is degraded, gain 0");
        for (float ld : {-25.f, -24.f, -23.f, -10.f, -24.5f}) R.add(ld);   // one spike (sigma_theta swings): the median ignores it
        R.close_room();
        check(R.reference.has_value() and *R.reference == -24.f, "the reference is the previous room's median log-det");
        check(R.samples.empty(), "the new room starts with no samples");
        check(R.gain(-24.5f) == 0.f, "a pose as good as typical is worth nothing");
        check(std::abs(R.gain(-18.f) - 3.f) < 1e-5f, "a degraded pose (ln det 6 above typical) is worth 3 nats");
        R.close_room();
        check(R.reference.has_value() and *R.reference == -24.f, "a room that gave no samples keeps the old reference");
    }
    std::println("{}/{} checks passed", checks - fails, checks);
    return fails == 0 ? 0 : 1;
}
