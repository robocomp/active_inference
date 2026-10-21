// test_rt_ring.cpp — decoding the NEWEST slot of a cortex RT history ring (src/rt_ring.h).
// Build + run (from controller/):
//   g++ -std=c++23 -O0 -g -Wall -Wextra -Isrc tests/test_rt_ring.cpp -o /tmp/trr && /tmp/trr
// ★ WHY: cortex stores rt_head_index as an offset into the PACKED 3-vectors of the slot it writes NEXT
//   (dsr_rt_api.cpp: write slot k, then head = 3*(k+1)), not as an index into rt_timestamps. Reading
//   rt_timestamps[head] picked an old slot on 8 of every 25 writes (25-slot ring) and reported a pose up
//   to 1.2 s stale; the policy then projected the robot 0.3 m into a doorway jamb (rtD1a, 2026-10-21).
#include <cstdint>
#include <print>
#include <string>
#include <vector>

#include "../src/rt_ring.h"

static int checks = 0, failures = 0;
static void check(bool ok, const std::string &what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL: {}", what);
    }
}

int main()
{
    // Mirror cortex's writer: slot = (head/3) % n is written, then head = 3*slot + 3 (head may reach 3*n).
    constexpr std::size_t n = 25;
    std::vector<std::uint64_t> stamps(n, 0);
    std::size_t head = 0;
    for (std::uint64_t t = 1; t <= 3 * n + 7; ++t)
    {
        const std::size_t slot = (head / 3) % n;
        stamps[slot] = 1000 + 50 * t;
        head = 3 * slot + 3;
        const auto newest = rc::rt::latest_slot_from_head(head, n);
        check(newest == slot, std::format("write {}: newest slot {} (expected {})", t, newest, slot));
        check(stamps[newest] == 1000 + 50 * t, std::format("write {}: newest stamp is the last written", t));
    }
    // The wrap value 3*n decodes to the last slot.
    check(rc::rt::latest_slot_from_head(3 * n, n) == n - 1, "head 3n -> slot n-1");
    check(rc::rt::latest_slot_from_head(0, n) == n - 1, "head 0 -> slot n-1");
    check(rc::rt::latest_slot_from_head(3, n) == 0, "head 3 -> slot 0");

    std::println("{} checks, {} failures — {}", checks, failures, failures == 0 ? "ALL PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}
