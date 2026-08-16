/*
 * Standalone runner for rc::KitchenManager::self_test() (cabinet_kitchen.h).
 *
 * ★WHY THIS FILE EXISTS. The self-test had NO caller — `grep -rn "KitchenManager::self_test"` over the whole
 * tree found only its own definition. That is how it came to hold an expectation that ENCODED a defect
 * ("north run fills corner-to-corner to both shared vertices"), and how every one of its corner-fill checks
 * came to select a run by `std::abs(b.cy - 2.45f) < 0.2f` — a FRONT-FACE coordinate — while KitchenBox.cx/cy
 * is the box CENTRE, half a depth further back. The predicates matched nothing, so the test printed
 * "all checks passed" while asserting nothing at all about the corner it was written to guard.
 *
 * Deliberately NOT wired into CMakeLists: the agent must not run self-tests at startup, and this needs no
 * DSR, no Qt and no DDS. One command, from this directory:
 *
 *   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I../src kitchen_selftest.cpp -o kitchen_selftest && ./kitchen_selftest
 *
 * Exit status is the result; it prints one FAIL line per failed check.
 */
#include <clocale>
#include <cstdio>

#include "cabinet_kitchen.h"

int main()
{
    // CLAUDE.md: a harness has no Qt, so it stays in the "C" locale while the agent runs under es_ES. This
    // test parses no files, but matching the agent's locale keeps the two answering the same question.
    std::setlocale(LC_ALL, "");
    const bool ok = rc::KitchenManager::self_test();
    std::printf("kitchen self_test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
