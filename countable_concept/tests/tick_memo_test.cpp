// Per-tick memo (tick_memo.h): same key => one underlying call; clear() (a new tick) => recompute; failures not stored.
#include "tick_memo.h"
#include <cstdio>
#include <cstdlib>

static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main()
{
    rc::TickMemo<double> memo;
    int calls = 0;
    bool hit = false;
    const auto q = [&] { ++calls; return std::optional<double>{3.5}; };

    CHECK(memo.get("room_1|body|0", q, hit).value() == 3.5 and not hit and calls == 1);
    CHECK(memo.get("room_1|body|0", q, hit).value() == 3.5 and hit and calls == 1);   // same key: no new call
    CHECK(memo.get("room_1|body|7", q, hit).has_value() and not hit and calls == 2);  // other stamp: own entry
    CHECK(memo.get("room_2|body|0", q, hit).has_value() and not hit and calls == 3);  // other frame: own entry
    memo.clear();                                                                      // new tick
    CHECK(memo.get("room_1|body|0", q, hit).has_value() and not hit and calls == 4);

    int fcalls = 0;
    const auto fail = [&] { ++fcalls; return std::optional<double>{}; };
    CHECK(not memo.get("x", fail, hit).has_value() and fcalls == 1);
    CHECK(not memo.get("x", fail, hit).has_value() and fcalls == 2);                  // a failure is retried
    if (fails == 0) std::puts("tick_memo_test OK");
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
