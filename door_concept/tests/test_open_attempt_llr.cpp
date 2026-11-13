// open_attempt_llr: one `open` claim's evidence for the door's existence.
// g++ -std=c++23 -O1 -I/usr/include/eigen3 -Isrc tests/test_open_attempt_llr.cpp -o /tmp/toal && /tmp/toal   (run from door_concept/)
#include "../src/core/existence_update.h"
#include <cstdlib>
#include <print>
using rc::door::core::open_attempt_llr;
static int failures = 0;
#define CHECK(x) do { if (not (x)) { ++failures; std::println("FAIL line {}: {}", __LINE__, #x); } } while (false)
int main()
{
    const float l = 3.0f;
    CHECK(open_attempt_llr(true, false, true, l) == l);       // asked, seen open: it exists
    CHECK(open_attempt_llr(true, false, false, l) == -l);     // asked, never seen open
    CHECK(open_attempt_llr(false, true, false, l) == -l);     // nothing that opens is here (clean5's door_4)
    CHECK(open_attempt_llr(false, false, false, l) == 0.0f);  // no request made (unreachable, infeasible): says nothing
    std::println("{}", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
