// promotion_due: the promotion trigger is room_concept's robot RT room differing from `current`.
//   run from the component root:  ./bin/promotion_trigger_test
#include "../src/room_eviction.h"
#include <QCoreApplication>
#include <cstdio>
#include <cstdlib>
static int failures = 0;
static void check(bool ok, const char *what) { if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }
int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    check(ltsm::promotion_due(3, 1), "(room_3, room_1) due");
    check(not ltsm::promotion_due(1, 1), "(room_1, room_1) not due");
    check(not ltsm::promotion_due(std::nullopt, 1), "(none, room_1) not due");
    check(ltsm::promotion_due(3, std::nullopt), "(room_3, none) due");
    std::printf("%s\n", failures ? "FAILED" : "PASSED");
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
