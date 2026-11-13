// Timing (no assertions): the per-cycle memory publish for 10 doors, nothing changed vs one door changed, against a realistic
// 50-view ledger. "old" = serialise every cycle (the first version); "new" = serialise only when ViewLedger::version() moved.
#include "core/view_ledger.h"
#include "door_memory_note.h"
#include "door_memory_graph.h"   // changed_since / apply_changes (the DSR write itself is not timed: same call, fewer fields)

#include <chrono>
#include <cstdio>
#include <vector>

using namespace rc::door;
using Clock = std::chrono::steady_clock;

static core::ViewLedger make_ledger()
{
    core::ViewLedger L;
    const core::DetectorModel m{};
    for (int i = 0; i < 50; ++i)
    {
        const auto v = L.observe({3.f * std::cos(0.12f * i), 3.f * std::sin(0.12f * i)}, {0.f, 0.f}, {0.f, 1.f}, 0.5f, 0.3f, i);
        if (v == core::ViewLedger::npos) continue;
        for (int k = 0; k < 5; ++k) L.add_bernoulli(v, core::Channel::Zed, 0.8f, m, 0.9f);
        L.add_geometry(v, 0.4f, 1.f, 0.f, 1.f, 0.2f);
        L.recompute(v, true);
    }
    return L;
}

int main()
{
    constexpr int kDoors = 10, kCycles = 2000;
    std::vector<core::ViewLedger> led(kDoors, make_ledger());
    std::printf("ledger views = %zu, string bytes = %zu\n", led[0].views().size(), led[0].serialise().size());
    struct Pub { memnote::NoteAttrs last; std::uint64_t ver = ~0ull; bool has = false; };
    const auto run = [&](bool changed, bool cached) {
        std::vector<Pub> pub(kDoors);
        const auto cycle = [&](int c) {
            for (int d = 0; d < kDoors; ++d)
            {
                if (changed and d == 0 and c > 0) { led[0].add_actuation(0, 0.001f); }
                IdentityRow r; r.exist_logodds = 1.f; r.visit = 1;
                const bool moved = not cached or not pub[d].has or pub[d].ver != led[d].version();
                if (moved) r.ledger = led[d].serialise();
                const auto now = memnote::note_of(r, 0.03f);
                const auto df = memnote::changed_since(now, pub[d].has ? &pub[d].last : nullptr);
                if (memnote::any_change(df)) memnote::apply_changes(pub[d].last, df);
                pub[d].ver = led[d].version(); pub[d].has = true;
            }
        };
        cycle(0);
        const auto t0 = Clock::now();
        for (int c = 1; c <= kCycles; ++c) cycle(c);
        return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / kCycles;
    };
    std::printf("10 doors, per cycle: serialise-always unchanged %.1f us | cached unchanged %.1f us\n", run(false, false), run(false, true));
    std::printf("10 doors, per cycle: serialise-always 1 door changed %.1f us | cached 1 door changed %.1f us\n", run(true, false), run(true, true));
}
