/*
 * merge_instances_test.cpp — the pairwise merge sweep. Standalone:
 *
 *     g++ -std=c++23 -O1 merge_instances_test.cpp -o merge_instances_test && ./merge_instances_test
 *
 * The sweep mutates the very map it is walking, which is where a duplicated-seven-times loop earns its
 * test: the id snapshot, the re-`find` guard, and `if (drop == ids[i]) break;` are each load-bearing and
 * each invisible in a passing run. The last one especially — dropping `i` itself and then continuing to
 * compare `i` against `j+1..n` reads fine and silently merges into a corpse.
 *
 * EvidenceGlobals is mirrored rather than included so this stays dependency-free (the real header pulls Qt).
 */

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rc { struct EvidenceGlobals { int merges = 0; long merges_cum = 0; }; }

// ── the sweep, mirrored from merge_instances.h ────────────────────────────────────────────────────
namespace rc::track
{
template <class Insts, class DecideFn, class ObservedFn, class ApplyFn>
int merge_overlapping(Insts& insts, rc::EvidenceGlobals& ev,
                      DecideFn&& decide, ObservedFn&& observed, ApplyFn&& apply)
{
    if (insts.size() < 2) return 0;
    std::vector<std::uint64_t> ids;
    ids.reserve(insts.size());
    for (auto& [id, _] : insts) ids.push_back(id);
    int merged = 0;
    std::unordered_set<std::uint64_t> removed;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (removed.count(ids[i])) continue;
        for (std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if (removed.count(ids[j])) continue;
            const auto ia = insts.find(ids[i]), ib = insts.find(ids[j]);
            if (ia == insts.end() or ib == insts.end()) continue;
            auto payload = decide(ia->second, ib->second);
            if (not payload.has_value()) continue;
            const bool keep_i = observed(ia->second) >= observed(ib->second);
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            apply(keep, drop, keep_i ? ia->second : ib->second,
                              keep_i ? ib->second : ia->second, *payload);
            removed.insert(drop);
            ++merged;
            if (drop == ids[i]) break;
        }
    }
    ev.merges += merged; ev.merges_cum += merged;
    return merged;
}
}  // namespace rc::track

// ── a toy instance: a 1-D interval with an observation count ──────────────────────────────────────
struct Inst { float x = 0.0f; int frames = 0; float len = 1.0f; };
using Map = std::unordered_map<std::uint64_t, Inst>;

static int fails = 0;
static void ck(bool c, const std::string& what) { if (not c) { std::printf("FAIL: %s\n", what.c_str()); ++fails; } }

// merge when centres are within 1.0; retire = erase.
static int sweep(Map& m, rc::EvidenceGlobals& ev, std::vector<std::string>* trace = nullptr)
{
    return rc::track::merge_overlapping(
        m, ev,
        [](const Inst& a, const Inst& b) -> std::optional<float>
        { const float d = a.x - b.x; return (d * d <= 1.0f) ? std::optional<float>(d) : std::nullopt; },
        [](const Inst& i) { return i.frames; },
        [&](std::uint64_t keep, std::uint64_t drop, Inst& keeper, const Inst& dropped, float)
        {
            if (trace) trace->push_back(std::to_string(drop) + "->" + std::to_string(keep));
            keeper.len += dropped.len;          // a graft: read `dropped` BEFORE retiring it
            m.erase(drop);
        });
}

int main()
{
    // ── the plain case ────────────────────────────────────────────────────────────────────────────
    {
        Map m{{1, {0.0f, 5, 1}}, {2, {0.2f, 9, 1}}, {3, {50.0f, 3, 1}}};
        rc::EvidenceGlobals ev;
        const int n = sweep(m, ev);
        ck(n == 1, "two near instances merge, the far one is untouched");
        ck(m.size() == 2 and m.count(2) and m.count(3), "the MORE-OBSERVED id survives (9 frames beats 5)");
        ck(m.at(2).len == 2.0f, "the graft ran: `dropped` was readable inside apply");
        ck(ev.merges == 1 and ev.merges_cum == 1, "★the counter is filled BY THE SWEEP — the thing three "
                                                  "agents forgot to do at the call site");
    }

    // ── ★the `break`: when `i` ITSELF is the one dropped ──────────────────────────────────────────
    // id 1 has the FEWEST frames, so pairing it with 2 drops 1 — and 1 must not go on to be compared
    // against 3. Without the break, `insts.find(ids[i])` is the guard that saves it; with BOTH removed,
    // nothing does. The trace pins the exact sequence, not just the final size.
    {
        Map m{{1, {0.0f, 1, 1}}, {2, {0.2f, 9, 1}}, {3, {0.4f, 5, 1}}};
        rc::EvidenceGlobals ev;
        std::vector<std::string> trace;
        const int n = sweep(m, ev, &trace);
        ck(n == 2 and m.size() == 1 and m.count(2), "three mutually-near instances collapse onto the best");
        ck(trace.size() == 2, "exactly two merges are applied, never a third against a retired instance");
        for (const auto& t : trace)
            ck(t.substr(t.find("->") + 2) == "2", "every merge targets the surviving id, never a corpse: " + t);
        ck(m.at(2).len == 3.0f, "both grafts landed on the survivor");
    }

    // ── nothing to do ─────────────────────────────────────────────────────────────────────────────
    {
        Map m{{1, {0.0f, 5, 1}}};
        rc::EvidenceGlobals ev;
        ck(sweep(m, ev) == 0 and ev.merges == 0, "a single instance is never merged and never counted");
        Map empty;
        ck(sweep(empty, ev) == 0, "an empty map is not a special case at the call site");
        Map far{{1, {0.0f, 5, 1}}, {2, {50.0f, 5, 1}}};
        ck(sweep(far, ev) == 0 and far.size() == 2, "instances that decide() rejects are both kept");
    }

    // ── the decision may REFUSE on evidence, not only on distance ─────────────────────────────────
    // cabinet gates merging on both runs having a measured axis; an unfit instance must never be fused.
    {
        Map m{{1, {0.0f, 5, 1}}, {2, {0.2f, 0, 1}}};      // frames == 0 ⇒ "not yet fitted"
        rc::EvidenceGlobals ev;
        const int n = rc::track::merge_overlapping(
            m, ev,
            [](const Inst& a, const Inst& b) -> std::optional<float>
            { if (a.frames == 0 or b.frames == 0) return std::nullopt; return 0.0f; },
            [](const Inst& i) { return i.frames; },
            [&](std::uint64_t, std::uint64_t drop, Inst&, const Inst&, float) { m.erase(drop); });
        ck(n == 0 and m.size() == 2, "an evidence gate inside decide() blocks the merge, sweep obeys it");
    }

    // ── ties ──────────────────────────────────────────────────────────────────────────────────────
    {
        Map m{{1, {0.0f, 7, 1}}, {2, {0.2f, 7, 1}}};
        rc::EvidenceGlobals ev;
        ck(sweep(m, ev) == 1 and m.size() == 1, "an equal-evidence tie still resolves to exactly one survivor");
    }

    // ── the counter ACCUMULATES across cycles ─────────────────────────────────────────────────────
    {
        rc::EvidenceGlobals ev;
        for (int cycle = 0; cycle < 3; ++cycle)
        {
            ev.merges = 0;                                   // agents zero the per-cycle field, not the cum
            Map m{{1, {0.0f, 5, 1}}, {2, {0.2f, 9, 1}}};
            sweep(m, ev);
        }
        ck(ev.merges == 1 and ev.merges_cum == 3, "per-cycle resets, cumulative does not");
    }

    std::printf(fails ? "merge_instances: %d FAILED\n" : "merge_instances: all checks passed\n", fails);
    return fails ? 1 : 0;
}
