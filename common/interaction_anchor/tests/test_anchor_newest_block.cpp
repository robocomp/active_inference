// read_anchor / read_anchor_latest (interaction_anchor.cpp): is the anchor's newest block DRIVABLE?
// Final review I-2: the owner writes a PREDICTED block stamped `now` every cycle, and when it stopped publishing it wrote the
// wait pose BORN STALE (stamped now − kAnchorStaleMs − 1) — an OLDER stamp than the block written one cycle earlier. The RT
// edge is a CHRONOLOGICAL history ring: cortex re-sorts or DROPS a block older than the newest, so the wait pose never became
// the newest block and the last predicted block read as fresh for ~kAnchorStaleMs after the owner stopped. Now: stamps are
// monotonic, and "not drivable" is the node's `valid` (Anchor::set_drivable). This test reproduces the old encoding's failure
// (as a documented fact about the ring) and checks the new protocol stops the reader at once.
// Build+run (a real DSR graph on a private DDS domain; from common/interaction_anchor/tests):
//   W=$PWD/../../..; g++ -std=c++23 -O0 -g -fPIC -I/usr/include/eigen3 -I$W/common \
//       -DSHADOW_JSON="\"$W/robot_concept/shadow.json\"" $(pkg-config --cflags Qt6Core) \
//       -o /tmp/tanb test_anchor_newest_block.cpp ../interaction_anchor.cpp \
//       -ldsr_api -ldsr_core -lfastcdr -lfastdds $(pkg-config --libs Qt6Core) && /tmp/tanb
#include "interaction_anchor/interaction_anchor.h"

#include <cmath>
#include <memory>
#include <print>
#include <string>
#include <string_view>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

using rc::interaction::kAnchorStaleMs;
using rc::interaction::RelPose;

RelPose pose(float x, float y, std::int64_t stamp)
{
    RelPose r;
    r.x = {x, y, 0.f};
    r.P = Eigen::Matrix3f::Identity() * 1e-4f;
    r.stamp_ms = stamp;
    return r;
}

struct Scene
{
    std::shared_ptr<DSR::DSRGraph> G;
    std::uint64_t door = 0, aff = 0;
};
Scene scene(int k)
{
    Scene s;
    s.G = std::make_shared<DSR::DSRGraph>("anchor_newest_block_test_" + std::to_string(k), 977 + k, std::string(SHADOW_JSON),
                                          true, static_cast<int8_t>(67 + k), DSR::Queue);
    DSR::Node door = DSR::Node::create<object_node_type>("door_" + std::to_string(k));
    s.G->add_or_modify_attrib_local<width_m_att>(door, 0.9f);
    s.door = s.G->insert_node(door).value_or(0);
    DSR::Node aff = DSR::Node::create<affordance_node_type>("aff_door_" + std::to_string(k) + "_cross");
    s.aff = s.G->insert_node(aff).value_or(0);
    return s;
}
}   // namespace

int main()
{
    const std::int64_t t0 = 5'000'000;

    // ── 1. The OLD encoding, as a fact about the ring: a born-stale block written AFTER a fresh one is not the newest. ──
    {
        auto s = scene(0);
        CHECK(s.door != 0 and s.aff != 0);
        auto a = rc::interaction::Anchor::create(*s.G, s.door, s.aff, 977, pose(0.f, 2.f, t0));
        CHECK(a.has_value());
        if (not a.has_value()) return 1;
        CHECK(a->write(*s.G, pose(0.f, 1.9f, t0 + 50)));                              // predicted, stamped now
        CHECK(a->write(*s.G, pose(0.3f, 1.5f, t0 + 100 - kAnchorStaleMs - 1)));       // "born stale", one cycle later
        const auto latest = rc::interaction::read_anchor_latest(*s.G, a->id());
        CHECK(latest.has_value() and latest->first.stamp_ms == t0 + 50);             // the predicted block still wins
        a->remove(*s.G);
    }

    // ── 2. The NEW protocol: monotonic stamps + the drivable mark. ───────────────────────────────────────────────────────
    {
        auto s = scene(1);
        CHECK(s.door != 0 and s.aff != 0);
        auto a = rc::interaction::Anchor::create(*s.G, s.door, s.aff, 977, pose(0.f, 2.f, t0 - kAnchorStaleMs - 1));
        CHECK(a.has_value());
        if (not a.has_value()) return 1;
        CHECK(not rc::interaction::read_anchor(*s.G, a->id(), t0).has_value());    // born not drivable

        // Seen, then PREDICTED: block first, then the drivable mark — fresh.
        std::int64_t now = t0;
        for (int k = 0; k < 7; ++k, now += 50)   // more cycles than the ring has slots: it wraps
        {
            CHECK(a->write(*s.G, pose(0.f, 2.f - 0.01f * k, now)) and a->set_drivable(*s.G, true));
            const auto r = rc::interaction::read_anchor(*s.G, a->id(), now);
            CHECK(r.has_value() and r->first.stamp_ms == now);
            CHECK(r.has_value() and std::abs(r->first.x.y() - (2.f - 0.01f * k)) < 1e-5f);
        }
        // The review's scenario: one cycle (50 ms) after the last predicted block the mode drops to None. The mark is
        // cleared FIRST — between the two writes the reader already refuses it — then the wait pose is stamped `now`.
        CHECK(a->set_drivable(*s.G, false));
        CHECK(not rc::interaction::read_anchor(*s.G, a->id(), now).has_value());
        CHECK(a->write(*s.G, pose(0.3f, 1.5f, now)));
        {
            const auto latest = rc::interaction::read_anchor_latest(*s.G, a->id());
            CHECK(latest.has_value() and latest->first.stamp_ms == now);              // the wait pose is the newest block
            CHECK(latest.has_value() and std::abs(latest->first.x.x() - 0.3f) < 1e-5f);
            CHECK(not rc::interaction::read_anchor(*s.G, a->id(), now).has_value()); // ...and NOT drivable
            CHECK(not rc::interaction::read_anchor(*s.G, a->id(), now + 1).has_value());
        }
        for (int k = 0; k < 7; ++k)   // rewritten every cycle: never drivable
        {
            now += 50;
            CHECK(a->set_drivable(*s.G, false) and a->write(*s.G, pose(0.3f, 1.5f, now)));
            CHECK(not rc::interaction::read_anchor(*s.G, a->id(), now).has_value());
        }
        // Seen again: block, then mark — fresh again.
        now += 50;
        CHECK(a->write(*s.G, pose(0.f, 1.2f, now)) and a->set_drivable(*s.G, true));
        {
            const auto r = rc::interaction::read_anchor(*s.G, a->id(), now);
            CHECK(r.has_value() and r->first.stamp_ms == now and std::abs(r->first.x.y() - 1.2f) < 1e-5f);
        }
        // A DEAD producer (nothing written): the stamp rule still retires it after kAnchorStaleMs.
        CHECK(rc::interaction::read_anchor(*s.G, a->id(), now + kAnchorStaleMs).has_value());
        CHECK(not rc::interaction::read_anchor(*s.G, a->id(), now + kAnchorStaleMs + 1).has_value());
        const auto anchor_id = a->id();
        a->remove(*s.G);
        CHECK(not rc::interaction::read_anchor_latest(*s.G, anchor_id).has_value());
    }

    std::println("test_anchor_newest_block: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
